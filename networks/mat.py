import sys

import numpy as np

sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_utils import misc
from torch_utils import persistence
from networks.basic_module import FullyConnectedLayer, Conv2dLayer, MappingNet, MinibatchStdLayer, DisFromRGB, DisBlock, StyleConv, ToRGB, get_style_code
from networks.ffc import ResnetBlock_remove_IN, FourierUnitN_NoNorm


# @misc.profiled_function
# def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
#     NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
#     return NF[2 ** stage]

@misc.profiled_function
def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 32, 256: 64, 128: 128, 64: 256, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2 ** stage]


@persistence.persistent_class
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(in_features=in_features, out_features=hidden_features, activation='lrelu')
        self.fc2 = FullyConnectedLayer(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@persistence.persistent_class
class Conv2dLayerPartial(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
                 trainable=True,  # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down, resample_filter,
                                conv_clamp, trainable)

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size ** 2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding)
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None


@persistence.persistent_class
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       down=down,
                                       )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistence.persistent_class
class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       up=up,
                                       )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask


@persistence.persistent_class
class ToToken(nn.Module):
    def __init__(self, in_channels=3, dim=128, kernel_size=5, stride=1):
        super().__init__()

        self.proj = Conv2dLayerPartial(in_channels=in_channels, out_channels=dim, kernel_size=kernel_size, activation='lrelu')

    def forward(self, x, mask):
        x, mask = self.proj(x, mask)

        return x, mask


# ----------------------------------------------------------------------------

@persistence.persistent_class
class EncFromRGB(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=1,
                                 activation=activation,
                                 )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


@persistence.persistent_class
class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=2,
                                 )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


class FFCBlock(nn.Module):

    def __init__(self, in_channels, out_channels, activation, stride=1, groups=1):
        # bn_layer not used
        super(FFCBlock, self).__init__()
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels // 2,
                                 kernel_size=1, bias=False, activation=activation)
        self.fu = FourierUnitN_NoNorm(out_channels // 2, out_channels // 2, groups)
        self.conv2 = Conv2dLayer(in_channels=out_channels // 2,
                                 out_channels=out_channels,
                                 kernel_size=1, bias=False, activation=activation)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)
        output = self.conv2(x + output)

        return output


@persistence.persistent_class
class ConvFFCBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, activation):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=2)
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 activation=activation)
        self.ffc = FFCBlock(out_channels, out_channels, activation)

    def forward(self, x):
        x = self.conv0(x)
        x_conv = self.conv1(x)
        x_ffc = self.ffc(x)
        x = x_conv + x_ffc

        return x


def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x


@persistence.persistent_class
class Encoder(nn.Module):
    def __init__(self, res_log2, img_channels, activation, patch_size=5, channels=16, drop_path_rate=0.1,
                 enc_ffc=False, min_size=4):
        super().__init__()

        self.resolution = []

        for idx, i in enumerate(range(res_log2, min_size - 1, -1)):  # from input size to 2^minsize
            res = 2 ** i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels + 1, nf(i), activation)
            else:
                if enc_ffc:
                    block = ConvFFCBlockDown(nf(i + 1), nf(i), activation)
                else:
                    block = ConvBlockDown(nf(i + 1), nf(i), activation)
            setattr(self, 'EncConv_Block_%dx%d' % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, 'EncConv_Block_%dx%d' % (res, res))(x)
            out[res_log2] = x

        return out


@persistence.persistent_class
class FFCEncoder(nn.Module):
    def __init__(self, res_log2, img_channels, activation, patch_size=5, channels=16, drop_path_rate=0.1):
        super().__init__()

        self.resolution = []

        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2 ** i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels + 1, nf(i), activation)
            else:
                block = ConvBlockDown(nf(i + 1), nf(i), activation)
            setattr(self, 'EncConv_Block_%dx%d' % (res, res), block)

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))
            x = getattr(self, 'EncConv_Block_%dx%d' % (res, res))(x)
            out[res_log2] = x

        return out


@persistence.persistent_class
class ToStyle(nn.Module):
    def __init__(self, in_channels, out_channels, activation, drop_rate):
        super().__init__()
        self.conv = nn.Sequential(
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
            Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, activation=activation, down=2),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(in_features=in_channels,
                                      out_features=out_channels,
                                      activation=activation)
        # self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))
        # x = self.dropout(x)

        return x


@persistence.persistent_class
class ToStyle4x4(nn.Module):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__()
        self.fc = FullyConnectedLayer(in_features=in_channels,
                                      out_features=out_channels,
                                      activation=activation)

    def forward(self, x):
        x = self.fc(x.flatten(start_dim=1))

        return x


@persistence.persistent_class
class DecBlockFirstV2(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 )
        self.conv1 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2 ** res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


@persistence.persistent_class
class DecBlockFirstV3(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.fc = FullyConnectedLayer(in_features=in_channels * 2,
                                      out_features=in_channels * 4 * 4,
                                      activation=activation)
        self.conv1 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2 ** res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.fc(x).reshape(x.shape[0], -1, 4, 4)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, 0], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style, skip=None)

        return x, img


# ----------------------------------------------------------------------------

@persistence.persistent_class
class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2 ** res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2 ** res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, ws, gs, E_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + E_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistence.persistent_class
class Decoder(nn.Module):
    def __init__(self, res_log2, activation, style_dim, use_noise, demodulate, img_channels, min_size=4):
        super().__init__()
        self.min_size = min_size
        self.Dec_16x16 = DecBlockFirstV2(min_size, nf(min_size), nf(min_size), activation,
                                         style_dim, use_noise, demodulate, img_channels)
        for res in range(min_size + 1, res_log2 + 1):
            setattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res),
                    DecBlock(res, nf(res - 1), nf(res), activation, style_dim, use_noise, demodulate, img_channels))
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x, img = self.Dec_16x16(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(self.min_size + 1, self.res_log2 + 1):
            block = getattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img


@persistence.persistent_class
class Decoder4x4(nn.Module):
    def __init__(self, res_log2, activation, style_dim, use_noise, demodulate, img_channels, min_size=2):
        super().__init__()
        self.min_size = min_size
        self.Dec_4x4 = DecBlockFirstV3(min_size, nf(min_size), nf(min_size), activation,
                                       style_dim, use_noise, demodulate, img_channels)
        for res in range(min_size + 1, res_log2 + 1):
            setattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res),
                    DecBlock(res, nf(res - 1), nf(res), activation, style_dim, use_noise, demodulate, img_channels))
        self.res_log2 = res_log2

    def forward(self, x, ws, gs, E_features, noise_mode='random'):
        x, img = self.Dec_4x4(x, ws, gs, E_features, noise_mode=noise_mode)
        for res in range(self.min_size + 1, self.res_log2 + 1):
            block = getattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res))
            x, img = block(x, img, ws, gs, E_features, noise_mode=noise_mode)

        return img


@persistence.persistent_class
class DecStyleBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, activation, style_dim, use_noise, demodulate, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2 ** res,
                               kernel_size=3,
                               up=2,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2 ** res,
                               kernel_size=3,
                               use_noise=use_noise,
                               activation=activation,
                               demodulate=demodulate,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           kernel_size=1,
                           demodulate=False,
                           )

    def forward(self, x, img, style, skip, noise_mode='random'):
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistence.persistent_class
class SynthesisFFCNet(nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels=3,  # Number of color channels.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 drop_rate=0.5,
                 use_noise=True,
                 demodulate=True,
                 style_ffc=False):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2
        self.style_ffc = style_ffc
        if self.style_ffc:
            self.num_layers += 8

        self.enc = Encoder(resolution_log2, img_channels, activation, patch_size=5, channels=16)

        if self.style_ffc:
            self.ffc_32x32 = []
            for _ in range(4):
                self.ffc_32x32.append(ResnetBlock_remove_IN(nf(5)))
                self.ffc_32x32.append(StyleConv(in_channels=nf(5),
                                                out_channels=nf(5),
                                                style_dim=w_dim,
                                                resolution=32,
                                                kernel_size=3,
                                                use_noise=use_noise,
                                                activation=activation,
                                                demodulate=demodulate))
            self.ffc_32x32 = nn.ModuleList(self.ffc_32x32)
            self.ffc_16x16 = []
            for _ in range(4):
                self.ffc_16x16.append(ResnetBlock_remove_IN(nf(4)))
                self.ffc_16x16.append(StyleConv(in_channels=nf(4),
                                                out_channels=nf(4),
                                                style_dim=w_dim,
                                                resolution=16,
                                                kernel_size=3,
                                                use_noise=use_noise,
                                                activation=activation,
                                                demodulate=demodulate))
            self.ffc_16x16 = nn.ModuleList(self.ffc_16x16)
        else:
            self.ffc_32x32 = nn.Sequential(*[ResnetBlock_remove_IN(nf(5)) for _ in range(4)])
            self.ffc_16x16 = nn.Sequential(*[ResnetBlock_remove_IN(nf(4)) for _ in range(4)])
        self.to_square = FullyConnectedLayer(in_features=w_dim, out_features=16 * 16, activation=activation)
        self.to_style = ToStyle(in_channels=nf(4), out_channels=nf(2) * 2, activation=activation, drop_rate=drop_rate)
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(resolution_log2, activation, style_dim, use_noise, demodulate, img_channels)

    def forward(self, img, mask, ws, noise_mode='random'):
        # encoder
        masked_img = img * (1 - mask)
        x = torch.cat([masked_img, mask], dim=1)
        E_features = self.enc(x)
        if self.style_ffc:
            wfs = ws[:, :8]
            ws = ws[:, 8:]

        # FFC
        if self.style_ffc:
            for i in range(len(self.ffc_32x32)):
                if i % 2 == 0:
                    E_features[5] = self.ffc_32x32[i](E_features[5])
                else:
                    E_features[5] = self.ffc_32x32[i](E_features[5], wfs[:, i // 2], noise_mode=noise_mode)
            for i in range(len(self.ffc_16x16)):
                if i % 2 == 0:
                    E_features[4] = self.ffc_16x16[i](E_features[4])
                else:
                    E_features[4] = self.ffc_16x16[i](E_features[4], wfs[:, i // 2 + 4], noise_mode=noise_mode)
        else:
            E_features[5] = self.ffc_32x32(E_features[5])
            E_features[4] = self.ffc_16x16(E_features[4])

        fea_16 = E_features[4]
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(add_n, size=fea_16.size()[-2:], mode='bilinear', align_corners=False)
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16

        # style
        gs = self.to_style(fea_16)

        # decoder
        output = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)

        # ensemble
        combined_output = output * mask + img * (1 - mask)

        return combined_output, output


@persistence.persistent_class
class SynthesisNet(nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels=3,  # Number of color channels.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 drop_rate=0.5,
                 use_noise=True,
                 demodulate=True,
                 enc_ffc=False):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        self.enc = Encoder(resolution_log2, img_channels, activation, patch_size=5, channels=16, enc_ffc=enc_ffc)
        self.to_square = FullyConnectedLayer(in_features=w_dim, out_features=16 * 16, activation=activation)
        self.to_style = ToStyle(in_channels=nf(4), out_channels=nf(2) * 2, activation=activation, drop_rate=drop_rate)
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder(resolution_log2, activation, style_dim, use_noise, demodulate, img_channels)

    def forward(self, img, mask, ws, noise_mode='random'):
        # encoder
        masked_img = img * (1 - mask)
        x = torch.cat([masked_img, mask], dim=1)
        E_features = self.enc(x)

        fea_16 = E_features[4]
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(add_n, size=fea_16.size()[-2:], mode='bilinear', align_corners=False)
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        E_features[4] = fea_16

        # style
        gs = self.to_style(fea_16)

        # decoder
        output = self.dec(fea_16, ws, gs, E_features, noise_mode=noise_mode)

        # ensemble
        combined_output = output * mask + img * (1 - mask)

        return combined_output, output


@persistence.persistent_class
class SynthesisNet4x4(nn.Module):
    def __init__(self,
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # Output image resolution.
                 img_channels=3,  # Number of color channels.
                 activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 drop_rate=0.5,
                 use_noise=True,
                 demodulate=True,
                 enc_ffc=False,
                 no_style=False):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        self.drop_rate = drop_rate
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4
        self.no_style = no_style

        self.num_layers = resolution_log2 * 2 - 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        self.enc = Encoder(resolution_log2, img_channels, activation, patch_size=5, channels=16, enc_ffc=enc_ffc, min_size=2)
        self.enc_4x4 = Conv2dLayer(in_channels=nf(2), out_channels=nf(2), kernel_size=3, activation=activation)
        self.to_style = ToStyle4x4(in_channels=nf(2) * 4 * 4, out_channels=nf(2) * 2, activation=activation)
        style_dim = w_dim + nf(2) * 2
        self.dec = Decoder4x4(resolution_log2, activation, style_dim, use_noise, demodulate, img_channels, min_size=2)

    def forward(self, img, mask, ws, noise_mode='random'):
        if self.no_style:
            ws = torch.ones_like(ws)
        # encoder
        masked_img = img * (1 - mask)
        x = torch.cat([masked_img, mask], dim=1)
        E_features = self.enc(x)

        # Enc 4x4
        fea_4 = self.enc_4x4(E_features[2])
        E_features[2] = fea_4

        # style
        global_x = self.to_style(fea_4)
        global_x = F.dropout(global_x, p=self.drop_rate, training=True)
        fea_4 = global_x

        # decoder
        output = self.dec(fea_4, ws, global_x, E_features, noise_mode=noise_mode)

        # ensemble
        combined_output = output * mask + img * (1 - mask)

        return combined_output, output



@persistence.persistent_class
class Generator(nn.Module):
    def __init__(self,
                 config,
                 z_dim,  # Input latent (Z) dimensionality, 0 = no latent.
                 c_dim,  # Conditioning label (C) dimensionality, 0 = no label.
                 w_dim,  # Intermediate latent (W) dimensionality.
                 img_resolution,  # resolution of generated image
                 img_channels,  # Number of input color channels.
                 synthesis_kwargs={},  # Arguments for SynthesisNetwork.
                 mapping_kwargs={},  # Arguments for MappingNetwork.
                 ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        if config.get('ffc', False):
            style_ffc = config.get('style_ffc', False)
            self.synthesis = SynthesisFFCNet(w_dim=w_dim,
                                             img_resolution=img_resolution,
                                             img_channels=img_channels,
                                             style_ffc=style_ffc,
                                             **synthesis_kwargs)
        else:
            synthesis_kwargs['enc_ffc'] = config.get('enc_ffc', False)
            if config.get('encoder_4x4', False):
                no_style = config.get('no_style', False)
                self.synthesis = SynthesisNet4x4(w_dim=w_dim,
                                                 img_resolution=img_resolution,
                                                 img_channels=img_channels,
                                                 no_style=no_style,
                                                 **synthesis_kwargs)
            else:
                self.synthesis = SynthesisNet(w_dim=w_dim,
                                              img_resolution=img_resolution,
                                              img_channels=img_channels,
                                              **synthesis_kwargs)
        self.mapping = MappingNet(z_dim=z_dim,
                                  c_dim=c_dim,
                                  w_dim=w_dim,
                                  num_ws=self.synthesis.num_layers,
                                  **mapping_kwargs)

    def forward(self, images_in, masks_in, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False,
                noise_mode='random', sym=False):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                          skip_w_avg_update=skip_w_avg_update, sym=sym)
        combined_out, out = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode)
        return combined_out, out


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
                 c_dim,  # Conditioning label (C) dimensionality.
                 img_resolution,  # Input resolution.
                 img_channels,  # Number of input color channels.
                 cmap_dim=None,  # Dimensionality of mapped conditioning label, None = default.
                 activation='lrelu',
                 mbstd_group_size=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
                 mbstd_num_channels=1,  # Number of features for the minibatch standard deviation layer, 0 = disable.
                 ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4
        self.resolution_log2 = resolution_log2

        if cmap_dim == None:
            cmap_dim = nf(2)
        if c_dim == 0:
            cmap_dim = 0
        self.cmap_dim = cmap_dim

        if c_dim > 0:
            self.mapping = MappingNet(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None)

        Dis = [DisFromRGB(img_channels + 1, nf(resolution_log2), activation)]
        for res in range(resolution_log2, 2, -1):
            Dis.append(DisBlock(nf(res), nf(res - 1), activation))

        if mbstd_num_channels > 0:
            Dis.append(MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels))
        Dis.append(Conv2dLayer(nf(2) + mbstd_num_channels, nf(2), kernel_size=3, activation=activation))
        self.Dis = nn.Sequential(*Dis)

        self.fc0 = FullyConnectedLayer(nf(2) * 4 ** 2, nf(2), activation=activation)
        self.fc1 = FullyConnectedLayer(nf(2), 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, img, mask):
        x = self.Dis(torch.cat([mask, img], dim=1))
        x = self.fc1(self.fc0(x.flatten(start_dim=1)))

        return x


if __name__ == '__main__':
    device = torch.device('cuda:0')
    batch = 1
    res = 256
    G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=res, img_channels=3).to(device)
    D = Discriminator(c_dim=0, img_resolution=res, img_channels=3).to(device)
    img = torch.randn(batch, 3, res, res).to(device)
    mask = torch.randn(batch, 1, res, res).to(device)
    z = torch.randn(batch, 512).to(device)
    G.eval()


    def count(block):
        return sum(p.numel() for p in block.parameters())


    print('Generator', count(G))
    # print('Generator SQU', count(G.synthesis.to_square))
    # print('Generator sty', count(G.synthesis.to_style))
    # print('Generator dec', count(G.synthesis.dec))
    # print('Generator mapping', count(G.mapping))
    print('discriminator', count(D))

    with torch.no_grad():
        img = G(img, mask, z, None)
    print('output of G:', img.shape)
    score = D(img, mask)
    print('output of D:', score.shape)
