# import torch_dct as dct
from networks.basic_module import Conv2dLayer
from .layers import *


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
                                              self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
                                              self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.fft_norm = fft_norm
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        if torch.__version__ > '1.7.1':
            x = x.to(torch.float32)
            batch = x.shape[0]

            # (batch, c, h, w/2+1, 2)
            fft_dim = (-2, -1)
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted.to(torch.float32)))
            ffted = ffted.to(torch.float32)

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])

            ifft_shape_slice = x.shape[-2:]
            output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        else:
            batch, c, h, w = x.size()
            r_size = x.size()

            # (batch, c, h, w/2+1, 2)
            ffted = torch.rfft(x, signal_ndim=2, normalized=True)
            # (batch, c, 2, h, w/2+1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
            ffted = self.relu(self.bn(ffted))

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
                0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

            output = torch.irfft(ffted, signal_ndim=2,
                                 signal_sizes=r_size[2:], normalized=True)

        return output


class FourierUnitN_NoNorm(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spectral_pos_encoding=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnitN_NoNorm, self).__init__()
        self.groups = groups
        self.fft_norm = fft_norm

        self.conv_layer = Conv2dLayer(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                      out_channels=out_channels * 2, kernel_size=1, bias=False, activation='relu')

    def forward(self, x):
        if torch.__version__ > '1.7.1':
            x = x.to(torch.float32)
            batch = x.shape[0]

            # (batch, c, h, w/2+1, 2)
            fft_dim = (-2, -1)
            ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
            ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted).to(torch.float32)  # (batch, c*2, h, w/2+1)

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
            ffted = torch.complex(ffted[..., 0], ffted[..., 1])

            ifft_shape_slice = x.shape[-2:]
            output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        else:
            batch, c, h, w = x.size()
            r_size = x.size()

            # (batch, c, h, w/2+1, 2)
            ffted = torch.rfft(x, signal_ndim=2, normalized=True)
            # (batch, c, 2, h, w/2+1)
            ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
            ffted = ffted.view((batch, -1,) + ffted.size()[3:])

            ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)

            ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)

            output = torch.irfft(ffted, signal_ndim=2,
                                 signal_sizes=r_size[2:], normalized=True)

        return output


class DCTUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(DCTUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        # (batch, c, h, w)
        ffted = dct.dct_2d(x, norm='ortho')
        # (batch, c, h, w)
        ffted = self.conv_layer(ffted)  # (batch, c, h, w)
        ffted = self.relu(self.bn(ffted))

        output = dct.idct_2d(ffted, norm='ortho')

        return output


class SeparableFourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3):
        # bn_layer not used
        super(SeparableFourierUnit, self).__init__()
        self.groups = groups
        row_out_channels = out_channels // 2
        col_out_channels = out_channels - row_out_channels
        self.row_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=row_out_channels * 2,
                                        kernel_size=(kernel_size, 1),
                                        # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.col_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=col_out_channels * 2,
                                        kernel_size=(kernel_size, 1),
                                        # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.row_bn = torch.nn.BatchNorm2d(row_out_channels * 2)
        self.col_bn = torch.nn.BatchNorm2d(col_out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def process_branch(self, x, conv, bn):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft(x, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.relu(bn(conv(ffted)))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfft(ffted, s=x.shape[-1:], norm="ortho")
        return output

    def forward(self, x):
        rowwise = self.process_branch(x, self.row_conv, self.row_bn)
        colwise = self.process_branch(x.permute(0, 1, 3, 2), self.col_conv, self.col_bn).permute(0, 1, 3, 2)
        out = torch.cat((rowwise, colwise), dim=1)
        return out


# class SpectralTransform(nn.Module):

#     def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, separable_fu=False, **fu_kwargs):
#         # bn_layer not used
#         super(SpectralTransform, self).__init__()
#         self.enable_lfu = enable_lfu
#         if stride == 2:
#             self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
#         else:
#             self.downsample = nn.Identity()

#         self.stride = stride
#         self.conv1 = nn.Conv2d(in_channels, out_channels //
#                                2, kernel_size=1, groups=groups, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels // 2)
#         self.act1 = nn.ReLU(inplace=True)
#         fu_class = FourierUnit
#         self.fu = fu_class(
#             out_channels // 2, out_channels // 2, groups, **fu_kwargs)
#         if self.enable_lfu:
#             self.lfu = fu_class(
#                 out_channels // 2, out_channels // 2, groups)
#         self.conv2 = torch.nn.Conv2d(
#             out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

#     def forward(self, x):

#         x = self.downsample(x)
#         x = self.conv1(x)
#         x = self.bn1(x.to(torch.float32))
#         x = self.act1(x)
#         output = self.fu(x)

#         if self.enable_lfu:
#             n, c, h, w = x.shape
#             split_no = 2
#             split_s = h // split_no
#             xs = torch.cat(torch.split(
#                 x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
#             xs = torch.cat(torch.split(xs, split_s, dim=-1),
#                            dim=1).contiguous()
#             xs = self.lfu(xs)
#             xs = xs.repeat(1, 1, split_no, split_no).contiguous()
#         else:
#             xs = 0

#         output = self.conv2(x + output + xs)

#         return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, separable_fu=False, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1), dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class DCTSpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(DCTSpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Identity()
        self.fu = DCTUnit(in_channels, out_channels, groups)
        if self.enable_lfu:
            pass
        self.conv2 = nn.Identity()

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_AXIAL(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, axial_type='raw'):
        super(FFC_AXIAL, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        self.axial = AxialZero(n_embd=in_cl, n_head=4, attn_pdrop=0., resid_pdrop=0.,
                               rel_pos_size=32, axial_type=axial_type, add_rel_pos=True)
        self.convl2g = nn.Conv2d(in_cl, out_cg, kernel_size,
                                 stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.convg2l = nn.Conv2d(in_cg, out_cl, kernel_size,
                                 stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.ffc_conv = SpectralTransform(in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)

        out_xl = self.axial(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.ffc_conv(x_g)

        return out_xl, out_xg


class FFC_atten(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC_atten, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        # self.atten = CrossAttention(out_cl, out_cg, 0.0, 0.0, n_head=8)
        self.atten = EfficientAttention(out_cl, out_cg, 4)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl = self.convl2l(x_l)
        out_xg = self.convg2g(x_g)
        out_xl, out_xg = self.atten(out_xl, out_xg)
        return out_xl, out_xg


class FFCDCT(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFCDCT, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg_ffc = int(in_channels * ratio_gin / 2)
        in_cg_dct = int(in_channels * ratio_gin) - in_cg_ffc
        in_cl = in_channels - in_cg_ffc - in_cg_dct
        out_cg_ffc = int(out_channels * ratio_gout / 2)
        out_cg_dct = int(out_channels * ratio_gout) - out_cg_ffc
        out_cl = out_channels - out_cg_ffc - out_cg_dct
        # groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        # groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num_ffc = in_cg_ffc
        self.global_in_num_dct = in_cg_dct

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg_ffc == 0 else nn.Conv2d
        self.convl2g_ffc = module(in_cl, out_cg_ffc, kernel_size,
                                  stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg_ffc == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l_ffc = module(in_cg_ffc, out_cl, kernel_size,
                                  stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg_ffc == 0 or out_cg_ffc == 0 else SpectralTransform
        self.convg2g_ffc = module(
            in_cg_ffc, out_cg_ffc, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        module = nn.Identity if in_cl == 0 or out_cg_dct == 0 else nn.Conv2d
        self.convl2g_dct = module(in_cl, out_cg_dct, kernel_size,
                                  stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg_dct == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l_dct = module(in_cg_dct, out_cl, kernel_size,
                                  stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg_dct == 0 or out_cg_dct == 0 else DCTSpectralTransform
        self.convg2g_dct = module(
            in_cg_dct, out_cg_dct, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        module = nn.Identity if in_cg_dct == 0 or out_cg_ffc == 0 else nn.Conv2d
        self.convdct2ffc = module(in_cg_dct, out_cg_ffc, kernel_size,
                                  stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg_ffc == 0 or out_cg_dct == 0 else nn.Conv2d
        self.convffc2dct = module(in_cg_ffc, out_cg_dct, kernel_size,
                                  stride, padding, dilation, groups, bias, padding_mode=padding_type)

        assert gated is False, 'gated is not False'
        self.gated = gated
        module = nn.Identity
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        if torch.is_tensor(x_g):
            x_g_ffc, x_g_dct = torch.split(x_g, [self.global_in_num_ffc, self.global_in_num_dct], dim=1)
        else:
            x_g_ffc, x_g_dct = 0, 0

        out_xl, out_xg = 0, 0

        assert self.gated is False, 'gated is not False'

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l_ffc(x_g_ffc) + self.convg2l_dct(x_g_dct)
        if self.ratio_gout != 0:
            out_xg_ffc = self.convl2g_ffc(x_l) + self.convg2g_ffc(x_g_ffc) + self.convdct2ffc(x_g_dct)
            out_xg_dct = self.convl2g_dct(x_l) + self.convg2g_dct(x_g_dct) + self.convffc2dct(x_g_ffc)
            out_xg = torch.cat((out_xg_ffc, out_xg_dct), dim=1)

        return out_xl, out_xg


class FFC_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l.to(torch.float32)))
        x_g = self.act_g(self.bn_g(x_g.to(torch.float32)))
        return x_l, x_g


class FFC_AXIAL_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 padding_type='reflect', enable_lfu=True, axial_type='raw'):
        super(FFC_AXIAL_BN_ACT, self).__init__()
        self.ffc = FFC_AXIAL(in_channels, out_channels, kernel_size,
                             ratio_gin, ratio_gout, stride, padding, dilation,
                             groups, bias, enable_lfu, padding_type=padding_type,
                             axial_type=axial_type)
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = nn.BatchNorm2d(out_channels - global_channels)
        self.bn_g = nn.BatchNorm2d(global_channels)

        self.act_l = nn.ReLU(inplace=True)
        self.act_g = nn.ReLU(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l.to(torch.float32)))
        x_g = self.act_g(self.bn_g(x_g.to(torch.float32)))
        return x_l, x_g


class FFC_BN_ACT_atten(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT_atten, self).__init__()
        self.ffc = FFC_atten(in_channels, out_channels, kernel_size,
                             ratio_gin, ratio_gout, stride, padding, dilation,
                             groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l.to(torch.float32)))
        x_g = self.act_g(self.bn_g(x_g.to(torch.float32)))
        return x_l, x_g


class FFCDCT_BN_ACT(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFCDCT_BN_ACT, self).__init__()
        self.ffc = FFCDCT(in_channels, out_channels, kernel_size,
                          ratio_gin, ratio_gout, stride, padding, dilation,
                          groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class ResnetBlock_remove_IN(nn.Module):
    def __init__(self, dim, dilation=1, activation_layer=nn.ReLU):
        super(ResnetBlock_remove_IN, self).__init__()

        self.ffc1 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=dilation, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=activation_layer, enable_lfu=False)

        self.ffc2 = FFC_BN_ACT(dim, dim, 3, 0.75, 0.75, stride=1, padding=1, dilation=1, groups=1, bias=False,
                               norm_layer=nn.BatchNorm2d, activation_layer=activation_layer, enable_lfu=False)

    def forward(self, x):
        output = x
        _, c, _, _ = output.shape
        output = torch.split(output, [c - int(c * 0.75), int(c * 0.75)], dim=1)
        x_l, x_g = self.ffc1(output)
        output = self.ffc2((x_l, x_g))
        output = torch.cat(output, dim=1)
        output = x + output

        return output


class GlobalFilter(nn.Module):
    def __init__(self, dim, h=32, w=17):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02)
        self.w = w
        self.h = h

    def forward(self, x):
        _, _, old_H, old_W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')

        _, H, W, _ = x.shape
        if H != self.h or W != self.w:
            new_weight = self.complex_weight.reshape(1, self.h, self.w, -1).permute(0, 3, 1, 2)
            new_weight = torch.nn.functional.interpolate(
                new_weight, size=(H, W), mode='bicubic', align_corners=True).permute(0, 2, 3,
                                                                                     1).reshape(
                H, W, -1, 2).contiguous()
        else:
            new_weight = self.complex_weight

        weight = torch.view_as_complex(new_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(old_H, old_W), dim=(1, 2), norm='ortho')
        x = x.permute(0, 3, 1, 2)
        return x


class GFBlock(nn.Module):

    def __init__(self, dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_type='reflect',
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, h=32, w=17):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = GlobalFilter(dim, h=h, w=w)
        self.norm2 = norm_layer(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        self.act = act_layer(inplace=True)

    def forward(self, x):
        x = x + self.act(self.norm2(self.conv(self.norm1(self.filter(x)))))
        return x