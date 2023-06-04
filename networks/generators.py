import torch
import torch.nn as nn

from networks.ffc import ResnetBlock_remove_IN, GFBlock
from networks.layers import GateConv, MaskedSinusoidalPositionalEmbedding, MultiLabelEmbedding, ResnetBlock
from networks.van import VANBlock


class myFFCResNetGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config['activation'] == 'relu':
            print('use relu')
            act = nn.ReLU
        elif config['activation'] == 'swish':
            print('use swish')
            act = nn.SiLU
        else:
            print('use identity')
            act = nn.Identity

        self.enconv1 = nn.Sequential(*[nn.ReflectionPad2d(3),
                                       nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
                                       nn.BatchNorm2d(64), act()])
        self.enconv2 = nn.Sequential(*[nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(128), act()])
        self.enconv3 = nn.Sequential(*[nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(256), act()])
        self.enconv4 = nn.Sequential(*[nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(512), act()])

        blocks = []
        # resnet blocks
        if config['use_GFBlock'] is False:
            print('use FFC')
            for i in range(9):
                blocks.append(ResnetBlock_remove_IN(512, 1, activation_layer=act))
        else:
            print('use Global Filter')
            for i in range(int(config['num_GFBlock'])):
                blocks.append(GFBlock(dim=512))

        self.middle = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(*[nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(256), act()])
        self.deconv2 = nn.Sequential(*[nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(128), act()])
        self.deconv3 = nn.Sequential(*[nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(64), act()])
        self.deconv4 = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)])

        self.act_last = nn.Tanh()

    def forward(self, x):
        x = self.enconv1(x)
        x = self.enconv2(x)
        x = self.enconv3(x)
        x = self.enconv4(x)

        x = self.middle(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.act_last(x)
        return x

    def forward_with_prior(self, x, rel_pos_emb, direct_emb, str_feats):
        x = self.enconv1[:2](x)
        inp = x.to(torch.float32) + rel_pos_emb + direct_emb
        x = self.enconv1[2:](inp)

        x = self.enconv2[:1](x + str_feats[0])
        x = self.enconv2[1:](x)

        x = self.enconv3[:1](x + str_feats[1])
        x = self.enconv3[1:](x)

        x = self.enconv4[:1](x + str_feats[2])
        x = self.enconv4[1:](x)

        x = self.middle(x + str_feats[3])

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.act_last(x)

        return x


class VANFFCResNetGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        print('VANFFCResNetGenerator initialized')

        if config['activation'] == 'relu':
            print('use relu')
            act = nn.ReLU
        elif config['activation'] == 'swish':
            print('use swish')
            act = nn.SiLU
        else:
            print('use identity')
            act = nn.Identity

        self.enconv1 = nn.Sequential(*[nn.ReflectionPad2d(3),
                                       nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
                                       nn.BatchNorm2d(64), act(True)])
        self.enconv2 = nn.Sequential(*[nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(128), act(True),
                                       VANBlock(dim=128, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act)])
        self.enconv3 = nn.Sequential(*[nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(256), act(True),
                                       VANBlock(dim=256, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act)])
        self.enconv4 = nn.Sequential(*[nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                                       nn.BatchNorm2d(512), act(True),
                                       VANBlock(dim=512, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act)])

        blocks = []
        # resnet blocks
        if config['use_GFBlock'] is False:
            if config['use_VAN_between_FFC'] is False:
                print('use pure FFC')
                for i in range(9):
                    blocks.append(ResnetBlock_remove_IN(512, 1, activation_layer=act))

        self.middle = nn.Sequential(*blocks)

        self.deconv1 = nn.Sequential(*[VANBlock(dim=512, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act),
                                       nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(256), act(True)])
        self.deconv2 = nn.Sequential(*[VANBlock(dim=256, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act),
                                       nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(128), act(True)])
        self.deconv3 = nn.Sequential(*[VANBlock(dim=128, kernel_size=config['van_kernel_size'],
                                                dilation=config['van_dilation'], act_layer=act),
                                       nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                                       nn.BatchNorm2d(64), act(True)])
        self.deconv4 = nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)])

        self.act_last = nn.Tanh()

    def forward(self, x):
        x = self.enconv1(x)
        x = self.enconv2(x)
        x = self.enconv3(x)
        x = self.enconv4(x)

        x = self.middle(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.act_last(x)
        return x

    def forward_with_prior(self, x, rel_pos_emb, direct_emb, str_feats):
        x = self.enconv1[:2](x)
        inp = x.to(torch.float32) + rel_pos_emb + direct_emb
        x = self.enconv1[2:](inp)

        x = self.enconv2[:1](x + str_feats[0])
        x = self.enconv2[1:](x)

        x = self.enconv3[:1](x + str_feats[1])
        x = self.enconv3[1:](x)

        x = self.enconv4[:1](x + str_feats[2])
        x = self.enconv4[1:](x)

        x = self.middle(x + str_feats[3])

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.act_last(x)

        return x


class StructureEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = GateConv(in_channels=config['prior_ch'], out_channels=64, kernel_size=7, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.act = nn.ReLU(True)

        self.conv2 = GateConv(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        blocks = []
        # resnet blocks
        for i in range(3):
            blocks.append(ResnetBlock(input_dim=512, out_dim=None, dilation=2))

        self.middle = nn.Sequential(*blocks)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt1 = GateConv(512, 256, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt1 = nn.BatchNorm2d(256)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt2 = GateConv(256, 128, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt2 = nn.BatchNorm2d(128)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        self.convt3 = GateConv(128, 64, kernel_size=4, stride=2, padding=1, transpose=True)
        self.bnt3 = nn.BatchNorm2d(64)
        self.alpha4 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

        if self.config['rezero_for_mpe']:
            self.rel_pos_emb = MaskedSinusoidalPositionalEmbedding(num_embeddings=self.config['rel_pos_num'],
                                                                   embedding_dim=64)
            self.direct_emb = MultiLabelEmbedding(num_positions=4, embedding_dim=64)
            self.alpha5 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
            self.alpha6 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def forward(self, x, rel_pos=None, direct=None):
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x.to(torch.float32))
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x.to(torch.float32))
        x = self.act(x)

        x = self.conv3(x)
        x = self.bn3(x.to(torch.float32))
        x = self.act(x)

        x = self.conv4(x)
        x = self.bn4(x.to(torch.float32))
        x = self.act(x)

        return_feats = []
        x = self.middle(x)
        return_feats.append(x * self.alpha1)

        x = self.convt1(x)
        x = self.bnt1(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha2)

        x = self.convt2(x)
        x = self.bnt2(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha3)

        x = self.convt3(x)
        x = self.bnt3(x.to(torch.float32))
        x = self.act(x)
        return_feats.append(x * self.alpha4)

        return_feats = return_feats[::-1]

        b, h, w = rel_pos.shape
        rel_pos = rel_pos.reshape(b, h * w)
        rel_pos_emb = self.rel_pos_emb(rel_pos).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha5
        direct = direct.reshape(b, h * w, 4).to(torch.float32)
        direct_emb = self.direct_emb(direct).reshape(b, h, w, -1).permute(0, 3, 1, 2) * self.alpha6

        return return_feats, rel_pos_emb, direct_emb


class FTRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if 'not_use_van' in self.config and self.config['not_use_van'] is True:
            self.G = myFFCResNetGenerator(config)
        else:
            self.G = VANFFCResNetGenerator(config)
        self.GCs = StructureEncoder(config)

    def forward(self, batch):
        img = batch['image']
        mask = batch['mask']
        masked_img = img * (1 - mask)

        masked_img = torch.cat([masked_img, mask], dim=1)
        if self.config['use_gradient'] is True:
            if 'not_use_line' in self.config and self.config['not_use_line'] is True:
                masked_str = torch.cat([batch['edge'], batch['gradientx'], batch['gradienty'], mask], dim=1)
            else:
                masked_str = torch.cat([batch['edge'], batch['gradientx'], batch['gradienty'], batch['line'], mask], dim=1)
        else:
            masked_str = torch.cat([batch['edge'], batch['line'], mask], dim=1)

        str_feats, rel_pos_emb, direct_emb = self.GCs(masked_str, batch['rel_pos'], batch['direct'])
        gen_img = self.G.forward_with_prior(masked_img.to(torch.float32), rel_pos_emb, direct_emb, str_feats)

        return gen_img
