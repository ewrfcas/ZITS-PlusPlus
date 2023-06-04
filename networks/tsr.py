import torch
import torch.nn as nn

from .transformer_layers import BlockAxial, my_Block_2, model_config


class EdgeLineGPT256RelBCE_edge_pred_infer(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config=model_config):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=64, kernel_size=7, padding=0)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer, input: 32*32*config.n_embd
        self.blocks = []
        for _ in range(config.n_layer // 2):
            self.blocks.append(BlockAxial(config))
            self.blocks.append(my_Block_2(config))
        self.blocks = nn.Sequential(*self.blocks)
        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(256)

        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0)

        self.act_last = nn.Sigmoid()

        self.block_size = 32
        self.config = config

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, img_idx, line_idx, masks=None):
        img_idx = img_idx * (1 - masks)
        line_idx = line_idx * (1 - masks)
        x = torch.cat((img_idx, line_idx, masks), dim=1)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.act(x)

        [b, c, h, w] = x.shape
        x = x.view(b, c, h * w).transpose(1, 2).contiguous()

        position_embeddings = self.pos_emb[:, :h * w, :]  # each position maps to a (learnable) vector
        x = self.drop(x + position_embeddings)  # [b,hw,c]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        x = self.blocks(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln_f(x).permute(0, 3, 1, 2).contiguous()

        x = self.convt1(x)
        x = self.act(x)

        x = self.convt2(x)
        x = self.act(x)

        x = self.convt3(x)
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)

        edge, line = torch.split(x, [1, 1], dim=1)

        edge, line = self.act_last(edge), self.act_last(line)
        return edge, line


class GradientGPT256RelBCE(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config=model_config):
        super().__init__()

        self.pad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=7, padding=0)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, 256))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer, input: 32*32*config.n_embd
        self.blocks = []
        for _ in range(config.n_layer // 2):
            self.blocks.append(BlockAxial(config))
            self.blocks.append(my_Block_2(config))
        self.blocks = nn.Sequential(*self.blocks)
        # decoder, input: 32*32*config.n_embd
        self.ln_f = nn.LayerNorm(256)

        self.convt1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)

        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)

        self.padt = nn.ReflectionPad2d(3)
        self.convt4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=7, padding=0)

        self.block_size = 32
        self.config = config

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv2d, nn.ConvTranspose2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, img_idx, edge_idx, line_idx, masks=None):
        img_idx = img_idx * (1 - masks)
        edge_idx = edge_idx * (1 - masks)
        line_idx = line_idx * (1 - masks)
        x = torch.cat((img_idx, edge_idx, line_idx, masks), dim=1)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)

        x = self.conv3(x)
        x = self.act(x)

        x = self.conv4(x)
        x = self.act(x)

        [b, c, h, w] = x.shape
        x = x.view(b, c, h * w).transpose(1, 2).contiguous()

        position_embeddings = self.pos_emb[:, :h * w, :]  # each position maps to a (learnable) vector
        x = self.drop(x + position_embeddings)  # [b,hw,c]
        x = x.permute(0, 2, 1).reshape(b, c, h, w)

        x = self.blocks(x)
        x = x.permute(0, 2, 3, 1)
        x = self.ln_f(x).permute(0, 3, 1, 2).contiguous()

        x = self.convt1(x)
        x = self.act(x)

        x = self.convt2(x)
        x = self.act(x)

        x = self.convt3(x)
        x = self.act(x)

        x = self.padt(x)
        x = self.convt4(x)

        edge, line = torch.split(x, [1, 1], dim=1)

        return edge, line
