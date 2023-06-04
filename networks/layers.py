import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# try:
#     from apex import amp
# except ImportError:
#     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


class GateConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False):
        super(GateConv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.gate_conv = nn.ConvTranspose2d(in_channels, out_channels * 2,
                                                kernel_size=kernel_size,
                                                stride=stride, padding=padding)
        else:
            self.gate_conv = nn.Conv2d(in_channels, out_channels * 2,
                                       kernel_size=kernel_size,
                                       stride=stride, padding=padding)

    def forward(self, x):
        x = self.gate_conv(x)
        (x, g) = torch.split(x, self.out_channels, dim=1)
        return x * torch.sigmoid(g)


class CrossAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, in_channels1, in_channels2, attn_pdrop, resid_pdrop, n_head=8):
        super().__init__()
        # key, query, value projections for all heads
        self.key1 = nn.Conv2d(in_channels=in_channels1, out_channels=in_channels1,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.query1 = nn.Conv2d(in_channels=in_channels1, out_channels=in_channels1,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.value1 = nn.Conv2d(in_channels=in_channels1, out_channels=in_channels1,
                                kernel_size=3, stride=1, padding=1, bias=False)

        self.key2 = nn.Conv2d(in_channels=in_channels2, out_channels=in_channels2,
                              kernel_size=3, stride=1, padding=1, bias=False)
        self.query2 = nn.Conv2d(in_channels=in_channels2, out_channels=in_channels2,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.value2 = nn.Conv2d(in_channels=in_channels2, out_channels=in_channels2,
                                kernel_size=3, stride=1, padding=1, bias=False)
        # regularization
        self.attn_drop1 = nn.Dropout(attn_pdrop)
        self.resid_drop1 = nn.Dropout(resid_pdrop)
        self.attn_drop2 = nn.Dropout(attn_pdrop)
        self.resid_drop2 = nn.Dropout(resid_pdrop)
        # output projection
        self.proj1 = nn.Conv2d(in_channels=in_channels1, out_channels=in_channels1,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.proj2 = nn.Conv2d(in_channels=in_channels2, out_channels=in_channels2,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.n_head = n_head

        # rezero
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x_l, x_g):
        n, c1, h, w = x_l.size()
        _, c2, _, _ = x_g.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k_l = self.key1(x_l).view(n, c1, self.n_head, h*w // self.n_head).transpose(1, 2)  # (B, nh, T1, hs)
        q_l = self.query1(x_l).view(n, c1, self.n_head, h*w // self.n_head).transpose(1, 2)  # (B, nh, T1, hs)
        v_l = self.value1(x_l).view(n, c1, self.n_head, h*w // self.n_head).transpose(1, 2)  # (B, nh, T1, hs)

        k_g = self.key2(x_g).view(n, c2, self.n_head, h * w // self.n_head).transpose(1, 2)  # (B, nh, T2, hs)
        q_g = self.query2(x_g).view(n, c2, self.n_head, h * w // self.n_head).transpose(1, 2)  # (B, nh, T2, hs)
        v_g = self.value2(x_g).view(n, c2, self.n_head, h * w // self.n_head).transpose(1, 2)  # (B, nh, T2, hs)

        # causal self-attention; Self-attend: (B, nh, T2, hs) x (B, nh, hs, T1) -> (B, nh, T2, T1)
        with amp.handle.disable_casts():
            att2 = torch.matmul(q_g.to(torch.float32), k_l.transpose(-2, -1).to(torch.float32)) * (1.0 / math.sqrt(k_l.size(-1)))
            att2 = F.softmax(att2, dim=-1)
            att2 = self.attn_drop2(att2)
            y_g = torch.matmul(att2.to(torch.float32), v_l.to(torch.float32))  # (B, nh, T2, T1) x (B, nh, T1, hs) -> (B, nh, T2, hs)
        y_g = y_g.transpose(1, 2).contiguous().view(n, c2, h, w)  # re-assemble all head outputs side by side
        # output projection
        y_g = self.resid_drop2(self.proj2(y_g))

        # causal self-attention; Self-attend: (B, nh, T1, hs) x (B, nh, hs, T2) -> (B, nh, T1, T2)
        with amp.handle.disable_casts():
            att1 = torch.matmul(q_l.to(torch.float32), k_g.transpose(-2, -1).to(torch.float32)) * (1.0 / math.sqrt(k_g.size(-1)))
            att1 = F.softmax(att1, dim=-1)
            att1 = self.attn_drop1(att1)
            y_l = torch.matmul(att1.to(torch.float32), v_g.to(torch.float32))  # (B, nh, T1, T2) x (B, nh, T2, hs) -> (B, nh, T1, hs)
        y_l = y_l.transpose(1, 2).contiguous().view(n, c1, h, w)  # re-assemble all head outputs side by side
        # output projection
        y_l = self.resid_drop1(self.proj1(y_l))
        return x_l + y_l * self.resweight, x_g + y_g * self.resweight


class EfficientAttention(nn.Module):
    def __init__(self, in_channels1, in_channels2, head_count):
        super().__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.head_count = head_count

        self.keys1 = nn.Conv2d(in_channels1, in_channels1, 1)
        self.queries1 = nn.Conv2d(in_channels1, in_channels1, 1)
        self.values1 = nn.Conv2d(in_channels1, in_channels1, 1)

        self.keys2 = nn.Conv2d(in_channels2, in_channels2, 1)
        self.queries2 = nn.Conv2d(in_channels2, in_channels2, 1)
        self.values2 = nn.Conv2d(in_channels2, in_channels2, 1)

        # rezero
        self.resweight = nn.Parameter(torch.Tensor([0]))

    def forward(self, x_l, x_g):
        n, c1, h, w = x_l.size()
        _, c2, _, _ = x_g.size()

        keys1 = self.keys1(x_l)
        queries1 = self.queries1(x_l)
        values1 = self.values1(x_l)

        keys2 = self.keys2(x_g)
        queries2 = self.queries2(x_g)
        values2 = self.values2(x_g)

        head_channels1 = c1 // self.head_count
        head_channels2 = c2 // self.head_count

        keys1 = keys1.reshape((n, c1, h * w)).to(torch.float32)  # [b,d,h*w]
        queries1 = queries1.reshape(n, c1, h * w).to(torch.float32)
        values1 = values1.reshape((n, c1, h * w)).to(torch.float32)

        keys2 = keys2.reshape((n, c2, h * w)).to(torch.float32)  # [b,d,h*w]
        queries2 = queries2.reshape(n, c2, h * w).to(torch.float32)
        values2 = values2.reshape((n, c2, h * w)).to(torch.float32)

        attended_values1 = []
        attended_values2 = []
        # with amp.handle.disable_casts():
        for i in range(self.head_count):
            key = F.softmax(keys1[:, i * head_channels1: (i + 1) * head_channels1, :], dim=2)
            query = F.softmax(queries1[:, i * head_channels1: (i + 1) * head_channels1, :], dim=1)
            value = values2[:, i * head_channels2: (i + 1) * head_channels2, :]
            context = key @ value.transpose(1, 2)  # [b, d1, d2]
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_channels2, h, w)
            attended_values2.append(attended_value)

            key = F.softmax(keys2[:, i * head_channels2: (i + 1) * head_channels2, :], dim=2)
            query = F.softmax(queries2[:, i * head_channels2: (i + 1) * head_channels2, :], dim=1)
            value = values1[:, i * head_channels1: (i + 1) * head_channels1, :]
            context = key @ value.transpose(1, 2)  # [b, d1, d2]
            attended_value = (context.transpose(1, 2) @ query).reshape(n, head_channels1, h, w)
            attended_values1.append(attended_value)

        aggregated_values1 = torch.cat(attended_values1, dim=1)
        aggregated_values2 = torch.cat(attended_values2, dim=1)

        return x_l + aggregated_values1 * self.resweight, x_g + aggregated_values2 * self.resweight


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, transpose=False,
                 use_spectral_norm=False):
        super(Conv, self).__init__()
        self.out_channels = out_channels
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                           kernel_size=kernel_size, stride=stride,
                                           padding=padding, bias=not use_spectral_norm)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=padding, bias=not use_spectral_norm)
        if use_spectral_norm:
            self.conv = spectral_norm(self.conv)

    def forward(self, x):
        return self.conv(x)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            padding,
            demodulate=True,
            upsample=False,
            downsample=False,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = padding

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        self.demodulate = demodulate

    def forward(self, input):
        _, in_channel, height, width = input.shape
        weight = self.scale * self.weight

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(1, self.out_channel, 1, 1, 1)

        weight = weight.view(self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            weight = weight.view(self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(in_channel, self.out_channel,
                                                    self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=self.padding, stride=2)

        elif self.downsample:
            _, _, height, width = input.shape
            out = F.conv2d(input, weight, padding=self.padding, stride=2)

        else:
            out = F.conv2d(input, weight, padding=self.padding)
        return out


class MaskedSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids):
        """`input_ids` is expected to be [bsz x seqlen]."""
        return super().forward(input_ids)


class MultiLabelEmbedding(nn.Module):
    def __init__(self, num_positions: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_positions, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight)

    def forward(self, input_ids):
        # input_ids:[B,HW,4](onehot)
        out = torch.matmul(input_ids, self.weight)  # [B,HW,dim]
        return out


class ResnetBlock(nn.Module):
    def __init__(self, input_dim, out_dim=None, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        if out_dim is not None:
            self.proj = nn.Conv2d(in_channels=input_dim, out_channels=out_dim, kernel_size=1, bias=False)
        else:
            self.proj = None
            out_dim = input_dim

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=0, dilation=dilation,
                                    bias=not use_spectral_norm), use_spectral_norm)
        )
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.act = nn.ReLU(True)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   spectral_norm(nn.Conv2d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=0, dilation=1,
                                                           bias=not use_spectral_norm), use_spectral_norm))
        self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)
        y = self.conv1(x)
        y = self.bn1(y.to(torch.float32))
        y = self.act(y)
        y = self.conv2(y)
        y = self.bn2(y.to(torch.float32))
        out = x + y

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out