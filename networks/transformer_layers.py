import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None, rel_pos=None, return_att=False, debug=False):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if rel_pos is not None:
            att += rel_pos
        if mask is not None:  # maybe we don't need mask in axial-transformer
            # mask:[B,1,L(1),L]
            att = att.masked_fill(mask == 1, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        if return_att:
            return y, att
        else:
            return y


class TransformerBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=1e-4)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=1e-4)
        self.attn = SelfAttention(config.n_embd, config.n_head,
                                  config.attn_pdrop, config.resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class PositionEmbedding(nn.Module):
    # add pos_emb and reshape
    def __init__(self, height, width, ch):
        super().__init__()
        self.height = height
        self.width = width
        self.h_emb = nn.Embedding(height, ch)
        self.w_emb = nn.Embedding(width, ch)

    def forward(self, x):  # [B,C,H,W]->[B,HW,C]
        [b, c, h, w] = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        h_ids = torch.arange(self.height, dtype=torch.long, device=x.device)  # [H,]
        h_ids = h_ids.reshape(1, self.height, 1).repeat(x.shape[0], 1, self.width)  # [B,H,W]
        h_ids = h_ids.reshape(x.shape[0], -1)  # [B,HW]
        w_ids = torch.arange(self.width, dtype=torch.long, device=x.device)  # [W,]
        w_ids = w_ids.reshape(1, self.width).repeat(x.shape[0], self.height)  # [B,WH]
        h_feat = self.h_emb(h_ids)
        w_feat = self.w_emb(w_ids)
        x = x + h_feat + w_feat
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)  # [B,C,H,W]
        return x


class AxialAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H, W,
                 add_rel_pos=True, rel_pos_bins=32):
        super().__init__()

        self.rln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.cln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
        self.attn_row = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn_col = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.add_rel_pos = add_rel_pos
        # self.rel_pos_bins = rel_pos_bins
        self.row_rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
        self.col_rel_pos_bias = nn.Linear(2 * W - 1, n_head, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
        # hidden_states:[B,L,D], [1,L]
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
        # [1,1,L]-[1,L,1]-->[1,L,L]
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        # [1,L,L]->[1,L,L,D]
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
        if row:
            rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        rel_pos = rel_pos.contiguous()
        return rel_pos

    def forward(self, x, return_att=False, debug=False):  # x:[B,C,H,W], mask:[B,1,H,W]
        [b, c, h, w] = x.shape
        x0 = x.clone()
        x0 = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)
        mask_row = None
        mask_col = None

        # ROW ATTENTION
        x = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        if self.add_rel_pos:
            row_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=h, row=True)
        else:
            row_rel_pos = None
        x_row = self.attn_row(self.rln1(x), mask_row, row_rel_pos, return_att=return_att, debug=debug)
        if return_att:
            x_row, att_row = x_row
        else:
            att_row = None
        x_row = x_row.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)

        # COL ATTENTION
        x = x.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b * h, w, c)
        if self.add_rel_pos:
            col_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=w, row=False)
        else:
            col_rel_pos = None
        x_col = self.attn_col(self.cln1(x), mask_col, col_rel_pos, return_att=return_att)
        if return_att:
            x_col, att_col = x_col
        else:
            att_col = None
        x_col = x_col.reshape(b, h, w, c).reshape(b, h * w, c)

        # [B,HW,C]
        x = x0 + x_row + x_col
        x = x + self.ff(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x.contiguous()

        if return_att:
            # att_row:[BW,head,H,H]
            att_row = torch.mean(att_row, dim=1).reshape(b, w, h, h)
            att_row = torch.sum(att_row, dim=2).permute(0, 2, 1)  # [b,h,w]
            # att_col:[BH,head,W,W]
            att_col = torch.mean(att_col, dim=1).reshape(b, h, w, w)
            att_col = torch.sum(att_col, dim=2)
            att_score = att_row * att_col
            return x, att_score
        else:
            return x


class AxialZeroAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H, W,
                 add_rel_pos=True, rel_pos_bins=32):
        super().__init__()

        self.attn_row = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn_col = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.add_rel_pos = add_rel_pos
        # self.rel_pos_bins = rel_pos_bins
        self.row_rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
        self.col_rel_pos_bias = nn.Linear(2 * W - 1, n_head, bias=False)
        self.alpha1 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)
        self.alpha3 = nn.Parameter(torch.tensor(0, dtype=torch.float32), requires_grad=True)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
        # hidden_states:[B,L,D], [1,L]
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
        # [1,1,L]-[1,L,1]-->[1,L,L]
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        # [1,L,L]->[1,L,L,D]
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
        if row:
            rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        rel_pos = rel_pos.contiguous()
        return rel_pos

    def forward(self, x, return_att=False, debug=False):  # x:[B,C,H,W], mask:[B,1,H,W]
        # print(self.alpha1, self.alpha2, self.alpha3)
        [b, c, h, w] = x.shape
        x0 = x.clone()
        x0 = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)
        mask_row = None
        mask_col = None

        # ROW ATTENTION
        x = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        if self.add_rel_pos:
            row_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=h, row=True)
        else:
            row_rel_pos = None
        x_row = self.attn_row(x, mask_row, row_rel_pos, return_att=return_att, debug=debug)
        if return_att:
            x_row, att_row = x_row
        else:
            att_row = None
        x_row = x_row.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)

        # COL ATTENTION
        x = x.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b * h, w, c)
        if self.add_rel_pos:
            col_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=w, row=False)
        else:
            col_rel_pos = None
        x_col = self.attn_col(x, mask_col, col_rel_pos, return_att=return_att)
        if return_att:
            x_col, att_col = x_col
        else:
            att_col = None
        x_col = x_col.reshape(b, h, w, c).reshape(b, h * w, c)

        # [B,HW,C]
        x = x0 + self.alpha1 * x_row + self.alpha2 * x_col
        x = x + self.alpha3 * self.ff(x)

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x.contiguous()

        if return_att:
            # att_row:[BW,head,H,H]
            att_row = torch.mean(att_row, dim=1).reshape(b, w, h, h)
            att_row = torch.sum(att_row, dim=2).permute(0, 2, 1)  # [b,h,w]
            # att_col:[BH,head,W,W]
            att_col = torch.mean(att_col, dim=1).reshape(b, h, w, w)
            att_col = torch.sum(att_col, dim=2)
            att_score = att_row * att_col
            return x, att_score
        else:
            return x


class SingleAxialAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H,
                 add_rel_pos=True, rel_pos_bins=32):
        super().__init__()

        # self.ln1 = nn.LayerNorm(n_embd, eps=1e-4)
        # self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.add_rel_pos = add_rel_pos
        # self.rel_pos_bins = rel_pos_bins
        self.rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
        self.alpha = nn.Parameter(torch.FloatTensor([0]), requires_grad=True)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size):
        # hidden_states:[B,L,D], [1,L]
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
        # [1,1,L]-[1,L,1]-->[1,L,L]
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        # [1,L,L]->[1,L,L,D]
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        rel_pos = rel_pos.contiguous()
        return rel_pos

    def forward(self, x, return_att=False, debug=False):  # x:[B,C,R,T], mask:[B,1,R,T]
        [b, c, r, t] = x.shape
        x0 = x.clone()
        x0 = x0.permute(0, 2, 3, 1).reshape(b, r * t, c)

        # ROW ATTENTION
        x = x.permute(0, 3, 2, 1).reshape(b * t, r, c)
        if self.add_rel_pos:
            rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=r)
        else:
            rel_pos = None
        x = self.attn(x, None, rel_pos, return_att=return_att, debug=debug)
        if return_att:
            x, att_row = x
        else:
            att_row = None
        x = x.reshape(b, t, r, c).permute(0, 2, 1, 3).reshape(b, r * t, c)

        # [B,HW,C]
        x = x0 + self.alpha * x
        x = x + self.alpha * self.ff(x)
        x = x.reshape(b, r, t, c).permute(0, 3, 1, 2)

        x = x.contiguous()

        if return_att:
            # att_row:[BW,head,H,H]
            att_row = torch.mean(att_row, dim=1).reshape(b, t, r, r)
            att_score = torch.sum(att_row, dim=2).permute(0, 2, 1)  # [b,r,t]
            return x, att_score
        else:
            return x


class VanillaAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, HW,
                 add_rel_pos=True, rel_pos_bins=32):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.add_rel_pos = add_rel_pos
        self.rel_pos_bias = nn.Linear(2 * HW - 1, n_head, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size):
        # hidden_states:[B,L,D], [1,L]
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
        # [1,1,L]-[1,L,1]-->[1,L,L]
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        # [1,L,L]->[1,L,L,D]
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
        rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos

    def forward(self, x, return_att=False, debug=False):  # x:[B,C,H,W], mask:[B,1,H,W]
        [b, c, h, w] = x.shape
        x0 = x.clone()
        x0 = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)

        # ATTENTION
        if self.add_rel_pos:
            rel_pos = self._cal_1d_pos_emb(x0, rel_pos_onehot_size=h * w)
        else:
            rel_pos = None

        x = self.attn(self.ln1(x0), None, rel_pos, return_att=return_att, debug=debug)
        if return_att:
            x, att_score = x
        else:
            att_score = None

        # [B,HW,C]
        x = x0 + x
        x = x + self.ff(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x.contiguous()

        if return_att:
            # x:[BW,head,H,H]
            return x, att_score
        else:
            return x


class BlockRel(VanillaAttention):

    def __init__(self, config):
        super().__init__(config.n_embd, config.n_head, config.attn_pdrop, config.resid_pdrop, 32*32)


class BlockAxial(AxialAttention):

    def __init__(self, config):
        super().__init__(config.n_embd, config.n_head, config.attn_pdrop, config.resid_pdrop, 32, 32)


class SequenceAxialAttention(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, H, W,
                 add_rel_pos=True, rel_pos_bins=32):
        super().__init__()

        self.rln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.cln1 = nn.LayerNorm(n_embd, eps=1e-4)
        self.ln2 = nn.LayerNorm(n_embd, eps=1e-4)
        self.attn_row = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.attn_col = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.add_rel_pos = add_rel_pos
        # self.rel_pos_bins = rel_pos_bins
        self.row_rel_pos_bias = nn.Linear(2 * H - 1, n_head, bias=False)
        self.col_rel_pos_bias = nn.Linear(2 * W - 1, n_head, bias=False)

    def _cal_1d_pos_emb(self, hidden_states, rel_pos_onehot_size, row=True):
        # hidden_states:[B,L,D], [1,L]
        position_ids = torch.arange(hidden_states.shape[1], dtype=torch.long).unsqueeze(0)
        # [1,1,L]-[1,L,1]-->[1,L,L]
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        rel_pos_mat -= torch.min(rel_pos_mat)
        # [1,L,L]->[1,L,L,D]
        rel_pos = F.one_hot(rel_pos_mat, num_classes=rel_pos_onehot_size * 2 - 1).type_as(hidden_states)
        # [1,L,L,D]->[1,L,L,H]->[1,H,L,L]
        if row:
            rel_pos = self.row_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)
        else:
            rel_pos = self.col_rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        rel_pos = rel_pos.contiguous()
        return rel_pos

    def forward(self, x, return_att=False, debug=False):  # x:[B,C,H,W], mask:[B,1,H,W]
        [b, n, c] = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        # [b, c, h, w] = x.shape
        x0 = x.clone()
        x0 = x0.permute(0, 2, 3, 1).reshape(b, h * w, c)
        mask_row = None
        mask_col = None

        # ROW ATTENTION
        x = x.permute(0, 3, 2, 1).reshape(b * w, h, c)
        if self.add_rel_pos:
            row_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=h, row=True)
        else:
            row_rel_pos = None
        x_row = self.attn_row(self.rln1(x), mask_row, row_rel_pos, return_att=return_att, debug=debug)
        if return_att:
            x_row, att_row = x_row
        else:
            att_row = None
        x_row = x_row.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b, h * w, c)

        # COL ATTENTION
        x = x.reshape(b, w, h, c).permute(0, 2, 1, 3).reshape(b * h, w, c)
        if self.add_rel_pos:
            col_rel_pos = self._cal_1d_pos_emb(x, rel_pos_onehot_size=w, row=False)
        else:
            col_rel_pos = None
        x_col = self.attn_col(self.cln1(x), mask_col, col_rel_pos, return_att=return_att)
        if return_att:
            x_col, att_col = x_col
        else:
            att_col = None
        x_col = x_col.reshape(b, h, w, c).reshape(b, h * w, c)

        # [B,HW,C]
        x = x0 + x_row + x_col
        x = x + self.ff(self.ln2(x))
        # x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = x.contiguous()

        if return_att:
            # att_row:[BW,head,H,H]
            att_row = torch.mean(att_row, dim=1).reshape(b, w, h, h)
            att_row = torch.sum(att_row, dim=2).permute(0, 2, 1)  # [b,h,w]
            # att_col:[BH,head,W,W]
            att_col = torch.mean(att_col, dim=1).reshape(b, h, w, w)
            att_col = torch.sum(att_col, dim=2)
            att_score = att_row * att_col
            return x, att_score
        else:
            return x


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

        self.config = config

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class GELU2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class my_Block_2(nn.Module):
    """ Transformer block with original GELU2 """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU2(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        [b, c, h, w] = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x.contiguous()
        return x


class EdgeLineGPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


model_config = EdgeLineGPTConfig(embd_pdrop=0.0, resid_pdrop=0.0, n_embd=256, block_size=32,
                                         attn_pdrop=0.0, n_layer=16, n_head=8)