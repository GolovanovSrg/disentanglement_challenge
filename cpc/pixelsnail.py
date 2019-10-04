# Based on https://github.com/rosinality/vq-vae-2-pytorch/blob/master/pixelsnail.py

from math import sqrt
from functools import partial, lru_cache

import torch
from torch import nn
from torch.nn import functional as F


class WNConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size,
                                                   stride=stride, padding=padding, bias=bias))

    @property
    def weight_g(self):
        return self.conv.weight_g

    @property
    def weight_v(self):
        return self.conv.weight_v

    def forward(self, input):
        out = self.conv(input)
        return out


class CausalConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding='downright'):
        super().__init__()

        if padding not in ['downright', 'down', 'causal']:
            raise ValueError(f'Wrong padding: expected one of ["downright", "down", "causal"], got {padding}')

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2

        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]
        elif padding == 'down' or padding == 'causal':
            pad = [kernel_size[1] // 2, kernel_size[1] // 2, kernel_size[0] - 1, 0]

        self.causal = kernel_size[1] // 2 if padding == 'causal' else 0
        self.pad = nn.ZeroPad2d(pad)
        self.conv = WNConv2d(in_channel, out_channel, kernel_size, stride=stride)

    def forward(self, input):
        out = self.pad(input)
        if self.causal > 0:
            self.conv.weight_v.data[:, :, -1, self.causal:].zero_()
        out = self.conv(out)

        return out


class GatedResBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, conv='wnconv2d', activation=nn.ELU(inplace=True),
                 dropout=0.1, auxiliary_channel=0, condition_dim=0):
        super().__init__()

        if conv not in ['wnconv2d', 'causal_downright', 'causal']:
            raise ValueError(f'Wrong conv: expected one of ["wnconv2d", "causal_downright", "causal"], got {conv}')

        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)
        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')
        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation
        self.conv1 = conv_module(in_channel, channel, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)
        self.gate = nn.GLU(1)

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        if condition_dim > 0:
            self.condition = nn.Linear(condition_dim, in_channel * 2, bias=False)

    def forward(self, input, aux_input=None, condition=None):
        out = self.activation(input)
        out = self.conv1(out)

        if aux_input is not None:
            aux_input = self.activation(aux_input)
            out = out + self.aux_conv(aux_input)

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if condition is not None:
            condition = self.condition(condition)
            out = out + condition.view(condition.shape[0], 1, 1, condition.shape[1])

        out = self.gate(out)
        out += input

        return out


class CausalAttention(nn.Module):
    @staticmethod
    @lru_cache(maxsize=64)
    def causal_mask(size, device):
        mask = torch.ones(size, size, dtype=torch.uint8, device=device)
        mask = torch.triu(mask, diagonal=1).t()
        mask = mask.unsqueeze(0)

        start_mask = torch.ones(size, dtype=torch.float32, device=device)
        start_mask.data[0] = 0
        start_mask = start_mask.unsqueeze(1)

        return mask, start_mask

    def __init__(self, query_channel, key_channel, channel, n_head=8, dropout=0.1):
        super().__init__()

        self.query = nn.utils.weight_norm(nn.Linear(query_channel, channel))
        self.key = nn.utils.weight_norm(nn.Linear(key_channel, channel))
        self.value = nn.utils.weight_norm(nn.Linear(key_channel, channel))
        self.dropout = nn.Dropout(dropout)
        self.dim_head = channel // n_head
        self.n_head = n_head

    def forward(self, query, key):
        def reshape(input):
            return input.view(input.shape[0], -1, self.n_head, self.dim_head).transpose(1, 2)

        batch, _, height, width = key.shape

        query_flat = query.view(batch, query.shape[1], -1).transpose(1, 2)
        key_flat = key.view(batch, key.shape[1], -1).transpose(1, 2)

        query = reshape(self.query(query_flat))
        key = reshape(self.key(key_flat)).transpose(2, 3)
        value = reshape(self.value(key_flat))

        attn = torch.matmul(query, key) / sqrt(self.dim_head)
        mask, start_mask = CausalAttention.causal_mask(height * width, attn.device)
        attn = attn.masked_fill(mask == 0, -1e8)
        attn = torch.softmax(attn, 3) * start_mask
        attn = self.dropout(attn)

        out = attn @ value
        out = out.transpose(1, 2).reshape(batch, height, width, self.dim_head * self.n_head)
        out = out.permute(0, 3, 1, 2)

        return out


class PixelBlock(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, n_res_block, attention=True, n_head=8,
                 activation=nn.ELU(inplace=True), dropout=0.1, condition_dim=0):
        super().__init__()

        self.resblocks = nn.ModuleList()
        for _ in range(n_res_block):
            block = GatedResBlock(in_channel, channel, kernel_size, conv='causal',
                                  activation=activation, dropout=dropout, condition_dim=condition_dim)
            self.resblocks.append(block)

        self.attention = attention
        if attention:
            self.key_resblock = GatedResBlock(in_channel * 2 + 2, in_channel, 1, activation=activation, dropout=dropout)
            self.query_resblock = GatedResBlock(in_channel + 2, in_channel, 1, activation=activation, dropout=dropout)
            self.causal_attention = CausalAttention(in_channel + 2, in_channel * 2 + 2, in_channel // 2, n_head=n_head, dropout=dropout)
            self.out_resblock = GatedResBlock(in_channel, in_channel, 1, activation=activation, dropout=dropout, auxiliary_channel=in_channel // 2)
        else:
            self.out = WNConv2d(in_channel + 2, in_channel, 1)

    def forward(self, input, background, condition=None):
        out = input
        for resblock in self.resblocks:
            out = resblock(out, condition=condition)

        if self.attention:
            key_cat = torch.cat([input, out, background], 1)
            key = self.key_resblock(key_cat)
            query_cat = torch.cat([out, background], 1)
            query = self.query_resblock(query_cat)
            attn_out = self.causal_attention(query, key)
            out = self.out_resblock(out, attn_out)
        else:
            bg_cat = torch.cat([out, background], 1)
            out = self.out(bg_cat)

        return out


class PixelSNAIL(nn.Module):
    @staticmethod
    def shift_down(input, size=1):
        return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]

    @staticmethod
    def shift_right(input, size=1):
        return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]

    @staticmethod
    def positions(height, width, device):
        # TODO: positional embedding
        coord_x = (torch.arange(height, dtype=torch.float32, device=device) - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(-1, -1, -1, width)
        coord_y = (torch.arange(width, dtype=torch.float32, device=device) - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(-1, -1, height, -1)
        positions = torch.cat([coord_x, coord_y], 1)

        return positions

    def __init__(self, in_channel, channel, kernel_size, n_block, n_res_block, res_channel, attention=True,
                 n_head=8, activation=nn.ReLU(inplace=True), dropout=0.1, condition_dim=0):
        super().__init__()

        kernel = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.horizontal = CausalConv2d(in_channel, channel, [kernel // 2, kernel], padding='down')
        self.vertical = CausalConv2d(in_channel, channel, [(kernel + 1) // 2, kernel // 2], padding='downright')

        self.blocks = nn.ModuleList()
        for _ in range(n_block):
            block = PixelBlock(channel, channel, kernel_size, n_res_block, attention=attention,
                               n_head=n_head, activation=activation, dropout=dropout, condition_dim=condition_dim)
        self.blocks.append(block)

        self.out = nn.Sequential(activation, nn.Conv2d(channel, in_channel, 1))
        self.background = None

    def forward(self, input, condition=None):
        horizontal = PixelSNAIL.shift_down(self.horizontal(input))
        vertical = PixelSNAIL.shift_right(self.vertical(input))
        out = horizontal + vertical

        if self.background is None or self.background.shape[2:] != input.shape[2:]:
            self.background = PixelSNAIL.positions(*input.shape[2:], input.device)
        background = self.background.expand(input.shape[0], -1, -1, -1)

        for block in self.blocks:
            out = block(out, background, condition=condition)
        out = self.out(out)

        return out
