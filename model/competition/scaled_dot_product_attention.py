import torch
import torch.nn as nn

import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        # q: Queries张量，形状为[B, L_q, D_q]
        # k: Keys张量，形状为[B, L_k, D_k]
        # v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        # scale: 缩放因子，一个浮点标量
        # attn_mask: Masking张量，形状为[B, L_q, L_k]
        attention = torch.bmm(q, k.transpose(1, 2))

        # attention [B, L_q, L_k]

        if scale:
            attention = attention * scale

        if attn_mask:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)

        # 计算softmax
        attention = attention.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)

        # context [B, L_q, D_v]

        return context, attention
