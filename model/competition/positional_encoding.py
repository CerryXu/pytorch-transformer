import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        # d_model: 一个标量。模型的维度，论文默认是512
        # max_seq_len: 一个标量。文本序列的最大长度

        super(PositionalEncoding, self).__init__()

        position_encoding = np.array([
            [pos / np.pow(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
        for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        # input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])

        return self.position_encoding(input_pos)