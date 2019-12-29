import torch
import torch.nn as nn
from model.competition.multi_head_attention import MultiHeadAttention
from model.competition.positional_wise_feed_forward import PositionalWiseFeedForward
from model.competition.positional_encoding import PositionalEncoding

from model.tool.padding_mask import padding_mask

class EncoderLayer(nn.Module):
    def __init__(self, model_dim=512, num_head=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_head, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):

        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )

        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, input_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(input_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []

        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions