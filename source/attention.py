import torch
from torch.nn import CrossEntropyLoss, Dropout, Linear, Conv2d, LayerNorm
import torch.nn.functional as F
from torch import nn

import math

from config import Config

config = Config()


class Attention(nn.Module):
    def __init__(self, num_attention_heads, hidden_size, attention_dropout_rate):

        super(Attention, self).__init__()

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(hidden_size, self.all_head_size)
        self.key = Linear(hidden_size, self.all_head_size)
        self.value = Linear(hidden_size, self.all_head_size)

        self.out = Linear(hidden_size, self.all_head_size)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = torch.nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


if __name__ == "__main__":
    from embeddings import Embeddings

    x = torch.randn(1, config.IN_CHANNELS * config.IMG_SIZE * config.IMG_SIZE)
    x = x.reshape(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    embeddings = Embeddings(
        img_size=(config.IMG_SIZE, config.IMG_SIZE),
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
    )

    att = Attention(
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
    )

    print(att(embeddings(x)))
