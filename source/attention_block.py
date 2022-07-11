import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Dropout, Linear, Conv2d, LayerNorm
import torch.nn.functional as F

from linear import Mlp
from attention import Attention


class Block(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        hidden_size,
        linear_dim,
        dropout_rate,
        attention_dropout_rate,
        eps,
        std_norm,
    ):

        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=eps)
        self.ffn_norm = LayerNorm(hidden_size, eps=eps)
        self.ffn = Mlp(
            hidden_size=hidden_size,
            linear_dim=linear_dim,
            dropout_rate=dropout_rate,
            std_norm=std_norm,
        )
        self.attn = Attention(
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            attention_dropout_rate=attention_dropout_rate,
        )

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


if __name__ == "__main__":
    from embeddings import Embeddings
    from config import Config

    config = Config()

    x = torch.randn(1, config.IN_CHANNELS * config.IMG_SIZE * config.IMG_SIZE)
    x = x.reshape(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    embeddings = Embeddings(
        img_size=(config.IMG_SIZE, config.IMG_SIZE),
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
    )

    b = Block(
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        hidden_size=config.HIDDEN_SIZE,
        linear_dim=config.LINEAR_DIM,
        dropout_rate=config.DROPOUT_RATE,
        attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
        eps=config.EPS,
        std_norm=config.STD_NORM,
    )
    x = embeddings(x)
    x = b(x)
    print(x)
