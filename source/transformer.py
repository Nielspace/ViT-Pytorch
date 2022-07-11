import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Dropout, Linear, Conv2d, LayerNorm
import torch.nn.functional as F

import math
import copy

from embeddings import Embeddings
from attention_block import Block


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        num_attention_heads,
        linear_dim,
        dropout_rate,
        attention_dropout_rate,
        eps,
        std_norm,
    ):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=eps)
        for _ in range(num_layers):
            layer = Block(
                num_attention_heads,
                hidden_size,
                linear_dim,
                dropout_rate,
                attention_dropout_rate,
                eps,
                std_norm,
            )
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(
        self,
        img_size,
        hidden_size,
        in_channels,
        num_layers,
        num_attention_heads,
        linear_dim,
        dropout_rate,
        attention_dropout_rate,
        eps,
        std_norm,
    ):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size, hidden_size, in_channels)

        self.encoder = Encoder(
            num_layers,
            hidden_size,
            num_attention_heads,
            linear_dim,
            dropout_rate,
            attention_dropout_rate,
            eps,
            std_norm,
        )

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        num_classes,
        hidden_size,
        in_channels,
        num_layers,
        num_attention_heads,
        linear_dim,
        dropout_rate,
        attention_dropout_rate,
        eps,
        std_norm,
    ):
        super(VisionTransformer, self).__init__()
        self.classifier = "token"

        self.transformer = Transformer(
            img_size,
            hidden_size,
            in_channels,
            num_layers,
            num_attention_heads,
            linear_dim,
            dropout_rate,
            attention_dropout_rate,
            eps,
            std_norm,
        )
        self.head = Linear(hidden_size, num_classes)

    def forward(self, x, labels=None):
        x, attn_weights = self.transformer(x)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 400), labels.view(-1))
            return loss
        else:
            return logits, attn_weights


if __name__ == "__main__":
    from config import Config

    config = Config()
    x = torch.randn(1, config.IN_CHANNELS * config.IMG_SIZE * config.IMG_SIZE)
    x = x.reshape(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    model = VisionTransformer(
        img_size=config.IMG_SIZE,
        num_classes=config.NUM_CLASSES,
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
        num_layers=config.NUM_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        linear_dim=config.LINEAR_DIM,
        dropout_rate=config.DROPOUT_RATE,
        attention_dropout_rate=config.ATTENTION_DROPOUT_RATE,
        eps=config.EPS,
        std_norm=config.STD_NORM,
    )

    x = model(x)
    print(x)
