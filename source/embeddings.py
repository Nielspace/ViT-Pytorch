import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm

from config import Config

config = Config()


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""

    def __init__(self, img_size: int, hidden_size: int, in_channels: int):

        super(Embeddings, self).__init__()
        img_size = _pair(img_size)

        patch_size = _pair(img_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches + 1, hidden_size)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.dropout = Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


if __name__ == "__main__":
    x = torch.randn(1, config.IN_CHANNELS * config.IMG_SIZE * config.IMG_SIZE)
    x = x.reshape(1, config.IN_CHANNELS, config.IMG_SIZE, config.IMG_SIZE)

    embeddings = Embeddings(
        img_size=(config.IMG_SIZE, config.IMG_SIZE),
        hidden_size=config.HIDDEN_SIZE,
        in_channels=config.IN_CHANNELS,
    )
    print(embeddings(x))
