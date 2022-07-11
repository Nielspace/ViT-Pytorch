import torch
from torch import nn
from torch.nn import CrossEntropyLoss, Dropout, Linear, Conv2d, LayerNorm
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, hidden_size, linear_dim, dropout_rate, std_norm):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, linear_dim)
        self.fc2 = Linear(linear_dim, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(dropout_rate)
        self.std_norm = std_norm
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=self.std_norm)
        nn.init.normal_(self.fc2.bias, std=self.std_norm)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
