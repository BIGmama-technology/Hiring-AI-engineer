import torch
import torch.nn as nn
from torch.nn import functional as F


class Mlp(nn.Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) < 3:
            raise ValueError("at least need two values to create an mlp")
        else:
            num_features = args

        self.layers = []
        for i in range(len(num_features) - 1):
            if i < (len(num_features) - 2):
                layer = nn.Sequential(
                    nn.Linear(num_features[i], num_features[i + 1]),
                    nn.BatchNorm1d(num_features[i + 1]),
                    nn.ReLU(),
                    nn.Dropout(0.1),  # (1 - 10/num_features[i+1]))
                )
            else:
                layer = nn.Linear(num_features[i], num_features[i + 1])

            self.layers.append(layer)
        self.layers = nn.ParameterList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
