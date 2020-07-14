import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, featur_dim, n_classes, hidden_dims=[]):
        super(Head, self).__init__()
        self.head = nn.Linear(featur_dim, n_classes)

    def forward(self, x):
        return self.head(x)


class MLP(nn.Module):

    def __init__(self, in_dim, n_classes, hidden_dims=[]):
        super(MLP, self).__init__()
        assert len(hidden_dims) > 0
        layers = [nn.Linear(in_dim, hidden_dims[0]), nn.BatchNorm1d(hidden_dims[0])]
        layers.append(nn.ReLU())
        for l_idx in range(1, len(hidden_dims)):
            layers.append(
                nn.Linear(hidden_dims[l_idx-1], hidden_dims[l_idx])
            )
            layers.append(nn.BatchNorm1d(hidden_dims[l_idx]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims[-1], n_classes))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)


if __name__ == "__main__":
    import resnet
    model = resnet.resnet18().eval()
    head = MLP(512, 10, [256, 128])
    x = torch.randn(1, 3, 32, 32)
    logits = model(x)
    score = head(logits)
    print(logits.size(), score.size())
