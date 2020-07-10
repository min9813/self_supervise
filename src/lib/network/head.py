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
