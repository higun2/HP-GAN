import numpy as np
import torch
import torch.nn as nn
from torch_utils import misc
from torch_utils import persistence
import torch.nn.functional as F

#----------------------------------------------------------------------------
class FT_Head(nn.Module):
    def __init__(self, gpus, batch_size=64, projector='512-1024-1024-1024', lambd=0.0051):
        super().__init__()
        self.batch_size = batch_size
        self.lambd = lambd
        self.gpus = gpus
        # projector
        sizes = list(map(int, projector.split('-'))) # ex) sizes = [512, 8192, 8192, 8192]

        # projector
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        if y1.ndim > 2:
            y1 = y1.mean([2,3])
        if y2.ndim > 2:
            y2 = y2.mean([2,3])

        z1 = self.projector(y1) # (batch, feature size)
        z2 = self.projector(y2)

        c = self.bn(z1).T @ self.bn(z2) # 512, 512

        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        if self.gpus > 1:
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum() # sum or mean
        off_diag = off_diagonal(c).pow_(2).sum() # sum or mean
        loss = on_diag + self.lambd * off_diag

        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
