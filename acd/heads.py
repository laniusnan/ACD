"""Angular Coordinate Interface; optional simplex ETF initialization."""
import torch
from torch import nn
from torch.nn import functional as F

class CosineHead(nn.Module):
    def __init__(self, weight, scale=30., etf=False):
        super().__init__()
        if scale <= 0:
            raise ValueError('scale must be positive')
        c, d = weight.shape
        initial = F.normalize(weight.detach().clone(), dim=1)
        if etf:
            if d < c - 1:
                raise ValueError('Simplex ETF needs feature dimension >= classes - 1')
            # Eigenbasis of the centering matrix spans the C-1 simplex directions.
            center = torch.eye(c, device=weight.device) - torch.ones(c, c, device=weight.device)/c
            _, basis = torch.linalg.eigh(center)
            rotation, _ = torch.linalg.qr(torch.randn(d, c-1, device=weight.device))
            initial = (c/(c-1))**.5 * basis[:, 1:] @ rotation.T
        self.weight = nn.Parameter(initial)
        self.register_buffer('scale', torch.tensor(float(scale)))

    def forward(self, features):
        return self.scale * F.linear(F.normalize(features, dim=-1), F.normalize(self.weight, dim=1))
