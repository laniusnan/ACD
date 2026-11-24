"""Angular Coordinate Interface; optional simplex ETF initialization."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def simplex_frame(weight):
    """Return a simplex basis and an orthonormal frame using the caller's RNG."""
    classes, dimension = weight.shape
    if dimension < classes - 1:
        raise ValueError(f'Simplex ETF needs feature dimension >= classes - 1; '
                         f'got {dimension} features and {classes} classes')
    dtype = torch.float64 if weight.dtype == torch.float64 else torch.float32
    center = torch.eye(classes, device=weight.device, dtype=dtype)
    center -= torch.ones_like(center) / classes
    _, basis = torch.linalg.eigh(center)
    frame, _ = torch.linalg.qr(torch.randn(dimension, classes-1,
                                          device=weight.device, dtype=dtype))
    return (((classes/(classes-1))**.5 * basis[:, 1:]).to(weight.dtype),
            frame.to(weight.dtype))


class CosineHead(nn.Module):
    def __init__(self, weight, scale=30., etf=False):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be finite and positive')
        if weight.ndim != 2 or not weight.is_floating_point():
            raise ValueError('weight must be a floating-point class-by-feature matrix')
        classes, feature_dimension = weight.shape
        if classes < 2 or feature_dimension < 1:
            raise ValueError('At least two classes and one feature are required')
        initial = F.normalize(weight.detach().clone(), dim=1)
        if etf:
            basis, frame = simplex_frame(weight)
            initial = basis @ frame.T
        self.weight = nn.Parameter(initial)
        self.register_buffer('scale', weight.new_tensor(float(scale)))

    def forward(self, features):
        return self.scale * F.linear(F.normalize(features, dim=-1), F.normalize(self.weight, dim=1))


class RotatingETFHead(nn.Module):
    """Learn the orientation of a strict simplex on the Stiefel manifold.

    The minimal C-1 frame spans the simplex. Project the gradient before SGD,
    then retract both the frame and momentum after every optimizer step.
    """
    def __init__(self, weight, scale=30.):
        super().__init__()
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be finite and positive')
        if weight.ndim != 2 or not weight.is_floating_point() or weight.shape[0] < 2:
            raise ValueError('weight must have at least two floating-point class rows')
        basis, frame = simplex_frame(weight)
        self.register_buffer('basis', basis)
        self.register_buffer('scale', weight.new_tensor(float(scale)))
        self.frame = nn.Parameter(frame)

    def directions(self):
        return F.normalize(self.basis @ self.frame.T, dim=1)

    def forward(self, features):
        return self.scale * F.linear(F.normalize(features, dim=-1), self.directions())

    @staticmethod
    def tangent(frame, value):
        inner = frame.T @ value
        return value - frame @ ((inner + inner.T) * .5)

    @torch.no_grad()
    def project_gradient(self):
        if self.frame.grad is None:
            raise RuntimeError('Rotating ETF frame received no gradient')
        self.frame.grad.copy_(self.tangent(self.frame, self.frame.grad))

    @torch.no_grad()
    def retract(self, optimizer):
        q, r = torch.linalg.qr(self.frame.double(), mode='reduced')
        diagonal = r.diagonal()
        if not torch.isfinite(r).all() or (diagonal.abs() < 1e-12).any():
            raise FloatingPointError('Rank-deficient/nonfinite ETF retraction')
        self.frame.copy_((q * diagonal.sign()).to(self.frame.dtype))
        momentum = optimizer.state[self.frame].get('momentum_buffer')
        if momentum is not None:
            momentum.copy_(self.tangent(self.frame, momentum))

    @torch.no_grad()
    def geometry_error(self):
        directions = self.directions().double()
        classes = len(directions)
        expected = torch.eye(classes, device=directions.device, dtype=directions.dtype)
        expected = expected * classes/(classes-1) - 1/(classes-1)
        error = float((directions @ directions.T - expected).abs().max())
        if not math.isfinite(error) or error > 2e-5:
            raise FloatingPointError(f'ETF geometry invariant violated: {error}')
        return error


class LiftedClassifier(nn.Module):
    """A shared nonlinear feature lift, used in every arm of a low-dim pair."""
    def __init__(self, input_dimension, output_dimension, classes, bias=True):
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_dimension, output_dimension), nn.ReLU())
        self.classifier = nn.Linear(output_dimension, classes, bias=bias)

    def forward(self, features):
        return self.classifier(self.projection(features))


def optimizer_groups(model):
    frames = [m.frame for m in model.modules() if isinstance(m, RotatingETFHead)]
    if not frames:
        return model.parameters()
    frame_ids = {id(p) for p in frames}
    return [{'params': [p for p in model.parameters() if id(p) not in frame_ids]},
            # Squared frame norm is constant on Stiefel: its tangent decay is zero.
            {'params': frames, 'weight_decay': 0.}]
