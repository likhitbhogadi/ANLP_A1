"""
Normalization layers implemented from fundamental ops (no nn.LayerNorm).

  - CustomLayerNorm: standard LayerNorm (Ba et al. 2016), normalizes over the
    last dimension using mean and variance, then applies a learned affine
    transform (gamma, beta).

  - RMSNorm (Zhang & Sennrich, 2019): drops the mean-centering step and only
    rescales by the root-mean-square of the activations, with a single
    learned scale (gamma). Cheaper than LayerNorm and used in place of it in
    C4.
"""
import torch
import torch.nn as nn


class CustomLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt((x.pow(2)).mean(dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


def get_norm_layer(norm_type, d_model):
    if norm_type == "layernorm":
        return CustomLayerNorm(d_model)
    elif norm_type == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"Unknown norm_type: {norm_type}")
