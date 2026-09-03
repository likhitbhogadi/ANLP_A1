"""
Positional encoding schemes implemented from scratch.

  - SinusoidalPositionalEncoding: classic additive absolute encoding
    (Vaswani et al. 2017), added to token embeddings once before the stack.

  - RotaryPositionalEmbedding (RoPE, Su et al. 2021): rotates pairs of
    dimensions of Q and K by an angle proportional to absolute position,
    which makes the dot product Q.K depend only on relative position.
    Applied inside attention (see attention.py), not added to embeddings.
"""
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        # handle odd d_model gracefully
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        """x: (B, T, d_model)"""
        return x + self.pe[:, : x.size(1)].to(x.dtype)


class RotaryPositionalEmbedding(nn.Module):
    """
    Precomputes cos/sin tables of shape (1, 1, max_len, d_k) and rotates
    query/key tensors of shape (B, H, T, d_k).
    """

    def __init__(self, d_k, max_len=4096, base=10000.0):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires an even head dimension"
        inv_freq = 1.0 / (base ** (torch.arange(0, d_k, 2).float() / d_k))
        t = torch.arange(max_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)  # (max_len, d_k/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_len, d_k)
        self.register_buffer("cos", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """q: (B, Hq, Tq, d_k), k: (B, Hk, Tk, d_k)"""
        Tq, Tk = q.size(2), k.size(2)
        cos_q = self.cos[:, :, :Tq].to(q.dtype)
        sin_q = self.sin[:, :, :Tq].to(q.dtype)
        cos_k = self.cos[:, :, :Tk].to(k.dtype)
        sin_k = self.sin[:, :, :Tk].to(k.dtype)

        q_rot = q * cos_q + self._rotate_half(q) * sin_q
        k_rot = k * cos_k + self._rotate_half(k) * sin_k
        return q_rot, k_rot
