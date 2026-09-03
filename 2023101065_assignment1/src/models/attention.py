"""
Attention modules implemented from fundamental PyTorch ops (no nn.MultiheadAttention).

Implements:
  - scaled_dot_product_attention: the core Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V
  - MultiHeadAttention (MHA)
  - GroupedQueryAttention (GQA): fewer K/V heads than Q heads, K/V heads are repeated
    to match the number of query heads (as in the GQA paper, Ainslie et al. 2023).

Both attention blocks optionally apply Rotary Positional Embeddings (RoPE) to
Q/K right before the dot product, so that positional information is injected
inside attention rather than added to the input embeddings.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """
    q: (B, H, Tq, d_k)
    k: (B, H, Tk, d_k)
    v: (B, H, Tk, d_k)
    mask: broadcastable to (B, H, Tq, Tk); positions with 0 are masked out.
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    out = torch.matmul(attn, v)
    return out, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1, use_rope=False, max_len=4096):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            from .positional import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(self.d_k, max_len)

    def _split_heads(self, x, num_heads):
        B, T, _ = x.shape
        return x.view(B, T, num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        q = self._split_heads(self.w_q(query), self.num_heads)
        k = self._split_heads(self.w_k(key), self.num_heads)
        v = self._split_heads(self.w_v(value), self.num_heads)

        if self.use_rope:
            q, k = self.rope(q, k)

        out, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(out), attn


class GroupedQueryAttention(nn.Module):
    """
    Same interface as MultiHeadAttention but K/V are projected into a smaller
    number of heads (num_kv_heads) and then repeated (n_rep = num_heads // num_kv_heads)
    times so every query head still has a matching key/value head to attend to.
    When num_kv_heads == num_heads this degenerates to standard MHA;
    when num_kv_heads == 1 this degenerates to Multi-Query Attention.
    """

    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1, use_rope=False, max_len=4096):
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            from .positional import RotaryPositionalEmbedding
            self.rope = RotaryPositionalEmbedding(self.d_k, max_len)

    def _repeat_kv(self, x):
        # x: (B, num_kv_heads, T, d_k) -> (B, num_heads, T, d_k)
        B, H_kv, T, D = x.shape
        if self.n_rep == 1:
            return x
        x = x[:, :, None, :, :].expand(B, H_kv, self.n_rep, T, D)
        return x.reshape(B, H_kv * self.n_rep, T, D)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        q = self.w_q(query).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(B, -1, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(B, -1, self.num_kv_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            q, k = self.rope(q, k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        out, attn = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        return self.w_o(out), attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if activation == "gelu" else F.relu

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))

