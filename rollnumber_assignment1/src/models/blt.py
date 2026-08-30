"""
Simplified Byte Latent Transformer (BLT) building blocks.

Real BLT (Pagnoni et al., 2024) uses an entropy-based dynamic patcher. For
this assignment we use a *fixed-size* patching scheme (patch_size bytes per
patch) which keeps the architecture tractable while preserving the key idea:
bytes are grouped into local "patches", a small local transformer encodes
each patch into a single latent vector, the *global* transformer (shared
architecture with C1-C4) operates over the sequence of patch vectors, and a
local decoder expands the global transformer's patch-level outputs back into
raw bytes/bits autoregressively.

No subword vocabulary is used anywhere in this pipeline - the only "vocab"
is the raw alphabet (e.g. 2 symbols for bits, or 256 for bytes) plus a
handful of special ids.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadAttention, PositionwiseFeedForward
from .norm import CustomLayerNorm


class LocalTransformerBlock(nn.Module):
    """A tiny pre-LN transformer block used inside the local encoder/decoder."""

    def __init__(self, d_model, num_heads=4, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = CustomLayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.norm2 = CustomLayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout=dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, mask=mask)
        x = x + attn_out
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class LocalEncoder(nn.Module):
    """
    Byte-level embedding + local self-attention within each fixed-size patch,
    pooled (mean) into a single patch embedding of size d_model that is fed
    to the global encoder/decoder.

    Input:  byte_ids (B, T)  with T divisible by patch_size (padded upstream)
    Output: patch_embeddings (B, T // patch_size, d_model)
    """

    def __init__(self, byte_vocab_size, d_model, patch_size=4, num_local_layers=1,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.byte_emb = nn.Embedding(byte_vocab_size, d_model)
        self.local_pos = nn.Parameter(torch.randn(1, patch_size, d_model) * 0.02)
        self.layers = nn.ModuleList(
            [LocalTransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_local_layers)]
        )

    def forward(self, byte_ids):
        B, T = byte_ids.shape
        assert T % self.patch_size == 0, "sequence length must be padded to a multiple of patch_size"
        n_patches = T // self.patch_size

        x = self.byte_emb(byte_ids)  # (B, T, d_model)
        x = x.view(B * n_patches, self.patch_size, self.d_model)
        x = x + self.local_pos

        for layer in self.layers:
            x = layer(x)

        patch_emb = x.mean(dim=1)  # mean-pool within patch -> (B * n_patches, d_model)
        patch_emb = patch_emb.view(B, n_patches, self.d_model)
        return patch_emb


class LocalDecoder(nn.Module):
    """
    Expands each patch-level hidden state produced by the global decoder back
    into `patch_size` raw byte/bit logits. Each patch's bytes are decoded
    with a small autoregressive local transformer conditioned (via cross
    attention) on that patch's global hidden state, and teacher-forced with
    the previous ground-truth byte of the same patch during training.
    """

    def __init__(self, byte_vocab_size, d_model, patch_size=4, num_local_layers=1,
                 num_heads=4, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.byte_vocab_size = byte_vocab_size
        self.byte_emb = nn.Embedding(byte_vocab_size, d_model)
        self.local_pos = nn.Parameter(torch.randn(1, patch_size, d_model) * 0.02)

        self.self_layers = nn.ModuleList(
            [LocalTransformerBlock(d_model, num_heads, dropout=dropout) for _ in range(num_local_layers)]
        )
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_norm = CustomLayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, byte_vocab_size)

        causal = torch.tril(torch.ones(patch_size, patch_size)).view(1, 1, patch_size, patch_size)
        self.register_buffer("causal_mask", causal, persistent=False)

    def forward(self, patch_hidden, target_byte_ids=None):
        """
        patch_hidden:    (B, n_patches, d_model) - global decoder output, one vector per patch
        target_byte_ids: (B, n_patches * patch_size) ground-truth bytes for teacher forcing (train time)
        returns logits: (B, n_patches * patch_size, byte_vocab_size)
        """
        B, n_patches, D = patch_hidden.shape
        device = patch_hidden.device

        if target_byte_ids is not None:
            tgt = target_byte_ids.view(B * n_patches, self.patch_size)
            # shift right: prepend a zero ("start") byte id, drop last
            start = torch.zeros(B * n_patches, 1, dtype=torch.long, device=device)
            shifted = torch.cat([start, tgt[:, :-1]], dim=1)
            x = self.byte_emb(shifted) + self.local_pos
        else:
            x = self.local_pos.expand(B * n_patches, self.patch_size, D).clone()

        for layer in self.self_layers:
            layer.attn_mask = self.causal_mask
            h = layer.norm1(x)
            attn_out, _ = layer.attn(h, h, h, mask=self.causal_mask)
            x = x + attn_out
            h = layer.norm2(x)
            x = x + layer.ffn(h)

        cond = patch_hidden.reshape(B * n_patches, 1, D)
        h = self.cross_norm(x)
        cross_out, _ = self.cross_attn(h, cond, cond)
        x = x + cross_out

        logits = self.out_proj(x)  # (B*n_patches, patch_size, vocab)
        logits = logits.view(B, n_patches * self.patch_size, self.byte_vocab_size)
        return logits

    @torch.no_grad()
    def greedy_decode(self, patch_hidden):
        """Autoregressively decode bytes for each patch without teacher forcing."""
        B, n_patches, D = patch_hidden.shape
        device = patch_hidden.device
        generated = torch.zeros(B * n_patches, self.patch_size, dtype=torch.long, device=device)
        cur = torch.zeros(B * n_patches, 1, dtype=torch.long, device=device)

        cond = patch_hidden.reshape(B * n_patches, 1, D)

        for t in range(self.patch_size):
            x = self.byte_emb(
                torch.cat([cur, torch.zeros(B * n_patches, self.patch_size - 1 - t, dtype=torch.long, device=device)], dim=1)
                if t > 0 else cur
            )
            pad_len = self.patch_size - x.size(1)
            if pad_len > 0:
                x = torch.cat([x, torch.zeros(B * n_patches, pad_len, D, device=device)], dim=1)
            x = x + self.local_pos
            causal = self.causal_mask
            for layer in self.self_layers:
                h = layer.norm1(x)
                attn_out, _ = layer.attn(h, h, h, mask=causal)
                x = x + attn_out
                h = layer.norm2(x)
                x = x + layer.ffn(h)
            h = self.cross_norm(x)
            cross_out, _ = self.cross_attn(h, cond, cond)
            x = x + cross_out
            step_logits = self.out_proj(x[:, t, :])
            next_byte = step_logits.argmax(dim=-1)
            generated[:, t] = next_byte
            if t + 1 < self.patch_size:
                cur = torch.cat([cur, next_byte.unsqueeze(1)], dim=1)

        return generated.view(B, n_patches * self.patch_size)
