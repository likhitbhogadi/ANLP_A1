"""
Encoder-Decoder Transformer built entirely from the custom modules in
attention.py / positional.py / norm.py (no nn.Transformer, no
nn.MultiheadAttention). A single `TransformerConfig` selects between the
five ablation configurations C1-C5 described in the assignment:

    positional_encoding: "sinusoidal" | "rope"
    attention_type:      "mha" | "gqa"
    norm_type:           "layernorm" | "rmsnorm"
    tokenization:        "subword" | "blt"   (handled by build_model / dataset.py)

Pre-LN residual connections are used throughout (norm -> sublayer -> add),
which is what Table 1 refers to as "LayerNorm" / "RMSNorm" (i.e. which norm
implementation sits inside the Pre-LN residual block).
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention, GroupedQueryAttention, PositionwiseFeedForward
from .positional import SinusoidalPositionalEncoding
from .norm import get_norm_layer
from .blt import LocalEncoder, LocalDecoder


@dataclass
class TransformerConfig:
    src_vocab_size: int
    tgt_vocab_size: int
    d_model: int = 256
    num_heads: int = 8
    num_kv_heads: int = 2          # only used when attention_type == "gqa"
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    max_len: int = 512
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    positional_encoding: str = "sinusoidal"   # "sinusoidal" | "rope"
    attention_type: str = "mha"               # "mha" | "gqa"
    norm_type: str = "layernorm"              # "layernorm" | "rmsnorm"
    tokenization: str = "subword"             # "subword" | "blt"

    # BLT-only
    blt_patch_size: int = 4
    blt_local_layers: int = 1


def build_attention(cfg, use_rope):
    if cfg.attention_type == "mha":
        return MultiHeadAttention(cfg.d_model, cfg.num_heads, cfg.dropout, use_rope=use_rope, max_len=cfg.max_len)
    elif cfg.attention_type == "gqa":
        return GroupedQueryAttention(
            cfg.d_model, cfg.num_heads, cfg.num_kv_heads, cfg.dropout, use_rope=use_rope, max_len=cfg.max_len
        )
    raise ValueError(f"Unknown attention_type: {cfg.attention_type}")


class EncoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        use_rope = cfg.positional_encoding == "rope"
        self.norm1 = get_norm_layer(cfg.norm_type, cfg.d_model)
        self.self_attn = build_attention(cfg, use_rope)
        self.norm2 = get_norm_layer(cfg.norm_type, cfg.d_model)
        self.ffn = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, mask=None):
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, mask=mask)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        x = x + self.dropout(self.ffn(h))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        use_rope = cfg.positional_encoding == "rope"
        self.norm1 = get_norm_layer(cfg.norm_type, cfg.d_model)
        self.self_attn = build_attention(cfg, use_rope)
        self.norm2 = get_norm_layer(cfg.norm_type, cfg.d_model)
        self.cross_attn = build_attention(cfg, use_rope=False)  # cross-attn keys/values from encoder, no RoPE needed
        self.norm3 = get_norm_layer(cfg.norm_type, cfg.d_model)
        self.ffn = PositionwiseFeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, mask=self_mask)
        x = x + self.dropout(attn_out)

        h = self.norm2(x)
        cross_out, _ = self.cross_attn(h, memory, memory, mask=cross_mask)
        x = x + self.dropout(cross_out)

        h = self.norm3(x)
        x = x + self.dropout(self.ffn(h))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.src_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.scale = cfg.d_model ** 0.5
        self.use_sinusoidal = cfg.positional_encoding == "sinusoidal"
        if self.use_sinusoidal:
            self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.num_encoder_layers)])
        self.final_norm = get_norm_layer(cfg.norm_type, cfg.d_model)

    def forward(self, src_ids, src_mask=None):
        x = self.embed(src_ids) * self.scale
        if self.use_sinusoidal:
            x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask=src_mask)
        return self.final_norm(x)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.tgt_vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.scale = cfg.d_model ** 0.5
        self.use_sinusoidal = cfg.positional_encoding == "sinusoidal"
        if self.use_sinusoidal:
            self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_decoder_layers)])
        self.final_norm = get_norm_layer(cfg.norm_type, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.tgt_vocab_size)

    def forward(self, tgt_ids, memory, self_mask=None, cross_mask=None):
        x = self.embed(tgt_ids) * self.scale
        if self.use_sinusoidal:
            x = self.pos_enc(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, memory, self_mask=self_mask, cross_mask=cross_mask)
        x = self.final_norm(x)
        return self.out_proj(x)


def make_padding_mask(ids, pad_id):
    # (B, 1, 1, T) - broadcastable over heads and query positions
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(T, device):
    return torch.tril(torch.ones(T, T, device=device)).view(1, 1, T, T)


class Seq2SeqTransformer(nn.Module):
    """Used for C1, C2, C3, C4 (standard subword tokenization)."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.tokenization == "subword"
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def forward(self, src_ids, tgt_in_ids):
        src_mask = make_padding_mask(src_ids, self.cfg.pad_id)
        tgt_pad_mask = make_padding_mask(tgt_in_ids, self.cfg.pad_id)
        causal = make_causal_mask(tgt_in_ids.size(1), tgt_in_ids.device)
        tgt_mask = tgt_pad_mask * causal

        memory = self.encoder(src_ids, src_mask)
        logits = self.decoder(tgt_in_ids, memory, self_mask=tgt_mask, cross_mask=src_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_ids, max_len=128):
        self.eval()
        device = src_ids.device
        B = src_ids.size(0)
        src_mask = make_padding_mask(src_ids, self.cfg.pad_id)
        memory = self.encoder(src_ids, src_mask)

        ys = torch.full((B, 1), self.cfg.bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            causal = make_causal_mask(ys.size(1), device)
            logits = self.decoder(ys, memory, self_mask=causal, cross_mask=src_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.cfg.pad_id), next_token)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == self.cfg.eos_id)
            if finished.all():
                break
        return ys


class BLTSeq2Seq(nn.Module):
    """
    Used for C5 (token-free). Bytes/bits -> LocalEncoder -> patch embeddings
    -> shared global Encoder/Decoder (same architecture family as C1) ->
    per-patch hidden states -> LocalDecoder -> bytes/bits.
    """

    def __init__(self, cfg: TransformerConfig, src_byte_vocab=2, tgt_byte_vocab=258):
        super().__init__()
        assert cfg.tokenization == "blt"
        self.cfg = cfg
        self.patch_size = cfg.blt_patch_size

        self.local_encoder = LocalEncoder(
            src_byte_vocab, cfg.d_model, patch_size=cfg.blt_patch_size,
            num_local_layers=cfg.blt_local_layers, num_heads=max(1, cfg.num_heads // 2), dropout=cfg.dropout,
        )
        self.local_decoder = LocalDecoder(
            tgt_byte_vocab, cfg.d_model, patch_size=cfg.blt_patch_size,
            num_local_layers=cfg.blt_local_layers, num_heads=max(1, cfg.num_heads // 2), dropout=cfg.dropout,
        )

        # global transformer operates purely on continuous patch vectors, so we
        # re-use Encoder/Decoder but skip the embedding lookup (vocab size is
        # irrelevant / unused for the global stage).
        global_cfg = cfg
        self.global_encoder = Encoder(global_cfg)
        self.global_decoder = Decoder(global_cfg)
        # global_decoder.out_proj/embed on raw patch vectors are not meaningful for
        # BLT, so we bypass them and use the pre-final-norm hidden state directly.
        self.global_encoder.embed = nn.Identity()
        self.global_decoder.embed = nn.Identity()
        self.global_decoder.out_proj = nn.Identity()

    def _global_encode(self, patch_emb):
        x = patch_emb * self.global_encoder.scale
        if self.global_encoder.use_sinusoidal:
            x = self.global_encoder.pos_enc(x)
        x = self.global_encoder.dropout(x)
        for layer in self.global_encoder.layers:
            x = layer(x)
        return self.global_encoder.final_norm(x)

    def _global_decode(self, patch_emb, memory):
        x = patch_emb * self.global_decoder.scale
        if self.global_decoder.use_sinusoidal:
            x = self.global_decoder.pos_enc(x)
        x = self.global_decoder.dropout(x)
        T = x.size(1)
        causal = make_causal_mask(T, x.device)
        for layer in self.global_decoder.layers:
            x = layer(x, memory, self_mask=causal)
        return self.global_decoder.final_norm(x)

    def forward(self, src_bytes, tgt_bytes):
        """
        src_bytes: (B, T_src) raw input symbols, T_src % patch_size == 0
        tgt_bytes: (B, T_tgt) raw target symbols, T_tgt % patch_size == 0
        returns byte-level logits (B, T_tgt, tgt_byte_vocab)
        """
        src_patches = self.local_encoder(src_bytes)          # (B, n_src_patches, d_model)
        memory = self._global_encode(src_patches)

        # simple continuous surrogate patch embedding for the decoder's *input*
        # side (shifted by one patch so the model can't see the current patch).
        # Uses the local_decoder's byte embedding table since tgt_bytes can take
        # plaintext-side ids (up to tgt_byte_vocab), unlike the src byte_emb.
        tgt_patch_emb = self.local_decoder.byte_emb(tgt_bytes)
        B, T, D = tgt_patch_emb.shape
        tgt_patch_emb = tgt_patch_emb.view(B, T // self.patch_size, self.patch_size, D).mean(dim=2)
        # shift right by one patch (prepend zero patch, drop last) for autoregression at the patch level
        zero_patch = torch.zeros(B, 1, D, device=tgt_patch_emb.device, dtype=tgt_patch_emb.dtype)
        tgt_patch_emb_shifted = torch.cat([zero_patch, tgt_patch_emb[:, :-1]], dim=1)

        patch_hidden = self._global_decode(tgt_patch_emb_shifted, memory)
        logits = self.local_decoder(patch_hidden, target_byte_ids=tgt_bytes)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_bytes, max_patches=64):
        self.eval()
        device = src_bytes.device
        B = src_bytes.size(0)
        src_patches = self.local_encoder(src_bytes)
        memory = self._global_encode(src_patches)

        D = self.cfg.d_model
        generated_bytes = []
        prev_patch_emb = torch.zeros(B, 1, D, device=device)

        for _ in range(max_patches):
            patch_hidden = self._global_decode(prev_patch_emb, memory)  # (B, t, D)
            last_hidden = patch_hidden[:, -1:, :]
            bytes_t = self.local_decoder.greedy_decode(last_hidden)  # (B, patch_size)
            generated_bytes.append(bytes_t)

            new_patch_emb = self.local_decoder.byte_emb(bytes_t).mean(dim=1, keepdim=True)
            prev_patch_emb = torch.cat([prev_patch_emb, new_patch_emb], dim=1)

        return torch.cat(generated_bytes, dim=1)
