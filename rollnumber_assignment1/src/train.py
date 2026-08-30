"""
Main training entry point.

Usage:
    python -m src.train --config C1 --data_path /path/to/dataset.csv --epochs 10
    python -m src.train --config C5 --data_path /path/to/dataset.csv --epochs 10

Configs C1-C5 map exactly to Table 1 of the assignment:
    C1: sinusoidal + MHA + LayerNorm + subword BPE   (base)
    C2: RoPE       + MHA + LayerNorm + subword BPE
    C3: sinusoidal + GQA + LayerNorm + subword BPE
    C4: sinusoidal + MHA + RMSNorm   + subword BPE
    C5: sinusoidal + MHA + LayerNorm + BLT (token-free)
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import (
    load_pairs, train_val_test_split,
    TokenizedSeq2SeqDataset, make_tokenized_collate_fn,
    ByteSeq2SeqDataset, make_byte_collate_fn,
)
from .tokenizer import BPETokenizer
from .models import TransformerConfig, Seq2SeqTransformer, BLTSeq2Seq
from .utils import compute_metrics, plot_training_curves


CONFIGS = {
    "C1": dict(positional_encoding="sinusoidal", attention_type="mha", norm_type="layernorm", tokenization="subword"),
    "C2": dict(positional_encoding="rope",       attention_type="mha", norm_type="layernorm", tokenization="subword"),
    "C3": dict(positional_encoding="sinusoidal", attention_type="gqa", norm_type="layernorm", tokenization="subword"),
    "C4": dict(positional_encoding="sinusoidal", attention_type="mha", norm_type="rmsnorm",   tokenization="subword"),
    "C5": dict(positional_encoding="sinusoidal", attention_type="mha", norm_type="layernorm", tokenization="blt"),
}


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets_tokenized(train_rows, val_rows, test_rows, args):
    src_tok = BPETokenizer(vocab_size=args.src_vocab_size)
    tgt_tok = BPETokenizer(vocab_size=args.tgt_vocab_size)
    src_tok.train([r["ciphertext"] for r in train_rows])
    tgt_tok.train([r["plaintext"] for r in train_rows])

    train_ds = TokenizedSeq2SeqDataset(train_rows, src_tok, tgt_tok, max_len=args.max_len)
    val_ds = TokenizedSeq2SeqDataset(val_rows, src_tok, tgt_tok, max_len=args.max_len)
    test_ds = TokenizedSeq2SeqDataset(test_rows, src_tok, tgt_tok, max_len=args.max_len)
    collate = make_tokenized_collate_fn(src_tok.pad_id, tgt_tok.pad_id)
    return train_ds, val_ds, test_ds, collate, src_tok, tgt_tok


def build_datasets_byte(train_rows, val_rows, test_rows, args):
    train_ds = ByteSeq2SeqDataset(train_rows, patch_size=args.blt_patch_size, max_len=args.max_len)
    val_ds = ByteSeq2SeqDataset(val_rows, patch_size=args.blt_patch_size, max_len=args.max_len)
    test_ds = ByteSeq2SeqDataset(test_rows, patch_size=args.blt_patch_size, max_len=args.max_len)
    collate = make_byte_collate_fn(args.blt_patch_size)
    return train_ds, val_ds, test_ds, collate


def run_epoch(model, loader, optimizer, device, tokenization, pad_id_for_loss=0, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            if tokenization == "subword":
                src = batch["src_ids"].to(device)
                tgt_in = batch["tgt_in"].to(device)
                tgt_out = batch["tgt_out"].to(device)
                logits = model(src, tgt_in)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1), ignore_index=pad_id_for_loss
                )
            else:  # blt
                src = batch["src_bytes"].to(device)
                tgt = batch["tgt_bytes"].to(device)
                logits = model(src, tgt)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(1, n_batches)


@torch.no_grad()
def evaluate_generation(model, loader, device, tokenization, tgt_tok=None, max_len=128):
    model.eval()
    all_preds, all_targets = [], []
    for batch in loader:
        if tokenization == "subword":
            src = batch["src_ids"].to(device)
            gen = model.greedy_decode(src, max_len=max_len)
            tgt_out = batch["tgt_out"].to(device)
            for i in range(src.size(0)):
                pred_str = tgt_tok.decode(gen[i].tolist())
                target_ids = tgt_out[i].tolist()
                target_ids = [t for t in target_ids if t != tgt_tok.pad_id]
                tgt_str = tgt_tok.decode(target_ids)
                all_preds.append(pred_str)
                all_targets.append(tgt_str)
        else:
            src = batch["src_bytes"].to(device)
            tgt = batch["tgt_bytes"].to(device)
            n_patches = max(1, tgt.size(1) // model.patch_size)
            gen = model.greedy_decode(src, max_patches=n_patches)
            for i in range(src.size(0)):
                pred_bytes = [b for b in gen[i].tolist() if b < 256]
                tgt_bytes = [b for b in tgt[i].tolist() if b < 256]
                all_preds.append("".join(chr(b) for b in pred_bytes))
                all_targets.append("".join(chr(b) for b in tgt_bytes))
    return all_preds, all_targets


def build_model(config_name, src_vocab_size, tgt_vocab_size, args, device):
    overrides = CONFIGS[config_name]
    cfg = TransformerConfig(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + 8,
        blt_patch_size=args.blt_patch_size,
        **overrides,
    )
    if cfg.tokenization == "subword":
        model = Seq2SeqTransformer(cfg)
    else:
        model = BLTSeq2Seq(cfg, src_byte_vocab=2, tgt_byte_vocab=ByteSeq2SeqDataset.PLAINTEXT_VOCAB)
    return model.to(device), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), required=True)
    parser.add_argument("--data_path", required=True, help="CSV/JSON/JSONL with 'ciphertext','plaintext' columns")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--src_vocab_size", type=int, default=512)
    parser.add_argument("--tgt_vocab_size", type=int, default=512)
    parser.add_argument("--blt_patch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="anlp-a1-seq2seq-crypto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_pairs(args.data_path)
    train_rows, val_rows, test_rows = train_val_test_split(rows, seed=args.seed)

    tokenization = CONFIGS[args.config]["tokenization"]
    tgt_tok = None
    if tokenization == "subword":
        train_ds, val_ds, test_ds, collate, src_tok, tgt_tok = build_datasets_tokenized(
            train_rows, val_rows, test_rows, args
        )
        src_vocab_size, tgt_vocab_size = src_tok.vocab_size_actual, tgt_tok.vocab_size_actual
        src_tok.save(os.path.join(args.output_dir, f"{args.config}_src_tokenizer.json"))
        tgt_tok.save(os.path.join(args.output_dir, f"{args.config}_tgt_tokenizer.json"))
    else:
        train_ds, val_ds, test_ds, collate = build_datasets_byte(train_rows, val_rows, test_rows, args)
        src_vocab_size, tgt_vocab_size = 2, ByteSeq2SeqDataset.PLAINTEXT_VOCAB

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model, cfg = build_model(args.config, src_vocab_size, tgt_vocab_size, args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.config}] {n_params:,} parameters, tokenization={tokenization}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, name=args.config, config=vars(args))

    history = {"train_loss": [], "val_loss": []}
    pad_id_for_loss = tgt_tok.pad_id if tgt_tok is not None else 0

    peak_mem_mb = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loss = run_epoch(model, train_loader, optimizer, device, tokenization, pad_id_for_loss, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, tokenization, pad_id_for_loss, train=False)

        if device.type == "cuda":
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / 1e6)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"[{args.config}] epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if wandb_run is not None:
            wandb_run.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "peak_mem_mb": peak_mem_mb})

    train_time_sec = time.time() - t0

    preds, targets = evaluate_generation(model, test_loader, device, tokenization, tgt_tok, max_len=args.max_len)
    metrics = compute_metrics(preds, targets, tokenized=(tokenization == "subword"))
    metrics["train_time_sec"] = train_time_sec
    metrics["peak_mem_mb"] = peak_mem_mb
    metrics["num_params"] = n_params
    print(f"[{args.config}] test metrics: {metrics}")

    if wandb_run is not None:
        wandb_run.log({f"test/{k}": v for k, v in metrics.items()})
        wandb_run.finish()

    plot_training_curves(history, os.path.join(args.output_dir, f"{args.config}_loss_curve.png"))
    with open(os.path.join(args.output_dir, f"{args.config}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.output_dir, f"{args.config}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    ckpt_path = os.path.join(args.output_dir, f"{args.config}_checkpoint.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": vars(cfg)}, ckpt_path)
    print(f"[{args.config}] saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
