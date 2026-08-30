# Assignment 1 — Custom Transformers & BLT

Sequence-to-sequence transformer (built from scratch in PyTorch, no
`nn.Transformer` / `nn.MultiheadAttention`) that learns to map encrypted
binary sequences to plaintext, plus a 5-way architectural ablation (C1–C5)
covering positional encoding (sinusoidal vs RoPE), attention (MHA vs GQA),
normalization (LayerNorm vs RMSNorm), and tokenization (learned subword BPE
vs token-free BLT).

## 1. Setup

```bash
python -m venv venv && source venv/bin/activate
pip install torch matplotlib nltk rouge-score wandb huggingface_hub
```

## 2. Data

Download the assignment dataset and convert/point to a single file with one
row per example and two fields:

```json
{"ciphertext": "0100101101000110...", "plaintext": "attack at dawn"}
```

Supported formats: `.jsonl`, `.json` (list of dicts), `.csv` / `.tsv` (with
`ciphertext`,`plaintext` header columns). If the raw dataset ships in a
different shape, write a small one-off script to reshape it into this format
before running the commands below — `src/dataset.py::load_pairs` is the only
place that needs the file to look like this.

The loader does a 80/10/10 train/val/test split internally
(`train_val_test_split`, seeded, in `src/dataset.py`).

## 3. Train a single configuration

```bash
python -m src.train --config C1 --data_path data/dataset.jsonl \
    --epochs 20 --batch_size 32 --d_model 256 --num_heads 8 \
    --num_kv_heads 2 --num_layers 4 --d_ff 1024 --max_len 256 \
    --src_vocab_size 512 --tgt_vocab_size 512 \
    --output_dir outputs --use_wandb --wandb_project anlp-a1-seq2seq-crypto
```

`--config` selects one of `C1, C2, C3, C4, C5` (see table below); every other
flag is shared across configs so the ablation is controlled. For `C5`
(BLT / token-free) the `--src_vocab_size` / `--tgt_vocab_size` flags are
ignored (vocab is fixed to raw bits/bytes) and `--blt_patch_size` (default 4)
controls the fixed patch size used by the local encoder/decoder.

| Config | Positional Encoding | Attention | Normalization | Tokenization |
|--------|---------------------|-----------|----------------|--------------|
| C1 (base) | Sinusoidal | MHA | LayerNorm | Subword BPE (learned) |
| C2 | RoPE | MHA | LayerNorm | Subword BPE |
| C3 | Sinusoidal | GQA | LayerNorm | Subword BPE |
| C4 | Sinusoidal | MHA | RMSNorm | Subword BPE |
| C5 | Sinusoidal | MHA | LayerNorm | BLT (token-free) |

## 4. Run the full ablation (C1–C5) in one go

```bash
python -m src.run_ablation --data_path data/dataset.jsonl --epochs 20 \
    --output_dir outputs --use_wandb \
    --extra_args "--batch_size 32 --d_model 256 --num_heads 8 --num_layers 4 --d_ff 1024"
```

This produces, per config, in `outputs/`: `*_metrics.json`, `*_history.json`,
`*_loss_curve.png`, `*_checkpoint.pt`, tokenizer JSON files, and, after all
five runs finish, `ablation_summary.json` plus `compare_*.png` bar charts.

## 5. Evaluation metrics

Computed with greedy decoding on the held-out test split
(`src/utils.py::compute_metrics`):
Bit-level accuracy, Sequence accuracy, Levenshtein distance, BLEU, ROUGE-1/L
(BLEU/ROUGE only reported for the tokenized configs C1–C4, as specified).

## 6. Pushing checkpoints to Hugging Face

```bash
huggingface-cli login
python -m src.push_to_hf --repo_id <your-username>/anlp-a1-C1 --output_dir outputs --config C1
# repeat for C2..C5
```

## 7. Code layout

```
src/
  models/
    attention.py    # scaled dot-product attention, MHA, GQA (from scratch)
    positional.py   # sinusoidal absolute PE, RoPE (from scratch)
    norm.py         # LayerNorm, RMSNorm (from scratch)
    blt.py          # BLT local encoder / local decoder (patchified bytes)
    transformer.py  # Encoder/Decoder stacks + Seq2SeqTransformer + BLTSeq2Seq,
                     # config-driven switch between C1-C5
  tokenizer.py      # from-scratch BPE (subword) tokenizer
  dataset.py        # tokenized + byte-level dataset/dataloader utilities
  train.py          # training loop, greedy decode eval, WandB logging
  run_ablation.py   # runs C1-C5 back to back + aggregates comparison plots
  push_to_hf.py     # uploads a checkpoint + tokenizer to the HF Hub
  utils.py          # metrics (bit acc, seq acc, Levenshtein, BLEU, ROUGE) + plots
outputs/            # metrics json, loss curves, comparison plots, checkpoints
Report.pdf
```

## 8. Links (fill in after training on the full dataset)

- WandB runs: `<add link here>`
- Hugging Face checkpoints: `<add link here>` (one repo per config, or one
  repo with 5 subfolders)

## 9. Notes / simplifications

- **BLT patching**: the assignment's BLT is simplified to *fixed-size*
  patches (`--blt_patch_size`, default 4) rather than the entropy-based
  dynamic patcher from the original BLT paper, to keep the architecture
  tractable within the assignment scope. This is called out explicitly in
  the report.
- **Tokenizer**: `src/tokenizer.py` is a from-scratch BPE implementation
  (no `sentencepiece` / `tokenizers` / `tiktoken`). For the ciphertext side
  the base alphabet is `{'0','1'}` so learned merges yield variable-length
  bit n-grams as subword units (never fixed 8-bit chunks). For the
  plaintext side the base alphabet is characters, with `</w>` word-boundary
  markers.
- All five configs share identical depth/width/optimizer/batch size; only
  the single component named in Table 1 differs between a config and C1.
