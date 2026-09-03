#!/bin/bash
set -e

PYTHON="${PYTHON:-python3}"

CONFIG="C5"
DATA_PATH="data"
CIPHER_PATH="data/brown_cipher.txt"
PLAIN_PATH="data/brown_plain.txt"

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

EPOCHS=60
BATCH_SIZE=64
LR=0.001
WARMUP=1000
WEIGHT_DECAY=0.01
CLIP=1.0
SEED=0

D_MODEL=256
NUM_HEADS=8
NUM_KV_HEADS=2
NUM_LAYERS=3
D_FF=1024
DROPOUT=0.1

# Drastically increased for uncompressed byte-level sequences
MAX_SRC=2048
MAX_TGT=512
VOCAB_SIZE=1000
BLT_PATCH_SIZE=4

OUTPUT_DIR="outputs"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="anlp-a1-transformers"

CMD_ARGS=(
    "--config" "$CONFIG"
    "--data_path" "$DATA_PATH"
    "--epochs" "$EPOCHS"
    "--batch_size" "$BATCH_SIZE"
    "--lr" "$LR"
    "--warmup" "$WARMUP"
    "--weight_decay" "$WEIGHT_DECAY"
    "--clip" "$CLIP"
    "--d_model" "$D_MODEL"
    "--num_heads" "$NUM_HEADS"
    "--num_kv_heads" "$NUM_KV_HEADS"
    "--num_layers" "$NUM_LAYERS"
    "--d_ff" "$D_FF"
    "--dropout" "$DROPOUT"
    "--max_src" "$MAX_SRC"
    "--max_tgt" "$MAX_TGT"
    "--vocab_size" "$VOCAB_SIZE"
    "--blt_patch_size" "$BLT_PATCH_SIZE"
    "--seed" "$SEED"
    "--output_dir" "$OUTPUT_DIR"
    "--wandb_project" "$WANDB_PROJECT"
)

if [ -n "$CIPHER_PATH" ] && [ -n "$PLAIN_PATH" ] && [ -f "$CIPHER_PATH" ] && [ -f "$PLAIN_PATH" ]; then
    CMD_ARGS+=("--cipher_path" "$CIPHER_PATH" "--plain_path" "$PLAIN_PATH")
fi

if [ "$USE_WANDB" = true ]; then
    CMD_ARGS+=("--use_wandb")
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "Starting C5 run (BLT Token-Free)"
echo "Device: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Batch Size: $BATCH_SIZE | Epochs: $EPOCHS | Vocab: Byte-Level"
echo "============================================================"

uv run python -m src.train "${CMD_ARGS[@]}"