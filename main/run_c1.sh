#!/bin/bash
# ==============================================================================
# Training Script for Configuration C1
# Config C1: Sinusoidal Positional Encoding + MHA + LayerNorm + Subword BPE
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# Python Executable (defaults to python3, can be overridden via environment variable)
PYTHON="${PYTHON:-python3}"

# ------------------------------------------------------------------------------
# Argument Configurations / Hyperparameters
# ------------------------------------------------------------------------------

# Config & Dataset Selection
CONFIG="C1"                                  # Options: C1, C2, C3, C4, C5
DATA_PATH="data"                             # Directory containing brown_cipher.txt & brown_plain.txt
CIPHER_PATH="data/brown_cipher.txt"         # Path to ciphertext file
PLAIN_PATH="data/brown_plain.txt"           # Path to plaintext file

# Hardware / GPU Selection
GPU_ID="${GPU_ID:-0}"                        # GPU index (e.g. 0, 1, 2)
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Basic Training Hyperparameters
EPOCHS=40                                    # Number of training epochs
BATCH_SIZE=64                                # Training and evaluation batch size
LR=0.0003                                    # Learning rate (3e-4)
SEED=42                                      # Random seed for reproducibility

# Architecture Hyperparameters
D_MODEL=256                                  # Hidden dimension size
NUM_HEADS=8                                  # Number of attention heads
NUM_KV_HEADS=2                               # Number of Key/Value heads (used for GQA in C3)
NUM_LAYERS=4                                 # Number of encoder and decoder layers
D_FF=1024                                    # Dimension of Feed-Forward Network
DROPOUT=0.1                                  # Dropout rate
MAX_LEN=256                                  # Maximum sequence length

# Vocabulary and Tokenization Configuration
SRC_VOCAB_SIZE=512                           # Source (ciphertext) BPE vocabulary size
TGT_VOCAB_SIZE=512                           # Target (plaintext) BPE vocabulary size
BLT_PATCH_SIZE=4                             # Patch size for BLT (used when config=C5)

# Output and Logging Configuration
OUTPUT_DIR="outputs/c1"                      # Directory to save checkpoints, plots, and metrics
USE_WANDB="${USE_WANDB:-true}"                    # Set to true to enable Weights & Biases logging
WANDB_PROJECT="anlp-a1-seq2seq-crypto"       # Weights & Biases project name

# ------------------------------------------------------------------------------
# Build and Execute Command
# ------------------------------------------------------------------------------

CMD_ARGS=(
    "--config" "$CONFIG"
    "--data_path" "$DATA_PATH"
    "--epochs" "$EPOCHS"
    "--batch_size" "$BATCH_SIZE"
    "--lr" "$LR"
    "--d_model" "$D_MODEL"
    "--num_heads" "$NUM_HEADS"
    "--num_kv_heads" "$NUM_KV_HEADS"
    "--num_layers" "$NUM_LAYERS"
    "--d_ff" "$D_FF"
    "--dropout" "$DROPOUT"
    "--max_len" "$MAX_LEN"
    "--src_vocab_size" "$SRC_VOCAB_SIZE"
    "--tgt_vocab_size" "$TGT_VOCAB_SIZE"
    "--blt_patch_size" "$BLT_PATCH_SIZE"
    "--seed" "$SEED"
    "--output_dir" "$OUTPUT_DIR"
    "--wandb_project" "$WANDB_PROJECT"
)

# Append specific cipher/plain paths if set and files exist
if [ -n "$CIPHER_PATH" ] && [ -n "$PLAIN_PATH" ] && [ -f "$CIPHER_PATH" ] && [ -f "$PLAIN_PATH" ]; then
    CMD_ARGS+=("--cipher_path" "$CIPHER_PATH" "--plain_path" "$PLAIN_PATH")
fi

# Append --use_wandb flag if enabled
if [ "$USE_WANDB" = true ]; then
    CMD_ARGS+=("--use_wandb")
fi

echo "============================================================"
echo "Starting training for Configuration: $CONFIG"
echo "Data Directory: $DATA_PATH"
if [ -n "$CIPHER_PATH" ] && [ -n "$PLAIN_PATH" ]; then
    echo "Ciphertext File: $CIPHER_PATH"
    echo "Plaintext File:  $PLAIN_PATH"
fi
echo "Output Directory: $OUTPUT_DIR"
echo "Target GPU ID:    $GPU_ID"
echo "============================================================"

# Execute Python training module using uv
uv run python -m src.train "${CMD_ARGS[@]}"
