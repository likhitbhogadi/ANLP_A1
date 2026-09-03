# ANLP Assignment 1: Custom Transformers & BLT

Likhit Bhogadi (Roll No: `2023101065`)

---

## **Environment Setup**

This repository utilizes `uv` for lightning-fast package management and dependency resolution.

1. **Clone and enter the submission directory:**
```bash
cd 2023101065_assignment1

```


2. **Initialize the virtual environment and install dependencies:**
```bash
uv venv
source .venv/bin/activate
uv pip install torch torchvision torchaudio wandb huggingface_hub evaluate scikit-learn matplotlib

```



---

## **Running the Experiments**

All five configurations (C1–C5) can be executed using the provided shell scripts or directly via `src.train`.

* **C1 (Base - Sinusoidal + MHA + LayerNorm):**
```bash
./run_c1.sh

```


* **C2 (RoPE Positional Encoding):**
```bash
./run_c2.sh

```


* **C3 (Grouped-Query Attention):**
```bash
./run_c3.sh

```


* **C4 (RMS Normalization):**
```bash
./run_c4.sh

```


* **C5 (Token-Free Byte Latent Transformer):**
```bash
./run_c5.sh

```



---

## **Hosted Model Checkpoints (Hugging Face Hub)**

Pre-trained model weights, tokenizer configurations, and evaluation metrics for each ablation study configuration are hosted on the Hugging Face Hub:

| Configuration | Architecture Details | Hugging Face Repository Link |
| --- | --- | --- |
| **C1** | Base (Sinusoidal + MHA + LayerNorm) | [likhitbhogadi/anlp-a1-C1](https://huggingface.co/likhitbhogadi/anlp-a1-C1) |
| **C2** | RoPE + MHA + LayerNorm | [likhitbhogadi/anlp-a1-C2](https://huggingface.co/likhitbhogadi/anlp-a1-C2) |
| **C3** | Sinusoidal + GQA + LayerNorm | [likhitbhogadi/anlp-a1-C3](https://huggingface.co/likhitbhogadi/anlp-a1-C3) |
| **C4** | Sinusoidal + MHA + RMSNorm | [likhitbhogadi/anlp-a1-C4](https://huggingface.co/likhitbhogadi/anlp-a1-C4) |
| **C5** | Sinusoidal + MHA + LayerNorm + BLT | [likhitbhogadi/anlp-a1-C5](https://huggingface.co/likhitbhogadi/anlp-a1-C5) |

---

## **Weights & Biases Tracking**

Training progress, loss trajectories, and system metrics were logged using Weights & Biases:

* **Main Ablation Study Project:** [anlp-a1-transformers Dashboard](https://wandb.ai/likhitbhogadi-iiit-hyderabad/anlp-a1-transformers)
* **Initial Exploratory Runs Project:** [anlp-a1-seq2seq-crypto Dashboard](https://wandb.ai/likhitbhogadi-iiit-hyderabad/anlp-a1-seq2seq-crypto)