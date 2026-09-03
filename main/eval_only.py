import torch
import os
import argparse
from src.train import build_datasets_tokenized, evaluate_generation, CONFIGS
from src.models import TransformerConfig, Seq2SeqTransformer
from src.utils import compute_metrics
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="C1")
    parser.add_argument("--data_path", default="data")
    parser.add_argument("--checkpoint", default="outputs/C1_checkpoint.pt")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--max_len", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, weights_only=False)
    cfg_dict = ckpt["config"]
    
    # Reconstruct TransformerConfig
    cfg = TransformerConfig(**cfg_dict)
    
    # 2. Load dataset to get test loader and tokenizer
    from src.dataset import load_pairs, train_val_test_split
    rows = load_pairs(path=args.data_path)
    train_rows, val_rows, test_rows = train_val_test_split(rows, seed=42)
    
    _, _, test_ds, collate, src_tok, tgt_tok, _ = build_datasets_tokenized(
        train_rows, val_rows, test_rows, args
    )
    
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate)

    # 3. Build model and load weights
    model = Seq2SeqTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 4. Run evaluation with debug prints enabled
    print("\n--- Running Evaluation & Inspecting Samples ---")
    preds, targets, decoding_tok_metrics = evaluate_generation(
        model, test_loader, device, tokenization="subword", tgt_tok=tgt_tok, max_len=args.max_len
    )
    
    metrics = compute_metrics(preds, targets, tokenized=True)
    metrics.update(decoding_tok_metrics)
    print("\nFinal Test Metrics:\n", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()