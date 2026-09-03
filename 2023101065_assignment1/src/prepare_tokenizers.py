import argparse
import os
from .dataset import load_pairs
from .tokenizer import BPETokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data", help="Directory containing dataset")
    parser.add_argument("--src_vocab_size", type=int, default=256)
    parser.add_argument("--tgt_vocab_size", type=int, default=1000)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_pairs(path=args.data_path)

    print(f"Training Source Tokenizer (Vocab Size: {args.src_vocab_size})...", flush=True)
    src_tok = BPETokenizer(vocab_size=args.src_vocab_size)
    src_tok.train([r["ciphertext"] for r in rows], name="Src-BPE", verbose=True)
    src_tok.save(os.path.join(args.output_dir, "src_tokenizer.json"))

    print(f"Training Target Tokenizer (Vocab Size: {args.tgt_vocab_size})...", flush=True)
    tgt_tok = BPETokenizer(vocab_size=args.tgt_vocab_size)
    tgt_tok.train([r["plaintext"] for r in rows], name="Tgt-BPE", verbose=True)
    tgt_tok.save(os.path.join(args.output_dir, "tgt_tokenizer.json"))

    print("Tokenizers successfully saved to", args.output_dir)

if __name__ == "__main__":
    main()