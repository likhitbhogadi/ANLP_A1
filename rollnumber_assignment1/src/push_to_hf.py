"""
Small helper to upload a trained checkpoint (+ tokenizer files, if any) to
the Hugging Face Hub, so the report can link to hosted weights.

Usage:
    huggingface-cli login   # once, with a token that has write access
    python -m src.push_to_hf --repo_id <username>/anlp-a1-C1 --output_dir outputs --config C1
"""
import argparse
import glob
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", required=True, help="e.g. yourname/anlp-a1-C1")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--config", required=True, choices=["C1", "C2", "C3", "C4", "C5"])
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo

    create_repo(args.repo_id, private=args.private, exist_ok=True)
    api = HfApi()

    files = glob.glob(os.path.join(args.output_dir, f"{args.config}_*"))
    if not files:
        raise FileNotFoundError(f"No files found for config {args.config} in {args.output_dir}")

    for path in files:
        print(f"Uploading {path} ...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=args.repo_id,
        )
    print(f"Done. View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
