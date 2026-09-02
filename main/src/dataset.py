"""
Dataset utilities.

Expects the encrypted-binary -> plaintext dataset to be provided as a CSV /
JSON / JSONL file with two columns/keys:
    "ciphertext": a string of '0'/'1' characters (the encrypted binary sequence)
    "plaintext":  the corresponding plaintext string

Two dataset flavours are provided:
    TokenizedSeq2SeqDataset - for C1-C4: both sides are encoded with a
        from-scratch BPE tokenizer (src/tokenizer.py).
    ByteSeq2SeqDataset - for C5 (BLT): both sides are kept as raw
        integer symbol sequences (no vocabulary / subword tokenization at
        all), padded to a multiple of `patch_size`.
"""
import os
import json
import csv
import random
from typing import List, Tuple, Dict, Optional

import torch
from torch.utils.data import Dataset

from .tokenizer import BPETokenizer


def load_pairs(path: Optional[str] = None, cipher_path: Optional[str] = None, plain_path: Optional[str] = None) -> List[Dict[str, str]]:
    if cipher_path and plain_path:
        with open(cipher_path, "r", encoding="utf-8") as fc, open(plain_path, "r", encoding="utf-8") as fp:
            c_lines = [line.strip() for line in fc]
            p_lines = [line.strip() for line in fp]
        assert len(c_lines) == len(p_lines), f"Mismatch in line counts: cipher ({len(c_lines)}) vs plain ({len(p_lines)})"
        return [{"ciphertext": c, "plaintext": p} for c, p in zip(c_lines, p_lines) if c and p]

    if not path:
        raise ValueError("Must provide either path or both cipher_path and plain_path")

    if os.path.isdir(path):
        c_path = os.path.join(path, "brown_cipher.txt")
        p_path = os.path.join(path, "brown_plain.txt")
        if not os.path.exists(c_path):
            c_path = os.path.join(path, "cipher.txt")
            p_path = os.path.join(path, "plain.txt")
        return load_pairs(cipher_path=c_path, plain_path=p_path)

    if "," in path:
        c_p, p_p = path.split(",", 1)
        return load_pairs(cipher_path=c_p.strip(), plain_path=p_p.strip())

    if path.endswith(".jsonl"):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    elif path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else data["data"]
    elif path.endswith(".csv") or path.endswith(".tsv"):
        delim = "\t" if path.endswith(".tsv") else ","
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            return list(reader)
    else:
        raise ValueError(f"Unsupported dataset file extension or path format for: {path}")



def train_val_test_split(rows, val_frac=0.1, test_frac=0.1, seed=42):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_val = int(n * val_frac)
    n_test = int(n * test_frac)
    val = rows[:n_val]
    test = rows[n_val:n_val + n_test]
    train = rows[n_val + n_test:]
    return train, val, test


# ---------------------------------------------------------------------- #
# C1-C4: tokenized datasets
# ---------------------------------------------------------------------- #
class TokenizedSeq2SeqDataset(Dataset):
    def __init__(self, rows, src_tokenizer: BPETokenizer, tgt_tokenizer: BPETokenizer, max_len=256):
        self.rows = rows
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        src_ids = self.src_tok.encode(row["ciphertext"])[: self.max_len]
        tgt_ids = self.tgt_tok.encode(row["plaintext"])[: self.max_len]
        return {
            "src_ids": torch.tensor(src_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(tgt_ids, dtype=torch.long),
        }


def make_tokenized_collate_fn(src_pad_id, tgt_pad_id):
    def collate(batch):
        src_lens = [len(b["src_ids"]) for b in batch]
        tgt_lens = [len(b["tgt_ids"]) for b in batch]
        max_src, max_tgt = max(src_lens), max(tgt_lens)

        src_batch = torch.full((len(batch), max_src), src_pad_id, dtype=torch.long)
        tgt_batch = torch.full((len(batch), max_tgt), tgt_pad_id, dtype=torch.long)

        for i, b in enumerate(batch):
            src_batch[i, : len(b["src_ids"])] = b["src_ids"]
            tgt_batch[i, : len(b["tgt_ids"])] = b["tgt_ids"]

        tgt_in = tgt_batch[:, :-1]
        tgt_out = tgt_batch[:, 1:]
        return {"src_ids": src_batch, "tgt_in": tgt_in, "tgt_out": tgt_out}

    return collate


# ---------------------------------------------------------------------- #
# C5: raw byte/bit dataset for BLT (no tokenizer / vocabulary at all)
# ---------------------------------------------------------------------- #
class ByteSeq2SeqDataset(Dataset):
    """
    Ciphertext side: sequence of ints in {0, 1} (raw bits).
    Plaintext side: sequence of ints in [0, 255] (raw byte / char codes),
                    reserving ids 256/257 as BOS/EOS for the plaintext side
                    (needed so the local decoder knows where to stop).
    """

    PLAINTEXT_BOS = 256
    PLAINTEXT_EOS = 257
    PLAINTEXT_VOCAB = 258

    def __init__(self, rows, patch_size=4, max_len=512):
        self.rows = rows
        self.patch_size = patch_size
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def _pad_to_patch(self, seq: List[int], pad_value: int) -> List[int]:
        rem = len(seq) % self.patch_size
        if rem != 0:
            seq = seq + [pad_value] * (self.patch_size - rem)
        return seq

    def __getitem__(self, idx):
        row = self.rows[idx]
        bits = [int(c) for c in row["ciphertext"].strip() if c in "01"][: self.max_len]
        chars = [self.PLAINTEXT_BOS] + [min(ord(c), 255) for c in row["plaintext"]][: self.max_len] + [self.PLAINTEXT_EOS]

        bits = self._pad_to_patch(bits, pad_value=0)
        chars = self._pad_to_patch(chars, pad_value=self.PLAINTEXT_EOS)

        return {
            "src_bytes": torch.tensor(bits, dtype=torch.long),
            "tgt_bytes": torch.tensor(chars, dtype=torch.long),
        }


def make_byte_collate_fn(patch_size, src_pad_value=0, tgt_pad_value=None):
    tgt_pad_value = ByteSeq2SeqDataset.PLAINTEXT_EOS if tgt_pad_value is None else tgt_pad_value

    def pad_to_multiple(x, multiple, value):
        rem = x.size(0) % multiple
        if rem == 0:
            return x
        pad = torch.full((multiple - rem,), value, dtype=x.dtype)
        return torch.cat([x, pad])

    def collate(batch):
        max_src = max(len(b["src_bytes"]) for b in batch)
        max_tgt = max(len(b["tgt_bytes"]) for b in batch)
        # round up to multiple of patch_size
        max_src = ((max_src + patch_size - 1) // patch_size) * patch_size
        max_tgt = ((max_tgt + patch_size - 1) // patch_size) * patch_size

        src_batch = torch.full((len(batch), max_src), src_pad_value, dtype=torch.long)
        tgt_batch = torch.full((len(batch), max_tgt), tgt_pad_value, dtype=torch.long)
        for i, b in enumerate(batch):
            src_batch[i, : len(b["src_bytes"])] = b["src_bytes"]
            tgt_batch[i, : len(b["tgt_bytes"])] = b["tgt_bytes"]

        return {"src_bytes": src_batch, "tgt_bytes": tgt_batch}

    return collate
