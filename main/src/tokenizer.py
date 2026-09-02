"""
Byte-Pair Encoding (BPE) tokenizer implemented completely from scratch
(no HuggingFace tokenizers / sentencepiece / tiktoken).

Standard BPE algorithm (Sennrich et al., 2016):
  1. Start from the base alphabet (individual characters / bits).
  2. Repeatedly find the most frequent adjacent symbol pair across the
     training corpus and merge it into a new symbol.
  3. Stop once `vocab_size` merges have been learned (or no pairs remain).

This is used for BOTH sides of the C1-C4 experiments:
  - Ciphertext side: base alphabet = {'0', '1'} (a bit-string), so learned
    merges produce *variable-length* bit n-grams as subword units -- this
    satisfies the "learned subword tokenization, not fixed 8-bit chunking"
    requirement.
  - Plaintext side: base alphabet = individual characters of the text.

A `</w>` end-of-word marker is appended to each whitespace-split word on the
plaintext side so BPE can learn word-boundary-aware merges (this marker is
omitted for the ciphertext side, which is treated as one long symbol
stream since it has no natural "words").
"""
import json
import collections
from typing import List, Dict, Tuple


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class BPETokenizer:
    def __init__(self, vocab_size: int = 512, use_word_boundary: bool = False):
        self.vocab_size = vocab_size
        self.use_word_boundary = use_word_boundary
        self.merges: List[Tuple[str, str]] = []          # ordered list of learned merges
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def _word_to_symbols(self, word: str) -> List[str]:
        symbols = list(word)
        if self.use_word_boundary:
            symbols = symbols + ["</w>"]
        return symbols

    def _get_word_freqs(self, corpus: List[str]) -> Dict[Tuple[str, ...], int]:
        freqs = collections.Counter()
        for line in corpus:
            if self.use_word_boundary:
                words = line.split(" ")
                words = [w for w in words if w != ""]
            else:
                words = [line]  # treat whole line (e.g. a bit-string) as one "word"
            for w in words:
                freqs[tuple(self._word_to_symbols(w))] += 1
        return freqs

    @staticmethod
    def _get_pair_stats(word_freqs: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        pairs = collections.Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_word(word: Tuple[str, ...], pair: Tuple[str, str]) -> Tuple[str, ...]:
        merged = []
        i = 0
        a, b = pair
        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                merged.append(a + b)
                i += 2
            else:
                merged.append(word[i])
                i += 1
        return tuple(merged)

    def get_avg_token_len(self) -> float:
        """Returns average character length of tokens in vocabulary (excluding special tokens)."""
        valid_tokens = [tok for tok in self.token_to_id.keys() if tok not in SPECIAL_TOKENS]
        if not valid_tokens:
            return 0.0
        return sum(len(tok) for tok in valid_tokens) / len(valid_tokens)

    def get_compression_ratio(self, corpus: List[str]) -> Tuple[float, float]:
        """Returns (compression_ratio, avg_sequence_length) over a corpus.
        compression_ratio = total_raw_chars / total_subword_tokens.
        """
        total_raw_chars = 0
        total_tokens = 0
        for text in corpus:
            total_raw_chars += len(text)
            toks = self.tokenize(text)
            total_tokens += len(toks)
        avg_seq_len = total_tokens / max(1, len(corpus))
        comp_ratio = total_raw_chars / max(1, total_tokens)
        return comp_ratio, avg_seq_len

    def train(self, corpus: List[str], name: str = "BPE", verbose: bool = True):
        """Learn merges from a list of raw strings (bit-strings or plaintext lines)."""
        if verbose:
            print(f"[{name}] Extracting word frequencies from {len(corpus)} corpus samples...", flush=True)
        word_freqs = self._get_word_freqs(corpus)

        base_symbols = set()
        for word in word_freqs:
            base_symbols.update(word)

        num_merges_target = max(0, self.vocab_size - len(SPECIAL_TOKENS) - len(base_symbols))
        if verbose:
            print(f"[{name}] Starting BPE training: base_vocab={len(base_symbols)}, target_merges={num_merges_target}, target_vocab={self.vocab_size}", flush=True)

        self.merges = []
        log_interval = max(1, num_merges_target // 10)

        for step in range(num_merges_target):
            pair_stats = self._get_pair_stats(word_freqs)
            if not pair_stats:
                if verbose:
                    print(f"[{name}] Early stop at step {step}: no more merge pairs available.", flush=True)
                break
            best_pair, freq = max(pair_stats.items(), key=lambda kv: (kv[1], kv[0]))
            self.merges.append(best_pair)
            word_freqs = {
                self._merge_word(word, best_pair): f for word, f in word_freqs.items()
            }
            if verbose and ((step + 1) % log_interval == 0 or (step + 1) == num_merges_target):
                pct = ((step + 1) / max(1, num_merges_target)) * 100
                pair_str = f"'{best_pair[0]}' + '{best_pair[1]}'"
                print(f"  └─ [{name}] Merge {step + 1}/{num_merges_target} ({pct:.0f}%) | Pair: {pair_str} (freq: {freq})", flush=True)

        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        # Build final vocabulary: specials + base symbols + merged symbols
        vocab = list(SPECIAL_TOKENS) + sorted(base_symbols)
        for a, b in self.merges:
            merged_tok = a + b
            if merged_tok not in vocab:
                vocab.append(merged_tok)

        self.token_to_id = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        if verbose:
            comp_ratio, avg_len = self.get_compression_ratio(corpus[:min(1000, len(corpus))])
            print(f"[{name}] BPE Finished! Final Vocab: {self.vocab_size_actual} | Avg Token Chars: {self.get_avg_token_len():.2f} | Sample Compression: {comp_ratio:.2f}x (Avg Tokens/Seq: {avg_len:.1f})\n", flush=True)

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def _bpe_word(self, symbols: List[str]) -> List[str]:
        symbols = list(symbols)
        if not self.merge_ranks:
            return symbols
        while len(symbols) > 1:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self.merge_ranks[p], idx) for idx, p in enumerate(pairs) if p in self.merge_ranks]
            if not ranked:
                break
            _, merge_idx = min(ranked, key=lambda x: x[0])
            a, b = symbols[merge_idx], symbols[merge_idx + 1]
            symbols = symbols[:merge_idx] + [a + b] + symbols[merge_idx + 2:]
        return symbols

    def tokenize(self, text: str) -> List[str]:
        if self.use_word_boundary:
            words = [w for w in text.split(" ") if w != ""]
        else:
            words = [text]
        tokens = []
        for w in words:
            tokens.extend(self._bpe_word(self._word_to_symbols(w)))
        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = self.tokenize(text)
        unk_id = self.token_to_id[UNK_TOKEN]
        ids = [self.token_to_id.get(tok, unk_id) for tok in tokens]
        if add_special_tokens:
            ids = [self.token_to_id[BOS_TOKEN]] + ids + [self.token_to_id[EOS_TOKEN]]
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = [self.id_to_token.get(i, UNK_TOKEN) for i in ids]
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        text = "".join(tokens)
        if self.use_word_boundary:
            text = text.replace("</w>", " ").strip()
        return text

    @property
    def vocab_size_actual(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self):
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self):
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self):
        return self.token_to_id[EOS_TOKEN]

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "vocab_size": self.vocab_size,
                    "use_word_boundary": self.use_word_boundary,
                    "merges": self.merges,
                    "token_to_id": self.token_to_id,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"], use_word_boundary=data["use_word_boundary"])
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.merge_ranks = {pair: i for i, pair in enumerate(tok.merges)}
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        return tok
