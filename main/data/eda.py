import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

CIPHER_PATH = "brown_cipher.txt"
PLAIN_PATH = "brown_plain.txt"

# 1. Load data and verify alignment
with open(CIPHER_PATH, "r", encoding="utf-8") as f:
    cipher_lines = [line.strip() for line in f]

with open(PLAIN_PATH, "r", encoding="utf-8") as f:
    plain_lines = [line.strip() for line in f]

assert len(cipher_lines) == len(
    plain_lines
), f"Line mismatch: {len(cipher_lines)} vs {len(plain_lines)}"
num_samples = len(cipher_lines)
print(f"Total samples: {num_samples}")

# 2. Ciphertext Analysis
cipher_chars = set("".join(cipher_lines))
cipher_lens = np.array([len(s) for s in cipher_lines])
total_bits = sum(cipher_lens)
total_ones = sum(s.count("1") for s in cipher_lines)

print("\n--- Ciphertext Statistics (Binary) ---")
print(f"Unique characters: {sorted(list(cipher_chars))}")
print(f"Bit balance: {total_ones / total_bits:.4f} ones, {1 - (total_ones / total_bits):.4f} zeros")
print(
    f"Bit length -> Min: {cipher_lens.min()}, Median: {np.median(cipher_lens):.1f}, "
    f"Mean: {cipher_lens.mean():.1f}, 95th%: {np.percentile(cipher_lens, 95):.1f}, Max: {cipher_lens.max()}"
)

# 3. Plaintext Analysis
plain_char_lens = np.array([len(s) for s in plain_lines])
plain_word_lens = np.array([len(s.split()) for s in plain_lines])
plain_byte_lens = np.array([len(s.encode("utf-8")) for s in plain_lines])
unique_plain_chars = set("".join(plain_lines))

print("\n--- Plaintext Statistics ---")
print(f"Unique characters count: {len(unique_plain_chars)}")
print(
    f"Word count    -> Min: {plain_word_lens.min()}, Median: {np.median(plain_word_lens):.1f}, "
    f"Mean: {plain_word_lens.mean():.1f}, 95th%: {np.percentile(plain_word_lens, 95):.1f}, Max: {plain_word_lens.max()}"
)
print(
    f"Char length   -> Min: {plain_char_lens.min()}, Median: {np.median(plain_char_lens):.1f}, "
    f"Mean: {plain_char_lens.mean():.1f}, 95th%: {np.percentile(plain_char_lens, 95):.1f}, Max: {plain_char_lens.max()}"
)
print(
    f"Byte length   -> Min: {plain_byte_lens.min()}, Median: {np.median(plain_byte_lens):.1f}, "
    f"Mean: {plain_byte_lens.mean():.1f}, 95th%: {np.percentile(plain_byte_lens, 95):.1f}, Max: {plain_byte_lens.max()}"
)

# 4. Cipher-to-Plain Relationship
ratios = cipher_lens / plain_byte_lens
print("\n--- Cipher / Plaintext Ratio ---")
print(
    f"Bits per plain byte -> Min: {ratios.min():.2f}, Median: {np.median(ratios):.2f}, "
    f"Mean: {ratios.mean():.2f}, Max: {ratios.max():.2f}"
)

# 5. Tokenization Simulation (BPE on Plaintext)
vocab_candidates = [500, 1000, 2000, 4000]
print("\n--- Subword Tokenizer Vocab Size Sweep (Plaintext) ---")
for vocab_size in vocab_candidates:
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tokenizer.train_from_iterator(plain_lines, trainer=trainer)

    token_lens = [len(tokenizer.encode(s).ids) for s in plain_lines]
    token_lens = np.array(token_lens)
    print(
        f"Vocab {vocab_size:4d} | Avg tokens: {token_lens.mean():.1f} | "
        f"95th%: {np.percentile(token_lens, 95):.1f} | Max tokens: {token_lens.max()}"
    )

