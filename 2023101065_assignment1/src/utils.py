"""
Evaluation metrics + small plotting helpers.

Bit-level accuracy and sequence accuracy and Levenshtein distance are
implemented directly (they are simple enough that a from-scratch
implementation is both correct and dependency-free). BLEU / ROUGE use the
standard `nltk` and `rouge-score` libraries as permitted by the assignment
("You can use standard libraries for the evaluation metrics").
"""
from typing import List, Dict
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------- #
# Core string/bit metrics
# ---------------------------------------------------------------------- #
def bit_level_accuracy(pred: str, target: str) -> float:
    """Percentage of matching characters at aligned positions (treats strings
    as bit/char sequences); sequences are compared up to the shorter length,
    with any length mismatch counted as additional errors."""
    n = max(len(pred), len(target))
    if n == 0:
        return 1.0
    matches = sum(1 for a, b in zip(pred, target) if a == b)
    return matches / n


def sequence_accuracy(preds: List[str], targets: List[str]) -> float:
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return correct / max(1, len(preds))


def levenshtein_distance(a: str, b: str) -> int:
    """Classic O(len(a) * len(b)) DP edit distance."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # deletion
                curr[j - 1] + 1,    # insertion
                prev[j - 1] + cost  # substitution
            )
        prev = curr
    return prev[m]


def avg_levenshtein(preds: List[str], targets: List[str]) -> float:
    if not preds:
        return 0.0
    return sum(levenshtein_distance(p, t) for p, t in zip(preds, targets)) / len(preds)


# ---------------------------------------------------------------------- #
# BLEU / ROUGE (tokenized models only, as required by the assignment)
# ---------------------------------------------------------------------- #
def compute_bleu(preds: List[str], targets: List[str]) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    except ImportError:
        return float("nan")
    smoothie = SmoothingFunction().method4
    scores = []
    for p, t in zip(preds, targets):
        p_tokens = p.split()
        t_tokens = t.split()
        if not p_tokens or not t_tokens:
            scores.append(0.0)
            continue
        scores.append(sentence_bleu([t_tokens], p_tokens, smoothing_function=smoothie))
    return sum(scores) / max(1, len(scores))


def compute_rouge(preds: List[str], targets: List[str]) -> Dict[str, float]:
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        return {"rouge1": float("nan"), "rougeL": float("nan")}
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    r1_scores, rl_scores = [], []
    for p, t in zip(preds, targets):
        scores = scorer.score(t, p)
        r1_scores.append(scores["rouge1"].fmeasure)
        rl_scores.append(scores["rougeL"].fmeasure)
    return {
        "rouge1": sum(r1_scores) / max(1, len(r1_scores)),
        "rougeL": sum(rl_scores) / max(1, len(rl_scores)),
    }


def compute_metrics(preds: List[str], targets: List[str], tokenized: bool = True) -> Dict[str, float]:
    metrics = {
        "bit_accuracy": sum(bit_level_accuracy(p, t) for p, t in zip(preds, targets)) / max(1, len(preds)),
        "sequence_accuracy": sequence_accuracy(preds, targets),
        "levenshtein": avg_levenshtein(preds, targets),
    }
    if tokenized:
        metrics["bleu"] = compute_bleu(preds, targets)
        metrics.update(compute_rouge(preds, targets))
    return metrics


# ---------------------------------------------------------------------- #
# Plotting
# ---------------------------------------------------------------------- #
def plot_training_curves(history: Dict[str, List[float]], out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    for key, values in history.items():
        plt.plot(values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_config_comparison(config_names: List[str], metric_values: List[float], metric_name: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(config_names, metric_values)
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} across configurations")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
