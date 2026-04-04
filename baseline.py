"""
Baseline: Naive Bayes single-residue predictor.
Predicts secondary structure label (H/E/C) from a single amino acid.
Accuracy is low because it ignores neighboring residues entirely.
"""

from collections import defaultdict
import math


def load_dataset(filepath):
    """
    Load a dataset file with alternating lines:
      Line 1: amino acid sequence  (e.g. ACDEFG...)
      Line 2: secondary structure  (e.g. HHECCC...)
    Returns list of (sequence, ss_string) tuples.
    """
    sequences, labels = [], []
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith(">")]
    for i in range(0, len(lines) - 1, 2):
        sequences.append(lines[i])
        labels.append(lines[i + 1])
    return sequences, labels


def train_naive_bayes(sequences, labels):
    """Count P(SS | amino_acid) from training data."""
    counts = defaultdict(lambda: defaultdict(int))
    ss_counts = defaultdict(int)
    for seq, ss in zip(sequences, labels):
        for aa, s in zip(seq, ss):
            counts[aa][s] += 1
            ss_counts[s] += 1
    return counts, ss_counts


def predict_naive_bayes(aa, counts, ss_counts):
    """Return most likely SS label for a single amino acid."""
    best_label, best_log_p = None, -math.inf
    total = sum(ss_counts.values())
    for s in ss_counts:
        aa_given_s = (counts[aa][s] + 1) / (ss_counts[s] + 21)   # Laplace smoothing
        log_p = math.log(ss_counts[s] / total) + math.log(aa_given_s)
        if log_p > best_log_p:
            best_log_p, best_label = log_p, s
    return best_label


def evaluate(sequences, labels, counts, ss_counts):
    correct, total = 0, 0
    for seq, ss in zip(sequences, labels):
        for aa, true_s in zip(seq, ss):
            pred = predict_naive_bayes(aa, counts, ss_counts)
            if pred == true_s:
                correct += 1
            total += 1
    return correct / total if total else 0.0


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python baseline.py <dataset_file>")
        print("  File format: alternating lines of sequence / SS string")
        sys.exit(1)

    filepath = sys.argv[1]
    seqs, labs = load_dataset(filepath)

    # 80/20 split
    split = int(0.8 * len(seqs))
    train_seqs, train_labs = seqs[:split], labs[:split]
    test_seqs, test_labs   = seqs[split:], labs[split:]

    counts, ss_counts = train_naive_bayes(train_seqs, train_labs)
    acc = evaluate(test_seqs, test_labs, counts, ss_counts)
    print(f"Baseline Naive Bayes Q3 accuracy: {acc:.4f} ({acc*100:.1f}%)")
