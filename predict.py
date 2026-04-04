"""
Improved Secondary Structure Predictor
=======================================
Approach: Sliding Window + Random Forest Classifier

Key insight:
  A residue's secondary structure depends not just on itself but on its
  neighbours. By looking at a window of W residues centred on each position,
  we capture local sequence context that the single-residue Naive Bayes
  baseline completely ignores.

Feature encoding:
  Each amino acid in the window is one-hot encoded over the 20 standard
  amino acids (unknown/gap → all zeros). A window of size W=17 gives
  17 × 20 = 340 features per sample.

Classifier:
  Random Forest — ensemble of decision trees that handles the high-
  dimensional one-hot features well without needing feature scaling, and
  is robust to class imbalance (coil 'C' dominates most SS datasets).

Evaluation metric:
  Q3 accuracy — fraction of residues whose predicted label (H/E/C)
  matches the true label. This is the standard metric for SS prediction.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import pickle


# ── Constants ────────────────────────────────────────────────────────────────

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX    = {aa: i for i, aa in enumerate(AMINO_ACIDS)}   # 20 standard AAs
SS_LABELS   = ["H", "E", "C"]

WINDOW_SIZE = 17   # must be odd; neighbours = (W-1)/2 = 8 on each side


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(filepath):
    """
    Expects a plain-text file with alternating lines:
      <amino acid sequence>
      <secondary structure string>
    Lines starting with '>' (FASTA headers) are skipped.
    Returns list of (sequence, ss_string) pairs.
    """
    sequences, labels = [], []
    with open(filepath) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith(">")]
    for i in range(0, len(lines) - 1, 2):
        seq, ss = lines[i].upper(), lines[i + 1].upper()
        if len(seq) == len(ss):
            sequences.append(seq)
            labels.append(ss)
        else:
            print(f"  [warn] skipping pair at line {i}: length mismatch")
    return sequences, labels


# ── Feature extraction ───────────────────────────────────────────────────────

def one_hot(aa):
    """Return a length-20 binary vector for an amino acid character."""
    vec = np.zeros(20, dtype=np.float32)
    idx = AA_INDEX.get(aa)
    if idx is not None:
        vec[idx] = 1.0
    return vec   # all-zero for unknown residues / gaps


def extract_features(sequences, labels, window=WINDOW_SIZE):
    """
    For every residue in every sequence, extract a flat feature vector
    by one-hot encoding each residue in the surrounding window.

    Returns:
      X : np.ndarray of shape (N_residues, window * 20)
      y : np.ndarray of shape (N_residues,)  with integer labels 0/1/2
    """
    half = window // 2
    X_list, y_list = [], []

    ss_to_int = {s: i for i, s in enumerate(SS_LABELS)}

    for seq, ss in zip(sequences, labels):
        n = len(seq)
        # Pad sequence with gap characters on both sides
        padded = "-" * half + seq + "-" * half

        for i in range(n):
            label_char = ss[i]
            if label_char not in ss_to_int:
                continue   # skip non-standard labels

            # Extract window centred on residue i (after padding offset)
            window_seq = padded[i : i + window]
            features = np.concatenate([one_hot(aa) for aa in window_seq])

            X_list.append(features)
            y_list.append(ss_to_int[label_char])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ── Training ─────────────────────────────────────────────────────────────────

def train(train_seqs, train_labs, window=WINDOW_SIZE, n_trees=200, seed=42):
    """
    Build feature matrix from training data and fit a Random Forest.
    Returns the fitted classifier.
    """
    print(f"\n[1/3] Extracting features (window={window}) ...")
    X_train, y_train = extract_features(train_seqs, train_labs, window)
    print(f"      Training set: {X_train.shape[0]:,} residues, "
          f"{X_train.shape[1]} features each")

    label_counts = np.bincount(y_train)
    for i, s in enumerate(SS_LABELS):
        print(f"      {s}: {label_counts[i]:,} ({100*label_counts[i]/len(y_train):.1f}%)")

    print(f"\n[2/3] Training Random Forest ({n_trees} trees) ...")
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=None,          # grow full trees
        class_weight="balanced", # compensate for coil class dominance
        n_jobs=-1,               # use all CPU cores
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    print("      Done.")
    return clf


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(clf, test_seqs, test_labs, window=WINDOW_SIZE):
    """
    Extract test features, run predictions, and print Q3 accuracy +
    per-class precision/recall/F1.
    """
    print(f"\n[3/3] Evaluating on test set ...")
    X_test, y_test = extract_features(test_seqs, test_labs, window)
    print(f"      Test set: {X_test.shape[0]:,} residues")

    y_pred = clf.predict(X_test)
    q3 = accuracy_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Q3 Accuracy : {q3:.4f}  ({q3*100:.2f}%)")
    print(f"{'='*50}")
    print("\nPer-class report:")
    print(classification_report(
        y_test, y_pred,
        target_names=SS_LABELS,
        digits=4
    ))
    return q3


# ── Inference helper ─────────────────────────────────────────────────────────

def predict_sequence(clf, sequence, window=WINDOW_SIZE):
    """
    Predict SS string for a single amino acid sequence string.
    Returns a string of H/E/C labels of the same length.
    """
    half = window // 2
    padded = "-" * half + sequence.upper() + "-" * half
    n = len(sequence)

    X = np.array([
        np.concatenate([one_hot(padded[i + k]) for k in range(window)])
        for i in range(n)
    ], dtype=np.float32)

    int_labels = clf.predict(X)
    return "".join(SS_LABELS[l] for l in int_labels)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <dataset_file> [--save model.pkl]")
        print("  File format: alternating lines of sequence / SS string")
        sys.exit(1)

    dataset_path = sys.argv[1]
    save_path    = None
    if "--save" in sys.argv:
        idx = sys.argv.index("--save")
        save_path = sys.argv[idx + 1]

    print(f"Loading dataset from: {dataset_path}")
    seqs, labs = load_dataset(dataset_path)
    print(f"Loaded {len(seqs)} sequences")

    # 80/20 train-test split (preserve order, no shuffle — common in SS prediction)
    split = int(0.8 * len(seqs))
    train_seqs, train_labs = seqs[:split], labs[:split]
    test_seqs,  test_labs  = seqs[split:],  labs[split:]
    print(f"Train: {len(train_seqs)} seqs | Test: {len(test_seqs)} seqs")

    clf = train(train_seqs, train_labs)
    q3  = evaluate(clf, test_seqs, test_labs)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"\nModel saved to {save_path}")

    # Quick demo prediction on first test sequence
    demo_seq = test_seqs[0]
    pred_ss  = predict_sequence(clf, demo_seq)
    true_ss  = test_labs[0]
    per_seq_acc = sum(p == t for p, t in zip(pred_ss, true_ss)) / len(true_ss)

    print(f"\nDemo prediction (first test sequence, len={len(demo_seq)}):")
    print(f"  Sequence : {demo_seq[:60]}...")
    print(f"  Predicted: {pred_ss[:60]}...")
    print(f"  True     : {true_ss[:60]}...")
    print(f"  Per-seq accuracy: {per_seq_acc:.3f}")
