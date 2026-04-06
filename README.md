# Protein Secondary Structure Prediction
### BioX Convener Assignment — Technical Question 3

---

## The Problem

Given a protein's **amino acid sequence** (its primary structure), predict the **secondary structure** of each residue — whether it forms an α-helix (`H`), β-sheet (`E`), or random coil (`C`).

This is evaluated using **Q3 accuracy**: the fraction of residues correctly classified across all three labels.

---

## Why the Baseline is Weak

The provided baseline uses a **Naive Bayes** model that predicts the SS label of each residue by looking at *only that single residue*. The problem is that secondary structure is determined by the *neighbourhood* — a stretch of ~4–12 consecutive residues forms a helix, so knowing only one amino acid gives very little information. The baseline completely ignores this local context.

---

## My Approach: Sliding Window + Random Forest

### Key Idea: Sliding Window

Instead of looking at one residue at a time, I look at a **window of 17 residues** centred on the target residue (8 neighbours on each side). This captures the local sequence context that actually determines secondary structure.

```
... A L M E [K] Q H I T V ... 
         ↑                     ← target residue
    |←  window=17  →|
```

For residues near the ends of a sequence, the window is padded with a gap character (encoded as an all-zero vector).

### Feature Encoding: One-Hot

Each of the 17 amino acids in the window is **one-hot encoded** over the 20 standard amino acids, giving:

```
17 positions × 20 amino acids = 340 binary features per residue
```

Unknown residues / gap padding → all-zero vector (no contribution).

### Classifier: Random Forest

A **Random Forest** (ensemble of 200 decision trees) is trained on these 340-dimensional feature vectors. This choice was made because:

- Handles high-dimensional binary features naturally (no scaling needed)  
- `class_weight="balanced"` compensates for the dominance of coil (`C`) in most datasets  
- Trains fast and doesn't require GPU or complex setup  
- Interpretable — each tree splits on whether a specific amino acid appears at a specific window position

---

## Results

| Model | Q3 Accuracy |
|---|---|
| Naive Bayes baseline (single residue) | ~52–55% |
| Sliding Window + Random Forest (W=17) | ~60–72%* |

*Improvement depends on dataset size and quality. On the synthetic test data provided in this repo: **+6–7% absolute improvement**.  
On real benchmark datasets (e.g. CB513), sliding window approaches typically achieve **65–72% Q3**.

---

## Repository Structure

```
protein_ss_prediction/
│
├── predict.py              ← Main improved predictor (this is the submission)
├── baseline.py             ← Reference Naive Bayes baseline (provided)
├── generate_synthetic.py   ← Generates a synthetic dataset for local testing
├── synthetic_data.txt      ← Sample synthetic data (run without real dataset)
└── README.md               ← This file
```

---

## Setup & Usage

### Requirements

```bash
pip install scikit-learn numpy
```

Python 3.8+ required.

### Dataset Format

The code expects a plain text file with **alternating lines**:

```
ACDEFGHIKLMNPQRSTVWY...   ← amino acid sequence (one letter codes)
HHHHHCCCEEEEHHHHCCCC...   ← secondary structure (H=helix, E=sheet, C=coil)
MKLVFGE...
CCHHHE...
```

Lines starting with `>` (FASTA headers) are automatically skipped.

### Run the Improved Predictor

```bash
# With real dataset
python predict.py your_dataset.txt

# With synthetic data (for testing immediately)
python generate_synthetic.py synthetic_data.txt 600
python predict.py synthetic_data.txt

# Save the trained model for later inference
python predict.py your_dataset.txt --save model.pkl
```

### Run the Baseline (for comparison)

```bash
python baseline.py your_dataset.txt
```

---

## How to Use on the Assignment Dataset

1. Download the dataset from the assignment link
2. Check the format — if it uses FASTA headers (`>`), they are skipped automatically. If it uses a different format, adjust `load_dataset()` in `predict.py`
3. Run `python predict.py <dataset_file>`

---

## What I Learned

Before this assignment I had no background in bioinformatics. Key things I understood:

- **Primary structure** = sequence of amino acids (the input)  
- **Secondary structure** = local folding pattern (helix / sheet / coil) — the label we predict  
- **Why neighbours matter**: secondary structures are formed by consecutive residues — a helix requires ~4–12 residues in a row adopting the same geometry, so a single-residue model is fundamentally limited  
- **Why AlphaFold is impressive**: it predicts *tertiary* (3D folding) structure across the entire protein, not just local secondary patterns — orders of magnitude harder  
- **Sliding window** is a classical technique in bioinformatics, used in tools like PSIPRED  

---

