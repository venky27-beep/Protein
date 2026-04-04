"""
generate_synthetic.py
─────────────────────
Generates a realistic-looking synthetic protein SS dataset for local testing.

Real biology encodes tendencies:
  - Alanine (A), Leucine (L), Glutamate (E), Methionine (M) → helix formers
  - Valine (V), Isoleucine (I), Phenylalanine (F), Tyrosine (Y) → sheet formers
  - Glycine (G), Proline (P), Serine (S), Threonine (T)       → coil formers

We use these tendencies with some random noise to build fake but bio-plausible
sequence/structure pairs. This lets you run and test the code before getting
the real assignment dataset.
"""

import random

HELIX_AA  = list("ALENMQK")
SHEET_AA  = list("VIVFYT")
COIL_AA   = list("GPSTDCH")
ALL_AA    = list("ACDEFGHIKLMNPQRSTVWY")

def generate_sequence(length=100):
    """Generate a fake protein sequence with biased SS tendencies."""
    seq, ss = [], []
    i = 0
    while i < length:
        r = random.random()
        if r < 0.33:
            # helix run: 4–12 residues
            run = random.randint(4, 12)
            for _ in range(min(run, length - i)):
                seq.append(random.choice(HELIX_AA + ALL_AA))
                ss.append("H")
            i += run
        elif r < 0.55:
            # sheet run: 3–8 residues
            run = random.randint(3, 8)
            for _ in range(min(run, length - i)):
                seq.append(random.choice(SHEET_AA + ALL_AA))
                ss.append("E")
            i += run
        else:
            # coil: 2–6 residues
            run = random.randint(2, 6)
            for _ in range(min(run, length - i)):
                seq.append(random.choice(COIL_AA + ALL_AA))
                ss.append("C")
            i += run
    return "".join(seq[:length]), "".join(ss[:length])


def write_dataset(filepath, n_sequences=500, seed=42):
    random.seed(seed)
    with open(filepath, "w") as f:
        for _ in range(n_sequences):
            length = random.randint(50, 200)
            seq, ss = generate_sequence(length)
            f.write(seq + "\n")
            f.write(ss  + "\n")
    print(f"Wrote {n_sequences} synthetic sequences to {filepath}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_data.txt"
    n   = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    write_dataset(out, n)
