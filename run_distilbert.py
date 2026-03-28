"""
run_distilbert.py
==================
Standalone DistilBERT Transformer Run
--------------------------------------
Runs ONLY the DistilBERT fine-tuning experiment, completely independent
of the classical pipeline in main.py.

Usage:
    python run_distilbert.py

Requirements:
    pip install torch transformers accelerate

Course: CSE3246 – Natural Language Processing
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Make sure src/ is importable regardless of where the script is called from
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split

from src.utils import ensure_dirs
from src.data_loader import download_and_prepare_dataset
from src.distilbert_model import run_distilbert

# ===========================================================================
# Configuration — mirrors DISTILBERT_CONFIG in main.py exactly
# ===========================================================================
DISTILBERT_CONFIG = dict(
    train_subset=None,     # None = full ~39 665 train rows (fair vs classical)
    test_subset=None,      # None = full ~9 917 test rows (fair vs classical)
    num_epochs=2,
    batch_size=8,          # reduce to 4 if you get OOM on CPU
    max_length=256,        # 128 for faster CPU; 512 for peak accuracy
    gradient_accumulation_steps=2,
    best_classical={
        "Model":     "Logistic Regression",
        "Features":  "TF-IDF",
        "Accuracy":  0.8979,
        "Precision": 0.9003,
        "Recall":    0.8990,
        "F1-Score":  0.8996,
        "MCC":       0.7960,
    },
)


def main():
    print("=" * 70)
    print("  DISTILBERT TRANSFORMER ONLY RUN")
    print("=" * 31)
    print()
    print("  This script runs ONLY DistilBERT.")
    print("  Classical models are NOT executed.")
    print()

    # -----------------------------------------------------------------------
    # Step 0 – ensure output directories exist
    # -----------------------------------------------------------------------
    ensure_dirs()

    # -----------------------------------------------------------------------
    # Step 1 – load the real IMDb dataset (same as main.py)
    # -----------------------------------------------------------------------
    print("[STEP 1] Loading IMDb dataset...")
    df = download_and_prepare_dataset()
    print(f"  Loaded : {len(df):,} reviews")

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            "Dataset must have 'review' and 'sentiment' columns.\n"
            "Check src/data_loader.py or delete data/imdb_reviews.csv and rerun."
        )

    # -----------------------------------------------------------------------
    # Step 2 – remove duplicates (identical to main.py Step 2)
    # -----------------------------------------------------------------------
    print("\n[STEP 2] Removing duplicate reviews...")
    n_before = len(df)
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"  Removed {n_removed:,} duplicates → {len(df):,} reviews remain.")

    # -----------------------------------------------------------------------
    # Step 3 – train / test split (byte-for-byte identical to main.py Step 4)
    #           random_state=42, stratify=labels, test_size=0.2
    # -----------------------------------------------------------------------
    print("\n[STEP 3] Splitting dataset (80/20 stratified, seed=42)...")
    texts  = df["review"].tolist()
    labels = df["sentiment"].tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print(f"  Train : {len(X_train_raw):,} reviews  |  Test : {len(X_test_raw):,} reviews")
    print(f"  Train pos: {sum(x == 1 for x in y_train):,}  "
          f"neg: {sum(x == 0 for x in y_train):,}")
    print(f"  Test  pos: {sum(x == 1 for x in y_test):,}   "
          f"neg: {sum(x == 0 for x in y_test):,}")

    # -----------------------------------------------------------------------
    # Step 4 – run DistilBERT
    # -----------------------------------------------------------------------
    print("\n[STEP 4] Starting DistilBERT fine-tuning...")
    _fmt = lambda v: f"{v:,}" if v is not None else "full"
    print(f"  train_subset = {_fmt(DISTILBERT_CONFIG['train_subset'])}")
    print(f"  test_subset  = {_fmt(DISTILBERT_CONFIG['test_subset'])}")
    print(f"  epochs       = {DISTILBERT_CONFIG['num_epochs']}")
    print(f"  batch_size   = {DISTILBERT_CONFIG['batch_size']}")
    print(f"  max_length   = {DISTILBERT_CONFIG['max_length']}")
    print()

    try:
        distilbert_metrics = run_distilbert(
            X_train_raw, X_test_raw,
            y_train, y_test,
            **DISTILBERT_CONFIG,
        )
    except Exception as exc:
        print("\n" + "!" * 70)
        print("  DistilBERT run FAILED with the following error:")
        print(f"  {type(exc).__name__}: {exc}")
        print()
        print("  Common fixes:")
        print("    • Install dependencies : pip install torch transformers accelerate")
        print("    • Reduce batch_size to 4 if you get an OOM / memory error")
        print("    • Reduce max_length to 128 for faster CPU runs")
        print("!" * 70)
        return

    # -----------------------------------------------------------------------
    # Step 5 – final comparison printout
    # -----------------------------------------------------------------------
    if not distilbert_metrics:
        print("\n[INFO] DistilBERT returned no metrics (torch/transformers missing?).")
        print("       Install: pip install torch transformers accelerate")
        return

    bc  = DISTILBERT_CONFIG["best_classical"]
    acc = distilbert_metrics.get("Accuracy", float("nan"))
    f1  = distilbert_metrics.get("F1-Score",  float("nan"))
    mcc = distilbert_metrics.get("MCC",        float("nan"))

    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print()
    print("  Best classical baseline:")
    print(f"  Logistic Regression + TF-IDF = {bc['Accuracy'] * 100:.2f}%")
    print()
    print("  DistilBERT result:")
    print(f"    Accuracy : {acc:.4f}  ({acc * 100:.2f}%)  "
          f"{'↑ BETTER than classical' if acc > bc['Accuracy'] else '↓ below classical'}")
    print(f"    F1-Score : {f1:.4f}  "
          f"{'↑ BETTER' if f1 > bc['F1-Score'] else '↓ below'}")
    print(f"    MCC      : {mcc:.4f}  "
          f"{'↑ BETTER' if mcc > bc['MCC'] else '↓ below'}")
    print()
    print("  Outputs saved:")
    print("    • outputs/experimental_results.csv  (DistilBERT row appended)")
    print("    • outputs/plots/cm_distilbert.png   (confusion matrix)")
    print("    • outputs/plots/distilbert_vs_classical.png")
    print("    • outputs/plots/distilbert_training_loss.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
