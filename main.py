"""
Main Pipeline Runner  (CORRECTED — leakage-free)
==================================================
From Bag-of-Words to Transformers:
A Comprehensive Comparative Study of Classical and Contextual Approaches
for Sentiment Analysis of Movie Reviews

Course: CSE3246 - Natural Language Processing

CORRECT ORDER OF OPERATIONS:
  1. Load dataset
  2. Remove duplicates
  3. Train/test split on RAW texts + labels   ← CRITICAL: split FIRST
  4. Preprocess train texts → cleaned_train
     Preprocess test  texts → cleaned_test    ← SEPARATELY, no cross-contamination
  5. Fit vectorizers on cleaned_train ONLY
     Transform cleaned_test using fitted vectorizers
  6. Train classifiers on (X_train_features, y_train)
  7. Evaluate on (X_test_features, y_test)
  8. Save metrics, plots, CSV

WHY THE ORIGINAL PIPELINE PRODUCED 100% ACCURACY:
  - Vectorizers were fit on the entire dataset (train + test combined)
    before the train/test split → test data was visible during training
  - The synthetic fallback dataset was trivially separable (fixed
    templates with class-exclusive vocabulary like "amazing" vs "terrible")
  - Stale cleaned_reviews.csv caches silently loaded the wrong data
  - Over-cleaning reduced vocabulary to ~182 tokens making the task trivial
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import ensure_dirs
from src.data_loader import download_and_prepare_dataset
from src.preprocessing import preprocess_series, run_integrity_checks, _build_stopwords
from src.eda import run_eda, print_dataset_summary
from src.feature_extraction import extract_all_features_split
from src.models import get_models, train_and_predict, plot_learning_curves, print_hyperparameters
from src.evaluation import (
    compute_all_metrics, plot_confusion_matrix, plot_comparative_metrics,
    plot_mcc_heatmap, error_analysis, print_results_table,
    print_confusion_matrix_details,
)
from nltk.stem import WordNetLemmatizer


def _debug_feature_shapes(name, X_train, X_test, y_train, y_test):
    """Print a quick shape/sanity debug block for a feature set."""
    print(f"\n  [DEBUG] Feature: {name}")
    print(f"    Train shape   : {X_train.shape}  labels: {len(y_train)}")
    print(f"    Test  shape   : {X_test.shape}   labels: {len(y_test)}")
    first = X_train[0]
    if hasattr(first, "toarray"):
        first = first.toarray().ravel()
    nonzero = int(np.count_nonzero(np.asarray(first).ravel()))
    print(f"    First train sample non-zero entries: {nonzero}")


def main():
    """Run the full leakage-free sentiment analysis pipeline."""
    print("=" * 70)
    print("  FROM BAG-OF-WORDS TO TRANSFORMERS")
    print("  A Comparative Study of Classical and Contextual Approaches")
    print("  for Sentiment Analysis of Movie Reviews")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Step 0: Create output directories
    # ------------------------------------------------------------------
    ensure_dirs()

    # ------------------------------------------------------------------
    # Step 1: Load dataset
    # ------------------------------------------------------------------
    print("[STEP 1] Loading dataset...")
    df = download_and_prepare_dataset()
    print(f"  Loaded : {len(df):,} reviews")

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError("Dataset must have 'review' and 'sentiment' columns.")

    # ------------------------------------------------------------------
    # Step 2: Remove duplicates on RAW reviews (BEFORE splitting)
    # ------------------------------------------------------------------
    print("\n[STEP 2] Removing duplicate reviews...")
    n_before = len(df)
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    n_removed = n_before - len(df)
    print(f"  Removed {n_removed:,} duplicates → {len(df):,} reviews remain.")

    # ------------------------------------------------------------------
    # Step 3: EDA on raw data
    # ------------------------------------------------------------------
    print("\n[STEP 3] Dataset summary...")
    print_dataset_summary(df)

    # ------------------------------------------------------------------
    # Step 4: TRAIN / TEST SPLIT on raw texts and labels
    #         → MUST happen BEFORE any preprocessing or vectorisation
    # ------------------------------------------------------------------
    print("\n[STEP 4] Splitting dataset (80/20 stratified, seed=42)...")
    # .tolist() ensures plain Python lists — avoids TypeError from
    # PyArrow-backed ArrowExtensionArray columns that sklearn cannot index.
    texts  = df["review"].tolist()
    labels = df["sentiment"].tolist()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        texts, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    print(f"  Train : {len(X_train_raw):,} reviews  |  "
          f"Test : {len(X_test_raw):,} reviews")
    print(f"  Train pos: {sum(x==1 for x in y_train):,}  neg: {sum(x==0 for x in y_train):,}")
    print(f"  Test  pos: {sum(x==1 for x in y_test):,}   neg: {sum(x==0 for x in y_test):,}")

    # ------------------------------------------------------------------
    # Step 5: Preprocess train and test SEPARATELY
    #         (preprocessing is stateless — no fitting involved)
    # ------------------------------------------------------------------
    print("\n[STEP 5] Preprocessing text data...")
    lemmatizer = WordNetLemmatizer()
    stop_words = _build_stopwords()

    print("  Preprocessing TRAIN texts...")
    X_train_cleaned = preprocess_series(X_train_raw, lemmatizer, stop_words)
    print(f"  Sample cleaned train[0]: '{X_train_cleaned[0][:100]}...'")

    print("  Preprocessing TEST texts...")
    X_test_cleaned = preprocess_series(X_test_raw, lemmatizer, stop_words)

    # ------------------------------------------------------------------
    # Step 5b: Integrity checks
    # ------------------------------------------------------------------
    print("\n[STEP 5b] Running dataset integrity checks...")
    df_check = pd.DataFrame({
        "review":         list(X_train_raw)     + list(X_test_raw),
        "cleaned_review": X_train_cleaned       + X_test_cleaned,
        "sentiment":      list(y_train)         + list(y_test),
    })
    run_integrity_checks(df_check)

    # ------------------------------------------------------------------
    # Step 6: Feature extraction — vectorizers fit on TRAIN only
    # ------------------------------------------------------------------
    print("\n[STEP 6] Extracting features (train-fit, test-transform)...")
    feature_sets = extract_all_features_split(
        X_train_cleaned=X_train_cleaned,
        X_test_cleaned=X_test_cleaned,
        X_train_original=list(X_train_raw),
        X_test_original=list(X_test_raw),
        max_features=10_000,
        w2v_dim=100,
    )

    # ------------------------------------------------------------------
    # Step 7: Train and evaluate all model × feature combinations
    # ------------------------------------------------------------------
    models = get_models()
    print_hyperparameters(models)

    all_results = {}
    learning_curve_done = set()

    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for feat_name, feat_data in feature_sets.items():
        X_train = feat_data["X_train"]
        X_test  = feat_data["X_test"]

        print(f"\n--- Feature: {feat_name} ---")
        _debug_feature_shapes(feat_name, X_train, X_test, y_train, y_test)

        for model_name, (model_instance, params) in models.items():
            print(f"\n  Training {model_name} with {feat_name}...")

            from sklearn.base import clone
            model_clone = clone(model_instance)

            trained_model, y_pred = train_and_predict(
                model_clone, X_train, X_test, y_train
            )

            metrics = compute_all_metrics(y_test, y_pred)
            all_results[(model_name, feat_name)] = metrics

            acc = metrics["Accuracy"]
            leakage_flag = "⚠ CHECK FOR LEAKAGE" if acc >= 0.999 else "✓"
            print(f"    Accuracy: {acc:.4f} {leakage_flag} | "
                  f"F1: {metrics['F1-Score']:.4f} | "
                  f"MCC: {metrics['MCC']:.4f}")

            if acc >= 0.999:
                print("    !! 100% ACCURACY IS NOT REALISTIC ON IMDb !!")
                print("    !! You may be running on SYNTHETIC fallback data. !!")
                print("    !! Delete data/imdb_reviews.csv and rerun.         !!")

            plot_confusion_matrix(y_test, y_pred, model_name, feat_name)

            if ((model_name == "Logistic Regression" and feat_name == "TF-IDF") or
                (model_name == "Random Forest" and feat_name == "BoW")):
                lc_key = f"{model_name}_{feat_name}"
                if lc_key not in learning_curve_done:
                    print(f"  Computing learning curve for {model_name} + {feat_name}...")
                    lc_model = clone(model_instance)
                    plot_learning_curves(lc_model, model_name, X_train, y_train, feat_name)
                    learning_curve_done.add(lc_key)

            if feat_name == "TF-IDF":
                test_df = pd.DataFrame({
                    "review":         list(X_test_raw),
                    "cleaned_review": X_test_cleaned,
                    "sentiment":      list(y_test),
                }).reset_index(drop=True)
                error_analysis(
                    test_df,
                    pd.Series(y_test),
                    y_pred,
                    model_name,
                    feat_name,
                )

    # ------------------------------------------------------------------
    # Step 8: Results and visualisations
    # ------------------------------------------------------------------
    print_results_table(all_results)
    print_confusion_matrix_details(all_results)

    print("\n[STEP 8] Generating comparative visualizations...")
    plot_comparative_metrics(all_results)
    plot_mcc_heatmap(all_results)

    # ------------------------------------------------------------------
    # Step 9: EDA on full dataset
    # ------------------------------------------------------------------
    print("\n[STEP 9] Running EDA visualizations...")
    run_eda(df)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  Total experiments : {len(all_results)}")
    print(f"  Results saved to  : outputs/experimental_results.csv")
    print(f"  Plots saved to    : outputs/plots/")

    best_key = max(all_results, key=lambda k: all_results[k]["F1-Score"])
    best_metrics = all_results[best_key]
    print(f"\n  Best Model : {best_key[0]} + {best_key[1]}")
    print(f"    Accuracy : {best_metrics['Accuracy']:.4f}")
    print(f"    F1-Score : {best_metrics['F1-Score']:.4f}")
    print(f"    MCC      : {best_metrics['MCC']:.4f}")

    if best_metrics["Accuracy"] >= 0.999:
        print("\n  ⚠ ALL MODELS SCORED ~100%.")
        print("  ACTION: Delete data/imdb_reviews.csv and data/cleaned_reviews.csv")
        print("          then re-run to download real IMDb reviews from HuggingFace.")
        print("  Expected real-IMDb accuracy: 86–91% for classical models.")

    print("=" * 70)


# =============================================================================
# STEP 10 — DistilBERT TRANSFORMER BASELINE  (optional)
# =============================================================================
# Set RUN_DISTILBERT = True to run DistilBERT after the classical pipeline.
# The flag is False by default — the classical pipeline is NEVER affected.
#
# Install once:
#   pip install torch transformers accelerate
#
# Laptop-safe defaults (CPU):
#   train_subset = 5000   → ~2–3 h on CPU, 10–15 min on GPU
#   batch_size   = 8      → reduce to 4 if you get OOM errors
#   num_epochs   = 3
# =============================================================================
RUN_DISTILBERT = False

DISTILBERT_CONFIG = dict(
    train_subset  = 5000,     # int or None (None = full 39 665 train examples)
    test_subset   = 1000,     # int or None (None = full 9 917 test examples)
    num_epochs    = 3,
    batch_size    = 8,        # increase to 16–32 if you have a GPU
    max_length    = 256,      # 128 for faster CPU runs; 512 for peak accuracy
    gradient_accumulation_steps = 2,
    best_classical = {
        "Model":     "Logistic Regression",
        "Features":  "TF-IDF",
        "Accuracy":  0.8979,
        "Precision": 0.9003,
        "Recall":    0.8990,
        "F1-Score":  0.8996,
        "MCC":       0.7960,
    },
)


if __name__ == "__main__":
    main()

    # -------------------------------------------------------------------------
    # Step 10 [OPTIONAL]: DistilBERT Fine-Tuning
    # -------------------------------------------------------------------------
    # X_train_raw / X_test_raw / y_train / y_test are already produced inside
    # main() but are local to that function.  We re-create them here from the
    # same CSV so the split is byte-for-byte identical (same seed=42).
    if RUN_DISTILBERT:
        from src.distilbert_model import run_distilbert
        from src.data_loader import download_and_prepare_dataset as _load
        from sklearn.model_selection import train_test_split as _split

        print("\n[STEP 10] DistilBERT transformer experiment...")

        _df = _load()
        _df = _df.drop_duplicates(subset=["review"]).reset_index(drop=True)
        _texts  = _df["review"].tolist()
        _labels = _df["sentiment"].tolist()

        _X_tr, _X_te, _y_tr, _y_te = _split(
            _texts, _labels,
            test_size=0.2, random_state=42, stratify=_labels,
        )

        distilbert_metrics = run_distilbert(
            _X_tr, _X_te, _y_tr, _y_te,
            **DISTILBERT_CONFIG,
        )

        if distilbert_metrics:
            print("\n[STEP 10] DistilBERT experiment complete.")
            print(f"  Accuracy : {distilbert_metrics.get('Accuracy', 'N/A')}")
            print(f"  F1-Score : {distilbert_metrics.get('F1-Score', 'N/A')}")
            print(f"  MCC      : {distilbert_metrics.get('MCC', 'N/A')}")

