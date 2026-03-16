"""
Main Pipeline Runner
====================
Sentiment Analysis of Movie Reviews:
A Comparative Study of Machine Learning Approaches
with Multiple Feature Representations

Course: CSE3246 - Natural Language Processing
Topic: Movie Review Sentiment Analysis

This script orchestrates the full NLP pipeline:
1. Load/download the IMDb dataset
2. Preprocess text data
3. Perform Exploratory Data Analysis
4. Extract features (BoW, TF-IDF, Word2Vec, Hybrid)
5. Train 3 ML models × 4 feature types = 12 experiments
6. Evaluate with comprehensive metrics
7. Perform error analysis and generate visualizations
"""
import os
import sys
import warnings
import numpy as np
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import ensure_dirs
from src.data_loader import download_and_prepare_dataset
from src.preprocessing import preprocess_dataset
from src.eda import run_eda, print_dataset_summary
from src.feature_extraction import extract_all_features
from src.models import get_models, split_data, train_and_predict, plot_learning_curves, print_hyperparameters
from src.evaluation import (
    compute_all_metrics, plot_confusion_matrix, plot_comparative_metrics,
    plot_mcc_heatmap, error_analysis, print_results_table,
    print_confusion_matrix_details
)


def main():
    """Run the full sentiment analysis pipeline."""
    print("=" * 70)
    print("  SENTIMENT ANALYSIS OF MOVIE REVIEWS")
    print("  A Comparative Study of Machine Learning Approaches")
    print("  with Multiple Feature Representations")
    print("=" * 70)
    print()

    # Step 0: Create directories
    ensure_dirs()

    # Step 1: Load dataset
    print("[STEP 1] Loading dataset...")
    df = download_and_prepare_dataset()
    print(f"  Dataset loaded: {df.shape[0]} reviews")

    # Step 2: Preprocess
    print("\n[STEP 2] Preprocessing text data...")
    df = preprocess_dataset(df)

    # Step 3: Dataset summary and EDA
    print("\n[STEP 3] Running Exploratory Data Analysis...")
    print_dataset_summary(df)
    run_eda(df)

    # Step 4: Feature extraction
    print("\n[STEP 4] Extracting features...")
    features = extract_all_features(df)

    # Step 5: Get models and print hyperparameters
    models = get_models()
    print_hyperparameters(models)

    # Step 6: Train and evaluate all combinations
    labels = df["sentiment"]
    all_results = {}
    learning_curve_done = set()  # Track which learning curves we've plotted

    print("=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)

    for feat_name, (X_features, feat_model) in features.items():
        print(f"\n--- Feature: {feat_name} ---")

        # Split data
        X_train, X_test, y_train, y_test = split_data(X_features, labels)

        for model_name, (model_instance, params) in models.items():
            print(f"\n  Training {model_name} with {feat_name}...")

            # Clone the model (fresh instance)
            from sklearn.base import clone
            model_clone = clone(model_instance)

            # Train and predict
            trained_model, y_pred = train_and_predict(
                model_clone, X_train, X_test, y_train
            )

            # Compute metrics
            metrics = compute_all_metrics(y_test, y_pred)
            all_results[(model_name, feat_name)] = metrics

            print(f"    Accuracy: {metrics['Accuracy']:.4f} | "
                  f"F1: {metrics['F1-Score']:.4f} | "
                  f"MCC: {metrics['MCC']:.4f}")

            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred, model_name, feat_name)

            # Plot learning curves for 2 selected combinations
            # (Logistic Regression + TF-IDF, and Random Forest + BoW)
            if ((model_name == "Logistic Regression" and feat_name == "TF-IDF") or
                (model_name == "Random Forest" and feat_name == "BoW")):
                lc_key = f"{model_name}_{feat_name}"
                if lc_key not in learning_curve_done:
                    print(f"  Computing learning curve for {model_name} + {feat_name}...")
                    lc_model = clone(model_instance)
                    plot_learning_curves(lc_model, model_name, X_features, labels, feat_name)
                    learning_curve_done.add(lc_key)

            # Error analysis for the best-performing feature type per model
            # (we'll do this for TF-IDF features only to keep it focused)
            if feat_name == "TF-IDF":
                error_analysis(
                    df.iloc[:len(y_test)].reset_index(drop=True),
                    y_test.reset_index(drop=True) if hasattr(y_test, "reset_index") else y_test,
                    y_pred, model_name, feat_name
                )

    # Step 7: Print results
    print_results_table(all_results)
    print_confusion_matrix_details(all_results)

    # Step 8: Generate comparative visualizations
    print("\n[STEP 8] Generating comparative visualizations...")
    plot_comparative_metrics(all_results)
    plot_mcc_heatmap(all_results)

    # Final summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"  Total experiments: {len(all_results)}")
    print(f"  Results saved to: outputs/experimental_results.csv")
    print(f"  Plots saved to:   outputs/plots/")

    # Find best model
    best_key = max(all_results, key=lambda k: all_results[k]["F1-Score"])
    best_metrics = all_results[best_key]
    print(f"\n  Best Model: {best_key[0]} + {best_key[1]}")
    print(f"    Accuracy:  {best_metrics['Accuracy']:.4f}")
    print(f"    F1-Score:  {best_metrics['F1-Score']:.4f}")
    print(f"    MCC:       {best_metrics['MCC']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
