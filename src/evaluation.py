"""
Evaluation module.
Computes all required classification metrics and generates visualizations.
Includes confusion matrices, comparative charts, MCC heatmap, and error analysis.
"""
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from tabulate import tabulate

from src.utils import set_plot_style

PLOTS_DIR = "outputs/plots"


# ---------------------------------------------------------------------------
# Core Metrics Computation
# ---------------------------------------------------------------------------
def compute_all_metrics(y_true, y_pred):
    """
    Compute all evaluation metrics from the confusion matrix.

    Returns:
        dict of metric_name -> value
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    P = tp + fn  # total actual positives
    N = tn + fp  # total actual negatives

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (fp + tp) if (fp + tp) > 0 else 0
    recall = tp / (fn + tp) if (fn + tp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    sensitivity = tp / P if P > 0 else 0
    specificity = tn / N if N > 0 else 0
    fpr = fp / N if N > 0 else 0
    fnr = fn / P if P > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0

    # MCC
    denom = np.sqrt(float((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
        "FPR": round(fpr, 4),
        "FNR": round(fnr, 4),
        "NPV": round(npv, 4),
        "FDR": round(fdr, 4),
        "MCC": round(mcc, 4),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


# ---------------------------------------------------------------------------
# Confusion Matrix Visualization
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, model_name, feature_name):
    """Plot and save a confusion matrix heatmap."""
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax, linewidths=0.5, linecolor="gray",
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"Confusion Matrix\n{model_name} + {feature_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    safe_feat = feature_name.replace(" ", "_").replace("-", "")
    filename = f"cm_{safe_name}_{safe_feat}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()


# ---------------------------------------------------------------------------
# Comparative Visualization
# ---------------------------------------------------------------------------
def plot_comparative_metrics(all_results):
    """
    Plot comparative bar charts for Accuracy, F1-Score, and MCC
    across all model-feature combinations.
    """
    set_plot_style()

    # Prepare data
    rows = []
    for (model_name, feat_name), metrics in all_results.items():
        rows.append({
            "Model": model_name,
            "Features": feat_name,
            "Label": f"{model_name}\n({feat_name})",
            "Accuracy": metrics["Accuracy"],
            "F1-Score": metrics["F1-Score"],
            "MCC": metrics["MCC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
        })
    df = pd.DataFrame(rows)

    # 1. Grouped bar chart — Accuracy, F1, MCC
    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(len(df))
    width = 0.25

    bars1 = ax.bar(x - width, df["Accuracy"], width, label="Accuracy",
                   color="#3498db", edgecolor="white")
    bars2 = ax.bar(x, df["F1-Score"], width, label="F1-Score",
                   color="#2ecc71", edgecolor="white")
    bars3 = ax.bar(x + width, df["MCC"], width, label="MCC",
                   color="#e67e22", edgecolor="white")

    ax.set_xlabel("Model + Feature Combination")
    ax.set_ylabel("Score")
    ax.set_title("Comparative Performance Across All Configurations",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=8, ha="center")
    ax.legend()
    ax.set_ylim(0, 1.15)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comparative_metrics.png"))
    plt.close()
    print("[EVAL] Saved comparative_metrics.png")

    # 2. All metrics per model grouped by feature
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics_to_plot = ["Accuracy", "Precision", "Recall"]

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        features = df["Features"].unique()
        models = df["Model"].unique()
        x = np.arange(len(features))
        w = 0.8 / len(models)
        colors = ["#3498db", "#e74c3c", "#2ecc71"]

        for i, model in enumerate(models):
            vals = []
            for feat in features:
                row = df[(df["Model"] == model) & (df["Features"] == feat)]
                vals.append(row[metric].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * w, vals, w, label=model, color=colors[i],
                   edgecolor="white")

        ax.set_title(metric, fontweight="bold")
        ax.set_xticks(x + w)
        ax.set_xticklabels(features, fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=7)

    plt.suptitle("Per-Metric Comparison by Feature Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "per_metric_comparison.png"))
    plt.close()
    print("[EVAL] Saved per_metric_comparison.png")


# ---------------------------------------------------------------------------
# MCC Heatmap (NOVEL CONTRIBUTION)
# ---------------------------------------------------------------------------
def plot_mcc_heatmap(all_results):
    """
    Plot a heatmap of MCC values: Models × Feature Types.
    This is the Cross-Representation Stability Analysis.
    """
    set_plot_style()

    models = sorted(set(k[0] for k in all_results.keys()))
    features = sorted(set(k[1] for k in all_results.keys()))

    mcc_matrix = np.zeros((len(models), len(features)))
    for i, model in enumerate(models):
        for j, feat in enumerate(features):
            key = (model, feat)
            if key in all_results:
                mcc_matrix[i, j] = all_results[key]["MCC"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(mcc_matrix, annot=True, fmt=".4f", cmap="YlOrRd",
                xticklabels=features, yticklabels=models,
                linewidths=1, linecolor="white", ax=ax,
                annot_kws={"size": 12, "weight": "bold"},
                vmin=0, vmax=1)
    ax.set_title("Cross-Representation Stability Analysis (MCC Heatmap)\nModels × Feature Types",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature Representation", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    plt.savefig(os.path.join(PLOTS_DIR, "mcc_heatmap.png"))
    plt.close()
    print("[EVAL] Saved mcc_heatmap.png")


# ---------------------------------------------------------------------------
# Error Analysis (NOVEL CONTRIBUTION)
# ---------------------------------------------------------------------------
def error_analysis(df, y_true, y_pred, model_name, feature_name):
    """
    Perform misclassification error analysis with linguistic patterns.

    Analyzes:
    - Review length differences (misclassified vs correct)
    - Presence of negation words in misclassified reviews
    - Mixed sentiment patterns
    """
    set_plot_style()
    print(f"\n  [ERROR ANALYSIS] {model_name} + {feature_name}")

    df_analysis = df.copy()
    df_analysis = df_analysis.iloc[:len(y_true)].copy()
    df_analysis["y_true"] = y_true.values if hasattr(y_true, "values") else y_true
    df_analysis["y_pred"] = y_pred
    df_analysis["correct"] = df_analysis["y_true"] == df_analysis["y_pred"]
    df_analysis["review_length"] = df_analysis["cleaned_review"].apply(
        lambda x: len(str(x).split()))

    # Negation words analysis
    negation_words = {"not", "no", "never", "neither", "nobody", "nothing",
                      "nowhere", "nor", "cannot", "cant", "wont", "dont",
                      "doesnt", "didnt", "wasnt", "werent", "isnt", "arent",
                      "shouldnt", "wouldnt", "couldnt", "hardly", "barely",
                      "scarcely"}

    def count_negations(text):
        words = str(text).lower().split()
        return sum(1 for w in words if w in negation_words)

    df_analysis["negation_count"] = df_analysis["review"].apply(count_negations)

    correct = df_analysis[df_analysis["correct"]]
    incorrect = df_analysis[~df_analysis["correct"]]

    if len(incorrect) == 0:
        print("    No misclassifications found!")
        return

    print(f"    Total misclassified: {len(incorrect)} / {len(df_analysis)} "
          f"({100 * len(incorrect) / len(df_analysis):.1f}%)")
    print(f"    Avg review length (correct):   {correct['review_length'].mean():.1f} words")
    print(f"    Avg review length (incorrect): {incorrect['review_length'].mean():.1f} words")
    print(f"    Avg negation words (correct):   {correct['negation_count'].mean():.2f}")
    print(f"    Avg negation words (incorrect): {incorrect['negation_count'].mean():.2f}")

    # Visualization: length distribution of correct vs incorrect
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(correct["review_length"], bins=40, alpha=0.7, color="#2ecc71",
                 label="Correct", edgecolor="white")
    axes[0].hist(incorrect["review_length"], bins=40, alpha=0.7, color="#e74c3c",
                 label="Misclassified", edgecolor="white")
    axes[0].set_title("Review Length: Correct vs Misclassified", fontweight="bold")
    axes[0].set_xlabel("Number of Words")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()

    # Negation count comparison
    categories = ["Correct", "Misclassified"]
    neg_means = [correct["negation_count"].mean(), incorrect["negation_count"].mean()]
    axes[1].bar(categories, neg_means, color=["#2ecc71", "#e74c3c"], edgecolor="white")
    axes[1].set_title("Avg Negation Words: Correct vs Misclassified", fontweight="bold")
    axes[1].set_ylabel("Average Negation Word Count")

    for i, v in enumerate(neg_means):
        axes[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")

    plt.suptitle(f"Error Analysis — {model_name} ({feature_name})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    safe_feat = feature_name.replace(" ", "_").replace("-", "")
    filename = f"error_analysis_{safe_name}_{safe_feat}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    print(f"    Saved {filename}")


# ---------------------------------------------------------------------------
# Results Table
# ---------------------------------------------------------------------------
def print_results_table(all_results):
    """Print a comprehensive results table for all model-feature combinations."""
    print("\n" + "=" * 80)
    print("EXPERIMENTAL RESULTS — ALL MODEL × FEATURE COMBINATIONS")
    print("=" * 80)

    headers = [
        "Model", "Features", "Accuracy", "Precision", "Recall",
        "F1-Score", "MCC", "Sensitivity", "Specificity",
        "FPR", "FNR", "NPV", "FDR",
    ]

    rows = []
    for (model_name, feat_name), metrics in all_results.items():
        rows.append([
            model_name, feat_name,
            metrics["Accuracy"], metrics["Precision"], metrics["Recall"],
            metrics["F1-Score"], metrics["MCC"], metrics["Sensitivity"],
            metrics["Specificity"], metrics["FPR"], metrics["FNR"],
            metrics["NPV"], metrics["FDR"],
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid", floatfmt=".4f"))

    # Also save as CSV
    results_df = pd.DataFrame(rows, columns=headers)
    results_df.to_csv("outputs/experimental_results.csv", index=False)
    print("\n[EVAL] Results saved to outputs/experimental_results.csv")


def print_confusion_matrix_details(all_results):
    """Print confusion matrix details for each combination."""
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX DETAILS")
    print("=" * 60)

    for (model_name, feat_name), metrics in all_results.items():
        print(f"\n  {model_name} + {feat_name}:")
        print(f"    TP={metrics['TP']}  FP={metrics['FP']}")
        print(f"    FN={metrics['FN']}  TN={metrics['TN']}")
