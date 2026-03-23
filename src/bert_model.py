"""
Transformer-Based Sentiment Analysis
======================================
Extension module for: Sentiment Analysis of Movie Reviews
  — From Classical ML to Contextual Transformers (DistilBERT / BERT)

Course: CSE3246 - Natural Language Processing

Overview
--------
This module adds a DistilBERT (or BERT) fine-tuning experiment that slots
neatly alongside the existing classical ML pipeline.  Call the public entry
point `run_bert_experiment()` from main.py, passing the same raw text arrays
used for classical feature extraction.

Model Choice
------------
* Default : distilbert-base-uncased
    - 40 % fewer parameters than BERT-base
    - ~60 % faster inference on CPU
    - Retains ~97 % of BERT-base performance on GLUE benchmarks
    - **Recommended for laptop / CPU-only environments**

* Optional: bert-base-uncased  (set use_distilbert=False)
    - Higher ceiling accuracy on large datasets
    - Requires a CUDA GPU for practical fine-tuning times
    - Expected convergence: ~3 epochs, 40-60 min on a T4 GPU

Expected Accuracy (IMDb)
------------------------
| Setting               | Expected Accuracy |
|-----------------------|-------------------|
| Full dataset, GPU     | 92 – 93 %         |
| Full dataset, CPU     | 91 – 93 %         |
| 2 000-sample subset   | 88 – 91 %         |
| 500-sample subset     | 85 – 89 %         |

Compare with classical best: Logistic Regression + Hybrid = 89.52 %
→ DistilBERT on the full dataset should comfortably exceed this.

Academic Title Suggestion
-------------------------
"From Bag-of-Words to Transformers: A Comprehensive Comparative Study of
Classical and Contextual Approaches for Sentiment Analysis of Movie Reviews"

Next Research-Grade Improvement (Post-BERT)
-------------------------------------------
Domain-Adaptive Pre-Training (DAPT) — Gururangan et al., 2020.
Continue pre-training DistilBERT on the unlabelled IMDb corpus using
Masked Language Modelling *before* fine-tuning for sentiment.  This purely
unsupervised step typically yields +1–2 % accuracy with no additional labels.

Usage
-----
    from src.bert_model import run_bert_experiment, train_test_bert_split

    # Split raw texts from the preprocessed dataframe
    X_train_text, X_test_text, y_train, y_test = train_test_bert_split(df)

    run_bert_experiment(
        X_train_text, X_test_text, y_train, y_test,
        sample_size=2000,        # None → full dataset
        num_epochs=3,
        train_batch_size=16,
        eval_batch_size=32,
        max_length=256,
        use_distilbert=True,
        best_classical={
            "Model": "Logistic Regression",
            "Features": "Hybrid",
            "Accuracy": 0.8952,
            "F1-Score": 0.8958,
            "MCC": 0.7904,
        },
    )
"""

import os
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, classification_report,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy imports — deferred so the rest of the pipeline still works
# if transformers is not installed.
# ---------------------------------------------------------------------------
_TRANSFORMERS_AVAILABLE = False
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass  # handled with a friendly error in run_bert_experiment()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLOTS_DIR = "outputs/plots"
RESULTS_CSV = "outputs/experimental_results.csv"
BERT_OUTPUT_DIR = "outputs/bert_checkpoints"

DISTILBERT_MODEL = "distilbert-base-uncased"
BERT_MODEL = "bert-base-uncased"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _ensure_dirs():
    """Create necessary output directories."""
    for d in [PLOTS_DIR, BERT_OUTPUT_DIR]:
        os.makedirs(d, exist_ok=True)


def _set_plot_style():
    """Mirror the plot style used throughout the existing codebase."""
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        plt.style.use("seaborn-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })


def train_test_bert_split(df, test_size=0.2, random_state=42):
    """
    Re-create the same 80/20 train/test split used by the classical pipeline
    but return raw text strings (not feature vectors) for transformer input.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataframe with columns 'review' and 'sentiment'.
    test_size : float
        Fraction for test split (default 0.2 matches classical pipeline).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train_text : list[str]
    X_test_text  : list[str]
    y_train      : list[int]
    y_test       : list[int]
    """
    texts = df["review"].tolist()
    labels = df["sentiment"].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    return X_tr, X_te, y_tr, y_te


def _stratified_sample(X_text, y, sample_size, random_state=42):
    """
    Return a stratified subsample of ``sample_size`` examples.
    Class balance is preserved.
    """
    if sample_size is None or sample_size >= len(X_text):
        return list(X_text), list(y)
    X_arr = np.array(X_text)
    y_arr = np.array(y)
    # sklearn's train_test_split can sample a fixed number via train_size
    _, X_sub, _, y_sub = train_test_split(
        X_arr, y_arr,
        test_size=sample_size,
        stratify=y_arr,
        random_state=random_state,
    )
    return X_sub.tolist(), y_sub.tolist()


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class IMDbDataset(Dataset):
    """
    Minimal torch Dataset wrapping tokenised IMDb reviews.

    Parameters
    ----------
    texts   : list[str]  — raw review strings
    labels  : list[int]  — 0 (negative) / 1 (positive)
    tokenizer            — HuggingFace tokenizer instance
    max_length : int     — truncation/padding length
    """

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Metrics for HuggingFace Trainer
# ---------------------------------------------------------------------------
def _build_compute_metrics():
    """
    Return a compute_metrics function compatible with HuggingFace Trainer.
    Computes Accuracy, Precision, Recall, F1, and MCC.
    """
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        prec = precision_score(labels, predictions, average="binary", zero_division=0)
        rec = recall_score(labels, predictions, average="binary", zero_division=0)
        f1 = f1_score(labels, predictions, average="binary", zero_division=0)
        mcc = matthews_corrcoef(labels, predictions)
        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "mcc": round(float(mcc), 4),
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------
def _plot_bert_confusion_matrix(y_true, y_pred):
    """Save confusion matrix to outputs/plots/cm_bert.png."""
    _set_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax, linewidths=0.5, linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title("Confusion Matrix\nDistilBERT + Contextual Embeddings",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    path = os.path.join(PLOTS_DIR, "cm_bert.png")
    plt.savefig(path)
    plt.close()
    print(f"[BERT] Confusion matrix saved → {path}")


def _plot_bert_vs_classical(bert_metrics, best_classical):
    """
    Grouped bar chart: BERT vs best classical model.

    Parameters
    ----------
    bert_metrics   : dict  — keys: Accuracy, Precision, Recall, F1-Score, MCC
    best_classical : dict  — same keys + Model, Features
    """
    _set_plot_style()
    metric_keys = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC"]
    bert_vals = [bert_metrics[k] for k in metric_keys]
    classic_vals = [best_classical.get(k, 0.0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, classic_vals, width,
                   label=f"{best_classical.get('Model', 'Classical')} ({best_classical.get('Features', '')})",
                   color="#3498db", edgecolor="white", alpha=0.9)
    bars2 = ax.bar(x + width / 2, bert_vals, width,
                   label="DistilBERT (Contextual Embeddings)",
                   color="#9b59b6", edgecolor="white", alpha=0.9)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.4f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_keys, fontsize=11)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("DistilBERT vs Best Classical Model\n(All Key Metrics)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "bert_vs_classical.png")
    plt.savefig(path)
    plt.close()
    print(f"[BERT] Comparison plot saved → {path}")


def _plot_training_loss(trainer):
    """Plot training & eval loss curves if history is available."""
    try:
        log_history = trainer.state.log_history
        train_losses = [(e["epoch"], e["loss"])
                        for e in log_history if "loss" in e]
        eval_losses = [(e["epoch"], e["eval_loss"])
                       for e in log_history if "eval_loss" in e]

        if not train_losses:
            return

        _set_plot_style()
        fig, ax = plt.subplots(figsize=(9, 5))

        if train_losses:
            t_ep, t_loss = zip(*train_losses)
            ax.plot(t_ep, t_loss, marker="o", label="Training Loss",
                    color="#e74c3c", linewidth=2)
        if eval_losses:
            v_ep, v_loss = zip(*eval_losses)
            ax.plot(v_ep, v_loss, marker="s", label="Validation Loss",
                    color="#3498db", linewidth=2, linestyle="--")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("DistilBERT Training & Validation Loss", fontweight="bold")
        ax.legend()

        path = os.path.join(PLOTS_DIR, "bert_training_loss.png")
        plt.savefig(path)
        plt.close()
        print(f"[BERT] Training loss curve saved → {path}")
    except Exception:
        pass  # non-critical visualisation


# ---------------------------------------------------------------------------
# CSV Result Saving
# ---------------------------------------------------------------------------
def _save_to_csv(bert_metrics, model_label="BERT"):
    """
    Append (or create) a row in outputs/experimental_results.csv.
    Columns match the format written by evaluation.print_results_table().
    """
    row = {
        "Model": model_label,
        "Features": "Contextual Embeddings",
        "Accuracy": bert_metrics["Accuracy"],
        "Precision": bert_metrics["Precision"],
        "Recall": bert_metrics["Recall"],
        "F1-Score": bert_metrics["F1-Score"],
        "MCC": bert_metrics["MCC"],
        "Sensitivity": bert_metrics.get("Recall", bert_metrics["Recall"]),
        "Specificity": bert_metrics.get("Specificity", "N/A"),
        "FPR": bert_metrics.get("FPR", "N/A"),
        "FNR": bert_metrics.get("FNR", "N/A"),
        "NPV": bert_metrics.get("NPV", "N/A"),
        "FDR": bert_metrics.get("FDR", "N/A"),
    }

    row_df = pd.DataFrame([row])

    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        # Remove any previous BERT row to avoid duplicates on re-run
        existing = existing[existing["Model"] != model_label]
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(RESULTS_CSV, index=False)
    print(f"[BERT] Results appended → {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Additional granular metrics from confusion matrix
# ---------------------------------------------------------------------------
def _full_metrics_from_cm(y_true, y_pred):
    """Compute the extended metric dict that matches compute_all_metrics()."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    P = tp + fn
    N = tn + fp
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (fp + tp) if (fp + tp) > 0 else 0
    recall = tp / (fn + tp) if (fn + tp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / N if N > 0 else 0
    fpr = fp / N if N > 0 else 0
    fnr = fn / P if P > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
    denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / (denom ** 0.5) if denom > 0 else 0
    return {
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Specificity": round(specificity, 4),
        "FPR": round(fpr, 4),
        "FNR": round(fnr, 4),
        "NPV": round(npv, 4),
        "FDR": round(fdr, 4),
        "MCC": round(mcc, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ---------------------------------------------------------------------------
# Main Public API
# ---------------------------------------------------------------------------
def run_bert_experiment(
    X_train_text,
    X_test_text,
    y_train,
    y_test,
    # ---- configurable parameters ----
    sample_size=2000,
    num_epochs=3,
    train_batch_size=16,
    eval_batch_size=32,
    max_length=256,
    use_distilbert=True,
    warmup_ratio=0.1,
    weight_decay=0.01,
    gradient_accumulation_steps=2,
    logging_steps=50,
    seed=42,
    # ---- comparison ----
    best_classical=None,
):
    """
    Fine-tune DistilBERT (or BERT) for binary sentiment classification and
    integrate results into the existing pipeline outputs.

    Parameters
    ----------
    X_train_text : list[str]
        Raw training review texts (same split used by classical pipeline).
    X_test_text  : list[str]
        Raw test review texts.
    y_train      : list[int]  (0 = negative, 1 = positive)
    y_test       : list[int]
    sample_size  : int or None
        If int, use a stratified subsample of this many training examples.
        Set to None to use the full dataset (slow on CPU).
    num_epochs   : int
        Number of fine-tuning epochs.
    train_batch_size : int
        Per-device training batch size.  Reduce to 8 if OOM on CPU.
    eval_batch_size  : int
        Per-device evaluation batch size.
    max_length   : int
        Token truncation/padding length.  256 is a good CPU/GPU balance.
        Use 512 for maximum accuracy on GPU.
    use_distilbert : bool
        True  → distilbert-base-uncased (recommended for CPU)
        False → bert-base-uncased
    warmup_ratio : float
        Fraction of steps used for linear LR warm-up.
    weight_decay : float
        L2 regularisation coefficient.
    gradient_accumulation_steps : int
        Accumulate gradients over N mini-batches before a weight update.
        Keeps effective batch size large while reducing peak VRAM/RAM.
    logging_steps : int
        Log training loss every N steps.
    seed : int
        Random seed for reproducibility.
    best_classical : dict or None
        Metrics for the best classical model used in the comparison plot.
        Expected keys: Model, Features, Accuracy, Precision, Recall,
                       F1-Score, MCC.

    Returns
    -------
    dict : BERT evaluation metrics (Accuracy, Precision, Recall, F1-Score, MCC, …)
    """
    # ---- guard against missing libraries ----
    if not _TRANSFORMERS_AVAILABLE:
        print(
            "\n[BERT] ERROR: Required libraries not found.\n"
            "  Please install them with:\n"
            "    pip install torch transformers datasets accelerate\n"
        )
        return {}

    _ensure_dirs()
    print("\n" + "=" * 70)
    print("  BERT / TRANSFORMER EXPERIMENT")
    print("=" * 70)

    # ---- device detection ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BERT] Device       : {device.upper()}")
    use_fp16 = (device == "cuda")
    if device == "cpu":
        print("[BERT] Running on CPU — this will be slow for large datasets.")
        print(f"       Using sample_size={sample_size} to keep it tractable.")

    model_name = DISTILBERT_MODEL if use_distilbert else BERT_MODEL
    model_label = "DistilBERT" if use_distilbert else "BERT"
    print(f"[BERT] Model        : {model_name}")
    print(f"[BERT] Epochs       : {num_epochs}")
    print(f"[BERT] Batch size   : {train_batch_size}  (accum. steps: {gradient_accumulation_steps})")
    print(f"[BERT] Max length   : {max_length}")
    print(f"[BERT] Train size   : {len(X_train_text)} → subset: {sample_size}")
    print(f"[BERT] Test size    : {len(X_test_text)}")

    # ---- subsample training data (stratified) ----
    X_tr, y_tr = _stratified_sample(X_train_text, y_train, sample_size, seed)
    # Use at most 20 % of training subset for intermediate eval during training
    val_size = max(50, int(len(X_tr) * 0.1))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr,
        test_size=val_size,
        stratify=y_tr,
        random_state=seed,
    )
    print(f"[BERT] Actual train : {len(X_tr)} | val: {len(X_val)} | test: {len(X_test_text)}")

    # ---- load tokenizer ----
    print("[BERT] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # ---- tokenise ----
    print("[BERT] Tokenising datasets...")
    train_dataset = IMDbDataset(X_tr, y_tr, tokenizer, max_length)
    val_dataset = IMDbDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = IMDbDataset(list(X_test_text), list(y_test), tokenizer, max_length)

    # ---- load model ----
    print("[BERT] Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

    # ---- training arguments ----
    training_args = TrainingArguments(
        output_dir=BERT_OUTPUT_DIR,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=logging_steps,
        logging_dir=os.path.join(BERT_OUTPUT_DIR, "logs"),
        fp16=use_fp16,
        dataloader_pin_memory=(device == "cuda"),
        report_to="none",
        seed=seed,
        # Disable unnecessary checkpointing to save disk space
        save_total_limit=1,
        # Suppress the push-to-hub prompt
        push_to_hub=False,
    )

    # ---- trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ---- train ----
    print("\n[BERT] Starting fine-tuning...")
    trainer.train()
    print("[BERT] Fine-tuning complete.")

    # ---- evaluate on test set ----
    print("\n[BERT] Evaluating on test set...")
    preds_output = trainer.predict(test_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = np.array(list(y_test))

    # ---- compute full metric suite ----
    bert_metrics = _full_metrics_from_cm(y_true, y_pred)

    # ---- print results ----
    print("\n" + "=" * 50)
    print(f"  {model_label} + Contextual Embeddings — TEST RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {bert_metrics['Accuracy']:.4f}")
    print(f"  Precision : {bert_metrics['Precision']:.4f}")
    print(f"  Recall    : {bert_metrics['Recall']:.4f}")
    print(f"  F1-Score  : {bert_metrics['F1-Score']:.4f}")
    print(f"  MCC       : {bert_metrics['MCC']:.4f}")
    print(f"  TP={bert_metrics['TP']}  FP={bert_metrics['FP']}")
    print(f"  FN={bert_metrics['FN']}  TN={bert_metrics['TN']}")
    print("=" * 50)
    print("\n[BERT] Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Negative", "Positive"]))

    # ---- visualisations ----
    _plot_bert_confusion_matrix(y_true, y_pred)
    _plot_training_loss(trainer)

    if best_classical:
        _plot_bert_vs_classical(bert_metrics, best_classical)

    # ---- save to CSV ----
    _save_to_csv(bert_metrics, model_label=model_label)

    # ---- memory cleanup ----
    del model, trainer, train_dataset, val_dataset, test_dataset
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[BERT] Done!  Results saved to {RESULTS_CSV}")
    print(f"[BERT] Plots saved to {PLOTS_DIR}/")
    print("=" * 70)

    return bert_metrics
