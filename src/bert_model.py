
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



PLOTS_DIR = "outputs/plots"
RESULTS_CSV = "outputs/experimental_results.csv"
BERT_OUTPUT_DIR = "outputs/bert_checkpoints"

DISTILBERT_MODEL = "distilbert-base-uncased"
BERT_MODEL = "bert-base-uncased"



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


class IMDbDataset(Dataset):

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



def _build_compute_metrics():
   
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


def _save_to_csv(bert_metrics, model_label="BERT"):
   
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
    best_classical=None,
):
   
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

    X_tr, y_tr = _stratified_sample(X_train_text, y_train, sample_size, seed)
    val_size = max(50, int(len(X_tr) * 0.1))
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr,
        test_size=val_size,
        stratify=y_tr,
        random_state=seed,
    )
    print(f"[BERT] Actual train : {len(X_tr)} | val: {len(X_val)} | test: {len(X_test_text)}")

    print("[BERT] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("[BERT] Tokenising datasets...")
    train_dataset = IMDbDataset(X_tr, y_tr, tokenizer, max_length)
    val_dataset = IMDbDataset(X_val, y_val, tokenizer, max_length)
    test_dataset = IMDbDataset(list(X_test_text), list(y_test), tokenizer, max_length)

    print("[BERT] Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )

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
        save_total_limit=1,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=_build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n[BERT] Starting fine-tuning...")
    trainer.train()
    print("[BERT] Fine-tuning complete.")

    print("\n[BERT] Evaluating on test set...")
    preds_output = trainer.predict(test_dataset)
    y_pred = np.argmax(preds_output.predictions, axis=-1)
    y_true = np.array(list(y_test))

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
