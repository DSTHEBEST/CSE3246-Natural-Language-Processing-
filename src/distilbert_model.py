

import os
import gc
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")


_TORCH_OK = False
try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
    )
    _TORCH_OK = True
except ImportError:
    pass


PLOTS_DIR   = "outputs/plots"
RESULTS_CSV = "outputs/experimental_results.csv"
CKPT_DIR    = "outputs/distilbert_checkpoints"
MODEL_NAME  = "distilbert-base-uncased"



class _IMDbDataset(Dataset):

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
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  round(float(accuracy_score(labels, preds)), 4),
        "precision": round(float(precision_score(labels, preds,
                                                 average="binary",
                                                 zero_division=0)), 4),
        "recall":    round(float(recall_score(labels, preds,
                                               average="binary",
                                               zero_division=0)), 4),
        "f1":        round(float(f1_score(labels, preds,
                                           average="binary",
                                           zero_division=0)), 4),
        "mcc":       round(float(matthews_corrcoef(labels, preds)), 4),
    }



def _stratified_sample(texts, labels, n, seed=42):
    if n is None or n >= len(texts):
        return list(texts), list(labels)
    random.seed(seed)
    pos = [(t, l) for t, l in zip(texts, labels) if l == 1]
    neg = [(t, l) for t, l in zip(texts, labels) if l == 0]
    half = n // 2
    pos_s = random.sample(pos, min(half, len(pos)))
    neg_s = random.sample(neg, min(n - len(pos_s), len(neg)))
    combined = pos_s + neg_s
    random.shuffle(combined)
    t, l = zip(*combined)
    return list(t), list(l)



def _set_style():
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        plt.style.use("seaborn-darkgrid")
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150,
        "savefig.bbox": "tight", "font.size": 12,
    })


def _plot_confusion_matrix(y_true, y_pred):
    _set_style()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax, linewidths=0.5, linecolor="gray",
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Confusion Matrix\nDistilBERT · Contextual Embeddings",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    path = os.path.join(PLOTS_DIR, "cm_distilbert.png")
    plt.savefig(path); plt.close()
    print(f"[DistilBERT] Confusion matrix → {path}")


def _plot_comparison(distilbert_metrics, best_classical):
    _set_style()
    keys     = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC"]
    db_vals  = [distilbert_metrics.get(k, 0) for k in keys]
    cl_vals  = [best_classical.get(k, 0)     for k in keys]
    x, w     = np.arange(len(keys)), 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - w/2, cl_vals, w, label=f"{best_classical.get('Model','')} "
                f"({best_classical.get('Features','')})",
                color="#3498db", alpha=0.9, edgecolor="white")
    b2 = ax.bar(x + w/2, db_vals, w, label="DistilBERT (Contextual Embeddings)",
                color="#9b59b6", alpha=0.9, edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.004,
                    f"{h:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(keys, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score")
    ax.set_title("DistilBERT vs Best Classical Model",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "distilbert_vs_classical.png")
    plt.savefig(path); plt.close()
    print(f"[DistilBERT] Comparison plot → {path}")


def _plot_training_loss(trainer):
    try:
        history = trainer.state.log_history
        tr  = [(e["epoch"], e["loss"])       for e in history if "loss"      in e]
        val = [(e["epoch"], e["eval_loss"])  for e in history if "eval_loss" in e]
        if not tr:
            return
        _set_style()
        fig, ax = plt.subplots(figsize=(9, 5))
        if tr:
            ep, lo = zip(*tr)
            ax.plot(ep, lo, "o-", color="#e74c3c", lw=2, label="Train Loss")
        if val:
            ep, lo = zip(*val)
            ax.plot(ep, lo, "s--", color="#3498db", lw=2, label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("DistilBERT Training & Validation Loss", fontweight="bold")
        ax.legend()
        path = os.path.join(PLOTS_DIR, "distilbert_training_loss.png")
        plt.savefig(path); plt.close()
        print(f"[DistilBERT] Loss curve → {path}")
    except Exception:
        pass



def _full_metrics(y_true, y_pred):
    cm          = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    P           = tp + fn; N = tn + fp
    acc         = (tp + tn) / (tp + tn + fp + fn)
    prec        = tp / (fp + tp) if (fp + tp) else 0
    rec         = tp / (fn + tp) if (fn + tp) else 0
    f1          = 2*prec*rec / (prec+rec) if (prec+rec) else 0
    spec        = tn / N if N else 0
    fpr         = fp / N if N else 0
    fnr         = fn / P if P else 0
    npv         = tn / (tn+fn) if (tn+fn) else 0
    fdr         = fp / (fp+tp) if (fp+tp) else 0
    denom       = float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc         = (tp*tn - fp*fn) / denom**0.5 if denom else 0
    return {
        "Accuracy":    round(acc,  4),
        "Precision":   round(prec, 4),
        "Recall":      round(rec,  4),
        "F1-Score":    round(f1,   4),
        "Sensitivity": round(rec,  4),
        "Specificity": round(spec, 4),
        "FPR":         round(fpr,  4),
        "FNR":         round(fnr,  4),
        "NPV":         round(npv,  4),
        "FDR":         round(fdr,  4),
        "MCC":         round(mcc,  4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }



def _append_csv(metrics):
    row = {
        "Model":       "DistilBERT",
        "Features":    "Contextual Embeddings",
        "Accuracy":    metrics["Accuracy"],
        "Precision":   metrics["Precision"],
        "Recall":      metrics["Recall"],
        "F1-Score":    metrics["F1-Score"],
        "MCC":         metrics["MCC"],
        "Sensitivity": metrics["Sensitivity"],
        "Specificity": metrics["Specificity"],
        "FPR":         metrics["FPR"],
        "FNR":         metrics["FNR"],
        "NPV":         metrics["NPV"],
        "FDR":         metrics["FDR"],
    }
    row_df = pd.DataFrame([row])
    if os.path.exists(RESULTS_CSV):
        existing = pd.read_csv(RESULTS_CSV)
        existing = existing[existing["Model"] != "DistilBERT"]  # no duplicates
        combined = pd.concat([existing, row_df], ignore_index=True)
    else:
        combined = row_df
    combined.to_csv(RESULTS_CSV, index=False)
    print(f"[DistilBERT] Results appended → {RESULTS_CSV}")


def run_distilbert(
    X_train_text,
    X_test_text,
    y_train,
    y_test,
    # ---- subset sizes (for laptop safety) ----
    train_subset: int = 5000,
    test_subset:  int = 1000,
    # ---- training knobs ----
    num_epochs:   int = 3,
    batch_size:   int = 8,
    max_length:   int = 256,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 2,
    seed: int = 42,
    # ---- comparison ----
    best_classical: dict = None,
):
   
    if not _TORCH_OK:
        print(
            "\n[DistilBERT] ✗ Required packages not found.\n"
            "  Install with:\n"
            "    pip install torch transformers accelerate\n"
        )
        return {}

    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,  exist_ok=True)

    print("\n" + "=" * 70)
    print("  DISTILBERT TRANSFORMER EXPERIMENT")
    print("=" * 70)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = (device == "cuda")
    print(f"[DistilBERT] Device     : {device.upper()}")
    print(f"[DistilBERT] Model      : {MODEL_NAME}")
    print(f"[DistilBERT] Epochs     : {num_epochs}")
    print(f"[DistilBERT] Batch size : {batch_size}  "
          f"(grad accum × {gradient_accumulation_steps} → "
          f"effective {batch_size * gradient_accumulation_steps})")
    print(f"[DistilBERT] Max length : {max_length}")

    X_tr, y_tr = _stratified_sample(X_train_text, y_train, train_subset, seed)
    X_te, y_te = _stratified_sample(X_test_text,  y_test,  test_subset,  seed)
    print(f"[DistilBERT] Train size : {len(X_tr):,}  |  Test size: {len(X_te):,}")

    val_n    = max(50, int(len(X_tr) * 0.1))
    X_val    = X_tr[-val_n:]; y_val = y_tr[-val_n:]
    X_tr     = X_tr[:-val_n]; y_tr  = y_tr[:-val_n]
    print(f"[DistilBERT] After val split: train={len(X_tr):,}, val={len(X_val):,}")

    print("[DistilBERT] Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    print("[DistilBERT] Tokenising...")
    train_ds = _IMDbDataset(X_tr,  y_tr,  tokenizer, max_length)
    val_ds   = _IMDbDataset(X_val, y_val, tokenizer, max_length)
    test_ds  = _IMDbDataset(X_te,  y_te,  tokenizer, max_length)

    print("[DistilBERT] Loading pre-trained DistilBERT...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)

    training_args = TrainingArguments(
        output_dir                  = CKPT_DIR,
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size * 2,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_ratio                = warmup_ratio,
        weight_decay                = weight_decay,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        fp16                        = use_fp16,
        dataloader_pin_memory       = (device == "cuda"),
        logging_steps               = 50,
        save_total_limit            = 1,
        report_to                   = "none",
        seed                        = seed,
        push_to_hub                 = False,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        compute_metrics = _compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\n[DistilBERT] Starting fine-tuning...")
    trainer.train()
    print("[DistilBERT] Fine-tuning complete.")

    print("[DistilBERT] Evaluating on test set...")
    pred_out  = trainer.predict(test_ds)
    y_pred    = np.argmax(pred_out.predictions, axis=-1)
    y_true    = np.asarray(y_te)
    metrics   = _full_metrics(y_true, y_pred)

    print("\n" + "=" * 55)
    print("  DistilBERT · Contextual Embeddings — TEST RESULTS")
    print("=" * 55)
    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1-Score  : {metrics['F1-Score']:.4f}")
    print(f"  MCC       : {metrics['MCC']:.4f}")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}"
          f"  FN={metrics['FN']}  TN={metrics['TN']}")
    print("=" * 55)
    print("\n[DistilBERT] Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=["Negative", "Positive"]))

    if best_classical:
        print("\n[DistilBERT] Comparison with Best Classical Model:")
        print(f"  {best_classical.get('Model')} + "
              f"{best_classical.get('Features')}:")
        print(f"    Accuracy : {best_classical.get('Accuracy', 0):.4f}")
        print(f"    F1-Score : {best_classical.get('F1-Score', 0):.4f}")
        print(f"    MCC      : {best_classical.get('MCC', 0):.4f}")
        print(f"  DistilBERT:")
        print(f"    Accuracy : {metrics['Accuracy']:.4f}  "
              f"({'↑ BETTER' if metrics['Accuracy'] > best_classical.get('Accuracy',0) else '↓ behind'})")
        print(f"    F1-Score : {metrics['F1-Score']:.4f}  "
              f"({'↑ BETTER' if metrics['F1-Score']  > best_classical.get('F1-Score', 0) else '↓ behind'})")
        print(f"    MCC      : {metrics['MCC']:.4f}  "
              f"({'↑ BETTER' if metrics['MCC']       > best_classical.get('MCC', 0)      else '↓ behind'})")

    _plot_confusion_matrix(y_true, y_pred)
    _plot_training_loss(trainer)
    if best_classical:
        _plot_comparison(metrics, best_classical)
    _append_csv(metrics)

    del model, trainer, train_ds, val_ds, test_ds
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\n[DistilBERT] Done!  Outputs → {PLOTS_DIR}/ and {RESULTS_CSV}")
    print("=" * 70)
    return metrics
