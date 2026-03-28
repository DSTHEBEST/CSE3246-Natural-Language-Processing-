# DistilBERT Sentiment Classification — Experiment Report

## Overview

This report summarises the fine-tuning and evaluation of a **DistilBERT** transformer model (`distilbert-base-uncased`) for binary sentiment classification on the IMDb movie reviews dataset. Results are compared against the best-performing classical baseline.

---

## Dataset

| Property | Detail |
|---|---|
| Source | IMDb Reviews |
| Total reviews (raw) | 50,000 |
| After deduplication | 49,582 |
| Train split | 39,665 (80%) |
| Test split | 9,917 (20%) |
| Stratified seed | 42 |

**Class balance:**

| Split | Positive | Negative |
|---|---|---|
| Train | 19,907 | 19,758 |
| Test | 4,977 | 4,940 |

---

## Model & Training Configuration

| Parameter | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Device | CUDA (GPU) |
| Epochs | 2 |
| Batch size | 8 (gradient accumulation ×2 → effective 16) |
| Max sequence length | 256 tokens |
| Train size (after val split) | 35,699 |
| Validation size | 3,966 |
| Initial learning rate | 5e-05 (linear decay) |
| Train loss (final) | 0.2545 |
| Train runtime | ~54 minutes |

---

## Validation Results (per epoch)

| Epoch | Loss | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| 1 | 0.2751 | 0.9072 | 0.9222 | 0.8886 | 0.9051 | 0.8149 |
| 2 | 0.3466 | 0.9150 | 0.9179 | 0.9109 | 0.9144 | 0.8301 |

> Validation loss increased slightly at epoch 2, suggesting mild overfitting, while accuracy and F1 continued to improve. The epoch 2 checkpoint was retained for final evaluation.

---

## Test Set Results

| Metric | Score |
|---|---|
| **Accuracy** | **0.9071 (90.71%)** |
| **Precision** | **0.9199** |
| **Recall** | **0.8927** |
| **F1-Score** | **0.9061** |
| **MCC** | **0.8146** |

### Confusion Matrix

| | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN = 4,553 | FP = 387 |
| **Actual Positive** | FN = 534 | TP = 4,443 |

### Per-class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.90 | 0.92 | 0.91 | 4,940 |
| Positive | 0.92 | 0.89 | 0.91 | 4,977 |
| **Macro avg** | **0.91** | **0.91** | **0.91** | **9,917** |

---

## Comparison: DistilBERT vs. Best Classical Baseline

| Model | Accuracy | F1-Score | MCC |
|---|---|---|---|
| Logistic Regression + TF-IDF | 0.8979 | 0.8996 | 0.7960 |
| **DistilBERT (fine-tuned)** | **0.9071 ↑** | **0.9061 ↑** | **0.8146 ↑** |

DistilBERT outperforms the classical baseline across all three metrics. The accuracy gain is **+0.92 percentage points**, the F1 gain is **+0.65 pp**, and the MCC gain is **+0.0186**, indicating a consistent and meaningful improvement in classification quality.

---

## Output Artefacts

| File | Description |
|---|---|
| `outputs/experimental_results.csv` | Full results table with DistilBERT row appended |
| `outputs/plots/cm_distilbert.png` | Confusion matrix heatmap |
| `outputs/plots/distilbert_training_loss.png` | Training loss curve |
| `outputs/plots/distilbert_vs_classical.png` | Side-by-side comparison plot |

---

## Notes

- The `UNEXPECTED` weight keys in the load report (`vocab_layer_norm`, `vocab_transform`, `vocab_projector`) are expected and benign — they belong to the masked language modelling head, which is not used for sequence classification.
- The `MISSING` keys (`classifier`, `pre_classifier`) were freshly initialised and trained during fine-tuning, which is standard practice for task-specific fine-tuning.
- A minor LayerNorm key rename (`gamma`/`beta` → `weight`/`bias`) was flagged during best-checkpoint loading; this is a known HuggingFace version compatibility artefact and does not affect model correctness.
