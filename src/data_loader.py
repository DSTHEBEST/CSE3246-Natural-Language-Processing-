"""
Data Loading Module  (CORRECTED — no synthetic fallback)
==========================================================
Always loads real IMDb reviews from HuggingFace.

Realism checks:
  - avg review length >= 50 words
  - vocabulary size  >= 5 000 unique tokens
  - row count        >= 40 000

If the local CSV fails any check, it is deleted and re-downloaded.
Synthetic fallback has been REMOVED permanently.
"""
import os
import re
import pandas as pd

DATA_DIR  = "data"
CSV_PATH  = os.path.join(DATA_DIR, "imdb_reviews.csv")

# ---------------------------------------------------------------------------
# Realism thresholds  (tuned for real 50K IMDb reviews)
# ---------------------------------------------------------------------------
MIN_AVG_LEN    = 50      # words per review (raw, before cleaning)
MIN_VOCAB_SIZE = 5_000   # unique whitespace-split tokens
MIN_ROWS       = 40_000  # combined train + test = 50 000


def _quick_realism_check(df):
    """
    Returns (is_real: bool, reason: str).
    Fails fast on any threshold breach.
    """
    n = len(df)
    if n < MIN_ROWS:
        return False, f"only {n:,} rows (need >= {MIN_ROWS:,})"

    avg_len = df["review"].dropna().apply(lambda x: len(str(x).split())).mean()
    if avg_len < MIN_AVG_LEN:
        return False, f"avg review length {avg_len:.1f} words (need >= {MIN_AVG_LEN})"

    # Vocabulary check on a 5 000-row sample for speed
    sample = df["review"].dropna().sample(min(5_000, n), random_state=42)
    tokens = set()
    for text in sample:
        tokens.update(str(text).lower().split())
    if len(tokens) < MIN_VOCAB_SIZE:
        return False, (f"vocab size {len(tokens):,} on sample "
                       f"(need >= {MIN_VOCAB_SIZE:,})")

    return True, "ok"


def _download_real_imdb():
    """
    Download the real IMDb dataset from HuggingFace and save to CSV.
    Combines train (25 000) + test (25 000) = 50 000 reviews.

    Returns: pd.DataFrame with columns ['review', 'sentiment']
    Raises RuntimeError if download fails.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is not installed.\n"
            "Fix: pip install datasets"
        )

    print("[DATA] Downloading real IMDb dataset from HuggingFace "
          "(~84 MB, one-time)...")
    ds = load_dataset("imdb")

    records = []
    for split in ["train", "test"]:
        for row in ds[split]:
            records.append({
                "review":    row["text"],
                "sentiment": int(row["label"]),   # 0 = negative, 1 = positive
            })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[DATA] Saved {len(df):,} real reviews → {CSV_PATH}")

    # Debug: show a real sample
    sample_text = str(df["review"].iloc[0])[:200].replace("\n", " ")
    print(f"[DATA] Sample review[0]: \"{sample_text}...\"")

    return df


def download_and_prepare_dataset():
    """
    Load the IMDb dataset, enforcing realism checks.

    Decision logic:
      1. If CSV exists  → load and run realism check.
         PASS  → return df.
         FAIL  → delete CSV, print warning, download fresh copy.
      2. If CSV missing → download fresh copy.

    Returns:
        pd.DataFrame with columns ['review', 'sentiment']
        Always real IMDb data — synthetic fallback permanently removed.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"[DATA] Found existing dataset at {CSV_PATH} — validating...")
        try:
            df = pd.read_csv(CSV_PATH)
        except Exception as e:
            print(f"[DATA] ⚠ Failed to read CSV ({e}) — re-downloading.")
            os.remove(CSV_PATH)
            return _download_real_imdb()

        is_real, reason = _quick_realism_check(df)
        if is_real:
            print(f"[DATA] ✓ Dataset passed realism checks ({len(df):,} reviews).")
            return df
        else:
            print(f"[DATA] ⚠ SYNTHETIC DATASET DETECTED — {reason}")
            print("[DATA] Deleting stale CSV and re-downloading real IMDb data...")
            os.remove(CSV_PATH)
            # Also delete the cleaned cache so it gets rebuilt from fresh data
            cleaned_csv = os.path.join(DATA_DIR, "cleaned_reviews.csv")
            if os.path.exists(cleaned_csv):
                os.remove(cleaned_csv)
                print("[DATA] Deleted stale cleaned_reviews.csv.")

    return _download_real_imdb()


def load_dataset():
    """Alias for backwards compatibility."""
    return download_and_prepare_dataset()
