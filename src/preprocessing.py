"""
Text Preprocessing Module  (CORRECTED)
========================================
Fixes over-cleaning that was destroying sentiment signal and reducing
vocabulary to only ~182 tokens.

Key changes vs original:
  1. Digits and numeric tokens are KEPT (e.g. "10/10", "7 out of 10")
  2. Contractions are expanded BEFORE removal (don't → do not)
  3. min token length raised is not applied blindly — short meaningful
     words like "ok", "no", "bad" are preserved with len>1 filter
  4. Negation words (not, no, never…) are explicitly EXCLUDED from
     stopword removal — they reverse sentiment polarity
  5. The original raw 'review' column is ALWAYS preserved for VADER/
     TextBlob lexicon features in the Hybrid feature set
  6. Cache invalidation: the CLEANED_CSV is only loaded if its row count
     matches the loaded DataFrame to catch stale caches

IMPORTANT: preprocessing must be applied SEPARATELY to train and test
texts after the split.  The preprocess_series() helper is stateless
(no fitting), so calling it on train and test independently is safe.
"""
import os
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# ---------------------------------------------------------------------------
# NLTK resource downloads
# ---------------------------------------------------------------------------
def _download_nltk_data():
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                 "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


_download_nltk_data()

CLEANED_CSV = os.path.join("data", "cleaned_reviews.csv")

# ---------------------------------------------------------------------------
# Negation words — MUST NOT be removed as stopwords (they flip sentiment)
# ---------------------------------------------------------------------------
NEGATION_WORDS = {
    "no", "not", "nor", "never", "nobody", "nothing", "nowhere",
    "neither", "hardly", "barely", "scarcely",
    "cannot", "cant", "wont", "dont", "doesnt", "didnt",
    "wasnt", "werent", "isnt", "arent", "shouldnt", "wouldnt", "couldnt",
}

# Build a custom stopword set that preserves negations
def _build_stopwords():
    sw = set(stopwords.words("english"))
    sw -= NEGATION_WORDS          # remove negation words from the drop-list
    return sw


# ---------------------------------------------------------------------------
# Low-level cleaning helpers
# ---------------------------------------------------------------------------
def remove_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    return re.sub(r"http\S+|www\.\S+", " ", text)


def expand_contractions(text: str) -> str:
    """
    Expand common English contractions so that 'don't' → 'do not',
    preserving negation signal before stopword removal.
    """
    contractions = {
        r"\bdon't\b": "do not",   r"\bDon't\b": "do not",
        r"\bdoesn't\b": "does not", r"\bDoesn't\b": "does not",
        r"\bdidn't\b": "did not",  r"\bDidn't\b": "did not",
        r"\bwon't\b": "will not",  r"\bWon't\b": "will not",
        r"\bcan't\b": "cannot",    r"\bCan't\b": "cannot",
        r"\bcannot\b": "cannot",
        r"\bisn't\b": "is not",    r"\bIsn't\b": "is not",
        r"\baren't\b": "are not",  r"\bAren't\b": "are not",
        r"\bwasn't\b": "was not",  r"\bWasn't\b": "was not",
        r"\bweren't\b": "were not", r"\bWeren't\b": "were not",
        r"\bwouldn't\b": "would not", r"\bWouldn't\b": "would not",
        r"\bcouldn't\b": "could not", r"\bCouldn't\b": "could not",
        r"\bshouldn't\b": "should not", r"\bShouldn't\b": "should not",
        r"\bhasn't\b": "has not",  r"\bHasn't\b": "has not",
        r"\bhaven't\b": "have not", r"\bHaven't\b": "have not",
        r"\bhadn't\b": "had not",  r"\bHadn't\b": "had not",
        r"\bI'm\b": "I am",        r"\bI've\b": "I have",
        r"\bI'll\b": "I will",     r"\bI'd\b": "I would",
        r"\bthey're\b": "they are", r"\bthey've\b": "they have",
        r"\bthey'll\b": "they will", r"\bit's\b": "it is",
        r"\bhe's\b": "he is",      r"\bshe's\b": "she is",
    }
    for pattern, replacement in contractions.items():
        text = re.sub(pattern, replacement, text)
    return text


def remove_special_chars_keep_digits(text: str) -> str:
    """
    Remove punctuation but KEEP digits and letters.
    Reason: expressions like '10/10', '7 out of 10', '5 stars'
    carry sentiment information and should not be stripped.
    """
    # Keep letters, digits, spaces; remove everything else
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Full single-text preprocessor
# ---------------------------------------------------------------------------
def preprocess_text(text: str,
                    lemmatizer=None,
                    stop_words=None,
                    min_token_len: int = 2) -> str:
    """
    Preprocess a single review text.

    Steps:
        1. Remove HTML tags
        2. Remove URLs
        3. Expand contractions (preserves negation)
        4. Lowercase
        5. Remove special chars (keep digits)
        6. Tokenise
        7. Remove stopwords EXCEPT negation words
        8. Lemmatise
        9. Keep tokens with len >= min_token_len

    Args:
        text           : raw review string
        lemmatizer     : WordNetLemmatizer instance (created if None)
        stop_words     : custom stopword set (built if None)
        min_token_len  : minimum token length to keep (default 2 to
                         preserve 'no', 'ok', 'bad' etc.)

    Returns:
        Cleaned, lemmatised string
    """
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = _build_stopwords()

    text = remove_html_tags(str(text))
    text = remove_urls(text)
    text = expand_contractions(text)
    text = text.lower()
    text = remove_special_chars_keep_digits(text)
    text = normalize_whitespace(text)

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words and len(t) >= min_token_len
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Dataset-level preprocessing
# ---------------------------------------------------------------------------
def preprocess_series(texts, lemmatizer=None, stop_words=None):
    """
    Preprocess an iterable of texts.  Stateless — safe to call
    independently on train and test splits.

    Returns: list of cleaned strings (same length as input)
    """
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = _build_stopwords()

    return [
        preprocess_text(str(t), lemmatizer, stop_words)
        for t in texts
    ]


def preprocess_dataset(df):
    """
    Preprocess the full dataset DataFrame (used before the train/test split).

    Adds a 'cleaned_review' column.  Caches result to CLEANED_CSV,
    but validates the cache row-count against the input to catch stale files.

    Args:
        df: DataFrame with at minimum columns ['review', 'sentiment']

    Returns:
        DataFrame with 'cleaned_review' added
    """
    # Cache validation — invalidate if row count differs
    if os.path.exists(CLEANED_CSV):
        cached = pd.read_csv(CLEANED_CSV)
        if len(cached) == len(df) and "cleaned_review" in cached.columns:
            print(f"[PREPROCESS] Loading valid cache from {CLEANED_CSV} "
                  f"({len(cached)} rows)")
            return cached
        else:
            print(f"[PREPROCESS] ⚠ Cache row count mismatch "
                  f"({len(cached)} cached vs {len(df)} loaded). "
                  f"Re-preprocessing...")

    print(f"[PREPROCESS] Preprocessing {len(df)} reviews...")
    lemmatizer = WordNetLemmatizer()
    stop_words = _build_stopwords()

    total = len(df)
    cleaned = []
    for i, text in enumerate(df["review"]):
        if (i + 1) % 5000 == 0 or i == 0:
            print(f"  {i+1}/{total} reviews processed...")
        cleaned.append(preprocess_text(str(text), lemmatizer, stop_words))

    df = df.copy()
    df["cleaned_review"] = cleaned
    df.to_csv(CLEANED_CSV, index=False)
    print(f"[PREPROCESS] Saved cleaned dataset → {CLEANED_CSV}")
    return df


# ---------------------------------------------------------------------------
# Dataset integrity checks
# ---------------------------------------------------------------------------
def run_integrity_checks(df, raw_df=None):
    """
    Run and print dataset health diagnostics before proceeding to modelling.

    Checks:
      1. Duplicate raw reviews
      2. Duplicate cleaned reviews
      3. Label leakage (sentiment words in text)
      4. Vocabulary sanity (via a quick CountVectorizer)
      5. Average review length
      6. Class balance

    Args:
        df     : DataFrame with 'cleaned_review' and 'sentiment' columns.
        raw_df : Optional original DataFrame before cleaning — used for
                 a more detailed duplicate check.

    Returns:
        df_deduped : DataFrame with exact duplicates removed from cleaned_review.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    print("\n" + "=" * 60)
    print("DATASET INTEGRITY AUDIT")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Duplicate detection
    # ------------------------------------------------------------------ #
    n_raw_dupes = int(df["review"].duplicated().sum()) if "review" in df else None
    n_clean_dupes = int(df["cleaned_review"].duplicated().sum())

    if n_raw_dupes is not None:
        print(f"\n[CHECK 1] Raw review duplicates    : {n_raw_dupes:,}")
    print(f"[CHECK 1] Cleaned review duplicates: {n_clean_dupes:,}")

    if n_clean_dupes > 0:
        print(f"  ⚠  Removing {n_clean_dupes} duplicate cleaned reviews...")
        df = df.drop_duplicates(subset=["cleaned_review"]).reset_index(drop=True)
        print(f"  → Dataset size after dedup: {len(df):,}")
    else:
        print("  ✓ No cleaned duplicates found.")

    # ------------------------------------------------------------------ #
    # 2. Label leakage detection
    # ------------------------------------------------------------------ #
    print("\n[CHECK 2] Label leakage detection...")
    # Check if the word 'positive' or 'negative' appears in cleaned reviews
    # with high correlation to the actual label
    leakage_words = {"positive", "negative", "label", "sentiment"}
    leaked = []
    for w in leakage_words:
        mask = df["cleaned_review"].str.contains(r"\b" + w + r"\b", case=False, na=False)
        count = int(mask.sum())
        if count > 0:
            # Check if counts are suspiciously correlated with label
            if "sentiment" in df.columns:
                pos_count = int(df[mask & (df["sentiment"] == 1)].shape[0])
                neg_count = int(df[mask & (df["sentiment"] == 0)].shape[0])
                leaked.append(f"'{w}': {count} reviews (pos={pos_count}, neg={neg_count})")
    if leaked:
        print(f"  ⚠ Potentially leaked label words found:")
        for entry in leaked:
            print(f"    {entry}")
    else:
        print("  ✓ No obvious label leakage words detected.")

    # ------------------------------------------------------------------ #
    # 3. Vocabulary sanity
    # ------------------------------------------------------------------ #
    print("\n[CHECK 3] Vocabulary sanity...")
    sample = df["cleaned_review"].fillna("").tolist()
    cv = CountVectorizer(max_features=50_000, min_df=1)
    cv.fit(sample)
    vocab_size = len(cv.vocabulary_)

    print(f"  Total vocabulary (min_df=1): {vocab_size:,}")
    if vocab_size < 500:
        print("  ⚠ CRITICAL: Vocabulary is too small for a real IMDb dataset!")
        print("    Expected: 20,000–80,000 unique tokens for 50K real reviews.")
        print("    This confirms you are likely running on SYNTHETIC data.")
        print("    → Delete data/imdb_reviews.csv and data/cleaned_reviews.csv")
        print("      and re-run to trigger a fresh download from HuggingFace.")
    elif vocab_size < 5_000:
        print("  ⚠ WARNING: Vocabulary may be too small. Check data source.")
    else:
        print("  ✓ Vocabulary size looks realistic.")

    # Top 15 tokens
    counts = cv.transform(sample).sum(axis=0).A1
    feature_names = cv.get_feature_names_out()
    top_idx = counts.argsort()[-15:][::-1]
    print(f"  Top 15 tokens: {list(feature_names[top_idx])}")

    # ------------------------------------------------------------------ #
    # 4. Average review length
    # ------------------------------------------------------------------ #
    lengths = df["cleaned_review"].fillna("").apply(lambda x: len(str(x).split()))
    avg_len = lengths.mean()
    min_len = lengths.min()
    max_len = lengths.max()

    print(f"\n[CHECK 4] Review length (words after cleaning):")
    print(f"  Average : {avg_len:.1f} words")
    print(f"  Min     : {min_len} words")
    print(f"  Max     : {max_len} words")
    if avg_len < 30:
        print("  ⚠ Average review length is very short.")
        print("    Real IMDb cleaned reviews average 80–150 words.")
        print("    This suggests over-cleaning or synthetic/demo data.")
    else:
        print("  ✓ Review length looks realistic.")

    # ------------------------------------------------------------------ #
    # 5. Class balance
    # ------------------------------------------------------------------ #
    if "sentiment" in df.columns:
        vc = df["sentiment"].value_counts()
        print(f"\n[CHECK 5] Class balance:")
        print(f"  Positive (1): {vc.get(1, 0):,}")
        print(f"  Negative (0): {vc.get(0, 0):,}")
        ratio = vc.get(1, 0) / max(vc.get(0, 0), 1)
        if 0.85 <= ratio <= 1.15:
            print("  ✓ Classes are balanced.")
        else:
            print("  ⚠ Classes are imbalanced — consider stratified splitting.")

    print("\n" + "=" * 60)
    print("EXPECTED REALISTIC ACCURACY RANGES (IMDb, 50K real reviews)")
    print("=" * 60)
    print("  Logistic Regression + TF-IDF : 87 – 90 %")
    print("  SVM (Linear) + TF-IDF        : 88 – 91 %")
    print("  Random Forest + BoW           : 82 – 86 %")
    print("  LR + Hybrid (TF-IDF+Lexicon)  : 88 – 91 %")
    print("  DistilBERT (fine-tuned)       : 91 – 93 %")
    print("  If ALL models score 100 % → DATA LEAKAGE or TRIVIAL DATA.")
    print("=" * 60)

    return df
