"""
Text preprocessing module.
Handles cleaning, tokenization, stopword removal, and lemmatization.
"""
import os
import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Download required NLTK data
def _download_nltk_data():
    """Download NLTK resources silently."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                 "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]
    for res in resources:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


_download_nltk_data()

CLEANED_CSV = os.path.join("data", "cleaned_reviews.csv")


def remove_html_tags(text):
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r"http\S+|www\.\S+", " ", text)


def remove_special_chars(text):
    """Remove special characters and digits, keep letters and spaces."""
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def normalize_whitespace(text):
    """Collapse multiple spaces into one."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text, lemmatizer=None, stop_words=None):
    """
    Full preprocessing pipeline for a single review.

    Steps:
        1. Remove HTML tags
        2. Remove URLs
        3. Lowercase
        4. Remove special characters and digits
        5. Tokenize
        6. Remove stopwords
        7. Lemmatize
        8. Rejoin
    """
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    if stop_words is None:
        stop_words = set(stopwords.words("english"))

    text = remove_html_tags(text)
    text = remove_urls(text)
    text = text.lower()
    text = remove_special_chars(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def preprocess_dataset(df):
    """
    Preprocess the entire dataset.

    Args:
        df: DataFrame with columns ['review', 'sentiment']

    Returns:
        DataFrame with added 'cleaned_review' column
    """
    if os.path.exists(CLEANED_CSV):
        print("[INFO] Loading pre-cleaned dataset...")
        return pd.read_csv(CLEANED_CSV)

    print("[INFO] Preprocessing reviews (this may take a few minutes)...")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    df = df.copy()
    total = len(df)
    cleaned = []
    for i, text in enumerate(df["review"]):
        if (i + 1) % 2000 == 0 or i == 0:
            print(f"  Processing review {i+1}/{total}...")
        cleaned.append(preprocess_text(str(text), lemmatizer, stop_words))

    df["cleaned_review"] = cleaned
    df.to_csv(CLEANED_CSV, index=False)
    print(f"[INFO] Saved cleaned dataset to {CLEANED_CSV}")
    return df
