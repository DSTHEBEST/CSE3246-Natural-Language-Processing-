"""
Feature extraction module.
Implements BoW, TF-IDF, Word2Vec, and Hybrid (TF-IDF + Lexicon) features.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix


# ---------------------------------------------------------------------------
# 1. Bag of Words
# ---------------------------------------------------------------------------
def extract_bow(texts, max_features=10000):
    """
    Extract Bag of Words features using CountVectorizer.

    Args:
        texts: list/Series of cleaned review texts
        max_features: maximum vocabulary size

    Returns:
        features (sparse matrix), vectorizer
    """
    print("[FEATURES] Extracting Bag of Words features...")
    vectorizer = CountVectorizer(max_features=max_features)
    features = vectorizer.fit_transform(texts)
    print(f"  BoW shape: {features.shape}")
    return features, vectorizer


# ---------------------------------------------------------------------------
# 2. TF-IDF
# ---------------------------------------------------------------------------
def extract_tfidf(texts, max_features=10000):
    """
    Extract TF-IDF features using TfidfVectorizer.

    Args:
        texts: list/Series of cleaned review texts
        max_features: maximum vocabulary size

    Returns:
        features (sparse matrix), vectorizer
    """
    print("[FEATURES] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    features = vectorizer.fit_transform(texts)
    print(f"  TF-IDF shape: {features.shape}")
    return features, vectorizer


# ---------------------------------------------------------------------------
# 3. Word2Vec (averaged word vectors)
# ---------------------------------------------------------------------------
def _train_word2vec(tokenized_texts, vector_size=100, window=5, min_count=2):
    """Train a Word2Vec model on the corpus."""
    model = Word2Vec(
        sentences=tokenized_texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42,
        epochs=10,
    )
    return model


def _document_vector(tokens, model, vector_size):
    """Average word vectors for a document."""
    valid = [model.wv[w] for w in tokens if w in model.wv]
    if valid:
        return np.mean(valid, axis=0)
    return np.zeros(vector_size)


def extract_word2vec(texts, vector_size=100):
    """
    Extract Word2Vec features by averaging word vectors per document.

    Args:
        texts: list/Series of cleaned review texts
        vector_size: dimensionality of word vectors

    Returns:
        features (numpy array), word2vec model
    """
    print("[FEATURES] Extracting Word2Vec features...")
    tokenized = [str(t).split() for t in texts]
    model = _train_word2vec(tokenized, vector_size=vector_size)
    features = np.array([_document_vector(toks, model, vector_size)
                         for toks in tokenized])
    print(f"  Word2Vec shape: {features.shape}")
    return features, model


# ---------------------------------------------------------------------------
# 4. Hybrid: TF-IDF + Sentiment Lexicon Scores (NOVEL CONTRIBUTION)
# ---------------------------------------------------------------------------
def _compute_lexicon_features(original_texts):
    """
    Compute sentiment lexicon features using VADER and TextBlob.

    Returns a DataFrame with columns:
        vader_compound, vader_pos, vader_neg, vader_neu,
        textblob_polarity, textblob_subjectivity
    """
    analyzer = SentimentIntensityAnalyzer()
    records = []
    for text in original_texts:
        text_str = str(text)
        # VADER scores
        vs = analyzer.polarity_scores(text_str)
        # TextBlob scores
        blob = TextBlob(text_str)
        records.append({
            "vader_compound": vs["compound"],
            "vader_pos": vs["pos"],
            "vader_neg": vs["neg"],
            "vader_neu": vs["neu"],
            "textblob_polarity": blob.sentiment.polarity,
            "textblob_subjectivity": blob.sentiment.subjectivity,
        })
    return pd.DataFrame(records)


def extract_hybrid(cleaned_texts, original_texts, max_features=10000):
    """
    Extract Hybrid features: TF-IDF + Sentiment Lexicon Scores.

    This is a NOVEL CONTRIBUTION that combines statistical text features
    with domain-aware sentiment signals.

    Args:
        cleaned_texts: preprocessed review texts (for TF-IDF)
        original_texts: original review texts (for lexicon analysis)
        max_features: max TF-IDF vocabulary size

    Returns:
        features (sparse matrix), (tfidf_vectorizer, lexicon_df)
    """
    print("[FEATURES] Extracting Hybrid (TF-IDF + Lexicon) features...")

    # TF-IDF part
    tfidf_vec = TfidfVectorizer(max_features=max_features, sublinear_tf=True)
    tfidf_features = tfidf_vec.fit_transform(cleaned_texts)

    # Lexicon part
    print("  Computing VADER and TextBlob scores...")
    lexicon_df = _compute_lexicon_features(original_texts)
    lexicon_sparse = csr_matrix(lexicon_df.values)

    # Combine
    features = hstack([tfidf_features, lexicon_sparse])
    print(f"  Hybrid shape: {features.shape}")
    return features, (tfidf_vec, lexicon_df)


# ---------------------------------------------------------------------------
# Master extraction function
# ---------------------------------------------------------------------------
def extract_all_features(df):
    """
    Extract all four feature representations from the dataset.

    Args:
        df: DataFrame with 'cleaned_review' and 'review' columns

    Returns:
        dict of {feature_name: (features, model/vectorizer)}
    """
    cleaned = df["cleaned_review"].fillna("")
    original = df["review"].fillna("")

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)

    results = {}
    results["BoW"] = extract_bow(cleaned)
    results["TF-IDF"] = extract_tfidf(cleaned)
    results["Word2Vec"] = extract_word2vec(cleaned)
    results["Hybrid"] = extract_hybrid(cleaned, original)

    print("[FEATURES] All features extracted successfully.\n")
    return results
