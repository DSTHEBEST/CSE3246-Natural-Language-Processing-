

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix


def fit_bow(X_train_texts, max_features=10_000):
 
    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=2,           # ignore tokens appearing in <2 docs (reduces noise)
        ngram_range=(1, 2), # unigrams + bigrams for richer features
    )
    vectorizer.fit(X_train_texts)
    return vectorizer


def transform_bow(texts, vectorizer):
    return vectorizer.transform(texts)


def extract_bow(X_train_texts, X_test_texts, max_features=10_000):

    print("[FEATURES] Bag of Words — fitting on TRAIN only...")
    vec = fit_bow(X_train_texts, max_features=max_features)
    X_tr = transform_bow(X_train_texts, vec)
    X_te = transform_bow(X_test_texts, vec)
    vocab_size = len(vec.vocabulary_)
    print(f"  Train BoW shape : {X_tr.shape}")
    print(f"  Test  BoW shape : {X_te.shape}")
    print(f"  Vocabulary size : {vocab_size:,}  "
          f"{'⚠ SUSPICIOUSLY SMALL — check dataset' if vocab_size < 500 else '✓ OK'}")
    return X_tr, X_te, vec


def fit_tfidf(X_train_texts, max_features=10_000):
    """Fit a TfidfVectorizer on TRAINING texts only."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        ngram_range=(1, 2),
    )
    vectorizer.fit(X_train_texts)
    return vectorizer


def transform_tfidf(texts, vectorizer):
    return vectorizer.transform(texts)


def extract_tfidf(X_train_texts, X_test_texts, max_features=10_000):

    print("[FEATURES] TF-IDF — fitting on TRAIN only...")
    vec = fit_tfidf(X_train_texts, max_features=max_features)
    X_tr = transform_tfidf(X_train_texts, vec)
    X_te = transform_tfidf(X_test_texts, vec)
    vocab_size = len(vec.vocabulary_)
    print(f"  Train TF-IDF shape : {X_tr.shape}")
    print(f"  Test  TF-IDF shape : {X_te.shape}")
    print(f"  Vocabulary size    : {vocab_size:,}  "
          f"{'⚠ SUSPICIOUSLY SMALL' if vocab_size < 500 else '✓ OK'}")
    # Show top 10 tokens by document frequency
    df_counts = np.asarray(X_tr.sum(axis=0)).ravel()
    feature_names = np.array(vec.get_feature_names_out())
    top_idx = df_counts.argsort()[-10:][::-1]
    print(f"  Top tokens: {list(feature_names[top_idx])}")
    return X_tr, X_te, vec



def fit_word2vec(X_train_texts, vector_size=100, window=5, min_count=2):
    
    tokenized_train = [str(t).split() for t in X_train_texts]
    model = Word2Vec(
        sentences=tokenized_train,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        seed=42,
        epochs=10,
    )
    return model


def _doc_vector(tokens, model, vector_size):
    valid = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(valid, axis=0) if valid else np.zeros(vector_size)


def transform_word2vec(texts, model):
    vector_size = model.vector_size
    tokenized = [str(t).split() for t in texts]
    return np.array([_doc_vector(toks, model, vector_size) for toks in tokenized])


def extract_word2vec(X_train_texts, X_test_texts, vector_size=100):

    print("[FEATURES] Word2Vec — training on TRAIN texts only...")
    model = fit_word2vec(X_train_texts, vector_size=vector_size)
    X_tr = transform_word2vec(X_train_texts, model)
    X_te = transform_word2vec(X_test_texts, model)
    print(f"  Train W2V shape : {X_tr.shape}")
    print(f"  Test  W2V shape : {X_te.shape}")
    return X_tr, X_te, model



def _compute_lexicon_features(texts):

    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for text in texts:
        text_str = str(text)
        vs = analyzer.polarity_scores(text_str)
        blob = TextBlob(text_str)
        rows.append([
            vs["compound"], vs["pos"], vs["neg"], vs["neu"],
            blob.sentiment.polarity, blob.sentiment.subjectivity,
        ])
    return np.array(rows, dtype=np.float32)


def extract_hybrid(X_train_cleaned, X_test_cleaned,
                   X_train_original, X_test_original,
                   max_features=10_000):

    print("[FEATURES] Hybrid (TF-IDF + Lexicon) — fitting TF-IDF on TRAIN only...")

    vec = fit_tfidf(X_train_cleaned, max_features=max_features)
    tfidf_tr = transform_tfidf(X_train_cleaned, vec)
    tfidf_te = transform_tfidf(X_test_cleaned, vec)

    print("  Computing VADER and TextBlob scores (train)...")
    lex_tr = csr_matrix(_compute_lexicon_features(X_train_original))
    print("  Computing VADER and TextBlob scores (test)...")
    lex_te = csr_matrix(_compute_lexicon_features(X_test_original))

    X_tr = hstack([tfidf_tr, lex_tr])
    X_te = hstack([tfidf_te, lex_te])
    print(f"  Train Hybrid shape : {X_tr.shape}")
    print(f"  Test  Hybrid shape : {X_te.shape}")
    return X_tr, X_te, (vec,)

def extract_all_features_split(
    X_train_cleaned, X_test_cleaned,
    X_train_original, X_test_original,
    max_features=10_000,
    w2v_dim=100,
):

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION  (train-only fit, test-only transform)")
    print("=" * 60)

    results = {}

    bow_tr, bow_te, bow_vec = extract_bow(
        X_train_cleaned, X_test_cleaned, max_features=max_features)
    results["BoW"] = {"X_train": bow_tr, "X_test": bow_te, "model": bow_vec}

    tfidf_tr, tfidf_te, tfidf_vec = extract_tfidf(
        X_train_cleaned, X_test_cleaned, max_features=max_features)
    results["TF-IDF"] = {"X_train": tfidf_tr, "X_test": tfidf_te, "model": tfidf_vec}

    w2v_tr, w2v_te, w2v_model = extract_word2vec(
        X_train_cleaned, X_test_cleaned, vector_size=w2v_dim)
    results["Word2Vec"] = {"X_train": w2v_tr, "X_test": w2v_te, "model": w2v_model}

    hyb_tr, hyb_te, hyb_model = extract_hybrid(
        X_train_cleaned, X_test_cleaned,
        X_train_original, X_test_original,
        max_features=max_features)
    results["Hybrid"] = {"X_train": hyb_tr, "X_test": hyb_te, "model": hyb_model}

    print("\n[FEATURES] All feature sets extracted successfully (NO LEAKAGE).\n")
    return results
