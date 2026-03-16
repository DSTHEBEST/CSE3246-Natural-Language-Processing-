"""
Data loading module.
Downloads and loads the IMDb 50K movie review dataset using HuggingFace datasets.
"""
import os
import pandas as pd


DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "imdb_reviews.csv")


def download_and_prepare_dataset():
    """
    Download the IMDb dataset via HuggingFace datasets library and save as CSV.
    Falls back to a synthetic dataset if download fails.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(CSV_PATH):
        print(f"[INFO] Dataset already exists at {CSV_PATH}")
        return pd.read_csv(CSV_PATH)

    try:
        print("[INFO] Downloading IMDb dataset via HuggingFace datasets...")
        from datasets import load_dataset
        ds = load_dataset("imdb")

        records = []
        for split in ["train", "test"]:
            for row in ds[split]:
                records.append({
                    "review": row["text"],
                    "sentiment": row["label"],  # 0=neg, 1=pos
                })

        df = pd.DataFrame(records)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        df.to_csv(CSV_PATH, index=False)
        print(f"[INFO] Saved {len(df)} reviews to {CSV_PATH}")
        return df

    except Exception as e:
        print(f"[WARN] Failed to download dataset: {e}")
        print("[INFO] Generating a synthetic demo dataset instead...")
        return _generate_synthetic_dataset()


def _generate_synthetic_dataset():
    """
    Generate a synthetic IMDb-like dataset for demonstration.
    Fallback when the real dataset cannot be downloaded.
    """
    import random
    random.seed(42)

    positive_templates = [
        "This movie was absolutely {adj}! The {noun} was {adj2} and I loved every minute of it. The director truly captured the essence of the story beautifully.",
        "A truly {adj} film with {adj2} performances. Highly recommended to anyone who appreciates quality cinema! The pacing kept me on the edge of my seat.",
        "One of the best movies I've seen this year. The {noun} was {adj} and the story was {adj2}. Would definitely watch again and recommend to friends.",
        "Incredible {noun}! This film is a {adj} masterpiece with {adj2} direction. Every scene was crafted with care and attention to detail.",
        "I was thoroughly impressed by this film. The {adj} storyline combined with {adj2} acting made it a truly unforgettable experience for the whole family.",
        "What a {adj} movie this turned out to be! The cast delivered {adj2} performances throughout. I was moved to tears by several scenes.",
        "Brilliant filmmaking at its finest. Every aspect of the {noun} was {adj} and {adj2}. This deserves all the awards it can get.",
        "A {adj} experience from start to finish. The {noun} alone is worth the price of admission. Do not miss this gem of a movie.",
        "Exceptional movie with {adj} visuals and a {adj2} soundtrack. A must-see for any film enthusiast. This is how cinema should be made.",
        "The director did a {adj} job creating this {adj2} cinematic experience. From the opening scene to the closing credits, I was captivated.",
        "I have watched many movies this year, but this one stands out as truly {adj}. The {noun} was {adj2} and perfectly executed.",
        "An absolutely {adj} piece of cinema. The {noun} is {adj2} and the emotional depth of every character shines through brilliantly.",
    ]
    negative_templates = [
        "This movie was absolutely {adj}. The {noun} was {adj2} and I hated every single minute of it. Completely waste of time and money.",
        "A truly {adj} film with {adj2} performances. Do not waste your time on this disaster. I wanted to leave the theater halfway through.",
        "One of the worst movies I've seen this year. The {noun} was {adj} and the story was {adj2}. I regret spending money on this film.",
        "Terrible {noun} throughout. This film is a {adj} disaster with {adj2} direction. No redeeming qualities whatsoever in this mess.",
        "I was thoroughly disappointed by this film. The {adj} storyline and {adj2} acting were unbearable to sit through. Avoid at all costs.",
        "What a {adj} movie! The cast delivered {adj2} performances that made me cringe. This should never have been greenlit as a project.",
        "Awful filmmaking at its worst. Every aspect of the {noun} was {adj} and {adj2}. I cannot believe anyone thought this was good.",
        "A {adj} experience from start to finish. The {noun} was painfully bad and poorly executed. Save your time and watch something else.",
        "Dreadful movie with {adj} visuals and a {adj2} soundtrack. Avoid at all costs. This is one of the worsts movies ever made.",
        "The director did a {adj} job ruining this {adj2} potentially decent concept. What could have been good became utterly unwatchable.",
        "I sat through the entire film hoping it would improve, but the {noun} remained {adj} and the story was {adj2} and predictable.",
        "A completely {adj} waste of talent. The {noun} was {adj2} despite the star cast. Not even worth streaming for free at home.",
    ]

    pos_adj = ["amazing", "wonderful", "fantastic", "brilliant", "outstanding", "excellent",
               "magnificent", "superb", "stunning", "captivating", "riveting", "marvelous"]
    neg_adj = ["terrible", "awful", "dreadful", "horrible", "atrocious", "abysmal",
               "pathetic", "miserable", "disappointing", "boring", "painful", "mediocre"]
    nouns = ["acting", "cinematography", "script", "plot", "soundtrack", "direction",
             "casting", "dialogue", "story", "performance", "editing", "pacing"]

    records = []
    for _ in range(2500):
        template = random.choice(positive_templates)
        text = template.format(
            adj=random.choice(pos_adj),
            adj2=random.choice(pos_adj),
            noun=random.choice(nouns),
        )
        records.append({"review": text, "sentiment": 1})

    for _ in range(2500):
        template = random.choice(negative_templates)
        text = template.format(
            adj=random.choice(neg_adj),
            adj2=random.choice(neg_adj),
            noun=random.choice(nouns),
        )
        records.append({"review": text, "sentiment": 0})

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] Generated synthetic dataset with {len(df)} reviews -> {CSV_PATH}")
    return df


def load_dataset():
    """Load the dataset CSV, downloading it first if necessary."""
    if not os.path.exists(CSV_PATH):
        return download_and_prepare_dataset()
    return pd.read_csv(CSV_PATH)
