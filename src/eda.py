
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

from src.utils import set_plot_style

PLOTS_DIR = "outputs/plots"


def plot_class_distribution(df):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    counts = df["sentiment"].value_counts()
    labels = ["Negative (0)", "Positive (1)"]
    colors = ["#e74c3c", "#2ecc71"]
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors,
                  edgecolor="white", linewidth=1.5)

    for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=13)

    ax.set_title("Class Distribution of Movie Reviews", fontsize=15, fontweight="bold")
    ax.set_ylabel("Number of Reviews")
    ax.set_xlabel("Sentiment Class")
    plt.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"))
    plt.close()
    print("[EDA] Saved class_distribution.png")


def plot_review_length_distribution(df):
    set_plot_style()
    df = df.copy()
    text_col = "cleaned_review" if "cleaned_review" in df.columns else "review"
    df["review_length"] = df[text_col].apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (label, color, sent) in enumerate([
        ("Negative", "#e74c3c", 0), ("Positive", "#2ecc71", 1)
    ]):
        subset = df[df["sentiment"] == sent]["review_length"]
        axes[idx].hist(subset, bins=50, color=color, alpha=0.8, edgecolor="white")
        axes[idx].set_title(f"{label} Reviews — Length Distribution", fontweight="bold")
        axes[idx].set_xlabel("Number of Words")
        axes[idx].set_ylabel("Frequency")
        axes[idx].axvline(subset.mean(), color="black", linestyle="--", linewidth=1.5,
                          label=f"Mean: {subset.mean():.0f}")
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "review_length_distribution.png"))
    plt.close()
    print("[EDA] Saved review_length_distribution.png")


def plot_word_clouds(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    text_col = "cleaned_review" if "cleaned_review" in df.columns else "review"

    for idx, (label, sent, cmap) in enumerate([
        ("Positive Reviews", 1, "Greens"),
        ("Negative Reviews", 0, "Reds"),
    ]):
        text = " ".join(df[df["sentiment"] == sent][text_col].dropna().tolist())
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=cmap, max_words=150, random_state=42)
        wc.generate(text)
        axes[idx].imshow(wc, interpolation="bilinear")
        axes[idx].set_title(label, fontsize=14, fontweight="bold")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "word_clouds.png"))
    plt.close()
    print("[EDA] Saved word_clouds.png")


def plot_top_words(df, top_n=20):
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    text_col = "cleaned_review" if "cleaned_review" in df.columns else "review"

    for idx, (label, sent, color) in enumerate([
        ("Positive", 1, "#2ecc71"), ("Negative", 0, "#e74c3c"),
    ]):
        words = " ".join(df[df["sentiment"] == sent][text_col].dropna()).split()
        counter = Counter(words).most_common(top_n)
        words_list, counts_list = zip(*counter)

        axes[idx].barh(range(top_n), counts_list, color=color, edgecolor="white")
        axes[idx].set_yticks(range(top_n))
        axes[idx].set_yticklabels(words_list)
        axes[idx].invert_yaxis()
        axes[idx].set_title(f"Top {top_n} Words — {label} Reviews", fontweight="bold")
        axes[idx].set_xlabel("Frequency")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "top_words.png"))
    plt.close()
    print("[EDA] Saved top_words.png")


def plot_review_length_boxplot(df):
    set_plot_style()
    df = df.copy()
    text_col = "cleaned_review" if "cleaned_review" in df.columns else "review"
    df["review_length"] = df[text_col].apply(lambda x: len(str(x).split()))
    df["Sentiment"] = df["sentiment"].map({0: "Negative", 1: "Positive"})

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="Sentiment", y="review_length",
                palette={"Negative": "#e74c3c", "Positive": "#2ecc71"}, ax=ax)
    ax.set_title("Review Length by Sentiment Class", fontweight="bold")
    ax.set_ylabel("Number of Words")
    ax.set_xlabel("Sentiment")
    plt.savefig(os.path.join(PLOTS_DIR, "review_length_boxplot.png"))
    plt.close()
    print("[EDA] Saved review_length_boxplot.png")


def run_eda(df):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    plot_class_distribution(df)
    plot_review_length_distribution(df)
    plot_word_clouds(df)
    plot_top_words(df)
    plot_review_length_boxplot(df)
    print("[EDA] All visualizations saved to outputs/plots/")


def print_dataset_summary(df):
   
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    df_temp = df.copy()
    text_col = "cleaned_review" if "cleaned_review" in df_temp.columns else "review"
    print(f"  (measuring length from column: '{text_col}')")
    df_temp["review_length"] = df_temp[text_col].apply(lambda x: len(str(x).split()))

    pos = df_temp[df_temp["sentiment"] == 1]
    neg = df_temp[df_temp["sentiment"] == 0]

    summary = {
        "Metric": [
            "Total Reviews", "Positive Reviews", "Negative Reviews",
            "Avg Length (Positive)", "Avg Length (Negative)",
            "Max Length (Positive)", "Max Length (Negative)",
            "Min Length (Positive)", "Min Length (Negative)",
        ],
        "Value": [
            len(df_temp), len(pos), len(neg),
            f"{pos['review_length'].mean():.1f} words",
            f"{neg['review_length'].mean():.1f} words",
            f"{pos['review_length'].max()} words",
            f"{neg['review_length'].max()} words",
            f"{pos['review_length'].min()} words",
            f"{neg['review_length'].min()} words",
        ],
    }
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print()
