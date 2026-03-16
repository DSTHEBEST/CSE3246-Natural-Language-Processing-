"""
Utility helper functions for the NLP project.
"""
import os
import pickle
import numpy as np


def ensure_dirs():
    """Create all required output directories."""
    dirs = [
        "data",
        "outputs",
        "outputs/plots",
        "outputs/models",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_object(obj, filepath):
    """Pickle an object to disk."""
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_object(filepath):
    """Load a pickled object from disk."""
    with open(filepath, "rb") as f:
        return pickle.load(f)


def set_plot_style():
    """Set a consistent, attractive matplotlib style."""
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    })
