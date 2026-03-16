"""
Machine Learning Models module.
Implements Logistic Regression, SVM, and Random Forest with training curves.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import issparse

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.utils import set_plot_style

PLOTS_DIR = "outputs/plots"


def get_models():
    """
    Return a dictionary of model name -> (model instance, hyperparameter description).
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            ),
            {
                "C (Regularization)": 1.0,
                "Penalty": "L2",
                "Solver": "liblinear",
                "Max Iterations": 1000,
            },
        ),
        "SVM (Linear)": (
            LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
            ),
            {
                "C (Regularization)": 1.0,
                "Kernel": "Linear",
                "Max Iterations": 2000,
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=50,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
            {
                "N Estimators": 200,
                "Max Depth": 50,
                "Min Samples Split": 5,
                "Min Samples Leaf": 2,
            },
        ),
    }
    return models


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Stratified train-test split.

    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_and_predict(model, X_train, X_test, y_train):
    """
    Train a model and return predictions.

    Returns:
        model, y_pred
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model, y_pred


def plot_learning_curves(model, model_name, X, y, feature_name):
    """
    Plot the learning curve (training vs validation accuracy) for a model.
    Used for visualizing the training process.
    """
    set_plot_style()

    # For large sparse matrices, subsample for speed
    n_samples = X.shape[0]
    if n_samples > 10000:
        # Subsample to 10000 for learning curve computation
        from sklearn.utils import resample
        indices = resample(range(n_samples), n_samples=10000, random_state=42, stratify=y)
        if issparse(X):
            X_sub = X[indices]
        else:
            X_sub = X[indices]
        y_sub = y[indices] if isinstance(y, np.ndarray) else y.iloc[indices].values
    else:
        X_sub = X
        y_sub = y if isinstance(y, np.ndarray) else y.values

    train_sizes = np.linspace(0.1, 1.0, 8)

    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_sub, y_sub,
            train_sizes=train_sizes,
            cv=5,
            scoring="accuracy",
            n_jobs=-1,
            random_state=42,
        )
    except Exception as e:
        print(f"  [WARN] Could not compute learning curve for {model_name}: {e}")
        return

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#2ecc71")
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#3498db")
    ax.plot(train_sizes_abs, train_mean, "o-", color="#2ecc71", linewidth=2,
            label="Training Accuracy")
    ax.plot(train_sizes_abs, val_mean, "o-", color="#3498db", linewidth=2,
            label="Validation Accuracy")

    ax.set_title(f"Learning Curve — {model_name} ({feature_name})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    safe_feat = feature_name.replace(" ", "_").replace("-", "")
    filename = f"learning_curve_{safe_name}_{safe_feat}.png"
    plt.savefig(os.path.join(PLOTS_DIR, filename))
    plt.close()
    print(f"  [MODEL] Saved {filename}")


def print_hyperparameters(models_dict):
    """Print hyperparameter descriptions for all models."""
    print("\n" + "=" * 60)
    print("MODEL HYPERPARAMETERS")
    print("=" * 60)

    for name, (model, params) in models_dict.items():
        print(f"\n  {name}:")
        for param, value in params.items():
            print(f"    {param}: {value}")
    print()
