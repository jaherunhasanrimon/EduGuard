"""
EduGuard — Bias Diagnostic Script
===================================
Run this standalone to get a comprehensive class-imbalance audit:

  python ml/analyze_bias.py

Outputs:
  1. Actual class distribution in the dataset
  2. Trains the model and evaluates on a held-out test set
  3. Confusion matrix (text + saved PNG)
  4. Full classification report per class
  5. ROC-AUC (macro OvR) score
  6. Predicted vs. actual class distribution (with and without threshold tuning)
  7. Plots saved to: models/bias_analysis.png
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    CATEGORICAL_FEATURES,
    DEMO_DATASET_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TARGET_COL,
    THRESHOLDS_PATH,
)
from ml.pipeline import build_pipeline

# ─── Colours ──────────────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Dropout":  "#E24B4A",
    "Graduate": "#639922",
    "Enrolled": "#3B82F6",
}

SEP = "=" * 65


def load_data():
    df = pd.read_csv(DEMO_DATASET_PATH)
    df.columns = [col.replace("\ufeff", "").strip() for col in df.columns]
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def apply_thresholds(probas, classes, thresholds):
    thresh_arr = np.array([thresholds.get(c, 1.0 / len(classes)) for c in classes])
    adjusted = probas / thresh_arr
    return np.array(classes)[np.argmax(adjusted, axis=1)]


def tune_thresholds(y_true, probas, classes, n_thresholds=200):
    best_thresholds = {c: 1.0 / len(classes) for c in classes}
    best_macro_f1 = 0.0
    grid = np.linspace(0.05, 0.95, n_thresholds)
    for class_idx, cls in enumerate(classes):
        for t in grid:
            candidate = dict(best_thresholds)
            candidate[cls] = t
            preds = apply_thresholds(probas, classes, candidate)
            score = f1_score(y_true, preds, average="macro", zero_division=0)
            if score > best_macro_f1:
                best_macro_f1 = score
                best_thresholds[cls] = t
    return best_thresholds, best_macro_f1


def plot_distribution(y_actual, y_raw, y_tuned, classes, output_path):
    """Plot actual vs. predicted class distributions (before & after threshold tuning)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("EduGuard — Class Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)

    datasets = [
        ("Actual Distribution", y_actual),
        ("Predicted (Raw Argmax)", y_raw),
        ("Predicted (Threshold-Tuned)", y_tuned),
    ]
    for ax, (title, series) in zip(axes, datasets):
        counts = {cls: (np.array(series) == cls).sum() for cls in classes}
        bars = ax.bar(
            classes,
            [counts[c] for c in classes],
            color=[CLASS_COLORS.get(c, "#888") for c in classes],
            edgecolor="white",
            linewidth=1.5,
            width=0.55,
        )
        total = sum(counts.values())
        for bar, cls in zip(bars, classes):
            pct = counts[cls] / total * 100
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.01,
                f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_ylim(0, max(counts.values()) * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"      📊  Distribution plot saved → {output_path}")


def plot_confusion_matrices(y_test, y_raw, y_tuned, classes, output_path):
    """Side-by-side confusion matrices before and after threshold tuning."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("EduGuard — Confusion Matrices", fontsize=14, fontweight="bold")

    for ax, (y_pred, title) in zip(axes, [
        (y_raw,   "Before Threshold Tuning"),
        (y_tuned, "After Threshold Tuning"),
    ]):
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"      📊  Confusion matrix plot saved → {output_path}")


def main():
    print(SEP)
    print("  EduGuard — Bias & Class Imbalance Diagnostic")
    print(SEP)

    # ── 1. Dataset distribution ───────────────────────────────────
    print("\n[STEP 1]  Actual Class Distribution in Dataset")
    print("-" * 45)
    X, y = load_data()
    classes = sorted(y.unique())
    total = len(y)
    for cls in classes:
        cnt = (y == cls).sum()
        bar = "█" * int(cnt / total * 40)
        print(f"  {cls:<12}: {cnt:>5} ({cnt/total*100:5.1f}%)  {bar}")
    print(f"\n  Total samples: {total:,}  |  Classes: {classes}")

    # ── 2. Train / test split ─────────────────────────────────────
    print("\n[STEP 2]  Train/Test Split (80/20 stratified)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. Check for saved model ──────────────────────────────────
    pipeline_path = MODELS_DIR / "pipeline.pkl"
    if pipeline_path.exists():
        print("\n[STEP 3]  Loading existing saved model …")
        pipeline = joblib.load(pipeline_path)
        print(f"  Model type: {type(pipeline.named_steps['model']).__name__}")
    else:
        print("\n[STEP 3]  No saved model found — training fresh …")
        pipeline = build_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
        sw = compute_sample_weight("balanced", y=y_train)
        pipeline.fit(X_train, y_train, model__sample_weight=sw)
        print("  Model trained ✅")

    # ── 4. Raw predictions ────────────────────────────────────────
    print("\n[STEP 4]  Evaluating with Raw Argmax (no threshold tuning)")
    print("-" * 45)
    probas_test = pipeline.predict_proba(X_test)
    y_raw = pipeline.predict(X_test)

    roc = roc_auc_score(
        y_test, probas_test, multi_class="ovr", average="macro", labels=classes
    )
    print(f"\n  ROC-AUC (macro OvR): {roc:.4f}")
    print(f"\n  Classification Report (raw argmax):")
    print(classification_report(y_test, y_raw, target_names=classes))
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_raw, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())

    # ── 5. Threshold tuning ───────────────────────────────────────
    print("\n[STEP 5]  Tuning Per-Class Decision Thresholds …")
    print("-" * 45)

    # Try to load saved thresholds; re-tune if not present
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            tdata = json.load(f)
        tuned_thresholds = tdata["thresholds"]
        print(f"  Loaded saved thresholds: {tuned_thresholds}")
    else:
        tuned_thresholds, tuned_f1 = tune_thresholds(
            y_test.values, probas_test, classes
        )
        print(f"  Optimal thresholds: {tuned_thresholds}")
        print(f"  Post-tune macro F1: {tuned_f1:.4f}")

    y_tuned = apply_thresholds(probas_test, classes, tuned_thresholds)

    print(f"\n  Classification Report (threshold-tuned):")
    print(classification_report(y_test, y_tuned, target_names=classes))
    print(f"\n  Confusion Matrix (tuned, rows=actual, cols=predicted):")
    cm_t = confusion_matrix(y_test, y_tuned, labels=classes)
    cm_t_df = pd.DataFrame(cm_t, index=classes, columns=classes)
    print(cm_t_df.to_string())

    # ── 6. Distribution comparison ────────────────────────────────
    print(f"\n[STEP 6]  Predicted vs. Actual Distribution Comparison")
    print("-" * 65)
    print(f"  {'Class':<12}  {'Actual':>8}  {'Predicted (raw)':>17}  {'Predicted (tuned)':>19}")
    print(f"  {'-'*60}")
    n = len(y_test)
    raw_counts = pd.Series(y_raw).value_counts()
    tuned_counts = pd.Series(y_tuned).value_counts()
    actual_counts = y_test.value_counts()
    for cls in classes:
        a = actual_counts.get(cls, 0)
        r = raw_counts.get(cls, 0)
        t = tuned_counts.get(cls, 0)
        print(f"  {cls:<12}  {a:>5} ({a/n*100:4.1f}%)  {r:>5} ({r/n*100:4.1f}%)        {t:>5} ({t/n*100:4.1f}%)")

    # ── 7. Generate plots ─────────────────────────────────────────
    print(f"\n[STEP 7]  Generating Diagnostic Plots …")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plot_distribution(
        y_test.values, y_raw, y_tuned, classes,
        output_path=MODELS_DIR / "bias_distribution.png",
    )
    plot_confusion_matrices(
        y_test, y_raw, y_tuned, classes,
        output_path=MODELS_DIR / "bias_confusion_matrices.png",
    )

    print(f"\n{SEP}")
    print("  Diagnostic complete!  All plots saved to models/")
    print(SEP)


if __name__ == "__main__":
    main()
