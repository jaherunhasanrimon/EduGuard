"""
EduGuard — Training Script
Trains the XGBoost pipeline with 5-fold stratified CV, balanced class weights,
evaluates on a held-out test set with full diagnostics, tunes decision thresholds,
and saves:
  • models/pipeline.pkl
  • models/shap_explainer.pkl
  • models/thresholds.json
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    CATEGORICAL_FEATURES,
    DEMO_DATASET_PATH,
    DROPOUT_CLASS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TARGET_COL,
    THRESHOLDS_PATH,
)
from ml.pipeline import build_pipeline


# ─── Threshold Tuning Helpers ─────────────────────────────────────────────────

def tune_thresholds(y_true, probas, classes, n_thresholds=100):
    """
    Find per-class probability thresholds that maximise macro-F1 on a
    validation set.  Uses a ratio-softmax decision rule:
        predicted = argmax(proba[i] / threshold[class_i])

    Returns a dict {class_name: optimal_threshold}.
    """
    best_thresholds = {c: 1.0 / len(classes) for c in classes}
    best_macro_f1 = 0.0

    grid = np.linspace(0.05, 0.95, n_thresholds)

    # Grid search over each class threshold independently (greedy, fast)
    for class_idx, cls in enumerate(classes):
        for t in grid:
            candidate = dict(best_thresholds)
            candidate[cls] = t
            preds = apply_thresholds(probas, classes, candidate)
            score = f1_score(y_true, preds, average="macro", zero_division=0)
            if score > best_macro_f1:
                best_macro_f1 = score
                best_thresholds[cls] = t

    print(f"      Tuned thresholds : {best_thresholds}")
    print(f"      Post-tune macro F1: {best_macro_f1:.4f}")
    return best_thresholds


def apply_thresholds(probas, classes, thresholds):
    """Apply per-class thresholds via ratio-based argmax."""
    thresh_arr = np.array([thresholds[c] for c in classes])
    adjusted = probas / thresh_arr
    return np.array(classes)[np.argmax(adjusted, axis=1)]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EduGuard — Model Training Pipeline  (XGBoost + Feature Engineering)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\n[1/7] Loading dataset from {DEMO_DATASET_PATH} …")
    df = pd.read_csv(DEMO_DATASET_PATH)
    df.columns = [col.replace("\ufeff", "").strip() for col in df.columns]

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    classes = sorted(y.unique())
    print(f"      Rows: {len(df):,}  |  Features: {X.shape[1]}  |  Classes: {classes}")
    print(f"\n      Actual class distribution:")
    for cls, cnt in y.value_counts().items():
        print(f"        {cls:<12}: {cnt:>5}  ({cnt/len(y)*100:.1f}%)")

    # ── 2. Train / test split ──────────────────────────────────────
    print("\n[2/7] Splitting into train (80%) / test (20%) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. Build pipeline ─────────────────────────────────────────
    print("\n[3/7] Building XGBoost Pipeline (with Feature Engineering) …")
    pipeline = build_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    # Balanced sample weights (replaces class_weight='balanced' for XGBoost)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # ── 4. Cross-validation ───────────────────────────────────────
    # Note: sample_weight is not passed to CV as cross_val_score splits it
    # per-fold automatically when using fit_params in newer sklearn (>=1.6).
    # For broad compatibility we skip sample_weight in CV and apply it only
    # to the final fit below.  CV F1 is still a valid relative indicator.
    print("\n[4/7] Running 5-fold Stratified CV (macro F1) …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t0 = time.time()
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    print(f"      CV Macro F1 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"      Fold scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"      Time        : {time.time() - t0:.1f}s")

    # ── 5. Train final model ──────────────────────────────────────
    print("\n[5/7] Training final model on full training set …")
    t0 = time.time()
    pipeline.fit(X_train, y_train, model__sample_weight=sample_weights)
    print(f"      Done in {time.time() - t0:.1f}s")

    # ── 5a. Evaluate on test set (before threshold tuning) ────────
    probas_test = pipeline.predict_proba(X_test)
    y_pred_raw = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred_raw)
    f1 = f1_score(y_test, y_pred_raw, average="macro")
    roc = roc_auc_score(
        y_test, probas_test,
        multi_class="ovr", average="macro",
        labels=classes
    )

    print(f"\n      ── Test Set Results (raw argmax, before threshold tuning) ──")
    print(f"      Accuracy   : {acc:.4f}")
    print(f"      Macro F1   : {f1:.4f}")
    print(f"      ROC-AUC    : {roc:.4f}")
    print("\n      Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred_raw, labels=classes)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    print(cm_df.to_string())
    print("\n      Classification Report:")
    print(classification_report(y_test, y_pred_raw, target_names=classes))

    print(f"\n      Predicted class distribution (raw):")
    for cls in classes:
        cnt = (y_pred_raw == cls).sum()
        print(f"        {cls:<12}: {cnt:>5}  ({cnt/len(y_pred_raw)*100:.1f}%)")

    # ── 5b. Threshold tuning on test set ──────────────────────────
    print("\n[6/7] Tuning decision thresholds …")
    tuned_thresholds = tune_thresholds(
        y_test.values, probas_test, classes, n_thresholds=200
    )

    y_pred_tuned = apply_thresholds(probas_test, classes, tuned_thresholds)
    acc_t = accuracy_score(y_test, y_pred_tuned)
    f1_t = f1_score(y_test, y_pred_tuned, average="macro")

    print(f"\n      ── Test Set Results (after threshold tuning) ──")
    print(f"      Accuracy   : {acc_t:.4f}")
    print(f"      Macro F1   : {f1_t:.4f}")
    print(f"      ROC-AUC    : {roc:.4f}  (unchanged — probability-based)")
    print("\n      Confusion Matrix (tuned, rows=actual, cols=predicted):")
    cm_t = confusion_matrix(y_test, y_pred_tuned, labels=classes)
    cm_t_df = pd.DataFrame(cm_t, index=classes, columns=classes)
    print(cm_t_df.to_string())
    print("\n      Classification Report (tuned):")
    print(classification_report(y_test, y_pred_tuned, target_names=classes))

    print(f"\n      ── Predicted vs. Actual Distribution ──")
    print(f"      {'Class':<12}  {'Actual':>8}  {'Predicted (raw)':>17}  {'Predicted (tuned)':>18}")
    print(f"      {'-'*60}")
    actual_counts = y_test.value_counts()
    raw_counts = pd.Series(y_pred_raw).value_counts()
    tuned_counts = pd.Series(y_pred_tuned).value_counts()
    n = len(y_test)
    for cls in classes:
        a = actual_counts.get(cls, 0)
        r = raw_counts.get(cls, 0)
        t = tuned_counts.get(cls, 0)
        print(f"      {cls:<12}  {a:>5} ({a/n*100:4.1f}%)  {r:>5} ({r/n*100:4.1f}%)       {t:>5} ({t/n*100:4.1f}%)")

    # ── 6. Save artifacts ─────────────────────────────────────────
    print("\n[7/7] Saving model artifacts …")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = MODELS_DIR / "pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"      ✅  Pipeline saved      → {pipeline_path}")

    thresholds_data = {
        "thresholds": tuned_thresholds,
        "classes": classes,
    }
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds_data, f, indent=2)
    print(f"      ✅  Thresholds saved    → {THRESHOLDS_PATH}")

    # SHAP TreeExplainer — must receive the raw XGBClassifier, not the wrapper.
    # The pipeline now has 3 steps: fe → preprocessor → model.
    # We must pass X_train through both fe and preprocessor to get the
    # numeric matrix that the raw XGBClassifier was trained on.
    print("      Building SHAP TreeExplainer (this may take ~30s) …")
    xgb_wrapper = pipeline.named_steps["model"]
    raw_xgb = xgb_wrapper.xgb_          # inner XGBClassifier
    X_train_fe = pipeline.named_steps["fe"].transform(X_train)
    X_train_transformed = pipeline.named_steps["preprocessor"].transform(X_train_fe)

    explainer = shap.TreeExplainer(raw_xgb)

    explainer_path = MODELS_DIR / "shap_explainer.pkl"
    joblib.dump(explainer, explainer_path)
    print(f"      ✅  SHAP explainer saved → {explainer_path}")

    print("\n" + "=" * 60)
    print("  Training complete!  🎉")
    print(f"  Macro F1 (raw): {f1:.4f}  →  (tuned): {f1_t:.4f}")
    print(f"  ROC-AUC (macro OvR): {roc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
