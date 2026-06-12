"""
EduGuard — Training Script
Trains the Random Forest pipeline with 5-fold stratified CV,
evaluates on a held-out test set, and saves:
  • models/pipeline.pkl
  • models/shap_explainer.pkl
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    CATEGORICAL_FEATURES,
    DEMO_DATASET_PATH,
    DROPOUT_CLASS,
    MODELS_DIR,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from ml.pipeline import build_pipeline


def main():
    print("=" * 60)
    print("  EduGuard — Model Training Pipeline")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\n[1/6] Loading dataset from {DEMO_DATASET_PATH} …")
    df = pd.read_csv(DEMO_DATASET_PATH)
    df.columns = [col.replace("\ufeff", "").strip() for col in df.columns]

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    print(f"      Rows: {len(df):,}  |  Features: {X.shape[1]}  |  Classes: {list(y.unique())}")
    print(f"      Class distribution:\n{y.value_counts().to_string()}")

    # ── 2. Train / test split ──────────────────────────────────────
    print("\n[2/6] Splitting into train (80%) / test (20%) …")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── 3. Build pipeline ─────────────────────────────────────────
    print("\n[3/6] Building sklearn Pipeline …")
    pipeline = build_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)

    # ── 4. Cross-validation ───────────────────────────────────────
    print("\n[4/6] Running 5-fold Stratified CV (macro F1) …")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    t0 = time.time()
    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1
    )
    print(f"      CV Macro F1 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"      Fold scores : {[f'{s:.4f}' for s in cv_scores]}")
    print(f"      Time        : {time.time() - t0:.1f}s")

    # ── 5. Train final model ──────────────────────────────────────
    print("\n[5/6] Training final model on full training set …")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"      Done in {time.time() - t0:.1f}s")

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n      ── Test Set Results ──")
    print(f"      Accuracy  : {acc:.4f}")
    print(f"      Macro F1  : {f1:.4f}")
    print("\n      Classification Report:")
    print(classification_report(y_test, y_pred, target_names=sorted(y.unique())))

    # ── 6. Save artifacts ─────────────────────────────────────────
    print("\n[6/6] Saving model artifacts …")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    pipeline_path = MODELS_DIR / "pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"      ✅  Pipeline saved    → {pipeline_path}")

    # SHAP TreeExplainer — fit on transformed training data
    print("      Building SHAP TreeExplainer (this may take ~30s) …")
    rf_model = pipeline.named_steps["model"]
    X_train_transformed = pipeline.named_steps["preprocessor"].transform(X_train)

    explainer = shap.TreeExplainer(rf_model)

    explainer_path = MODELS_DIR / "shap_explainer.pkl"
    joblib.dump(explainer, explainer_path)
    print(f"      ✅  SHAP explainer saved → {explainer_path}")

    print("\n" + "=" * 60)
    print("  Training complete!  🎉")
    print(f"  Test Accuracy: {acc:.2%}  |  Macro F1: {f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
