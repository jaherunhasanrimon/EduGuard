"""
EduGuard — Inference Helpers
Provides cached model/data loading and batch/single prediction functions.
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    DEMO_DATASET_PATH,
    MODELS_DIR,
    TARGET_COL,
    DROPOUT_CLASS,
    RISK_THRESHOLDS,
)

# ─── Cached loaders (Streamlit-agnostic; decorators applied in app layer) ──────

def load_pipeline():
    """Load the fitted sklearn pipeline from disk."""
    path = MODELS_DIR / "pipeline.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python ml/train.py` first."
        )
    return joblib.load(path)


def load_explainer():
    """Load the fitted SHAP TreeExplainer from disk."""
    path = MODELS_DIR / "shap_explainer.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"SHAP explainer not found at {path}. Run `python ml/train.py` first."
        )
    return joblib.load(path)


def load_demo_data() -> pd.DataFrame:
    """Load and clean the bundled UCI demo dataset."""
    df = pd.read_csv(DEMO_DATASET_PATH)
    # Strip BOM and whitespace from column names
    df.columns = [col.replace("\ufeff", "").strip() for col in df.columns]
    return df


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """Load and clean a user-uploaded CSV file."""
    df = pd.read_csv(uploaded_file)
    df.columns = [col.replace("\ufeff", "").strip() for col in df.columns]
    return df


# ─── Risk Tier ────────────────────────────────────────────────────────────────

def get_risk_tier(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["high"]:
        return "High"
    elif prob >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


# ─── Batch Prediction ─────────────────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, pipeline=None) -> pd.DataFrame:
    """
    Run predictions on a full DataFrame.

    Adds columns: dropout_prob, risk_tier, predicted_label, student_id.
    Original data columns are preserved.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    X = df.drop(columns=[TARGET_COL], errors="ignore")

    classes = list(pipeline.classes_)
    dropout_idx = classes.index(DROPOUT_CLASS)

    probas = pipeline.predict_proba(X)
    dropout_probs = probas[:, dropout_idx]
    predicted_labels = pipeline.predict(X)

    result = df.copy()
    result["dropout_prob"] = dropout_probs
    result["risk_tier"] = [get_risk_tier(p) for p in dropout_probs]
    result["predicted_label"] = predicted_labels
    result["student_id"] = range(1, len(df) + 1)

    return result


# ─── Single-Row Prediction ────────────────────────────────────────────────────

def predict_single(feature_dict: dict, pipeline=None) -> dict:
    """
    Run prediction for a single student (used by the What-If Simulator).

    Args:
        feature_dict: {column_name: value} for all required features.
        pipeline:     Optional pre-loaded pipeline.

    Returns:
        dict with dropout_prob, risk_tier, predicted_label, all_probas.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    row_df = pd.DataFrame([feature_dict])

    classes = list(pipeline.classes_)
    dropout_idx = classes.index(DROPOUT_CLASS)

    probas = pipeline.predict_proba(row_df)
    dropout_prob = float(probas[0, dropout_idx])
    predicted_label = pipeline.predict(row_df)[0]

    return {
        "dropout_prob": dropout_prob,
        "risk_tier": get_risk_tier(dropout_prob),
        "predicted_label": predicted_label,
        "all_probas": {cls: float(p) for cls, p in zip(classes, probas[0])},
    }
