"""
EduGuard — Inference Helpers
Provides cached model/data loading and batch prediction functions.
Applies tuned per-class decision thresholds (from models/thresholds.json)
instead of the raw argmax so that Graduate and Enrolled are not swallowed
by the dominant Dropout class.
"""
import json
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
    THRESHOLDS_PATH,
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


def load_thresholds() -> dict:
    """
    Load per-class decision thresholds from models/thresholds.json.
    Falls back to uniform thresholds (1/n_classes) if the file is missing
    (i.e. old model trained without threshold tuning).
    """
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            data = json.load(f)
        return data  # {"thresholds": {...}, "classes": [...]}
    return None


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


# ─── Threshold-Aware Prediction Helpers ──────────────────────────────────────

def _apply_thresholds(probas: np.ndarray, classes: list, thresholds: dict) -> np.ndarray:
    """
    Apply per-class thresholds via ratio-based argmax:
        predicted = argmax(proba[i] / threshold[class_i])

    This is equivalent to shifting decision boundaries so that classes with a
    lower threshold require less probability mass to be predicted.
    """
    thresh_arr = np.array([thresholds.get(c, 1.0 / len(classes)) for c in classes])
    adjusted = probas / thresh_arr
    return np.array(classes)[np.argmax(adjusted, axis=1)]


# ─── Risk Tier ────────────────────────────────────────────────────────────────

def get_risk_tier(prob: float) -> str:
    if prob >= RISK_THRESHOLDS["high"]:
        return "High"
    elif prob >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "Low"


# ─── Batch Prediction ─────────────────────────────────────────────────────────

def predict_batch(df: pd.DataFrame, pipeline=None, thresholds_data: dict = None) -> pd.DataFrame:
    """
    Run predictions on a full DataFrame.

    Adds columns: dropout_prob, risk_tier, predicted_label, student_id.
    Uses tuned per-class thresholds if thresholds_data is provided.
    Original data columns are preserved.
    """
    if pipeline is None:
        pipeline = load_pipeline()
    if thresholds_data is None:
        thresholds_data = load_thresholds()

    X = df.drop(columns=[TARGET_COL], errors="ignore")

    classes = list(pipeline.classes_)
    dropout_idx = classes.index(DROPOUT_CLASS)

    probas = pipeline.predict_proba(X)
    dropout_probs = probas[:, dropout_idx]

    # Use tuned thresholds if available, otherwise fall back to argmax
    if thresholds_data and "thresholds" in thresholds_data:
        predicted_labels = _apply_thresholds(
            probas, classes, thresholds_data["thresholds"]
        )
    else:
        predicted_labels = pipeline.predict(X)

    result = df.copy()
    result["dropout_prob"] = dropout_probs
    result["risk_tier"] = [get_risk_tier(p) for p in dropout_probs]
    result["predicted_label"] = predicted_labels
    result["student_id"] = range(1, len(df) + 1)

    return result

