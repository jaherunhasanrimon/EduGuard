"""
EduGuard — Inference Helpers
Provides cached model/data loading and batch prediction functions.
Applies tuned per-class decision thresholds (from models/thresholds.json)
instead of the raw argmax so that Graduate and Enrolled are not swallowed
by the dominant Dropout class.
"""
import io
import json
import logging
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

from config.settings import (
    DEMO_DATASET_PATH,
    MODELS_DIR,
    TARGET_COL,
    DROPOUT_CLASS,
    RISK_THRESHOLDS,
    THRESHOLDS_PATH,
    ALL_FEATURES,
)

# ─── Custom Exceptions ───────────────────────────────────────────────────────

class CSVValidationError(ValueError):
    """Raised when an uploaded CSV file fails schema validation."""
    pass


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
    Falls back to None if the file is missing (i.e. old model trained without
    threshold tuning); predict_batch will then use raw argmax.
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


# ─── Robust CSV Loading ────────────────────────────────────────────────────────

class CSVValidationError(ValueError):
    """Raised when an uploaded CSV cannot be used for inference."""


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    """
    Load and validate a user-uploaded CSV file.

    Performs the following sanity checks:
    1. Parses the CSV with error recovery (bad lines are skipped).
    2. Strips BOM and whitespace from column names.
    3. Checks that at least a minimum subset of expected features is present.
    4. Coerces numeric columns to numeric, filling non-parseable cells with
       the column median (so dirty data doesn't NaN-poison the whole batch).
    5. Drops completely empty rows.

    Raises:
        CSVValidationError: if the file cannot be parsed or lacks required columns.
    """
    if isinstance(uploaded_file, (bytes, bytearray)):
        uploaded_file = io.BytesIO(uploaded_file)

    # ── 1. Parse with on_bad_lines="skip" to survive encoding quirks ──────────
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines="skip", encoding_errors="replace")
    except Exception as exc:
        raise CSVValidationError(f"Could not parse CSV file: {exc}") from exc

    if df.empty:
        raise CSVValidationError("The uploaded CSV file is empty.")

    # ── 2. Normalise column names ──────────────────────────────────────────────
    df.columns = [col.replace("\ufeff", "").strip() for col in df.columns]

    # ── 3. Check minimum required feature overlap ──────────────────────────────
    present = set(df.columns)
    required = set(ALL_FEATURES)
    missing = required - present
    # Allow up to 30% missing features (e.g. optional columns); hard-fail beyond
    if len(missing) > 0.3 * len(required):
        sample = sorted(missing)[:6]
        raise CSVValidationError(
            f"Uploaded CSV is missing {len(missing)} required feature columns "
            f"(e.g. {', '.join(sample)}, …). "
            "Please ensure the file matches the expected EduGuard student dataset schema."
        )

    # ── 4. Coerce numeric columns, fill dirty cells with column median ─────────
    from config.settings import NUMERIC_FEATURES
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

    # ── 5. Drop completely empty rows ──────────────────────────────────────────
    df = df.dropna(how="all").reset_index(drop=True)

    if df.empty:
        raise CSVValidationError("After cleaning, the CSV contains no valid rows.")

    log.info("Uploaded CSV loaded: %d rows, %d columns.", len(df), len(df.columns))
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

    Memory optimisation: uses vectorised numpy operations rather than a
    Python-level loop for risk_tier assignment (important for large CSVs).
    """
    if pipeline is None:
        pipeline = load_pipeline()
    if thresholds_data is None:
        thresholds_data = load_thresholds()

    X = df.drop(columns=[TARGET_COL], errors="ignore")

    classes = list(pipeline.classes_)
    dropout_idx = classes.index(DROPOUT_CLASS)

    # ── Vectorised probability inference ──────────────────────────────────────
    probas = pipeline.predict_proba(X)
    dropout_probs = probas[:, dropout_idx]

    # ── Label assignment ───────────────────────────────────────────────────────
    if thresholds_data and "thresholds" in thresholds_data:
        predicted_labels = _apply_thresholds(
            probas, classes, thresholds_data["thresholds"]
        )
    else:
        predicted_labels = pipeline.predict(X)

    # ── Vectorised risk_tier (avoids per-row Python loop) ─────────────────────
    hi  = RISK_THRESHOLDS["high"]
    med = RISK_THRESHOLDS["medium"]
    risk_tiers = np.where(
        dropout_probs >= hi, "High",
        np.where(dropout_probs >= med, "Medium", "Low")
    )

    result = df.copy()
    result["dropout_prob"]    = dropout_probs
    result["risk_tier"]       = risk_tiers
    result["predicted_label"] = predicted_labels
    result["student_id"]      = np.arange(1, len(df) + 1)

    return result
