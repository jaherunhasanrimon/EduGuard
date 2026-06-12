"""
EduGuard — SHAP Explainer Wrapper
Provides per-student SHAP value extraction using a pre-fitted TreeExplainer.
"""
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import FEATURE_DISPLAY_NAMES, DROPOUT_CLASS


def get_shap_values_for_student(
    student_idx: int,
    X_df,
    pipeline,
    explainer,
) -> dict:
    """
    Compute SHAP values for a single student row.

    Args:
        student_idx: Row index in X_df (already without Target column).
        X_df:        DataFrame of features (no Target column).
        pipeline:    Fitted sklearn Pipeline.
        explainer:   Fitted shap.TreeExplainer.

    Returns:
        dict with keys: shap_values, feature_names, feature_values, base_value
    """
    row = X_df.iloc[[student_idx]]
    X_transformed = pipeline.named_steps["preprocessor"].transform(row)

    # Get SHAP values — multiclass RF returns list[class][samples, features]
    shap_vals = explainer.shap_values(X_transformed)

    classes = list(pipeline.classes_)
    dropout_idx = classes.index(DROPOUT_CLASS)

    if isinstance(shap_vals, list):
        student_shap = shap_vals[dropout_idx][0]
        base = (
            float(explainer.expected_value[dropout_idx])
            if hasattr(explainer.expected_value, "__len__")
            else float(explainer.expected_value)
        )
    else:
        # Single-output (binary pipeline)
        student_shap = shap_vals[0]
        base = float(explainer.expected_value)

    # Recover ordered feature names from the ColumnTransformer
    num_names = pipeline.named_steps["preprocessor"].transformers_[0][2]
    cat_names = pipeline.named_steps["preprocessor"].transformers_[1][2]
    feature_names = list(num_names) + list(cat_names)

    feature_values = {
        fname: row.iloc[0].get(fname, "N/A") if fname in row.columns else "N/A"
        for fname in feature_names
    }

    return {
        "shap_values": student_shap,
        "feature_names": feature_names,
        "feature_values": feature_values,
        "base_value": base,
    }


def get_top_factors(shap_data: dict, n: int = 5) -> list:
    """
    Return the top N features sorted by absolute SHAP value.

    Each entry is a dict:
        feature, display_name, shap_value, feature_value, direction
    """
    shap_vals = shap_data["shap_values"]
    feature_names = shap_data["feature_names"]
    feature_values = shap_data["feature_values"]

    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:n]

    factors = []
    for idx in sorted_idx:
        fname = feature_names[idx]
        factors.append(
            {
                "feature": fname,
                "display_name": FEATURE_DISPLAY_NAMES.get(fname, fname),
                "shap_value": float(shap_vals[idx]),
                "feature_value": feature_values.get(fname, "N/A"),
                "direction": "increases" if shap_vals[idx] > 0 else "decreases",
            }
        )

    return factors
