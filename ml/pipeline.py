"""
EduGuard — ML Pipeline
Builds a scikit-learn Pipeline combining preprocessing and the Random Forest model.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Build and return the full sklearn Pipeline.

    Preprocessing:
        - Numerical  → StandardScaler
        - Categorical → OrdinalEncoder (handles unseen values gracefully)
    Model:
        - RandomForestClassifier with balanced class weights
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
