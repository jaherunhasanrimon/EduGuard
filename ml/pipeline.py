"""
EduGuard — ML Pipeline
Builds a scikit-learn Pipeline combining preprocessing and the XGBoost model.
XGBoost requires numeric class labels, so we wrap it with a LabelEncoder
inside a custom estimator so the pipeline remains sklearn-compatible.
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


class XGBLabelEncoded(BaseEstimator, ClassifierMixin):
    """
    Thin wrapper around XGBClassifier that handles string class labels by
    applying a LabelEncoder internally.  The pipeline-facing API (fit, predict,
    predict_proba, classes_) behaves exactly like a native sklearn estimator.
    """

    def __init__(
        self,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.tree_method = tree_method
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_

        self.xgb_ = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            tree_method=self.tree_method,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.xgb_.fit(X, y_enc, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self.xgb_.predict_proba(X)

    def predict(self, X):
        proba = self.xgb_.predict_proba(X)
        return self.le_.inverse_transform(np.argmax(proba, axis=1))


def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Build and return the full sklearn Pipeline.

    Preprocessing:
        - Numerical  → StandardScaler
        - Categorical → OrdinalEncoder (handles unseen values gracefully)
    Model:
        - XGBLabelEncoded (XGBoost + internal LabelEncoder for string targets)
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

    model = XGBLabelEncoded(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        tree_method="hist",
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
