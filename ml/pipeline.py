"""
EduGuard — ML Pipeline
Builds a scikit-learn Pipeline combining feature engineering,
preprocessing, and the XGBoost model.

Feature engineering (5 new features derived from the two notebooks):
  • total_approved         = sum of approved units across both semesters
  • academic_success       = sum of semester grades (raw signal of grade level)
  • approval_rate          = approved / (enrolled + 1) — completion rate
  • performance_change     = 2nd sem grade − 1st sem grade — trajectory
  • evaluation_efficiency  = approved / (evaluations + 1) — exam efficiency

XGBoost requires numeric class labels, so we wrap it with a LabelEncoder
inside a custom estimator so the pipeline remains sklearn-compatible.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


# ── Feature Engineering ───────────────────────────────────────────────────────

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds 5 derived features inspired by the two best-performing Kaggle
    notebooks in trained/. Must sit before the ColumnTransformer in the
    pipeline so the new columns are available to the preprocessor.

    Works on both DataFrames and numpy arrays; always returns a DataFrame.
    """

    # The 5 new column names — referenced in build_pipeline so they are
    # kept in sync automatically.
    NEW_FEATURES = [
        "total_approved",
        "academic_success",
        "approval_rate",
        "performance_change",
        "evaluation_efficiency",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        # Fill NaN in source columns with 0 before computing derived features.
        # Real student records may be missing curricular unit data for semesters
        # not yet completed; 0 is the correct neutral value in all five formulas.
        sem_cols = [
            "Curricular units 1st sem (approved)",
            "Curricular units 2nd sem (approved)",
            "Curricular units 1st sem (grade)",
            "Curricular units 2nd sem (grade)",
            "Curricular units 1st sem (enrolled)",
            "Curricular units 2nd sem (enrolled)",
            "Curricular units 1st sem (evaluations)",
            "Curricular units 2nd sem (evaluations)",
        ]
        for col in sem_cols:
            if col in X_out.columns:
                X_out[col] = X_out[col].fillna(0)

        # 1. total_approved
        X_out["total_approved"] = (
            X_out["Curricular units 1st sem (approved)"]
            + X_out["Curricular units 2nd sem (approved)"]
        )

        # 2. academic_success (combined semester grade totals)
        X_out["academic_success"] = (
            X_out["Curricular units 1st sem (grade)"]
            + X_out["Curricular units 2nd sem (grade)"]
        )

        # 3. approval_rate (approved / enrolled, +1 to avoid div-by-zero)
        X_out["approval_rate"] = X_out["total_approved"] / (
            X_out["Curricular units 1st sem (enrolled)"]
            + X_out["Curricular units 2nd sem (enrolled)"]
            + 1
        )

        # 4. performance_change (grade trajectory across semesters)
        X_out["performance_change"] = (
            X_out["Curricular units 2nd sem (grade)"]
            - X_out["Curricular units 1st sem (grade)"]
        )

        # 5. evaluation_efficiency (how many approved per evaluation)
        X_out["evaluation_efficiency"] = X_out["total_approved"] / (
            X_out["Curricular units 1st sem (evaluations)"]
            + X_out["Curricular units 2nd sem (evaluations)"]
            + 1
        )

        # Replace any remaining NaN/inf in engineered cols with 0
        for col in self.NEW_FEATURES:
            X_out[col] = X_out[col].fillna(0).replace([float("inf"), float("-inf")], 0)

        return X_out


# ── XGBoost Wrapper ───────────────────────────────────────────────────────────

class XGBLabelEncoded(BaseEstimator, ClassifierMixin):
    """
    Thin wrapper around XGBClassifier that handles string class labels by
    applying a LabelEncoder internally.  The pipeline-facing API (fit, predict,
    predict_proba, classes_) behaves exactly like a native sklearn estimator.
    """

    def __init__(
        self,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=4,
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


# ── Pipeline Builder ──────────────────────────────────────────────────────────

def build_pipeline(numeric_features: list, categorical_features: list) -> Pipeline:
    """
    Build and return the full sklearn Pipeline.

    Steps:
        1. FeatureEngineer — adds 5 derived numeric features
        2. ColumnTransformer (preprocessor)
               Numerical  → StandardScaler  (original + 5 new features)
               Categorical → OrdinalEncoder (handles unseen values gracefully)
        3. XGBLabelEncoded — XGBoost + internal LabelEncoder for string targets

    Note: settings.py feature lists are NOT modified. The 5 new features are
    appended here so all downstream code (predict.py, app/**) stays intact.
    """
    # Extend the numeric feature list with the 5 engineered columns
    extended_numeric = numeric_features + FeatureEngineer.NEW_FEATURES

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                extended_numeric,
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
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=4,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("fe", FeatureEngineer()),
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline
