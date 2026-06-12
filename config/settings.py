"""
EduGuard — Central Configuration
Column mappings, feature lists, risk thresholds, and app constants.
"""
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# ─── Application Metadata ─────────────────────────────────────────────────────
APP_TITLE = "EduGuard"
APP_SUBTITLE = "AI-Powered Student Dropout Risk Intelligence"
UNIVERSITY_NAME = "Daffodil International University"
APP_VERSION = "1.0.0-MVP"
COMPETITION_LABEL = "DIU AI Project Competition 2026"

# ─── Authentication ───────────────────────────────────────────────────────────
DEFAULT_PASSWORD = "eduguard2026"

# ─── Dataset ──────────────────────────────────────────────────────────────────
DEMO_DATASET_PATH = DATA_DIR / "dataset_2.csv"
TARGET_COL = "Target"
DROPOUT_CLASS = "Dropout"
GRADUATE_CLASS = "Graduate"
ENROLLED_CLASS = "Enrolled"

# ─── Feature Lists ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "Application order",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ─── Risk Thresholds ──────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "high": 0.65,    # ≥ 65% → High Risk
    "medium": 0.35,  # 35–64% → Medium Risk
    # < 35% → Low Risk
}

RISK_COLORS = {
    "High": "#E24B4A",
    "Medium": "#BA7517",
    "Low": "#639922",
}

RISK_BG_COLORS = {
    "High": "#FEE2E2",
    "Medium": "#FEF3C7",
    "Low": "#DCFCE7",
}

TARGET_COLORS = {
    "Dropout": "#E24B4A",
    "Graduate": "#639922",
    "Enrolled": "#3B82F6",
}

# ─── Feature Display Names (UCI → DIU-Friendly) ───────────────────────────────
FEATURE_DISPLAY_NAMES = {
    "Marital status": "Marital Status",
    "Application mode": "Application Mode",
    "Application order": "Application Order",
    "Course": "Program Code",
    "Daytime/evening attendance": "Attendance Type",
    "Previous qualification": "Prior Qualification",
    "Nacionality": "Nationality",
    "Mother's qualification": "Mother's Education",
    "Father's qualification": "Father's Education",
    "Mother's occupation": "Mother's Occupation",
    "Father's occupation": "Father's Occupation",
    "Displaced": "Displaced Student",
    "Educational special needs": "Special Educational Needs",
    "Debtor": "Outstanding Debt",
    "Tuition fees up to date": "Tuition Fees Paid",
    "Gender": "Gender",
    "Scholarship holder": "Scholarship Holder",
    "Age at enrollment": "Age at Enrollment",
    "International": "International Student",
    "Curricular units 1st sem (credited)": "Sem 1 Units Credited",
    "Curricular units 1st sem (enrolled)": "Sem 1 Units Enrolled",
    "Curricular units 1st sem (evaluations)": "Sem 1 Units Evaluated",
    "Curricular units 1st sem (approved)": "Sem 1 Courses Passed",
    "Curricular units 1st sem (grade)": "Sem 1 Average Grade",
    "Curricular units 1st sem (without evaluations)": "Sem 1 Units w/o Eval",
    "Curricular units 2nd sem (credited)": "Sem 2 Units Credited",
    "Curricular units 2nd sem (enrolled)": "Sem 2 Units Enrolled",
    "Curricular units 2nd sem (evaluations)": "Sem 2 Units Evaluated",
    "Curricular units 2nd sem (approved)": "Sem 2 Courses Passed",
    "Curricular units 2nd sem (grade)": "Sem 2 Average Grade",
    "Curricular units 2nd sem (without evaluations)": "Sem 2 Units w/o Eval",
    "Unemployment rate": "Regional Unemployment (%)",
    "Inflation rate": "Regional Inflation (%)",
    "GDP": "Regional GDP",
}

# ─── What-If Simulator Features ───────────────────────────────────────────────
WHATIF_FEATURES = [
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Tuition fees up to date",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 1st sem (enrolled)",
    "Age at enrollment",
    "Debtor",
    "Scholarship holder",
    "Curricular units 2nd sem (evaluations)",
    "GDP",
]

WHATIF_CONFIG = {
    "Curricular units 2nd sem (approved)":  {"min": 0,    "max": 20,   "step": 1,   "type": "int"},
    "Curricular units 2nd sem (grade)":     {"min": 0.0,  "max": 20.0, "step": 0.5, "type": "float"},
    "Curricular units 1st sem (approved)":  {"min": 0,    "max": 20,   "step": 1,   "type": "int"},
    "Curricular units 1st sem (grade)":     {"min": 0.0,  "max": 20.0, "step": 0.5, "type": "float"},
    "Tuition fees up to date":              {"min": 0,    "max": 1,    "step": 1,   "type": "select", "options": [0, 1], "labels": ["No (Overdue)", "Yes (Paid)"]},
    "Curricular units 2nd sem (enrolled)":  {"min": 0,    "max": 20,   "step": 1,   "type": "int"},
    "Curricular units 1st sem (enrolled)":  {"min": 0,    "max": 20,   "step": 1,   "type": "int"},
    "Age at enrollment":                    {"min": 17,   "max": 70,   "step": 1,   "type": "int"},
    "Debtor":                               {"min": 0,    "max": 1,    "step": 1,   "type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "Scholarship holder":                   {"min": 0,    "max": 1,    "step": 1,   "type": "select", "options": [0, 1], "labels": ["No", "Yes"]},
    "Curricular units 2nd sem (evaluations)": {"min": 0,  "max": 33,   "step": 1,   "type": "int"},
    "GDP":                                  {"min": -4.0, "max": 4.0,  "step": 0.1, "type": "float"},
}
