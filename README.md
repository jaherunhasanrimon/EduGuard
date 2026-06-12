# EduGuard 🎓
### AI-Powered Student Dropout Risk Intelligence Platform
**DIU AI Project Competition 2026 · Department of CIS, Daffodil International University**

---

## Overview

EduGuard is a multi-page Streamlit web application that predicts student dropout risk using a trained Random Forest model and provides explainable AI (SHAP) explanations and rule-based intervention suggestions to academic advisors.

### Key Features
| Feature | Description |
|---|---|
| 📊 **Risk Overview Dashboard** | KPI cards, donut chart, histogram, top-10 at-risk preview |
| 🎓 **Student Risk Table** | Searchable/filterable table with progress-bar risk scores, CSV export |
| 🔍 **Student Deep Dive** | Per-student profile, interactive SHAP waterfall chart, intervention checklist |
| 🔮 **What-If Simulator** | Adjust feature sliders → live risk recalculation + before/after comparison |

---

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | **77.29%** |
| Macro F1 | **0.713** |
| Dropout F1 | **0.78** (Precision: 83%, Recall: 73%) |
| Graduate F1 | **0.86** |
| CV Macro F1 | 0.704 ± 0.015 (5-fold) |

**Model**: Random Forest (300 trees, balanced class weights)  
**Dataset**: UCI Student Dropout dataset — 4,424 records × 34 features

---

## Project Structure

```
EduGuard/
├── app/
│   ├── main.py                         # Entry point + Overview Dashboard
│   ├── auth.py                         # Password gate (shared across pages)
│   ├── components/
│   │   ├── risk_badge.py               # HTML risk badge + cause tag renderers
│   │   └── shap_plot.py                # Interactive Plotly SHAP waterfall
│   └── pages/
│       ├── 2_🎓_Student_List.py        # Filterable risk table
│       ├── 3_🔍_Student_Detail.py      # Per-student SHAP + interventions
│       └── 4_🔮_WhatIf.py             # What-If Simulator
├── ml/
│   ├── pipeline.py                     # sklearn Pipeline (preprocessor + RF)
│   ├── train.py                        # Training script → saves .pkl files
│   ├── predict.py                      # Batch + single inference
│   └── shap_explainer.py               # SHAP value extraction wrapper
├── engine/
│   └── intervention_rules.py           # Rule-based intervention engine
├── config/
│   └── settings.py                     # Column mappings, thresholds, constants
├── data/
│   └── dataset_2.csv                   # Bundled UCI demo dataset
├── models/                             # Auto-created by train.py (gitignored)
│   ├── pipeline.pkl
│   └── shap_explainer.pkl
├── assets/
│   ├── style.css                       # Global CSS
│   └── diu_logo.png                    # University emblem
├── .streamlit/
│   ├── config.toml                     # Theme config
│   ├── secrets.toml                    # Auth password (NOT committed)
│   └── secrets.toml.example            # Template
└── requirements.txt
```

---

## Setup & Running Locally

### Prerequisites
- Python 3.10+
- pip

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python ml/train.py
```
This will:
- Run 5-fold stratified cross-validation
- Train the final Random Forest pipeline
- Save `models/pipeline.pkl` and `models/shap_explainer.pkl`

### 3. Set the password
```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and set your password
```

### 4. Launch the app
```bash
streamlit run app/main.py
```
Open [http://localhost:8501](http://localhost:8501) and log in with the password from `secrets.toml`.  
**Demo password**: `eduguard2026`

---

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub (ensure `models/` **is** committed, or add a startup command to run `train.py`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path**: `app/main.py`
4. In **Advanced settings → Secrets**, paste:
   ```toml
   [auth]
   password = "your-secure-password"
   ```
5. Deploy — Streamlit Community Cloud provides a permanent public URL at no cost.

> **Note**: The `models/` directory must be present before the app starts. Either commit the `.pkl` files or add `python ml/train.py` as a pre-run command.

---

## Adapting for Real DIU Data

EduGuard is designed for re-deployment with minimal changes:

| Step | Action |
|---|---|
| 1 | Prepare a CSV with equivalent columns (see `config/settings.py` for the full column list) |
| 2 | Update `FEATURE_DISPLAY_NAMES` in `config/settings.py` to match your institution's terminology |
| 3 | Adjust `RISK_THRESHOLDS` if your institution's dropout rate differs significantly |
| 4 | Re-run `python ml/train.py` with the new dataset |
| 5 | Deploy — no code changes required |

---

## Ethical Considerations

- **No PII stored**: uploaded CSVs are processed in-memory and never persisted to disk
- **Human-in-the-loop**: EduGuard generates suggestions for advisors; it does not automate decisions
- **Explainability**: every risk score is accompanied by a SHAP explanation — the model is auditable
- **Fairness**: model performance metrics should be evaluated separately across demographic subgroups before production deployment

---

## References

- Martins et al. (2021). *Early prediction of student's performance in higher education.* Springer.
- [UCI ML Repository — Student Dropout Dataset](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
- Lundberg & Lee (2017). *A unified approach to interpreting model predictions.* NeurIPS.
- [SHAP Documentation](https://shap.readthedocs.io)
- [scikit-learn](https://scikit-learn.org) | [Streamlit](https://docs.streamlit.io) | [Plotly](https://plotly.com/python/)

---

*EduGuard · DIU AI Project Competition 2026 · Department of CIS, Daffodil International University*
