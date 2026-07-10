"""
EduGuard — Page 1: Student Risk Table
Searchable, filterable risk table with badges and cause tags.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="EduGuard — Student List",
    page_icon="🎓",
    layout="wide",
)

from app.auth import require_auth
from app.components.risk_badge import risk_badge_html, cause_tag_html, target_badge_html
from config.settings import ASSETS_DIR, COMPETITION_LABEL, UNIVERSITY_NAME
from engine.intervention_rules import get_cause_tags
from ml.predict import load_pipeline, load_demo_data, predict_batch
from ml.shap_explainer import get_shap_values_for_student, get_top_factors

if not require_auth():
    st.stop()

# ─── Page Header ─────────────────────────────────────────────────────────────
import base64
logo_path = ASSETS_DIR / "diu_logo.svg"
logo_html = ""
if logo_path.exists():
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" style="width:50px;height:50px;object-fit:contain;">'

st.markdown(
    f"""
    <div class="eg-page-header">
      {logo_html}
      <div class="eg-page-header-text">
        <h1>🎓 Student Risk Table</h1>
        <p>{UNIVERSITY_NAME} · {COMPETITION_LABEL}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Load data ────────────────────────────────────────────────────────────────
df_pred = st.session_state.get("df_pred")
if df_pred is None:
    st.warning("⚠️ Data not loaded. Please return to the **Overview** page first.")
    st.stop()

# ─── Sidebar Filters ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔽 Filters")
    filter_tier = st.multiselect(
        "Risk Tier",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
    filter_outcome = st.multiselect(
        "Predicted Outcome",
        options=sorted(df_pred["predicted_label"].unique().tolist()),
        default=sorted(df_pred["predicted_label"].unique().tolist()),
    )
    filter_gender = st.multiselect(
        "Gender (0=F, 1=M)",
        options=sorted(df_pred["Gender"].dropna().unique().tolist()) if "Gender" in df_pred.columns else [],
        default=sorted(df_pred["Gender"].dropna().unique().tolist()) if "Gender" in df_pred.columns else [],
    )
    filter_scholarship = st.multiselect(
        "Scholarship Holder",
        options=[0, 1],
        default=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
    )
    search_id = st.number_input("Jump to Student ID", min_value=1, max_value=len(df_pred), value=1, step=1)

    st.markdown("---")
    if st.button("🔍 View Student Detail", use_container_width=True):
        st.session_state["detail_student_idx"] = int(search_id) - 1
        st.switch_page("pages/3_🔍_Student_Detail.py")

# ─── Apply Filters ────────────────────────────────────────────────────────────
filtered = df_pred.copy()
if filter_tier:
    filtered = filtered[filtered["risk_tier"].isin(filter_tier)]
if filter_outcome:
    filtered = filtered[filtered["predicted_label"].isin(filter_outcome)]
if "Gender" in filtered.columns and filter_gender:
    filtered = filtered[filtered["Gender"].isin(filter_gender)]
if "Scholarship holder" in filtered.columns and filter_scholarship is not None:
    filtered = filtered[filtered["Scholarship holder"].isin(filter_scholarship)]

# ─── Summary Bar ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Showing", f"{len(filtered):,} students")
c2.metric("🔴 High Risk", (filtered["risk_tier"] == "High").sum())
c3.metric("🟡 Medium Risk", (filtered["risk_tier"] == "Medium").sum())
c4.metric("🟢 Low Risk", (filtered["risk_tier"] == "Low").sum())

st.markdown("<br>", unsafe_allow_html=True)

# ─── Build display table ──────────────────────────────────────────────────────
DISPLAY_COLS = [
    "student_id",
    "risk_tier",
    "dropout_prob",
    "predicted_label",
]
if "Curricular units 2nd sem (approved)" in filtered.columns:
    DISPLAY_COLS += ["Curricular units 2nd sem (approved)", "Curricular units 2nd sem (grade)"]
if "Tuition fees up to date" in filtered.columns:
    DISPLAY_COLS += ["Tuition fees up to date"]
if "Scholarship holder" in filtered.columns:
    DISPLAY_COLS += ["Scholarship holder"]

display_df = filtered[DISPLAY_COLS].copy()
display_df = display_df.rename(columns={
    "student_id": "ID",
    "risk_tier": "Risk",
    "dropout_prob": "Dropout %",
    "predicted_label": "Outcome",
    "Curricular units 2nd sem (approved)": "Sem2 Passed",
    "Curricular units 2nd sem (grade)": "Sem2 Grade",
    "Tuition fees up to date": "Fees Paid",
    "Scholarship holder": "Scholarship",
})
display_df["Dropout %"] = (display_df["Dropout %"] * 100).round(1)

# ─── Render table (st.dataframe with column config) ───────────────────────────
st.dataframe(
    display_df.reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    column_config={
        "ID": st.column_config.NumberColumn("Student ID", width="small"),
        "Risk": st.column_config.TextColumn("Risk Tier", width="small"),
        "Dropout %": st.column_config.ProgressColumn(
            "Dropout Risk %",
            format="%.1f%%",
            min_value=0,
            max_value=100,
            width="medium",
        ),
        "Outcome": st.column_config.TextColumn("Predicted Outcome"),
        "Sem2 Passed": st.column_config.NumberColumn("Sem2 Passed", format="%d", width="small"),
        "Sem2 Grade": st.column_config.NumberColumn("Sem2 Grade", format="%.1f", width="small"),
        "Fees Paid": st.column_config.CheckboxColumn("Fees Paid?", width="small"),
        "Scholarship": st.column_config.CheckboxColumn("Scholarship?", width="small"),
    },
)

# ─── Export ───────────────────────────────────────────────────────────────────
csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️  Export filtered data as CSV",
    data=csv,
    file_name="eduguard_risk_export.csv",
    mime="text/csv",
)

st.caption(
    f"Showing {len(filtered):,} of {len(df_pred):,} students. "
    f"Use the **Student Detail** page for per-student SHAP explanations and intervention plans."
)
