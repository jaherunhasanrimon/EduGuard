"""
EduGuard — Page 2: Individual Student Deep Dive
Per-student profile, interactive SHAP waterfall chart, and intervention checklist.
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="EduGuard — Student Detail",
    page_icon="🎓",
    layout="wide",
)

from app.auth import require_auth
from app.components.risk_badge import risk_badge_html, target_badge_html
from app.components.shap_plot import create_shap_waterfall
from config.settings import (
    ASSETS_DIR, COMPETITION_LABEL, FEATURE_DISPLAY_NAMES,
    RISK_COLORS, UNIVERSITY_NAME,
)
from engine.intervention_rules import get_cause_tags, get_interventions
from ml.predict import load_explainer, load_pipeline
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
        <h1>🔍 Student Deep Dive</h1>
        <p>{UNIVERSITY_NAME} · {COMPETITION_LABEL}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Load data & model ────────────────────────────────────────────────────────
df_pred = st.session_state.get("df_pred")
if df_pred is None:
    st.warning("⚠️ Data not loaded. Please return to the **Overview** page first.")
    st.stop()

@st.cache_resource(show_spinner="Loading model…")
def get_pipeline():
    return load_pipeline()

@st.cache_resource(show_spinner="Loading SHAP explainer…")
def get_explainer():
    return load_explainer()

pipeline = get_pipeline()
explainer = get_explainer()

# ─── Student Selector ────────────────────────────────────────────────────────
max_id = len(df_pred)
default_idx = st.session_state.get("detail_student_idx", 0)
default_idx = max(0, min(default_idx, max_id - 1))

col_sel, col_info = st.columns([1, 3])
with col_sel:
    student_num = st.number_input(
        "Select Student ID",
        min_value=1,
        max_value=max_id,
        value=int(default_idx + 1),
        step=1,
        key="detail_student_id",
    )
    student_idx = student_num - 1
    st.session_state["detail_student_idx"] = student_idx

student_row = df_pred.iloc[student_idx]
dropout_prob = float(student_row["dropout_prob"])
risk_tier    = student_row["risk_tier"]
prediction   = student_row["predicted_label"]

with col_info:
    risk_color = RISK_COLORS.get(risk_tier, "#64748B")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:16px;padding:16px;
                    background:#FFFFFF;border-radius:12px;border:1px solid #E2E8F0;
                    box-shadow:0 2px 10px rgba(0,0,0,0.05);">
          <div style="font-size:3rem;font-weight:900;color:{risk_color};">{dropout_prob:.1%}</div>
          <div>
            <div style="font-size:0.8rem;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Dropout Probability</div>
            {risk_badge_html(risk_tier)}
            &nbsp;&nbsp;
            {target_badge_html(prediction)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─── Risk Meter ───────────────────────────────────────────────────────────────
fill_class = f"eg-risk-fill-{risk_tier.lower()}"
st.markdown(
    f"""
    <div class="eg-risk-meter" style="margin:12px 0;">
      <div class="eg-risk-meter-bar">
        <div class="eg-risk-meter-fill {fill_class}" style="width:{dropout_prob*100:.1f}%;"></div>
      </div>
      <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#94A3B8;margin-top:4px;">
        <span>0% — Low Risk</span><span>35% — Medium</span><span>65% — High Risk</span><span>100%</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ─── Two-column layout: Profile | SHAP ───────────────────────────────────────
left_col, right_col = st.columns([1, 2])

with left_col:
    # Student Profile
    st.markdown('<div class="eg-card">', unsafe_allow_html=True)
    st.markdown('<div class="eg-card-title">👤 Student Profile</div>', unsafe_allow_html=True)

    profile_fields = [
        ("Age at Enrollment", student_row.get("Age at enrollment", "—")),
        ("Gender", "Male" if student_row.get("Gender", 0) == 1 else "Female"),
        ("Scholarship", "Yes" if student_row.get("Scholarship holder", 0) == 1 else "No"),
        ("Fees Paid", "Yes" if student_row.get("Tuition fees up to date", 1) == 1 else "⚠️ No"),
        ("Debtor", "Yes ⚠️" if student_row.get("Debtor", 0) == 1 else "No"),
        ("International", "Yes" if student_row.get("International", 0) == 1 else "No"),
        ("Sem 1 Passed", f"{int(student_row.get('Curricular units 1st sem (approved)', 0))} units"),
        ("Sem 2 Passed", f"{int(student_row.get('Curricular units 2nd sem (approved)', 0))} units"),
        ("Sem 1 Grade", f"{student_row.get('Curricular units 1st sem (grade)', 0):.1f} / 20"),
        ("Sem 2 Grade", f"{student_row.get('Curricular units 2nd sem (grade)', 0):.1f} / 20"),
    ]

    html_fields = ""
    for label, value in profile_fields:
        html_fields += (
            f'<div class="eg-profile-field">'
            f'<div class="eg-profile-field-label">{label}</div>'
            f'<div class="eg-profile-field-value">{value}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="eg-profile-grid">{html_fields}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)



with right_col:
    # SHAP Chart
    st.markdown('<div class="eg-card">', unsafe_allow_html=True)
    st.markdown('<div class="eg-card-title">🧠 SHAP Risk Factor Analysis</div>', unsafe_allow_html=True)
    st.caption("Features in red **increase** dropout risk. Features in green **decrease** it.")

    try:
        X_df = df_pred.drop(columns=["dropout_prob", "risk_tier", "predicted_label", "student_id"], errors="ignore")
        with st.spinner("Computing SHAP values…"):
            shap_data = get_shap_values_for_student(student_idx, X_df, pipeline, explainer)
        top_factors = get_top_factors(shap_data, n=5)

        shap_fig = create_shap_waterfall(shap_data, dropout_prob, n_display=12)
        st.plotly_chart(shap_fig, use_container_width=True)
    except Exception as e:
        st.error(f"SHAP computation failed: {e}")
        top_factors = []

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ─── Intervention Checklist ───────────────────────────────────────────────────
st.markdown('<div class="eg-section-title">📋 Advisor Intervention Checklist</div>', unsafe_allow_html=True)
st.caption(
    "The following actions are generated by EduGuard's rule engine based on this student's "
    "risk profile. All interventions require human advisor approval before action."
)

student_dict = student_row.to_dict()
interventions = get_interventions(student_dict, top_factors)

if not interventions:
    st.info("No specific intervention rules triggered for this student.")
else:
    priority_colors = {"HIGH": "eg-intervention-high", "MEDIUM": "eg-intervention-medium", "LOW": "eg-intervention-low"}
    priority_labels = {"HIGH": "🔴 HIGH PRIORITY", "MEDIUM": "🟡 MEDIUM PRIORITY", "LOW": "🟢 LOW PRIORITY"}

    for iv in interventions:
        p_class = priority_colors.get(iv["priority"], "eg-intervention-medium")
        p_label = priority_labels.get(iv["priority"], iv["priority"])
        st.markdown(
            f"""
            <div class="eg-intervention {p_class}">
              <div class="eg-intervention-icon">{iv['icon']}</div>
              <div class="eg-intervention-body">
                <div class="eg-intervention-title">{iv['title']}</div>
                <div style="font-size:0.7rem;font-weight:600;color:#94A3B8;margin-bottom:4px;">{p_label}</div>
                <div class="eg-intervention-text">{iv['action']}</div>
                <div class="eg-intervention-trigger">Trigger: {iv['trigger']}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    "<div style='margin-top:16px;padding:12px 16px;background:#F0FDF4;border-radius:10px;"
    "border-left:4px solid #639922;font-size:0.8rem;color:#166534;'>"
    "⚠️ <strong>Human-in-the-Loop:</strong> EduGuard generates suggestions only. "
    "Every intervention must be reviewed and approved by a qualified academic advisor "
    "before being communicated to the student."
    "</div>",
    unsafe_allow_html=True,
)
