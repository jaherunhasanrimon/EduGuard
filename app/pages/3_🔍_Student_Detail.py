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

# ─── Load model ──────────────────────────────────────────────────────────────
from config.settings import MODELS_DIR

def _pipeline_mtime():
    p = MODELS_DIR / "pipeline.pkl"
    return int(p.stat().st_mtime) if p.exists() else 0

def _explainer_mtime():
    p = MODELS_DIR / "shap_explainer.pkl"
    return int(p.stat().st_mtime) if p.exists() else 0

@st.cache_resource(show_spinner="Loading model…")
def get_pipeline(_mtime):
    return load_pipeline()

@st.cache_resource(show_spinner="Loading SHAP explainer…")
def get_explainer(_mtime):
    return load_explainer()

pipeline = get_pipeline(_pipeline_mtime())
explainer = get_explainer(_explainer_mtime())

# ─── Sidebar & Data Load ──────────────────────────────────────────────────────
from app.components.sidebar import render_sidebar

# Render standard sidebar without custom slots
render_sidebar("student_details")
df_pred = st.session_state["df_pred"]
max_id = len(df_pred)

# Default active student index
if "detail_student_idx" not in st.session_state:
    st.session_state["detail_student_idx"] = 0

# Deterministic student name generator
def get_student_name(student_id):
    first_names = ["John", "Jane", "Rahim", "Karim", "Amina", "Sajid", "Emily", "Michael", "Sarah", "David", "Fatima", "Tanvir", "Jessica", "Daniel", "Sofia", "Ahmed"]
    last_names = ["Doe", "Smith", "Hasan", "Ali", "Chowdhury", "Khan", "Johnson", "Brown", "Miller", "Davis", "Begum", "Rahman", "Wilson", "Taylor", "Gomez", "Islam"]
    f_idx = (int(student_id) * 7) % len(first_names)
    l_idx = (int(student_id) * 13) % len(last_names)
    return f"{first_names[f_idx]} {last_names[l_idx]}"

# Initialize search query in session state
if "search_input_val" not in st.session_state:
    st.session_state["search_input_val"] = ""

# ─── Search Toolbar UI ────────────────────────────────────────────────────────
st.markdown('<div class="search-toolbar-wrapper">', unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([2.5, 4.5, 2])

with col_left:
    st.markdown(
        """
        <div class="search-title-box">
          <div class="search-title-text">Search Student</div>
          <div class="search-subtitle-text">Search by Student ID or Name</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_center:
    search_query = st.text_input(
        "Search Input",
        value=st.session_state["search_input_val"],
        placeholder="Enter Student ID or Name…",
        label_visibility="collapsed",
        key="student_search_input"
    )

with col_right:
    c_search, c_clear = st.columns(2)
    with c_search:
        search_clicked = st.button("Search", type="primary", use_container_width=True)
    with c_clear:
        clear_clicked = st.button("Clear", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Search Logic Execution
search_triggered = search_clicked or (search_query and search_query != st.session_state.get("last_search_query"))

if search_triggered:
    st.session_state["last_search_query"] = search_query
    st.session_state["search_input_val"] = search_query
    
    if search_query.strip():
        with st.spinner("Searching cohort..."):
            query = search_query.strip().lower()
            
            # Check if search by ID
            if query.isdigit():
                target_id = int(query)
                if 1 <= target_id <= max_id:
                    st.session_state["detail_student_idx"] = target_id - 1
                    st.toast(f"✅ Loaded Student ID {target_id}", icon="🎉")
                    st.session_state.pop("search_matches", None)
                    st.rerun()
                else:
                    st.error(f"❌ Invalid Student ID (must be between 1 and {max_id})")
                    st.session_state.pop("search_matches", None)
            else:
                # Search by name
                matches = []
                for i in range(max_id):
                    s_id = int(df_pred.iloc[i]["student_id"])
                    s_name = get_student_name(s_id)
                    if query in s_name.lower():
                        matches.append((i, s_id, s_name))
                        
                if len(matches) == 1:
                    st.session_state["detail_student_idx"] = matches[0][0]
                    st.toast(f"✅ Loaded {matches[0][2]} (ID: {matches[0][1]})", icon="🎉")
                    st.session_state.pop("search_matches", None)
                    st.rerun()
                elif len(matches) > 1:
                    st.session_state["search_matches"] = matches[:8] # limit suggestions to 8
                    st.toast(f"ℹ️ Found {len(matches)} matching students. Select from suggestions below.", icon="🔍")
                else:
                    st.error(f"❌ No student found matching '{search_query}'")
                    st.session_state.pop("search_matches", None)

if clear_clicked:
    st.session_state["search_input_val"] = ""
    st.session_state.pop("last_search_query", None)
    st.session_state.pop("search_matches", None)
    st.rerun()

# Suggestions autocomplete list
if "search_matches" in st.session_state and st.session_state["search_matches"]:
    st.markdown("<div class='suggestions-label'>Matching Students:</div>", unsafe_allow_html=True)
    matches = st.session_state["search_matches"]
    
    # Render suggestion badges
    cols = st.columns(min(len(matches), 4))
    for idx, match in enumerate(matches[:4]):
        col_cell = cols[idx % 4]
        with col_cell:
            if st.button(f"👤 {match[2]} (ID: {match[1]})", key=f"suggest_{match[1]}_{idx}", use_container_width=True):
                st.session_state["detail_student_idx"] = match[0]
                st.session_state["search_input_val"] = f"{match[2]} (ID: {match[1]})"
                st.session_state.pop("search_matches", None)
                st.rerun()

# Get currently active student details
student_idx = st.session_state["detail_student_idx"]
student_row = df_pred.iloc[student_idx]
dropout_prob = float(student_row["dropout_prob"])
risk_tier    = student_row["risk_tier"]
prediction   = student_row["predicted_label"]

# Generate student name for display
student_name = get_student_name(student_row["student_id"])


# ─── Student Summary Card ─────────────────────────────────────────────────────
risk_color = RISK_COLORS.get(risk_tier, "#64748B")
st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:18px;padding:20px;
                background:#FFFFFF;border-radius:12px;border:1px solid #E2E8F0;
                box-shadow:0 2px 10px rgba(0,0,0,0.05); margin-bottom:20px;">
      <div style="font-size:3.2rem;font-weight:900;color:{risk_color};line-height:1;">{dropout_prob:.1%}</div>
      <div>
        <div style="font-size:0.8rem;color:#64748B;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">Dropout Probability</div>
        <div style="margin-top: 6px;">
          {risk_badge_html(risk_tier)}
          &nbsp;&nbsp;
          {target_badge_html(prediction)}
        </div>
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
        ("Full Name", student_name),
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
