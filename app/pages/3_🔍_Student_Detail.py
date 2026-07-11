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

def get_lucide_svg(emoji_icon):
    svgs = {
        "🚨": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-alert-triangle"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/></svg>',
        "📚": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-open"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/></svg>',
        "💰": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-credit-card"><rect width="20" height="14" x="2" y="5" rx="2"/><line x1="2" x2="22" y1="10" y2="10"/></svg>',
        "🏦": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-landmark"><line x1="3" x2="21" y1="22" y2="22"/><line x1="6" x2="6" y1="18" y2="11"/><line x1="10" x2="10" y1="18" y2="11"/><line x1="14" x2="14" y1="18" y2="11"/><line x1="18" x2="18" y1="18" y2="11"/><path d="m12 2-8 5h16Z"/></svg>',
        "📝": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-pencil"><path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/></svg>',
        "🎓": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-graduation-cap"><path d="M21.42 10.922a1 1 0 0 0-.019-1.838L12.83 5.18a2 2 0 0 0-1.66 0L2.6 9.08a1 1 0 0 0 0 1.832l8.57 3.908a2 2 0 0 0 1.66 0z"/><path d="M6 12v5c0 2 2 3 6 3s6-1 6-3v-5"/><path d="M21.5 12v6"/></svg>',
        "⚖️": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-scale"><path d="m16 16 3-8 3 8c-.87.65-2.24 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-2.24 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/></svg>',
        "♿": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-accessibility"><circle cx="16" cy="4" r="1"/><path d="m18 19 1-7-6 1"/><path d="m5 8 3-3 5.5 3-2.36 3.5"/><path d="M4 24h4v-7h4v-4H4z"/></svg>',
        "🌍": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-globe"><circle cx="12" cy="12" r="10"/><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"/><path d="M2 12h20"/></svg>',
        "⚠️": '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-alert-circle"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>',
    }
    return svgs.get(emoji_icon, svgs["⚠️"])

if not interventions:
    st.info("No specific intervention rules triggered for this student.")
else:
    priority_colors = {"HIGH": "eg-iv-high", "MEDIUM": "eg-iv-medium", "LOW": "eg-iv-low"}
    priority_labels = {"HIGH": "Critical", "MEDIUM": "Medium", "LOW": "Low"}
    priority_pills = {"HIGH": "iv-badge-high", "MEDIUM": "iv-badge-medium", "LOW": "iv-badge-low"}

    for iv in interventions:
        p_class = priority_colors.get(iv["priority"], "eg-iv-medium")
        p_label = priority_labels.get(iv["priority"], iv["priority"])
        p_badge = priority_pills.get(iv["priority"], "iv-badge-medium")
        svg_icon = get_lucide_svg(iv['icon'])
        
        st.markdown(
            f"""
            <div class="intervention-card-redesign {p_class}">
              <div class="intervention-header-redesign">
                <div class="intervention-icon-redesign">{svg_icon}</div>
                <div class="intervention-title-redesign">{iv['title']}</div>
                <div class="intervention-badge-redesign {p_badge}">{p_label}</div>
              </div>
              <div class="intervention-body-redesign">
                <p class="intervention-text-redesign">{iv['action']}</p>
                <div class="intervention-trigger-redesign">Triggered by: {iv['trigger']}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Notice banner
shield_check_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#166534" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-shield-check"><path d="M20 13c0 5-3.5 7.5-7.66 9.7a1 1 0 0 1-.68 0C7.5 20.5 4 18 4 13V6a1 1 0 0 1 .76-.97l8-2a1 1 0 0 1 .48 0l8 2A1 1 0 0 1 20 6z"/><path d="m9 12 2 2 4-4"/></svg>'

st.markdown(
    f"""
    <div class="notice-banner-redesign">
      <div class="notice-icon-redesign">{shield_check_svg}</div>
      <div class="notice-body-redesign">
        <strong>Human-in-the-Loop Validation Required:</strong> EduGuard suggestions are advisory only. 
        All academic and financial interventions must be authorized by a qualified advisor before action.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

