"""
EduGuard — Page 4: Analytics View (Mockup/Placeholder)
Provides institutional risk aggregation analytics and cohort comparisons.
"""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="EduGuard — Analytics",
    page_icon="🎓",
    layout="wide",
)

from app.auth import require_auth
from config.settings import ASSETS_DIR, COMPETITION_LABEL, UNIVERSITY_NAME
from app.components.sidebar import render_sidebar

if not require_auth():
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar("analytics")

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
        <h1>📊 Cohort Risk Analytics</h1>
        <p>{UNIVERSITY_NAME} · {COMPETITION_LABEL}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Page Content ─────────────────────────────────────────────────────────────
df_pred = st.session_state.get("df_pred")
if df_pred is None:
    st.warning("⚠️ Data not loaded. Please return to the **Dashboard** first.")
    st.stop()

import plotly.express as px
import plotly.graph_objects as go

st.markdown("### 📈 Cohort Segmentation Analysis")

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="eg-card">', unsafe_allow_html=True)
    st.markdown('<div class="eg-card-title">Age Demographic Risk Trend</div>', unsafe_allow_html=True)
    
    if "Age at enrollment" in df_pred.columns:
        fig_age = px.histogram(
            df_pred,
            x="Age at enrollment",
            color="risk_tier",
            color_discrete_map={"High": "#E24B4A", "Medium": "#BA7517", "Low": "#639922"},
            barmode="stack",
            height=280,
        )
        fig_age.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(title="Age at Enrollment", gridcolor="#F1F5F9"),
            yaxis=dict(title="Student Count", gridcolor="#F1F5F9"),
            legend=dict(title="Risk Tier")
        )
        st.plotly_chart(fig_age, use_container_width=True)
    else:
        st.info("Age data not found in current dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="eg-card">', unsafe_allow_html=True)
    st.markdown('<div class="eg-card-title">Academic Performance vs. Dropout Risk</div>', unsafe_allow_html=True)
    
    x_col = "Curricular units 1st sem (grade)"
    y_col = "Curricular units 2nd sem (grade)"
    if x_col in df_pred.columns and y_col in df_pred.columns:
        fig_scatter = px.scatter(
            df_pred,
            x=x_col,
            y=y_col,
            color="risk_tier",
            color_discrete_map={"High": "#E24B4A", "Medium": "#BA7517", "Low": "#639922"},
            size="dropout_prob",
            hover_data=["student_id"],
            height=280,
        )
        fig_scatter.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            xaxis=dict(title="Sem 1 Grade", gridcolor="#F1F5F9"),
            yaxis=dict(title="Sem 2 Grade", gridcolor="#F1F5F9"),
            legend=dict(title="Risk Tier")
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Grade details not found in current dataset.")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="eg-card">', unsafe_allow_html=True)
st.markdown('<div class="eg-card-title">💡 Institutional Intervention Impact Mockup</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="padding: 20px; text-align: center; color: #64748B;">
      <p style="font-size: 1.1rem; font-weight: 600; color: #1E293B;">Intervention Analytics coming in v1.1.0</p>
      <p style="font-size: 0.85rem; max-width: 600px; margin: 8px auto;">
        This module will track advisor actions, student contact history, and compare risk scores 
        before and after student counseling sessions to evaluate institutional retention KPIs.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)
