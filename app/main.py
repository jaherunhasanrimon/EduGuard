"""
EduGuard — Main Entry Point & Overview Dashboard (Page 1)
Run with:  streamlit run app/main.py
"""
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ─── Page Config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="EduGuard — DIU Student Risk Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "EduGuard MVP · DIU AI Project Competition 2026 · Dept. of CIS",
    },
)

from app.auth import require_auth, logout
from config.settings import (
    APP_TITLE, APP_SUBTITLE, UNIVERSITY_NAME, ASSETS_DIR,
    RISK_COLORS, TARGET_COLORS, COMPETITION_LABEL,
)
from ml.predict import load_pipeline, load_demo_data, load_uploaded_data, predict_batch

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ─── Auth Gate ────────────────────────────────────────────────────────────────
if not require_auth():
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = ASSETS_DIR / "diu_logo.svg"
    col1, col2 = st.columns([1, 3])
    with col1:
        if logo_path.exists():
            st.image(str(logo_path), width=40)
    with col2:
        st.markdown(
            f"<div style='font-weight:800;font-size:1rem;color:#1E293B;line-height:1.2;'>"
            f"{APP_TITLE}</div>"
            f"<div style='font-size:0.7rem;color:#64748B;'>{COMPETITION_LABEL}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**📂 Data Source**")

    source = st.radio(
        "Choose dataset",
        ["Demo dataset (UCI)", "Upload CSV"],
        key="data_source",
        label_visibility="collapsed",
    )

    uploaded_file = None
    if source == "Upload CSV":
        uploaded_file = st.file_uploader(
            "Upload student CSV",
            type=["csv"],
            help="CSV must include the same columns as the UCI student dataset.",
        )

    st.markdown("---")

    if st.button("🚪  Sign Out", use_container_width=True):
        logout()

# ─── Load data & run predictions ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_pipeline():
    return load_pipeline()

@st.cache_data(show_spinner="Running predictions…")
def get_predictions(source_key: str, file_bytes=None):
    pipeline = get_pipeline()
    if source_key == "upload" and file_bytes is not None:
        import io
        df = load_uploaded_data(io.BytesIO(file_bytes))
    else:
        df = load_demo_data()
    return predict_batch(df, pipeline)

# Decide data key
if source == "Upload CSV" and uploaded_file is not None:
    file_bytes = uploaded_file.read()
    df_pred = get_predictions("upload", file_bytes)
else:
    df_pred = get_predictions("demo")

st.session_state["df_pred"] = df_pred

# ─── Page Header ─────────────────────────────────────────────────────────────
logo_html = ""
if logo_path.exists():
    import base64
    with open(logo_path, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" style="width:50px;height:50px;object-fit:contain;">'

st.markdown(
    f"""
    <div class="eg-page-header">
      {logo_html}
      <div class="eg-page-header-text">
        <h1>📊 Risk Overview Dashboard</h1>
        <p>{UNIVERSITY_NAME} · {COMPETITION_LABEL}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── KPI Cards ────────────────────────────────────────────────────────────────
total    = len(df_pred)
n_high   = (df_pred["risk_tier"] == "High").sum()
n_medium = (df_pred["risk_tier"] == "Medium").sum()
n_low    = (df_pred["risk_tier"] == "Low").sum()
avg_prob = df_pred["dropout_prob"].mean()

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f'<div class="eg-kpi-card eg-kpi-total">'
        f'<div class="eg-kpi-value">{total:,}</div>'
        f'<div class="eg-kpi-label">Total Students</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c2:
    pct_high = n_high / total * 100
    st.markdown(
        f'<div class="eg-kpi-card eg-kpi-high">'
        f'<div class="eg-kpi-value">{n_high:,}</div>'
        f'<div class="eg-kpi-label">🔴 High Risk</div>'
        f'<div class="eg-kpi-delta eg-delta-up">{pct_high:.1f}% of cohort</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c3:
    pct_med = n_medium / total * 100
    st.markdown(
        f'<div class="eg-kpi-card eg-kpi-medium">'
        f'<div class="eg-kpi-value">{n_medium:,}</div>'
        f'<div class="eg-kpi-label">🟡 Medium Risk</div>'
        f'<div class="eg-kpi-delta" style="color:#BA7517;">{pct_med:.1f}% of cohort</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c4:
    pct_low = n_low / total * 100
    st.markdown(
        f'<div class="eg-kpi-card eg-kpi-low">'
        f'<div class="eg-kpi-value">{n_low:,}</div>'
        f'<div class="eg-kpi-label">🟢 Low Risk</div>'
        f'<div class="eg-kpi-delta eg-delta-down">{pct_low:.1f}% of cohort</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        f'<div class="eg-kpi-card">'
        f'<div class="eg-kpi-value" style="color:#4F46E5;">{avg_prob:.1%}</div>'
        f'<div class="eg-kpi-label">Avg Dropout Risk</div>'
        f'<div class="eg-kpi-delta" style="color:#64748B;">Across all students</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─── Charts Row ───────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 2])

with col_left:
    st.markdown('<div class="eg-card">', unsafe_allow_html=True)
    st.markdown('<div class="eg-card-title">📊 Risk Tier Distribution</div>', unsafe_allow_html=True)

    donut_fig = go.Figure(
        go.Pie(
            labels=["High Risk", "Medium Risk", "Low Risk"],
            values=[n_high, n_medium, n_low],
            hole=0.62,
            marker=dict(
                colors=["#E24B4A", "#BA7517", "#639922"],
                line=dict(color="#FFFFFF", width=3),
            ),
            textinfo="label+percent",
            textfont=dict(size=12, family="Inter, sans-serif"),
            hovertemplate="<b>%{label}</b><br>Students: %{value:,}<br>%{percent}<extra></extra>",
        )
    )
    donut_fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:10px'>Students</span>",
        x=0.5, y=0.5,
        font=dict(size=18, color="#1E293B", family="Inter, sans-serif"),
        showarrow=False,
    )
    donut_fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    st.plotly_chart(donut_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="eg-card">', unsafe_allow_html=True)
    st.markdown('<div class="eg-card-title">📈 Predicted Outcome Distribution</div>', unsafe_allow_html=True)

    outcome_counts = df_pred["predicted_label"].value_counts().reset_index()
    outcome_counts.columns = ["Outcome", "Count"]

    bar_fig = px.bar(
        outcome_counts,
        x="Outcome",
        y="Count",
        color="Outcome",
        color_discrete_map=TARGET_COLORS,
        text="Count",
    )
    bar_fig.update_traces(
        texttemplate="%{text:,}",
        textposition="outside",
        marker_line_width=0,
    )
    bar_fig.update_layout(
        showlegend=False,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=10, r=10, t=10, b=10),
        height=260,
        xaxis=dict(title="", gridcolor="#F1F5F9"),
        yaxis=dict(title="Number of Students", gridcolor="#F1F5F9"),
        font=dict(family="Inter, sans-serif", size=12),
    )
    st.plotly_chart(bar_fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ─── Dropout Probability Distribution ────────────────────────────────────────
st.markdown('<div class="eg-card">', unsafe_allow_html=True)
st.markdown('<div class="eg-card-title">📉 Dropout Risk Score Distribution</div>', unsafe_allow_html=True)

hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(
    x=df_pred["dropout_prob"],
    nbinsx=40,
    name="All Students",
    marker=dict(color="#4F46E5", opacity=0.7, line=dict(color="#3730A3", width=0.5)),
    hovertemplate="Risk %{x:.0%}–%{x:.0%}: %{y} students<extra></extra>",
))
# Threshold lines
hist_fig.add_vline(x=0.35, line_dash="dash", line_color="#BA7517",
                   annotation_text="Medium threshold (35%)",
                   annotation_position="top right",
                   annotation_font=dict(size=10, color="#BA7517"))
hist_fig.add_vline(x=0.65, line_dash="dash", line_color="#E24B4A",
                   annotation_text="High threshold (65%)",
                   annotation_position="top right",
                   annotation_font=dict(size=10, color="#E24B4A"))
hist_fig.update_layout(
    showlegend=False,
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(l=10, r=10, t=10, b=30),
    height=220,
    xaxis=dict(title="Dropout Probability", tickformat=".0%", gridcolor="#F1F5F9"),
    yaxis=dict(title="Number of Students", gridcolor="#F1F5F9"),
    font=dict(family="Inter, sans-serif", size=12),
    bargap=0.05,
)
st.plotly_chart(hist_fig, use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# ─── Top 10 At-Risk Students Preview ─────────────────────────────────────────
st.markdown('<div class="eg-card">', unsafe_allow_html=True)
st.markdown('<div class="eg-card-title">🚨 Top 10 Highest-Risk Students</div>', unsafe_allow_html=True)

top10 = (
    df_pred[df_pred["risk_tier"] == "High"]
    .nlargest(10, "dropout_prob")[["student_id", "dropout_prob", "risk_tier", "predicted_label"]]
    .reset_index(drop=True)
)

if top10.empty:
    st.info("No high-risk students in current dataset.")
else:
    top10_display = top10.copy()
    top10_display["Dropout Risk"] = top10_display["dropout_prob"].apply(lambda p: f"{p:.1%}")
    top10_display["Risk Tier"] = top10_display["risk_tier"]
    top10_display["Predicted Outcome"] = top10_display["predicted_label"]
    top10_display = top10_display.rename(columns={"student_id": "Student ID"})
    st.dataframe(
        top10_display[["Student ID", "Dropout Risk", "Risk Tier", "Predicted Outcome"]],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Navigate to **Student List** or **Student Detail** pages for full analysis.")

st.markdown("</div>", unsafe_allow_html=True)
