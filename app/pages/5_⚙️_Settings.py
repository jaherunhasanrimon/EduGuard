"""
EduGuard — Page 5: Settings (Mockup/Configuration)
Provides configuration adjustments for risk thresholds and advisor priorities.
"""
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="EduGuard — Settings",
    page_icon="🎓",
    layout="wide",
)

from app.auth import require_auth
from config.settings import ASSETS_DIR, COMPETITION_LABEL, UNIVERSITY_NAME
from app.components.sidebar import render_sidebar

if not require_auth():
    st.stop()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar("settings")

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
        <h1>⚙️ Platform Settings</h1>
        <p>{UNIVERSITY_NAME} · {COMPETITION_LABEL}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Page Content ─────────────────────────────────────────────────────────────
st.markdown("### 🔧 Model & Rules Customization")

st.markdown('<div class="eg-card">', unsafe_allow_html=True)
st.markdown('<div class="eg-card-title">Threshold Fine-Tuning</div>', unsafe_allow_html=True)
st.markdown("<p style='font-size:0.85rem; color:#64748B;'>Adjust the risk scores that trigger High and Medium warning states for students.</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    high_threshold = st.slider("High Risk Trigger (%)", min_value=50, max_value=95, value=65, step=5, help="Students equal to or above this score are flagged in red.")
with col2:
    medium_threshold = st.slider("Medium Risk Trigger (%)", min_value=10, max_value=49, value=35, step=5, help="Students equal to or above this score are flagged in orange.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="eg-card">', unsafe_allow_html=True)
st.markdown('<div class="eg-card-title">Advisor Notification Settings</div>', unsafe_allow_html=True)
notify_channels = st.multiselect("Enable Automated Alerts For High-Risk Students", options=["Email Notification", "SMS Dashboard Alert", "DIU Student Portal API Call"], default=["Email Notification"])
st.write("")
save_btn = st.button("💾 Save System Configuration", type="primary")
if save_btn:
    st.toast("✅ Settings saved successfully! (Simulated)", icon="🎉")
st.markdown('</div>', unsafe_allow_html=True)
