"""
EduGuard — Page 5: Settings
Configuration adjustments for risk thresholds and advisor notification priorities.
Values are persisted to disk (config/settings_user.json) so they survive page
refreshes and server restarts, and are kept in session state during a session.
"""
import sys
import json
import base64
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

# ─── Persistent config file ───────────────────────────────────────────────────
SETTINGS_PATH = ROOT / "config" / "settings_user.json"

DEFAULTS = {
    "high_threshold": 65,
    "medium_threshold": 35,
    "notify_channels": ["Email Notification"],
}

def _load_saved_config() -> dict:
    """Load persisted config from disk, falling back to defaults."""
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r") as f:
                saved = json.load(f)
            # Merge with defaults so new keys are always available
            return {**DEFAULTS, **saved}
        except Exception:
            pass
    return dict(DEFAULTS)

def _save_config(cfg: dict) -> None:
    """Write config to disk."""
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_PATH, "w") as f:
        json.dump(cfg, f, indent=2)

# ─── Initialize session state once per session (or after refresh) ─────────────
if "settings_loaded" not in st.session_state:
    cfg = _load_saved_config()
    st.session_state["settings_high_threshold"]   = cfg["high_threshold"]
    st.session_state["settings_medium_threshold"] = cfg["medium_threshold"]
    st.session_state["settings_notify_channels"]  = cfg["notify_channels"]
    st.session_state["settings_loaded"]           = True

# ─── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar("settings")

# ─── Page Header ─────────────────────────────────────────────────────────────
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

# ─── Threshold Fine-Tuning ────────────────────────────────────────────────────
st.markdown("### 🔧 Model & Rules Customization")

st.markdown('<div class="eg-card">', unsafe_allow_html=True)
st.markdown('<div class="eg-card-title">Threshold Fine-Tuning</div>', unsafe_allow_html=True)
st.markdown(
    "<p style='font-size:0.85rem; color:#64748B;'>"
    "Adjust the risk scores that trigger High and Medium warning states for students."
    "</p>",
    unsafe_allow_html=True,
)

col1, col2 = st.columns(2)
with col1:
    high_threshold = st.slider(
        "High Risk Trigger (%)",
        min_value=50,
        max_value=95,
        value=st.session_state["settings_high_threshold"],  # ← bound to state
        step=5,
        key="slider_high_threshold",
        help="Students equal to or above this score are flagged in red.",
    )
with col2:
    medium_threshold = st.slider(
        "Medium Risk Trigger (%)",
        min_value=10,
        max_value=49,
        value=st.session_state["settings_medium_threshold"],  # ← bound to state
        step=5,
        key="slider_medium_threshold",
        help="Students equal to or above this score are flagged in orange.",
    )

# Guard: medium must be below high
if medium_threshold >= high_threshold:
    st.warning(
        f"⚠️ Medium threshold ({medium_threshold}%) must be lower than the High threshold ({high_threshold}%). "
        "Please adjust before saving."
    )

st.markdown('</div>', unsafe_allow_html=True)

# ─── Notification Settings ────────────────────────────────────────────────────
st.markdown('<div class="eg-card">', unsafe_allow_html=True)
st.markdown('<div class="eg-card-title">Advisor Notification Settings</div>', unsafe_allow_html=True)

notify_channels = st.multiselect(
    "Enable Automated Alerts For High-Risk Students",
    options=["Email Notification", "SMS Dashboard Alert", "DIU Student Portal API Call"],
    default=st.session_state["settings_notify_channels"],  # ← bound to state
    key="multiselect_notify_channels",
)

st.write("")

save_btn = st.button("💾 Save System Configuration", type="primary")

if save_btn:
    if medium_threshold >= high_threshold:
        st.error("❌ Cannot save: Medium threshold must be lower than High threshold.")
    else:
        # 1. Commit to session state
        st.session_state["settings_high_threshold"]   = high_threshold
        st.session_state["settings_medium_threshold"] = medium_threshold
        st.session_state["settings_notify_channels"]  = notify_channels

        # 2. Persist to disk for cross-session/cross-refresh durability
        _save_config({
            "high_threshold":   high_threshold,
            "medium_threshold": medium_threshold,
            "notify_channels":  notify_channels,
        })

        st.toast(
            f"✅ Saved — High: {high_threshold}%, Medium: {medium_threshold}%",
            icon="🎉",
        )

st.markdown('</div>', unsafe_allow_html=True)

# ─── Current active config preview ───────────────────────────────────────────
with st.expander("📋 View Active Configuration", expanded=False):
    active_cfg = _load_saved_config()
    st.json(active_cfg)
