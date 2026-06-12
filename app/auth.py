"""
EduGuard — Authentication Guard
Call require_auth() at the top of every page to enforce the password gate.
Session state key: st.session_state["authenticated"]
"""
import streamlit as st
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import DEFAULT_PASSWORD, APP_TITLE, APP_SUBTITLE, UNIVERSITY_NAME, ASSETS_DIR


def _load_css():
    css_path = ROOT / "assets" / "style.css"
    if css_path.exists():
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def require_auth() -> bool:
    """
    Check if the user is authenticated.
    If not, render the login page and return False.
    If yes, inject global CSS and return True.

    Usage at top of every page:
        from app.auth import require_auth
        if not require_auth():
            st.stop()
    """
    _load_css()

    if st.session_state.get("authenticated", False):
        return True

    # ── Login UI ──────────────────────────────────────────────────────────────
    _, col, _ = st.columns([1, 1.8, 1])
    with col:
        logo_path = ASSETS_DIR / "diu_logo.svg"
        if logo_path.exists():
            st.image(str(logo_path), width=90)

        st.markdown(
            f"""
            <div style="text-align:center; margin-bottom:24px;">
              <div class="eg-login-title">{APP_TITLE}</div>
              <div class="eg-login-subtitle">{APP_SUBTITLE}</div>
              <div class="eg-login-university">🏛️ {UNIVERSITY_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        password = st.text_input(
            "Administrator Password",
            type="password",
            placeholder="Enter password…",
            key="auth_password_input",
        )

        if st.button("🔐  Sign In", use_container_width=True, type="primary"):
            try:
                correct = st.secrets.get("auth", {}).get("password", DEFAULT_PASSWORD)
            except Exception:
                correct = DEFAULT_PASSWORD

            if password == correct:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌  Incorrect password. Try **eduguard2026** for the demo.")

        st.markdown(
            "<p style='text-align:center;font-size:0.72rem;color:#94A3B8;margin-top:20px;'>"
            "EduGuard MVP · DIU AI Project Competition 2026</p>",
            unsafe_allow_html=True,
        )

    return False


def logout():
    """Clear authentication state."""
    st.session_state["authenticated"] = False
    st.rerun()
