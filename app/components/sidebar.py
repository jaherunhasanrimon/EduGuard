import io
import streamlit as st
import base64
from pathlib import Path
from app.auth import logout
from app.components.data_manager import (
    save_uploaded_dataset,
    clear_uploaded_dataset,
    update_selected_source,
    init_data_state,
)
from ml.predict import load_pipeline, load_demo_data, load_uploaded_data, predict_batch, CSVValidationError

ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = ROOT / "assets"


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return load_pipeline()


@st.cache_data(show_spinner=False, max_entries=3)
def get_predictions(source_key: str, file_bytes: bytes = None):
    """
    Cached batch prediction call.

    Keyed on (source_key, file_bytes) so that:
    - Demo dataset predictions are computed once and reused.
    - Uploaded dataset predictions are recomputed only when the bytes change.
    - max_entries=3 caps memory usage (evicts the oldest cache entry).

    Returns a tuple (df_pred, error_message).
    error_message is None on success, or a human-readable string on failure.
    """
    pipeline = get_pipeline()
    try:
        if source_key == "upload" and file_bytes is not None:
            df = load_uploaded_data(io.BytesIO(file_bytes))
        else:
            df = load_demo_data()
        return predict_batch(df, pipeline), None
    except CSVValidationError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Unexpected error during prediction: {exc}"


def render_sidebar(active_page: str, slot_fn=None):
    """
    Renders the custom premium SaaS sidebar for EduGuard.
    Handles branding, navigation, data source selection, custom slot, and sign-out.
    """
    # ── 1. Ensure df_pred is populated for this session ───────────────────────
    # Only re-run init_data_state if df_pred is not yet in session state.
    # This prevents re-running predictions on every widget interaction.
    if "df_pred" not in st.session_state:
        _reload_predictions()

    with st.sidebar:
        # ── Branding Header ───────────────────────────────────────────────────
        logo_path = ASSETS_DIR / "diu_logo.svg"
        logo_html = ""
        if logo_path.exists():
            with open(logo_path, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" class="sidebar-logo-img">'

        st.markdown(
            f"""
            <div class="sidebar-header">
              {logo_html}
              <div class="sidebar-header-text">
                <div class="sidebar-title">EduGuard</div>
                <div class="sidebar-subtitle">Student Risk Intelligence</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Navigation Menu ───────────────────────────────────────────────────
        st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="sidebar-nav-container" data-active-page="{active_page}">',
            unsafe_allow_html=True,
        )
        st.page_link("main.py",                          label="Dashboard",       icon=":material/dashboard:")
        st.page_link("pages/2_🎓_Student_List.py",      label="Student List",    icon=":material/group:")
        st.page_link("pages/3_🔍_Student_Detail.py",    label="Student Details", icon=":material/person:")
        st.page_link("pages/4_📊_Analytics.py",         label="Analytics",       icon=":material/analytics:")
        st.page_link("pages/5_⚙️_Settings.py",         label="Settings",        icon=":material/settings:")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

        # ── Data Source Card ──────────────────────────────────────────────────
        st.markdown('<div class="sidebar-nav-label">Data Source</div>', unsafe_allow_html=True)

        selected_source = st.session_state.get("data_source", "Demo dataset (UCI)")
        source_options  = ["Demo dataset (UCI)", "Upload CSV"]

        try:
            choice_idx = source_options.index(selected_source)
        except ValueError:
            choice_idx = 0

        st.markdown('<div class="source-card">', unsafe_allow_html=True)

        source_choice = st.radio(
            "Data Source Selection",
            source_options,
            index=choice_idx,
            key="sidebar_data_source_radio",
            label_visibility="collapsed",
        )

        # ── Source selection changed ──────────────────────────────────────────
        if source_choice != selected_source:
            update_selected_source(source_choice)
            # If user switched back to Demo, wipe any cached upload predictions
            if source_choice == "Demo dataset (UCI)" and "uploaded_bytes" in st.session_state:
                clear_uploaded_dataset()
            # Force df_pred to be recomputed on next render
            st.session_state.pop("df_pred", None)
            st.rerun()

        if source_choice == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload student CSV",
                type=["csv"],
                help="CSV must include the same columns as the UCI student dataset.",
                key="sidebar_csv_uploader",
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                new_bytes = uploaded_file.read()
                # Only process if the file content actually changed
                if new_bytes != st.session_state.get("uploaded_bytes"):
                    save_uploaded_dataset(new_bytes, uploaded_file.name)
                    # Invalidate df_pred so it's recomputed from the new file
                    st.session_state.pop("df_pred", None)
                    # Run predictions immediately with spinner feedback
                    with st.spinner("🔄 Analysing uploaded dataset…"):
                        _reload_predictions()
                    st.rerun()

            if "uploaded_bytes" in st.session_state:
                fname = st.session_state.get("uploaded_filename", "dataset.csv")
                st.markdown(
                    f"""
                    <div class="active-upload-badge">
                      <span class="active-upload-dot"></span>
                      <span class="active-upload-name" title="{fname}">
                        Active: {fname}
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("🔄 Reset to Demo Dataset", key="reset_demo_btn", use_container_width=True):
                    clear_uploaded_dataset()
                    st.session_state.pop("df_pred", None)
                    st.rerun()
        else:
            # Switched to Demo — clear any leftover upload state
            if "uploaded_bytes" in st.session_state:
                clear_uploaded_dataset()
                st.session_state.pop("df_pred", None)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # ── Slot for page-specific widgets (e.g. Filters) ─────────────────────
        if slot_fn is not None:
            st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-slot">', unsafe_allow_html=True)
            slot_fn()
            st.markdown("</div>", unsafe_allow_html=True)

        # ── Sign Out ──────────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="sidebar_signout_btn", use_container_width=True):
            logout()
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Internal: compute & store predictions, show errors inline ────────────────

def _reload_predictions():
    """
    Compute (or retrieve from cache) df_pred and store it in session_state.
    On validation/parsing errors, displays an inline sidebar error instead
    of crashing the page, and falls back to the demo dataset gracefully.
    """
    source = st.session_state.get("data_source", "Demo dataset (UCI)")

    if source == "Upload CSV" and "uploaded_bytes" in st.session_state:
        df_pred, err = get_predictions("upload", st.session_state["uploaded_bytes"])
    else:
        df_pred, err = get_predictions("demo")

    if err is not None:
        # Show a friendly inline error in the sidebar
        with st.sidebar:
            st.error(
                f"⚠️ **Upload Error**\n\n{err}\n\n"
                "_Falling back to demo dataset._"
            )
        # Fall back to demo so the rest of the app doesn't crash
        df_pred, _ = get_predictions("demo")
        # Reset source so the radio reflects reality
        update_selected_source("Demo dataset (UCI)")
        st.session_state.pop("uploaded_bytes", None)

    st.session_state["df_pred"] = df_pred
