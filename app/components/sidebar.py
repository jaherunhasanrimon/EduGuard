import streamlit as st
import base64
from pathlib import Path
from app.auth import logout
from app.components.data_manager import save_uploaded_dataset, clear_uploaded_dataset, update_selected_source, init_data_state
from ml.predict import load_pipeline, load_demo_data, load_uploaded_data, predict_batch

ROOT = Path(__file__).parent.parent.parent
ASSETS_DIR = ROOT / "assets"

@st.cache_resource(show_spinner=False)
def get_pipeline():
    return load_pipeline()

@st.cache_data(show_spinner=False)
def get_predictions(source_key: str, file_bytes=None):
    pipeline = get_pipeline()
    if source_key == "upload" and file_bytes is not None:
        import io
        df = load_uploaded_data(io.BytesIO(file_bytes))
    else:
        df = load_demo_data()
    return predict_batch(df, pipeline)

def render_sidebar(active_page: str, slot_fn=None):
    """
    Renders the custom premium SaaS sidebar for EduGuard.
    Handles branding, navigation, data source selection, custom slot, and sign-out.
    """
    # 1. Initialize dataset state (ensures st.session_state["df_pred"] is populated on every page load)
    init_data_state(get_predictions)
    
    with st.sidebar:
        # Branding Header
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
        
        # Navigation Menu
        st.markdown('<div class="sidebar-nav-label">Navigation</div>', unsafe_allow_html=True)
        
        # Custom css attributes to help style current active item
        st.markdown(
            f"""
            <div class="sidebar-nav-container" data-active-page="{active_page}">
            """,
            unsafe_allow_html=True,
        )
        
        # Render page links
        st.page_link("main.py", label="Dashboard", icon=":material/dashboard:")
        st.page_link("pages/2_🎓_Student_List.py", label="Student List", icon=":material/group:")
        st.page_link("pages/3_🔍_Student_Detail.py", label="Student Details", icon=":material/person:")
        st.page_link("pages/4_📊_Analytics.py", label="Analytics", icon=":material/analytics:")
        st.page_link("pages/5_⚙️_Settings.py", label="Settings", icon=":material/settings:")
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
        
        # Data Source Card
        st.markdown('<div class="sidebar-nav-label">Data Source</div>', unsafe_allow_html=True)
        
        # Determine source
        selected_source = st.session_state.get("data_source", "Demo dataset (UCI)")
        
        st.markdown('<div class="source-card">', unsafe_allow_html=True)
        
        # Custom styled radio/selector
        source_options = ["Demo dataset (UCI)", "Upload CSV"]
        try:
            choice_idx = source_options.index(selected_source)
        except ValueError:
            choice_idx = 0
            
        source_choice = st.radio(
            "Data Source Selection",
            source_options,
            index=choice_idx,
            key="sidebar_data_source_radio",
            label_visibility="collapsed",
        )
        
        # Update source selection on change
        if source_choice != selected_source:
            update_selected_source(source_choice)
            st.rerun()
            
        if source_choice == "Upload CSV":
            # File uploader
            uploaded_file = st.file_uploader(
                "Upload student CSV",
                type=["csv"],
                help="CSV must include the same columns as the UCI student dataset.",
                key="sidebar_csv_uploader",
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                save_uploaded_dataset(file_bytes, uploaded_file.name)
                st.rerun()
                
            if "uploaded_bytes" in st.session_state:
                st.markdown(
                    f"""
                    <div class="active-upload-badge">
                      <span class="active-upload-dot"></span>
                      <span class="active-upload-name" title="{st.session_state['uploaded_filename']}">
                        Active: {st.session_state['uploaded_filename']}
                      </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("🔄 Reset to Demo Dataset", key="reset_demo_btn", use_container_width=True):
                    clear_uploaded_dataset()
                    st.rerun()
        else:
            # If changed back to Demo, clear uploaded state
            if "uploaded_bytes" in st.session_state:
                clear_uploaded_dataset()
                st.rerun()
                
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Slot for additional custom components (e.g. Filters on specific pages)
        # These are rendered after the data source card and before Sign Out
        if slot_fn is not None:
            st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-slot">', unsafe_allow_html=True)
            slot_fn()
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Sign Out Button Container
        st.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)
        if st.button("🚪 Sign Out", key="sidebar_signout_btn", use_container_width=True):
            logout()
        st.markdown('</div>', unsafe_allow_html=True)

