import os
import json
import streamlit as st
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent.parent
PERSIST_CSV_PATH = ROOT / "data" / "active_upload.csv"
PERSIST_META_PATH = ROOT / "data" / "active_upload_meta.json"

def init_data_state(get_predictions_fn):
    """
    Initializes data states on startup/refresh.
    Restores session state from persistent files if they exist on disk.
    Ensures st.session_state['df_pred'] is populated.
    """
    # 1. Load from disk if available and session state is empty
    if PERSIST_CSV_PATH.exists() and PERSIST_META_PATH.exists():
        try:
            with open(PERSIST_META_PATH, "r") as f:
                meta = json.load(f)
            
            # Restore filename and selection status
            if "uploaded_filename" not in st.session_state:
                st.session_state["uploaded_filename"] = meta.get("filename", "Uploaded Dataset")
            
            if "uploaded_bytes" not in st.session_state:
                with open(PERSIST_CSV_PATH, "rb") as f:
                    st.session_state["uploaded_bytes"] = f.read()
            
            if "data_source" not in st.session_state:
                st.session_state["data_source"] = meta.get("selected_source", "Upload CSV")
        except Exception as e:
            # Handle potential corruption gracefully
            clear_uploaded_dataset()
    else:
        if "data_source" not in st.session_state:
            st.session_state["data_source"] = "Demo dataset (UCI)"

    # 2. Get predictions based on active source
    source = st.session_state.get("data_source", "Demo dataset (UCI)")
    if source == "Upload CSV" and "uploaded_bytes" in st.session_state:
        df_pred = get_predictions_fn("upload", st.session_state["uploaded_bytes"])
    else:
        df_pred = get_predictions_fn("demo")
    
    st.session_state["df_pred"] = df_pred
    return df_pred

def save_uploaded_dataset(file_bytes, filename):
    """
    Saves uploaded file bytes and metadata to disk and updates session state.
    """
    try:
        # Create directories if they do not exist
        PERSIST_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        with open(PERSIST_CSV_PATH, "wb") as f:
            f.write(file_bytes)
        
        meta = {
            "filename": filename,
            "selected_source": "Upload CSV"
        }
        with open(PERSIST_META_PATH, "w") as f:
            json.dump(meta, f)
            
        st.session_state["uploaded_bytes"] = file_bytes
        st.session_state["uploaded_filename"] = filename
        st.session_state["data_source"] = "Upload CSV"
    except Exception as e:
        st.error(f"Failed to persist uploaded dataset: {e}")

def clear_uploaded_dataset():
    """
    Deletes persisted files and clears upload session state, resetting to demo.
    """
    try:
        if PERSIST_CSV_PATH.exists():
            os.remove(PERSIST_CSV_PATH)
        if PERSIST_META_PATH.exists():
            os.remove(PERSIST_META_PATH)
    except Exception:
        pass
        
    st.session_state.pop("uploaded_bytes", None)
    st.session_state.pop("uploaded_filename", None)
    st.session_state["data_source"] = "Demo dataset (UCI)"

def update_selected_source(source):
    """
    Updates the selected source in session state and persists it to metadata if upload exists.
    """
    st.session_state["data_source"] = source
    if PERSIST_META_PATH.exists():
        try:
            with open(PERSIST_META_PATH, "r") as f:
                meta = json.load(f)
            meta["selected_source"] = source
            with open(PERSIST_META_PATH, "w") as f:
                json.dump(meta, f)
        except Exception:
            pass
