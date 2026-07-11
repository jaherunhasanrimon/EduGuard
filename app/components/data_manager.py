import os
import json
import streamlit as st
from pathlib import Path

# Paths
ROOT = Path(__file__).parent.parent.parent
PERSIST_CSV_PATH = ROOT / "data" / "active_upload.csv"
PERSIST_META_PATH = ROOT / "data" / "active_upload_meta.json"
SETTINGS_PATH = ROOT / "config" / "settings_user.json"

_THRESHOLD_DEFAULTS = {"high_threshold": 65, "medium_threshold": 35}

def get_active_thresholds() -> dict:
    """
    Return the user-saved risk thresholds as decimal fractions (0–1).
    Falls back to hardcoded defaults (0.65 / 0.35) if no config is saved yet.
    """
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r") as f:
                cfg = json.load(f)
            return {
                "high":   cfg.get("high_threshold",   _THRESHOLD_DEFAULTS["high_threshold"])   / 100,
                "medium": cfg.get("medium_threshold",  _THRESHOLD_DEFAULTS["medium_threshold"]) / 100,
            }
        except Exception:
            pass
    return {
        "high":   _THRESHOLD_DEFAULTS["high_threshold"]   / 100,
        "medium": _THRESHOLD_DEFAULTS["medium_threshold"] / 100,
    }


def init_data_state(get_predictions_fn=None):
    """
    Restore disk-persisted upload metadata into session state on startup/refresh.

    This function is now responsible ONLY for re-hydrating session_state from
    the on-disk persisted CSV/meta files (e.g. after a browser refresh).
    It does NOT trigger predictions itself — that is handled by
    sidebar._reload_predictions(), which runs only when df_pred is absent.

    The `get_predictions_fn` argument is kept for backward compatibility but
    is no longer used.
    """
    if PERSIST_CSV_PATH.exists() and PERSIST_META_PATH.exists():
        try:
            with open(PERSIST_META_PATH, "r") as f:
                meta = json.load(f)

            if "uploaded_filename" not in st.session_state:
                st.session_state["uploaded_filename"] = meta.get("filename", "Uploaded Dataset")

            if "uploaded_bytes" not in st.session_state:
                with open(PERSIST_CSV_PATH, "rb") as f:
                    st.session_state["uploaded_bytes"] = f.read()

            if "data_source" not in st.session_state:
                st.session_state["data_source"] = meta.get("selected_source", "Upload CSV")

        except Exception:
            # Corrupted files — wipe and fall back to demo
            clear_uploaded_dataset()
    else:
        if "data_source" not in st.session_state:
            st.session_state["data_source"] = "Demo dataset (UCI)"


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
