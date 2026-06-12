"""
EduGuard — Page 3: What-If Simulator
Interactive feature sliders → live risk recalculation + SHAP comparison.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="EduGuard — What-If Simulator",
    page_icon="🎓",
    layout="wide",
)

from app.auth import require_auth
from app.components.risk_badge import risk_badge_html
from app.components.shap_plot import create_shap_waterfall
from config.settings import (
    ASSETS_DIR, COMPETITION_LABEL, FEATURE_DISPLAY_NAMES,
    RISK_COLORS, UNIVERSITY_NAME, WHATIF_FEATURES, WHATIF_CONFIG,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL,
)
from ml.predict import load_pipeline, load_explainer, predict_single
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
        <h1>🔮 What-If Simulator</h1>
        <p>{UNIVERSITY_NAME} · {COMPETITION_LABEL}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='color:#64748B;font-size:0.875rem;margin-bottom:20px;'>"
    "Adjust student features using the controls below and run the simulation to see "
    "how changes affect the dropout risk prediction and SHAP explanations.</p>",
    unsafe_allow_html=True,
)

# ─── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_pipeline():
    return load_pipeline()

@st.cache_resource(show_spinner="Loading SHAP explainer…")
def get_explainer():
    return load_explainer()

pipeline = get_pipeline()
explainer = get_explainer()

# ─── Get seed student ────────────────────────────────────────────────────────
df_pred = st.session_state.get("df_pred")
whatif_idx = st.session_state.get("whatif_student_idx", 0)

seed_row = None
if df_pred is not None and whatif_idx < len(df_pred):
    seed_row = df_pred.iloc[whatif_idx]
    orig_prob = float(seed_row["dropout_prob"])
    orig_tier = seed_row["risk_tier"]
else:
    orig_prob = None
    orig_tier = None

# ─── Controls ────────────────────────────────────────────────────────────────
ctrl_col, result_col = st.columns([1, 1.4])

with ctrl_col:
    st.markdown('<div class="eg-card">', unsafe_allow_html=True)
    st.markdown('<div class="eg-card-title">⚙️ Feature Controls</div>', unsafe_allow_html=True)

    if seed_row is not None:
        st.caption(f"Pre-populated from Student #{whatif_idx + 1} (original risk: **{orig_prob:.1%}**)")
    else:
        st.caption("Configure features manually below.")

    feature_values = {}

    for feat in WHATIF_FEATURES:
        cfg = WHATIF_CONFIG.get(feat, {"min": 0, "max": 10, "step": 1, "type": "int"})
        display = FEATURE_DISPLAY_NAMES.get(feat, feat)
        # Get seed value
        seed_val = float(seed_row[feat]) if (seed_row is not None and feat in seed_row) else cfg["min"]

        if cfg["type"] == "select":
            options   = cfg["options"]
            labels    = cfg.get("labels", [str(o) for o in options])
            seed_idx  = options.index(int(seed_val)) if int(seed_val) in options else 0
            chosen    = st.selectbox(
                display,
                options=options,
                index=seed_idx,
                format_func=lambda x, cfg=cfg: cfg.get("labels", [str(o) for o in cfg["options"]])[cfg["options"].index(x)],
                key=f"wi_{feat}",
            )
            feature_values[feat] = int(chosen)

        elif cfg["type"] == "float":
            val = st.slider(
                display,
                min_value=float(cfg["min"]),
                max_value=float(cfg["max"]),
                value=float(max(cfg["min"], min(cfg["max"], seed_val))),
                step=float(cfg["step"]),
                key=f"wi_{feat}",
            )
            feature_values[feat] = val

        else:  # int
            val = st.slider(
                display,
                min_value=int(cfg["min"]),
                max_value=int(cfg["max"]),
                value=int(max(cfg["min"], min(cfg["max"], seed_val))),
                step=int(cfg["step"]),
                key=f"wi_{feat}",
            )
            feature_values[feat] = val

    st.markdown("</div>", unsafe_allow_html=True)

    run_sim = st.button("▶  Run Simulation", use_container_width=True, type="primary")

# ─── Result Panel ────────────────────────────────────────────────────────────
with result_col:
    st.markdown('<div class="eg-card">', unsafe_allow_html=True)
    st.markdown('<div class="eg-card-title">📊 Simulation Result</div>', unsafe_allow_html=True)

    # Build full feature dict by merging seed values + overrides
    if seed_row is not None:
        full_features = seed_row.drop(
            labels=["dropout_prob", "risk_tier", "predicted_label", "student_id", TARGET_COL],
            errors="ignore",
        ).to_dict()
    else:
        # Default fallback values for all features
        all_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        full_features = {col: 0 for col in all_cols}

    # Apply What-If overrides
    full_features.update(feature_values)

    if run_sim or "sim_result" in st.session_state:
        if run_sim:
            with st.spinner("Running model…"):
                result = predict_single(full_features, pipeline)
            st.session_state["sim_result"] = result
            st.session_state["sim_features"] = full_features
        else:
            result = st.session_state.get("sim_result")
            full_features = st.session_state.get("sim_features", full_features)

        if result:
            new_prob = result["dropout_prob"]
            new_tier = result["risk_tier"]
            new_label = result["predicted_label"]

            risk_color = RISK_COLORS.get(new_tier, "#4F46E5")

            # Big score display
            if orig_prob is not None:
                delta = new_prob - orig_prob
                delta_str = f"{'▲' if delta > 0 else '▼'} {abs(delta):.1%} vs original"
                delta_color = "#E24B4A" if delta > 0 else "#639922"
            else:
                delta_str = ""
                delta_color = "#64748B"

            st.markdown(
                f"""
                <div style="text-align:center;padding:20px 0;">
                  <div style="font-size:3.5rem;font-weight:900;color:{risk_color};line-height:1;">
                    {new_prob:.1%}
                  </div>
                  <div style="font-size:0.9rem;color:#64748B;margin:8px 0;">Dropout Probability</div>
                  {risk_badge_html(new_tier)}
                  &nbsp;
                  <span style="font-size:0.85rem;font-weight:600;color:{delta_color};">{delta_str}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Before / After comparison
            if orig_prob is not None:
                st.markdown("**Before → After Comparison**")
                comp_data = {
                    "Scenario": ["Original (Before)", "Simulated (After)"],
                    "Dropout Risk": [f"{orig_prob:.1%}", f"{new_prob:.1%}"],
                    "Risk Tier": [orig_tier, new_tier],
                    "Predicted Outcome": [
                        df_pred.iloc[whatif_idx]["predicted_label"] if df_pred is not None else "—",
                        new_label,
                    ],
                }
                st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

            # All class probabilities
            with st.expander("All class probabilities"):
                for cls, prob in result["all_probas"].items():
                    st.write(f"**{cls}**: {prob:.3%}")
    else:
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;color:#94A3B8;">'
            '<div style="font-size:3rem;">🔮</div>'
            '<div style="font-size:0.9rem;margin-top:12px;">Adjust the controls and click <strong>Run Simulation</strong></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ─── SHAP Chart for Simulated Student ────────────────────────────────────────
if "sim_result" in st.session_state and st.session_state["sim_result"] is not None:
    st.markdown("---")
    st.markdown('<div class="eg-section-title">🧠 SHAP Explanation — Simulated Scenario</div>', unsafe_allow_html=True)
    st.caption(
        "This SHAP chart shows which features are driving the **simulated** risk score. "
        "Compare against the original student on the Student Detail page."
    )

    try:
        sim_features = st.session_state.get("sim_features", full_features)
        sim_df = pd.DataFrame([sim_features])

        sim_result = st.session_state["sim_result"]
        new_prob = sim_result["dropout_prob"]

        X_transformed = pipeline.named_steps["preprocessor"].transform(sim_df)
        import shap as shap_lib
        shap_vals = explainer.shap_values(X_transformed)

        classes = list(pipeline.classes_)
        from config.settings import DROPOUT_CLASS
        dropout_idx = classes.index(DROPOUT_CLASS)

        if isinstance(shap_vals, list):
            student_shap = shap_vals[dropout_idx][0]
            base = (
                float(explainer.expected_value[dropout_idx])
                if hasattr(explainer.expected_value, "__len__")
                else float(explainer.expected_value)
            )
        else:
            student_shap = shap_vals[0]
            base = float(explainer.expected_value)

        num_names = pipeline.named_steps["preprocessor"].transformers_[0][2]
        cat_names = pipeline.named_steps["preprocessor"].transformers_[1][2]
        feature_names = list(num_names) + list(cat_names)

        feature_values_map = {f: sim_features.get(f, "N/A") for f in feature_names}

        shap_data = {
            "shap_values": student_shap,
            "feature_names": feature_names,
            "feature_values": feature_values_map,
            "base_value": base,
        }

        shap_fig = create_shap_waterfall(shap_data, new_prob, n_display=12)
        st.plotly_chart(shap_fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Could not render SHAP chart: {e}")

st.markdown(
    "<div style='margin-top:24px;padding:12px 16px;background:#EEF2FF;border-radius:10px;"
    "border-left:4px solid #4F46E5;font-size:0.8rem;color:#3730A3;'>"
    "💡 <strong>How to use this tool:</strong> Increase units passed or ensure fees are paid to see "
    "how targeted interventions could reduce a student's dropout probability."
    "</div>",
    unsafe_allow_html=True,
)
