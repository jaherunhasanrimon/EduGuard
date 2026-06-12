"""
EduGuard — Interactive SHAP Waterfall Chart (Plotly)
Renders an interactive horizontal bar chart showing per-feature SHAP contributions
for the dropout class.
"""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import FEATURE_DISPLAY_NAMES


def create_shap_waterfall(
    shap_data: dict,
    prediction_prob: float,
    n_display: int = 12,
) -> go.Figure:
    """
    Build an interactive Plotly horizontal bar chart of SHAP values.

    Args:
        shap_data:       Dict from ml.shap_explainer.get_shap_values_for_student()
        prediction_prob: Dropout probability (0–1) for chart title
        n_display:       Number of features to display

    Returns:
        plotly.graph_objects.Figure
    """
    shap_vals    = np.array(shap_data["shap_values"])
    feature_names = shap_data["feature_names"]
    feature_values = shap_data["feature_values"]
    base_value   = shap_data["base_value"]

    # Sort by absolute SHAP value, take top N
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1][:n_display]
    # Reverse for display (most impactful at top)
    display_idx = sorted_idx[::-1]

    labels = []
    values = []
    for idx in display_idx:
        fname = feature_names[idx]
        display = FEATURE_DISPLAY_NAMES.get(fname, fname)
        fval = feature_values.get(fname, "")
        # Format value nicely
        if isinstance(fval, float):
            fval_str = f"{fval:.2f}" if fval != int(fval) else str(int(fval))
        else:
            fval_str = str(fval)
        labels.append(f"{display}  =  {fval_str}")
        values.append(float(shap_vals[idx]))

    # Colours: red = increases dropout risk, green = decreases
    bar_colors = [
        "#E24B4A" if v > 0 else "#639922" for v in values
    ]
    border_colors = [
        "#C62828" if v > 0 else "#3D7A20" for v in values
    ]

    text_labels = [
        f"{'▲' if v > 0 else '▼'} {abs(v):.4f}" for v in values
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(color=border_colors, width=1),
                opacity=0.88,
            ),
            text=text_labels,
            textposition="outside",
            textfont=dict(size=11, family="Inter, sans-serif"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "SHAP Impact: %{x:+.4f}<br>"
                "<extra></extra>"
            ),
            cliponaxis=False,
        )
    )

    # Base value dashed line
    fig.add_vline(
        x=base_value,
        line_dash="dot",
        line_color="#94A3B8",
        line_width=2,
        annotation_text=f"Base {base_value:.3f}",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#64748B"),
    )

    risk_pct = f"{prediction_prob:.1%}"
    tier = (
        "🔴 HIGH RISK" if prediction_prob >= 0.65
        else "🟡 MEDIUM RISK" if prediction_prob >= 0.35
        else "🟢 LOW RISK"
    )

    fig.update_layout(
        title=dict(
            text=f"Risk Factor Analysis — Dropout Probability: <b>{risk_pct}</b>  {tier}",
            font=dict(size=15, color="#1E293B", family="Inter, sans-serif"),
            x=0,
        ),
        xaxis=dict(
            title="← Reduces Risk | Increases Risk →",
            title_font=dict(size=11, color="#64748B"),
            gridcolor="#F1F5F9",
            zeroline=True,
            zerolinecolor="#CBD5E1",
            zerolinewidth=2,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            tickfont=dict(size=11, family="Inter, sans-serif"),
            gridcolor="#F8FAFC",
        ),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        height=max(420, n_display * 38 + 120),
        margin=dict(l=20, r=80, t=70, b=40),
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
        bargap=0.25,
    )

    return fig
