"""
EduGuard — Risk Badge Component
Renders HTML risk badges and cause-tag pills.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import RISK_COLORS, RISK_BG_COLORS


RISK_ICONS = {
    "High": "🔴",
    "Medium": "🟡",
    "Low": "🟢",
}


def risk_badge_html(tier: str) -> str:
    """Return an HTML string for a coloured risk badge."""
    icon = RISK_ICONS.get(tier, "⚪")
    color = RISK_COLORS.get(tier, "#64748B")
    bg = RISK_BG_COLORS.get(tier, "#F1F5F9")
    css_class = f"eg-badge eg-badge-{tier.lower()}"
    return (
        f'<span class="{css_class}">'
        f'{icon} {tier} Risk'
        f"</span>"
    )


def cause_tag_html(tags: list) -> str:
    """Return HTML string of cause-tag pills."""
    if not tags:
        return '<span style="color:#94A3B8;font-size:0.75rem;">—</span>'
    return "".join(f'<span class="eg-tag">{t}</span>' for t in tags)


def target_badge_html(label: str) -> str:
    """Return a coloured badge for the predicted outcome label."""
    colors = {
        "Dropout": ("#FEE2E2", "#C62828"),
        "Graduate": ("#DCFCE7", "#166534"),
        "Enrolled": ("#DBEAFE", "#1E40AF"),
    }
    bg, fg = colors.get(label, ("#F1F5F9", "#475569"))
    return (
        f'<span style="background:{bg};color:{fg};padding:3px 10px;'
        f'border-radius:999px;font-size:0.72rem;font-weight:700;">'
        f"{label}</span>"
    )
