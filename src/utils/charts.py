import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

COLOR_MAP = {
    "Improving": "#10B981",    # Emerald
    "Stable": "#06B6D4",       # Cyan
    "Declining": "#EF4444",    # Crimson
    "Anomalous": "#A855F7",    # Purple
    "Tier 1: High-Priority / Aspirational": "#F59E0B",
    "Tier 2: Developing / Rural-Agricultural": "#3B82F6",
    "Tier 3: Progressing / Semi-Urban Transition": "#06B6D4",
    "Tier 4: Front-Runner / Advanced Socioeconomic": "#10B981",
}

def apply_dark_theme(fig: go.Figure, **custom_kwargs) -> go.Figure:
    """Applies unified luxury dark theme without keyword collisions."""
    base_layout = dict(
        paper_bgcolor="rgba(15, 23, 42, 0)",
        plot_bgcolor="rgba(15, 23, 42, 0)",
        font=dict(family="Inter, sans-serif", color="#E2E8F0"),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    base_layout.update(custom_kwargs)
    fig.update_layout(**base_layout)
    fig.update_xaxes(gridcolor="rgba(255, 255, 255, 0.07)", zerolinecolor="rgba(255, 255, 255, 0.15)", tickfont=dict(color="#94A3B8"))
    fig.update_yaxes(gridcolor="rgba(255, 255, 255, 0.07)", zerolinecolor="rgba(255, 255, 255, 0.15)", tickfont=dict(color="#94A3B8"))
    return fig

def create_gauge_chart(score: float, title: str = "BharatLens Score", prev_score: Optional[float] = None) -> go.Figure:
    """Creates a sleek semi-circular gauge chart."""
    delta_dict = {}
    if prev_score is not None:
        delta_dict = {"reference": prev_score, "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}}

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta" if prev_score is not None else "gauge+number",
        value=score,
        delta=delta_dict,
        title={"text": f"<b>{title}</b>", "font": {"size": 18, "color": "#F8FAFC"}},
        number={"font": {"size": 42, "color": "#FFFFFF", "family": "Outfit, sans-serif"}, "suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
            "bar": {"color": "#F59E0B", "thickness": 0.28},
            "bgcolor": "rgba(255, 255, 255, 0.05)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(239, 68, 68, 0.25)"},
                {"range": [40, 65], "color": "rgba(245, 158, 11, 0.25)"},
                {"range": [65, 80], "color": "rgba(6, 182, 212, 0.25)"},
                {"range": [80, 100], "color": "rgba(16, 185, 129, 0.25)"},
            ],
            "threshold": {
                "line": {"color": "#FFFFFF", "width": 3},
                "thickness": 0.75,
                "value": score
            }
        }
    ))
    apply_dark_theme(fig, height=260)
    return fig

def create_dimension_radar_chart(dim_data: Dict[str, float], title: str = "Socioeconomic Dimension Profile") -> go.Figure:
    """Creates an interactive radar spider chart for dimensions."""
    categories = list(dim_data.keys())
    values = list(dim_data.values())
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(245, 158, 11, 0.25)",
        line=dict(color="#F59E0B", width=2.5),
        marker=dict(size=6, color="#FBBF24")
    ))

    apply_dark_theme(
        fig,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#94A3B8", size=9), gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(tickfont=dict(color="#E2E8F0", size=11, family="Inter"), gridcolor="rgba(255,255,255,0.08)")
        ),
        showlegend=False,
        height=320,
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color="#F8FAFC"), x=0.5, y=0.98)
    )
    return fig

def create_shap_horizontal_bar(top_pos: List[Dict[str, Any]], top_neg: List[Dict[str, Any]]) -> go.Figure:
    """Creates a bi-directional horizontal bar chart for SHAP drivers."""
    items = []
    for p in reversed(top_pos):
        items.append({"name": p["display_name"], "val": p["shap_value"], "type": "Positive Factor (Upward)"})
    for n in top_neg:
        items.append({"name": n["display_name"], "val": n["shap_value"], "type": "Negative Factor (Downward)"})

    if not items:
        fig = go.Figure()
        apply_dark_theme(fig, height=280)
        return fig

    df = pd.DataFrame(items)
    colors = ["#10B981" if v > 0 else "#EF4444" for v in df["val"]]

    fig = go.Figure(go.Bar(
        x=df["val"],
        y=df["name"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="<b>%{y}</b><br>SHAP Impact: %{x:+.3f}<extra></extra>"
    ))

    apply_dark_theme(
        fig,
        height=320,
        title=dict(text="<b>Model Prediction Drivers (Local SHAP Feature Attribution)</b>", font=dict(size=14, color="#F8FAFC")),
        xaxis=dict(title="Contribution to Predicted Trajectory (Points)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(tickfont=dict(size=11))
    )
    return fig

def create_historical_comparison_chart(prev_vals: Dict[str, float], curr_vals: Dict[str, float]) -> go.Figure:
    """Creates paired bar chart comparing NFHS-4 baseline vs NFHS-5 latest."""
    indicators = list(curr_vals.keys())
    c_vals = [curr_vals[k] for k in indicators]
    p_vals = [prev_vals.get(k, 0.0) for k in indicators]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline (NFHS-4)",
        x=indicators,
        y=p_vals,
        marker_color="rgba(148, 163, 184, 0.6)"
    ))
    fig.add_trace(go.Bar(
        name="Latest (NFHS-5)",
        x=indicators,
        y=c_vals,
        marker_color="#06B6D4"
    ))

    apply_dark_theme(
        fig,
        barmode="group",
        height=320,
        title=dict(text="<b>Historical Comparison Across Key Tracking Indicators (%)</b>", font=dict(size=14, color="#F8FAFC")),
        yaxis=dict(title="Observed Rate (%)", range=[0, 105]),
        xaxis=dict(tickangle=-20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
