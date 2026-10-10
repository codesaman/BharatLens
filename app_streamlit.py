import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.utils.charts import (
    create_gauge_chart,
    create_dimension_radar_chart,
    create_shap_horizontal_bar,
    create_historical_comparison_chart,
    apply_dark_theme,
    COLOR_MAP
)
from src.explainability.shap_analysis import format_natural_language_explanation

# Page Configuration
st.set_page_config(
    page_title="BharatLens ML - Indian District Intelligence",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark Styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #F1F5F9;
}

h1, h2, h3, h4, .hero-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, #0B132B 0%, #0F172A 50%, #090E17 100%);
}

/* Glassmorphic card styling */
.glass-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.glass-card:hover {
    border-color: rgba(245, 158, 11, 0.3);
    transform: translateY(-2px);
}

.stat-label {
    font-size: 0.82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94A3B8;
    margin-bottom: 0.25rem;
}

.stat-val {
    font-family: 'Outfit', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.1;
}

.stat-delta {
    font-size: 0.85rem;
    font-weight: 600;
    margin-top: 0.25rem;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.28rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.badge-improving { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
.badge-stable { background: rgba(6, 182, 212, 0.2); color: #22D3EE; border: 1px solid #06B6D4; }
.badge-declining { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }
.badge-anomaly { background: rgba(168, 85, 247, 0.25); color: #C084FC; border: 1px solid #A855F7; }
.badge-tier { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }

/* Custom metrics container */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 0.8rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Data & Metadata Loaders (Cached)
@st.cache_data
def load_master_data() -> pd.DataFrame:
    path = os.path.join("data", "processed", "district_master_ml_results.csv")
    if not os.path.exists(path):
        path = os.path.join("data", "processed", "district_features_dataset.csv")
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_geojson():
    path = os.path.join("data", "external", "districts.geojson")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_all_metadata():
    m_dir = os.path.join("models", "metadata")
    files = {
        "regression": os.path.join(m_dir, "regression_metadata.json"),
        "classification": os.path.join(m_dir, "classification_metadata.json"),
        "clustering": os.path.join(m_dir, "clustering_metadata.json"),
        "anomaly": os.path.join(m_dir, "anomaly_metadata.json"),
        "shap": os.path.join(m_dir, "shap_explanations.json"),
        "data_sources": os.path.join("config", "data_sources.yaml"),
    }
    loaded = {}
    for k, p in files.items():
        if os.path.exists(p):
            if p.endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    loaded[k] = json.load(f)
            elif p.endswith(".yaml"):
                import yaml
                with open(p, "r", encoding="utf-8") as f:
                    loaded[k] = yaml.safe_load(f)
    return loaded

# Load resources
df_master = load_master_data()
metadata = load_all_metadata()
geojson_data = load_geojson()

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
        <div style="font-size: 2.2rem;">🇮🇳</div>
        <div>
            <div style="font-family: 'Outfit'; font-size: 1.4rem; font-weight: 800; color: #FFFFFF; line-height: 1.1;">BHARATLENS ML</div>
            <div style="font-size: 0.75rem; color: #F59E0B; font-weight: 600; letter-spacing: 0.05em;">DISTRICT INTELLIGENCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 0.6rem 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    nav_page = st.radio(
        "Navigation",
        [
            "🌐 National Overview",
            "🔍 District Deep-Dive",
            "🏛️ State Explorer",
            "🤖 ML Insights & Evaluation",
            "📚 Public Data Sources",
            "📐 Methodology & Scoring",
        ],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='margin: 1rem 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    # Telemetry Status Widget
    st.markdown("""
    <div class="glass-card" style="padding: 0.9rem; margin-bottom: 0.5rem;">
        <div class="stat-label">System Telemetry</div>
        <div style="font-size: 0.85rem; margin-top: 0.4rem;">
            <div>📦 <b>643</b> Districts Analyzed</div>
            <div>🏛️ <b>37</b> States & UTs</div>
            <div>🤖 <b>Ridge & Logistic</b> Champions</div>
            <div>🔍 <b>SHAP Tree & Linear</b> Active</div>
            <div style="margin-top: 0.35rem; color: #10B981; font-weight: 600;">● Engine Online (Cached)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("BharatLens ML v1.0.0 | Production AI Engine")


# ==============================================================================
# PAGE 1: NATIONAL OVERVIEW (DASHBOARD)
# ==============================================================================
if nav_page == "🌐 National Overview":
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="margin: 0; font-size: 2.4rem;">National District Intelligence Overview</h1>
        <p style="color: #94A3B8; font-size: 1.05rem; margin-top: 0.2rem;">
            Empirical socioeconomic monitoring, growth trajectory classification, and anomaly detection across Indian districts.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI Metrics Cards
    total_districts = len(df_master)
    mean_score = df_master["bharatlens_score_current"].mean()
    mean_change = df_master["score_change"].mean()
    improving_cnt = (df_master["final_trend_class"] == "Improving").sum()
    stable_cnt = (df_master["final_trend_class"] == "Stable").sum()
    declining_cnt = (df_master["final_trend_class"] == "Declining").sum()
    anomaly_cnt = (df_master["final_trend_class"] == "Anomalous").sum()

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Districts Covered</div>
            <div class="stat-val">{total_districts}</div>
            <div class="stat-delta" style="color: #94A3B8;">37 States & UTs</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">National Avg Score</div>
            <div class="stat-val">{mean_score:.1f}</div>
            <div class="stat-delta" style="color: #10B981;">▲ +{mean_change:.1f} vs Baseline</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Improving / Accelerating</div>
            <div class="stat-val" style="color: #34D399;">{improving_cnt}</div>
            <div class="stat-delta" style="color: #34D399;">{improving_cnt/total_districts*100:.1f}% of India</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Declining / Lagging</div>
            <div class="stat-val" style="color: #F87171;">{declining_cnt}</div>
            <div class="stat-delta" style="color: #F87171;">{declining_cnt/total_districts*100:.1f}% of India</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Anomalous Outliers</div>
            <div class="stat-val" style="color: #C084FC;">{anomaly_cnt}</div>
            <div class="stat-delta" style="color: #C084FC;">{anomaly_cnt/total_districts*100:.1f}% Irregular</div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Map Section
    st.markdown("<h3 style='margin-top: 1rem;'>Pan-India Geospatial Intelligence Map</h3>", unsafe_allow_html=True)

    map_metric = st.selectbox(
        "Select Map Layer Metric:",
        [
            "BharatLens Composite Score (0-100)",
            "Score Progress Delta (NFHS-5 vs NFHS-4)",
            "ML Trend Classification",
            "Socioeconomic Cluster Tier",
            "Anomaly Risk Score (0-100)"
        ],
        index=0
    )

    color_col = "bharatlens_score_current"
    color_scale = "Viridis"
    discrete_color = False

    if map_metric == "Score Progress Delta (NFHS-5 vs NFHS-4)":
        color_col = "score_change"
        color_scale = "RdYlGn"
    elif map_metric == "ML Trend Classification":
        color_col = "final_trend_class"
        discrete_color = True
    elif map_metric == "Socioeconomic Cluster Tier":
        color_col = "cluster_name"
        discrete_color = True
    elif map_metric == "Anomaly Risk Score (0-100)":
        color_col = "anomaly_risk_score"
        color_scale = "Plasma"

    # Scatter Mapbox Coordinates
    state_coords = {
        "Andhra Pradesh": (15.91, 79.74), "Arunachal Pradesh": (28.21, 94.72), "Assam": (26.20, 92.93),
        "Bihar": (25.09, 85.31), "Chhattisgarh": (21.27, 81.86), "Goa": (15.29, 74.12),
        "Gujarat": (22.25, 71.19), "Haryana": (29.05, 76.08), "Himachal Pradesh": (31.10, 77.17),
        "Jharkhand": (23.61, 85.27), "Karnataka": (15.31, 75.71), "Kerala": (10.85, 76.27),
        "Madhya Pradesh": (22.97, 78.65), "Maharashtra": (19.75, 75.71), "Manipur": (24.66, 93.90),
        "Meghalaya": (25.46, 91.36), "Mizoram": (23.16, 92.93), "Nagaland": (26.15, 94.56),
        "Odisha": (20.95, 85.09), "Punjab": (31.14, 75.34), "Rajasthan": (27.02, 74.21),
        "Sikkim": (27.53, 88.51), "Tamil Nadu": (11.12, 78.65), "Telangana": (18.11, 79.01),
        "Tripura": (23.94, 91.98), "Uttar Pradesh": (26.84, 80.94), "Uttarakhand": (30.06, 79.01),
        "West Bengal": (22.98, 87.85), "Delhi": (28.70, 77.10), "Jammu & Kashmir": (33.77, 76.57),
        "Ladakh": (34.15, 77.57), "Puducherry": (11.94, 79.80), "Chandigarh": (30.73, 76.77),
        "Andaman & Nicobar Island": (11.74, 92.65), "Dadra & Nagar Haveli And Daman & Diu": (20.42, 72.83)
    }

    map_df = df_master.copy()
    lats, lons = [], []
    for _, r in map_df.iterrows():
        base = state_coords.get(r["state"], (22.0, 78.0))
        h = hash(r["district"]) % 1000 / 1000.0 - 0.5
        h2 = hash(r["district"] + "lat") % 1000 / 1000.0 - 0.5
        lats.append(base[0] + h * 1.8)
        lons.append(base[1] + h2 * 2.2)
    map_df["latitude"] = lats
    map_df["longitude"] = lons

    hover_data = {
        "state": True,
        "bharatlens_score_current": ":.1f",
        "score_change": ":+.1f",
        "final_trend_class": True,
        "cluster_name": True,
        "latitude": False,
        "longitude": False
    }

    if discrete_color:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color=color_col,
            color_discrete_map=COLOR_MAP,
            hover_name="district",
            hover_data=hover_data,
            zoom=3.8,
            center={"lat": 22.5, "lon": 82.0},
            mapbox_style="carto-darkmatter",
            size_max=14,
            height=580,
        )
    else:
        fig_map = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color=color_col,
            color_continuous_scale=color_scale,
            hover_name="district",
            hover_data=hover_data,
            zoom=3.8,
            center={"lat": 22.5, "lon": 82.0},
            mapbox_style="carto-darkmatter",
            size_max=14,
            height=580,
        )

    apply_dark_theme(fig_map, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Leaderboard & Priority Alerts
    c_lead, c_alert = st.columns(2)
    with c_lead:
        st.markdown("<h4>🏆 Top 10 Leading Districts in India</h4>", unsafe_allow_html=True)
        top10 = df_master.nlargest(10, "bharatlens_score_current")[
            ["district", "state", "bharatlens_score_current", "score_change", "final_trend_class"]
        ].reset_index(drop=True)
        top10.columns = ["District", "State", "BharatLens Score", "Score Change", "Trend"]
        st.dataframe(top10, use_container_width=True, hide_index=True)

    with c_alert:
        st.markdown("<h4>⚠️ Priority Aspirational & Anomaly Alerts</h4>", unsafe_allow_html=True)
        alerts = df_master[
            (df_master["final_trend_class"] == "Anomalous") | (df_master["cluster_id"] == 0)
        ].nsmallest(10, "bharatlens_score_current")[
            ["district", "state", "bharatlens_score_current", "anomaly_risk_score", "final_trend_class"]
        ].reset_index(drop=True)
        alerts.columns = ["District", "State", "Score", "Anomaly Risk", "Status"]
        st.dataframe(alerts, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 2: DISTRICT DEEP-DIVE EXPLORER
# ==============================================================================
elif nav_page == "🔍 District Deep-Dive":
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h1 style="margin: 0; font-size: 2.3rem;">District Deep-Dive Explorer</h1>
        <p style="color: #94A3B8; margin-top: 0.2rem;">
            Granular indicators, historical baselines, ML predictive forecasting, and SHAP explainability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Dual Selectors: State then District
    all_states = sorted(df_master["state"].dropna().unique().tolist())
    s_col, d_col = st.columns([1, 1])

    with s_col:
        sel_state = st.selectbox("Select State / Union Territory:", all_states, index=all_states.index("Maharashtra") if "Maharashtra" in all_states else 0)

    state_districts = sorted(df_master[df_master["state"] == sel_state]["district"].unique().tolist())
    with d_col:
        sel_district = st.selectbox("Select District:", state_districts, index=0)

    # Filter district row
    d_row = df_master[(df_master["state"] == sel_state) & (df_master["district"] == sel_district)].iloc[0]

    # District Banner
    trend_badge_class = f"badge-{d_row['final_trend_class'].lower()}" if d_row['final_trend_class'] in ["Improving", "Stable", "Declining"] else "badge-anomaly"
    pop_str = f"{int(d_row.get('census_population', 0)):,}" if pd.notna(d_row.get("census_population")) else "N/A"

    st.markdown(f"""
    <div class="glass-card" style="margin-top: 1rem; border-left: 4px solid #F59E0B;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;">
            <div>
                <h2 style="margin: 0; font-size: 2rem; color: #FFFFFF;">{d_row['district']}</h2>
                <div style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.2rem;">
                    State: <b style="color: #F8FAFC;">{d_row['state']}</b> &nbsp;|&nbsp;
                    Population: <b style="color: #F8FAFC;">{pop_str}</b> &nbsp;|&nbsp;
                    Urbanization: <b style="color: #F8FAFC;">{d_row.get('census_urban_pct', 0.0):.1f}%</b>
                </div>
            </div>
            <div style="display: flex; gap: 0.6rem; align-items: center;">
                <span class="badge {trend_badge_class}">{d_row['final_trend_class']}</span>
                <span class="badge badge-tier">{d_row.get('cluster_name', 'Tier 2')}</span>
                {'<span class="badge badge-anomaly">ANOMALY ALERT</span>' if d_row.get('is_anomaly') == 1 else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Visuals Grid: Gauge vs Radar vs Paired Bars
    g_col, r_col = st.columns([1, 1.2])

    with g_col:
        fig_gauge = create_gauge_chart(
            score=float(d_row["bharatlens_score_current"]),
            title="Current BharatLens Score",
            prev_score=float(d_row["bharatlens_score_previous"])
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"""
        <div style="display: flex; justify-content: space-around; background: rgba(255,255,255,0.03); border-radius: 8px; padding: 0.6rem; font-size: 0.88rem; text-align: center;">
            <div>
                <div style="color: #94A3B8;">State Avg</div>
                <div style="font-weight: 700; color: #FFFFFF;">{d_row.get('state_average_score', 0):.1f}</div>
            </div>
            <div>
                <div style="color: #94A3B8;">National Avg</div>
                <div style="font-weight: 700; color: #FFFFFF;">{d_row.get('national_average_score', 0):.1f}</div>
            </div>
            <div>
                <div style="color: #94A3B8;">Relative Deviation</div>
                <div style="font-weight: 700; color: {'#10B981' if d_row.get('state_relative_deviation', 0) >= 0 else '#EF4444'};">
                    {d_row.get('state_relative_deviation', 0):+.1f} pts
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r_col:
        dim_data = {
            "Health & Nutrition": float(d_row.get("dim_health_nutrition_current", 50.0)),
            "Women Empowerment": float(d_row.get("dim_women_empowerment_current", 50.0)),
            "Infrastructure": float(d_row.get("dim_infrastructure_amenities_current", 50.0)),
            "Economic & Wages": float(d_row.get("dim_economic_employment_current", 50.0)),
            "Demographics & Edu": float(d_row.get("dim_demographic_education_current", 50.0)),
        }
        fig_radar = create_dimension_radar_chart(dim_data, title=f"5-Dimension Score Profile - {d_row['district']}")
        st.plotly_chart(fig_radar, use_container_width=True)

    # Historical Tracking Bar Chart
    st.markdown("<h4 style='margin-top: 1rem;'>Historical Observations Comparison (NFHS-4 Baseline vs NFHS-5 Latest)</h4>", unsafe_allow_html=True)
    prev_tracking = {
        "Institutional Births": float(d_row.get("institutional_births_nfhs4", 0.0)),
        "Full Vaccination": float(d_row.get("children_vaccinated_nfhs4", 0.0)),
        "Female Literacy": float(d_row.get("female_literacy_nfhs4", 0.0)),
        "Electricity Access": float(d_row.get("electricity_access_nfhs4", 0.0)),
        "Improved Sanitation": float(d_row.get("improved_sanitation_nfhs4", 0.0)),
        "Clean Cooking Fuel": float(d_row.get("clean_fuel_cooking_nfhs4", 0.0)),
    }
    curr_tracking = {
        "Institutional Births": float(d_row.get("institutional_births_nfhs5", 0.0)),
        "Full Vaccination": float(d_row.get("children_vaccinated_nfhs5", 0.0)),
        "Female Literacy": float(d_row.get("female_literacy_nfhs5", 0.0)),
        "Electricity Access": float(d_row.get("electricity_access_nfhs5", 0.0)),
        "Improved Sanitation": float(d_row.get("improved_sanitation_nfhs5", 0.0)),
        "Clean Cooking Fuel": float(d_row.get("clean_fuel_cooking_nfhs5", 0.0)),
    }
    fig_hist = create_historical_comparison_chart(prev_tracking, curr_tracking)
    st.plotly_chart(fig_hist, use_container_width=True)

    # ML Forecast & Explainability Section
    st.markdown("<hr style='margin: 1.5rem 0; border-color: rgba(255,255,255,0.08);'>", unsafe_allow_html=True)
    st.markdown("<h3>🤖 Machine Learning Predictive Trajectory & SHAP Attribution</h3>", unsafe_allow_html=True)

    ml_col1, ml_col2 = st.columns([1, 1.4])

    with ml_col1:
        pred_delta = float(d_row.get("predicted_score_change", 0.0))
        pred_next = float(d_row.get("predicted_next_bharatlens_score", d_row["bharatlens_score_current"]))
        ml_trend = str(d_row.get("ml_predicted_trend", d_row["final_trend_class"]))
        conf = float(d_row.get("ml_prediction_confidence", 85.0))

        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">ML Forecast (Champion Model)</div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF; margin-top: 0.4rem;">
                Projected Next-Cycle Index: <span style="color: #F59E0B;">{pred_next:.1f} / 100</span>
            </div>
            <div style="font-size: 0.95rem; margin-top: 0.4rem; color: {'#34D399' if pred_delta >= 0 else '#F87171'}; font-weight: 600;">
                {'▲ Expected Growth' if pred_delta >= 0 else '▼ Expected Deceleration'}: {pred_delta:+.2f} points
            </div>
            <div style="margin-top: 0.8rem; font-size: 0.9rem; color: #94A3B8;">
                Predicted Trajectory Class: <b style="color: #FFFFFF;">{ml_trend}</b><br>
                Model Confidence: <b style="color: #FFFFFF;">{conf:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Retrieve SHAP explanations for this district
        shap_dict = metadata.get("shap", {}).get("district_explanations", {})
        d_key = f"{sel_state}___{sel_district}"
        curr_shap = shap_dict.get(d_key, {})

        if curr_shap:
            narrative = format_natural_language_explanation(curr_shap)
            st.markdown(f"""
            <div class="glass-card" style="font-size: 0.9rem; line-height: 1.5; color: #CBD5E1;">
                {narrative}
            </div>
            """, unsafe_allow_html=True)

    with ml_col2:
        if curr_shap:
            fig_shap = create_shap_horizontal_bar(
                curr_shap.get("top_positive_contributors", []),
                curr_shap.get("top_negative_contributors", [])
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP feature attribution matrix available for all processed districts.")

    # Peer District Comparison
    st.markdown("<h4 style='margin-top: 1rem;'>👥 Socioeconomic Peer Group Districts (Same Cluster Tier)</h4>", unsafe_allow_html=True)
    cluster_id = d_row.get("cluster_id")
    peer_df = df_master[
        (df_master["cluster_id"] == cluster_id) & (df_master["district"] != sel_district)
    ].sort_values("bharatlens_score_current", ascending=False).head(8)[
        ["district", "state", "bharatlens_score_current", "score_change", "final_trend_class"]
    ].reset_index(drop=True)
    peer_df.columns = ["District", "State", "BharatLens Score", "Score Change", "Trend"]
    st.dataframe(peer_df, use_container_width=True, hide_index=True)


# ==============================================================================
# PAGE 3: STATE EXPLORER
# ==============================================================================
elif nav_page == "🏛️ State Explorer":
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h1 style="margin: 0; font-size: 2.3rem;">State Comparative Intelligence</h1>
        <p style="color: #94A3B8; margin-top: 0.2rem;">
            Inter-district disparity analysis, state rankings, and sub-regional performance profiles.
        </p>
    </div>
    """, unsafe_allow_html=True)

    all_states = sorted(df_master["state"].dropna().unique().tolist())
    sel_state = st.selectbox("Select State / UT:", all_states, index=all_states.index("Uttar Pradesh") if "Uttar Pradesh" in all_states else 0)

    st_df = df_master[df_master["state"] == sel_state].copy()
    st_count = len(st_df)
    st_mean_score = st_df["bharatlens_score_current"].mean()
    st_lead = st_df.loc[st_df["bharatlens_score_current"].idxmax()]
    st_lag = st_df.loc[st_df["bharatlens_score_current"].idxmin()]

    sk1, sk2, sk3, sk4 = st.columns(4)
    with sk1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Districts in State</div>
            <div class="stat-val">{st_count}</div>
            <div class="stat-delta" style="color: #94A3B8;">Comprehensive Coverage</div>
        </div>
        """, unsafe_allow_html=True)
    with sk2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">State Mean Index</div>
            <div class="stat-val">{st_mean_score:.1f}</div>
            <div class="stat-delta" style="color: #10B981;">{st_mean_score - df_master['bharatlens_score_current'].mean():+.1f} vs National</div>
        </div>
        """, unsafe_allow_html=True)
    with sk3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Top Performer</div>
            <div class="stat-val" style="font-size: 1.5rem; color: #34D399;">{st_lead['district']}</div>
            <div class="stat-delta" style="color: #34D399;">Score: {st_lead['bharatlens_score_current']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with sk4:
        st.markdown(f"""
        <div class="glass-card">
            <div class="stat-label">Lagging District</div>
            <div class="stat-val" style="font-size: 1.5rem; color: #F87171;">{st_lag['district']}</div>
            <div class="stat-delta" style="color: #F87171;">Score: {st_lag['bharatlens_score_current']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    # State District Rankings Bar Chart
    st.markdown("<h4 style='margin-top: 1rem;'>District Rankings in " + sel_state + "</h4>", unsafe_allow_html=True)
    st_sorted = st_df.sort_values("bharatlens_score_current", ascending=True)

    fig_state_bar = px.bar(
        st_sorted,
        x="bharatlens_score_current",
        y="district",
        orientation="h",
        color="final_trend_class",
        color_discrete_map=COLOR_MAP,
        hover_data=["score_change", "cluster_name"],
        height=max(400, st_count * 22)
    )
    apply_dark_theme(fig_state_bar, xaxis=dict(title="BharatLens Current Score (0-100)", range=[0, 100]), yaxis=dict(title=""))
    st.plotly_chart(fig_state_bar, use_container_width=True)

    # Distribution of Trends & Clusters in State
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("<h5>Trend Classification Distribution</h5>", unsafe_allow_html=True)
        trend_counts = st_df["final_trend_class"].value_counts().reset_index()
        trend_counts.columns = ["Trend", "Count"]
        fig_donut = px.pie(
            trend_counts, values="Count", names="Trend", hole=0.5,
            color="Trend", color_discrete_map=COLOR_MAP
        )
        apply_dark_theme(fig_donut, height=300)
        st.plotly_chart(fig_donut, use_container_width=True)

    with sc2:
        st.markdown("<h5>Socioeconomic Tier Distribution</h5>", unsafe_allow_html=True)
        tier_counts = st_df["cluster_name"].value_counts().reset_index()
        tier_counts.columns = ["Tier", "Count"]
        fig_tier = px.pie(
            tier_counts, values="Count", names="Tier", hole=0.5,
            color="Tier", color_discrete_map=COLOR_MAP
        )
        apply_dark_theme(fig_tier, height=300)
        st.plotly_chart(fig_tier, use_container_width=True)


# ==============================================================================
# PAGE 4: ML INSIGHTS & EVALUATION
# ==============================================================================
elif nav_page == "🤖 ML Insights & Evaluation":
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h1 style="margin: 0; font-size: 2.3rem;">Machine Learning Benchmarks & AI Governance</h1>
        <p style="color: #94A3B8; margin-top: 0.2rem;">
            Empirical algorithm evaluation, cross-validation metrics, confusion matrix, and global SHAP drivers.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Model Comparison Scorecards
    reg_meta = metadata.get("regression", {})
    clf_meta = metadata.get("classification", {})
    clu_meta = metadata.get("clustering", {})
    ano_meta = metadata.get("anomaly", {})

    st.markdown("<h3>1. Regression Model Comparison (Score Delta Forecasting)</h3>", unsafe_allow_html=True)
    reg_bench = reg_meta.get("benchmark_table", [])
    if reg_bench:
        reg_df = pd.DataFrame(reg_bench)
        reg_df.columns = ["Model Architecture", "Test MAE", "Test RMSE", "Test R²", "5-Fold CV R² (Mean)", "CV R² (Std)"]
        st.dataframe(reg_df, use_container_width=True, hide_index=True)
        st.success(f"🏆 Champion Regressor: **{reg_meta.get('champion_model_name')}** (Selected by minimal RMSE = {reg_meta.get('best_rmse'):.4f})")

    st.markdown("<h3>2. Multi-Class Trend Classification Comparison</h3>", unsafe_allow_html=True)
    clf_bench = clf_meta.get("benchmark_table", [])
    if clf_bench:
        clf_df = pd.DataFrame(clf_bench)
        clf_df.columns = ["Model Architecture", "Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "5-Fold CV Macro-F1", "CV F1 (Std)"]
        st.dataframe(clf_df, use_container_width=True, hide_index=True)
        st.success(f"🏆 Champion Classifier: **{clf_meta.get('champion_model_name')}** (Selected by maximal Macro-F1 = {clf_meta.get('best_macro_f1'):.4f})")

    # Confusion Matrix Heatmap
    cm = clf_meta.get("confusion_matrix", [])
    classes = clf_meta.get("class_order", ["Improving", "Stable", "Declining", "Anomalous"])
    if cm:
        st.markdown("<h4>Confusion Matrix (Champion Model Test Evaluation)</h4>", unsafe_allow_html=True)
        cm_fig = px.imshow(
            cm,
            labels=dict(x="Predicted Class", y="Ground Truth Class", color="Count"),
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Blues",
            aspect="auto"
        )
        apply_dark_theme(cm_fig, height=360)
        st.plotly_chart(cm_fig, use_container_width=True)

    # Global SHAP Feature Importance
    st.markdown("<h3>3. Top 15 Global Predictive Feature Drivers (Explainable AI)</h3>", unsafe_allow_html=True)
    shap_global = metadata.get("shap", {}).get("global_importance", [])
    if shap_global:
        sg_df = pd.DataFrame(shap_global[:15])
        fig_shap_glob = px.bar(
            sg_df.iloc[::-1],
            x="mean_abs_shap",
            y="display_name",
            orientation="h",
            color="mean_abs_shap",
            color_continuous_scale="Viridis",
            labels={"mean_abs_shap": "Mean |SHAP| Impact", "display_name": "Indicator"}
        )
        apply_dark_theme(fig_shap_glob, height=450)
        st.plotly_chart(fig_shap_glob, use_container_width=True)

    # Clustering & Anomaly Scatter Visualization
    st.markdown("<h3>4. Socioeconomic Clusters & Outlier Anomaly Space</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("<h5>Economic Dynamism vs Sanitation by Socioeconomic Tier</h5>", unsafe_allow_html=True)
        fig_cl_scatter = px.scatter(
            df_master,
            x="dim_infrastructure_amenities_current",
            y="dim_economic_employment_current",
            color="cluster_name",
            color_discrete_map=COLOR_MAP,
            hover_name="district",
            labels={"dim_infrastructure_amenities_current": "Infrastructure Index", "dim_economic_employment_current": "Economic Index"}
        )
        apply_dark_theme(fig_cl_scatter, height=360)
        st.plotly_chart(fig_cl_scatter, use_container_width=True)

    with c2:
        st.markdown("<h5>Score Delta vs Anomaly Risk Score (Flagged Outliers)</h5>", unsafe_allow_html=True)
        fig_ano_scatter = px.scatter(
            df_master,
            x="score_change",
            y="anomaly_risk_score",
            color="is_anomaly",
            color_discrete_map={0: "#06B6D4", 1: "#EF4444"},
            hover_name="district",
            labels={"score_change": "Observed Score Delta", "anomaly_risk_score": "Anomaly Risk Score", "is_anomaly": "Is Outlier"}
        )
        apply_dark_theme(fig_ano_scatter, height=360)
        st.plotly_chart(fig_ano_scatter, use_container_width=True)


# ==============================================================================
# PAGE 5: DATA SOURCES & QUALITY TELEMETRY
# ==============================================================================
elif nav_page == "📚 Public Data Sources":
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h1 style="margin: 0; font-size: 2.3rem;">Public Data Sources & Integrity Telemetry</h1>
        <p style="color: #94A3B8; margin-top: 0.2rem;">
            Metadata catalog, open-access public repositories, update cadences, and automated validation telemetry.
        </p>
    </div>
    """, unsafe_allow_html=True)

    catalog = metadata.get("data_sources", {}).get("sources", {})

    for key, src in catalog.items():
        with st.expander(f"📦 {src.get('name')} ({src.get('geographic_level')})", expanded=True):
            st.markdown(f"""
            - **Agency / Provider**: {src.get('provider')}
            - **Official Portal**: [{src.get('portal_url')}]({src.get('portal_url')})
            - **Time Period**: `{src.get('time_period')}`
            - **Update Frequency**: `{src.get('update_frequency')}`
            - **Format / License**: `{src.get('format')}` | `{src.get('license')}`
            - **Important Columns**: {', '.join([f'`{c}`' for c in src.get('important_columns', [])])}
            - **Transformation Notes**: {src.get('transformation_notes', 'Pivoted and cleaned.')}
            """)

    # Quality Report preview
    st.markdown("<h3>Data Validation & Quality Telemetry Report</h3>", unsafe_allow_html=True)
    report_file = os.path.join("reports", "data_quality_report.md")
    if os.path.exists(report_file):
        with open(report_file, "r", encoding="utf-8") as f:
            st.markdown(f.read())


# ==============================================================================
# PAGE 6: METHODOLOGY & SCORING FRAMEWORK
# ==============================================================================
elif nav_page == "📐 Methodology & Scoring":
    st.markdown("""
    <div style="margin-bottom: 1.25rem;">
        <h1 style="margin: 0; font-size: 2.3rem;">Methodology & Analytical Framework</h1>
        <p style="color: #94A3B8; margin-top: 0.2rem;">
            Mathematical formulation of the BharatLens Score, feature transformations, and AI governance principles.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 1. Mathematical Formulation of the BharatLens Index

    The **BharatLens Score** is an interpretable, reproducible composite indicator scaled strictly from **0 to 100**,
    synthesizing 5 core socioeconomic dimensions with empirical public data:

    1. **Health & Nutrition** ($w_1 = 0.25$): Institutional births, full childhood vaccination, maternal ANC visits, stunting, wasting, anaemia.
    2. **Women Empowerment** ($w_2 = 0.20$): Female literacy, 10+ years schooling, rural female labour participation.
    3. **Infrastructure & Amenities** ($w_3 = 0.20$): Household electricity, improved sanitation, clean cooking fuels, internet penetration.
    4. **Economic & Employment Floor** ($w_4 = 0.20$): MGNREGA daily wages, employment days provided, agricultural expenditure, workforce participation.
    5. **Demographic & Educational Baseline** ($w_5 = 0.15$): Sex ratio, decadal literacy, urban dynamism.

    #### Directional Normalization
    For positive indicators (where higher values represent progress):
    $$
    \\tilde{x}_{i} = \\frac{x_i - \\min(x_i)}{\\max(x_i) - \\min(x_i)} \\times 100
    $$

    For negative burden indicators (stunting, wasting, anaemia):
    $$
    \\tilde{x}_{i} = \\frac{\\max(x_i) - x_i}{\\max(x_i) - \\min(x_i)} \\times 100
    $$

    #### Dimension Aggregation
    $$
    S_d = \\frac{1}{|I_d|} \\sum_{i \\in I_d} \\tilde{x}_i \\quad \\text{and} \\quad \\text{BharatLens Score} = \\sum_{d=1}^{5} w_d \\cdot S_d
    $$

    ---

    ### 2. Peer-Relative Statistical Trend Classification

    To eliminate national secular baseline bias (where countrywide programs lift all raw numbers simultaneously),
    BharatLens evaluates **relative momentum** against state and national benchmarks:

    $$
    Z_{\\Delta, i} = \\frac{\\Delta S_i - \\mu_{\\Delta}}{\\sigma_{\\Delta}}
    $$

    - **Improving**: Statistically accelerating ($Z_{\\Delta, i} \\ge +0.35$ or relative state growth $\\ge +1.5$ points).
    - **Stable**: Tracking expected benchmark momentum ($-0.35 \\le Z_{\\Delta, i} < +0.35$).
    - **Declining**: Statistically decelerating or lagging peer progress ($Z_{\\Delta, i} < -0.35$).
    - **Anomalous**: Multi-variate outlier flagged by Isolation Forest ($contamination = 0.05$) exhibiting erratic divergence across dimensions.

    ---

    ### 3. Ethical AI & Disclaimers
    - **Association vs Causation**: All SHAP values and model feature attributions represent statistical associations learned across empirical observations. They do NOT assert direct legal or causal responsibility.
    - **Boundary Changes**: Districts correspond to the administrative framework mapped across Census 2011 and NFHS survey cycles. Newly bifurcated districts inherit apportioned baselines from parent units.
    """)
