import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, send_file
import yaml

app = Flask(__name__)

# State coordinates proxy for map rendering
STATE_COORDINATES = {
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

# Cache Data Assets on Startup
DATA_PATH = os.path.join("data", "processed", "district_master_ml_results.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("data", "processed", "district_features_dataset.csv")

DF_MASTER = pd.read_csv(DATA_PATH)

def load_json_meta(filename: str) -> Dict[str, Any]:
    path = os.path.join("models", "metadata", filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

REG_META = load_json_meta("regression_metadata.json")
CLF_META = load_json_meta("classification_metadata.json")
CLU_META = load_json_meta("clustering_metadata.json")
ANO_META = load_json_meta("anomaly_metadata.json")
SHAP_META = load_json_meta("shap_explanations.json")

DATA_SOURCES_CFG = {}
if os.path.exists("config/data_sources.yaml"):
    with open("config/data_sources.yaml", "r", encoding="utf-8") as f:
        DATA_SOURCES_CFG = yaml.safe_load(f).get("sources", {})


# -----------------------------------------------------------------------------
# Frontend Routes
# -----------------------------------------------------------------------------
@app.route("/")
def index():
    """Serves the India.gov.in themed main intelligence portal."""
    return render_template("index.html")


# -----------------------------------------------------------------------------
# RESTful API Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/summary")
def api_summary():
    """Returns National KPI counters, top leading, priority alert districts, and map coordinates."""
    total_dists = len(DF_MASTER)
    nat_avg_score = float(DF_MASTER["bharatlens_score_current"].mean())
    nat_avg_delta = float(DF_MASTER["score_change"].mean())

    improving = int((DF_MASTER["final_trend_class"] == "Improving").sum())
    stable = int((DF_MASTER["final_trend_class"] == "Stable").sum())
    declining = int((DF_MASTER["final_trend_class"] == "Declining").sum())
    anomalies = int((DF_MASTER["final_trend_class"] == "Anomalous").sum())

    # Top 10 Leading Districts
    top10_df = DF_MASTER.nlargest(10, "bharatlens_score_current")
    top_10 = [{
        "district": r["district"],
        "state": r["state"],
        "score": round(float(r["bharatlens_score_current"]), 1),
        "delta": round(float(r["score_change"]), 1),
        "trend": r["final_trend_class"],
        "tier": r.get("cluster_name", "Tier 4")
    } for _, r in top10_df.iterrows()]

    # Top 10 Alerts (Anomalous or Lowest Scores)
    alert_df = DF_MASTER[
        (DF_MASTER["final_trend_class"] == "Anomalous") | (DF_MASTER["cluster_id"] == 0)
    ].nsmallest(10, "bharatlens_score_current")
    top_alerts = [{
        "district": r["district"],
        "state": r["state"],
        "score": round(float(r["bharatlens_score_current"]), 1),
        "anomaly_risk": round(float(r.get("anomaly_risk_score", 0.0)), 1),
        "trend": r["final_trend_class"],
        "tier": r.get("cluster_name", "Tier 1")
    } for _, r in alert_df.iterrows()]

    # Map Points
    map_points = []
    np.random.seed(42)
    for _, r in DF_MASTER.iterrows():
        base = STATE_COORDINATES.get(r["state"], (22.0, 78.0))
        h1 = (hash(r["district"]) % 1000 / 1000.0 - 0.5) * 1.8
        h2 = (hash(r["district"] + "lon") % 1000 / 1000.0 - 0.5) * 2.2
        map_points.append({
            "district": r["district"],
            "state": r["state"],
            "lat": round(base[0] + h1, 3),
            "lon": round(base[1] + h2, 3),
            "score": round(float(r["bharatlens_score_current"]), 1),
            "delta": round(float(r["score_change"]), 1),
            "trend": r["final_trend_class"],
            "tier": r.get("cluster_name", "Tier 2"),
            "cluster_id": int(r.get("cluster_id", 1)),
            "anomaly_risk": round(float(r.get("anomaly_risk_score", 0.0)), 1)
        })

    return jsonify({
        "total_districts": total_dists,
        "national_avg_score": round(nat_avg_score, 1),
        "national_avg_delta": round(nat_avg_delta, 1),
        "improving_count": improving,
        "stable_count": stable,
        "declining_count": declining,
        "anomaly_count": anomalies,
        "top_10_leading": top_10,
        "top_10_alerts": top_alerts,
        "map_points": map_points
    })


@app.route("/api/states")
def api_states():
    """Returns sorted list of all States & Union Territories."""
    states = sorted(DF_MASTER["state"].dropna().unique().tolist())
    return jsonify(states)


@app.route("/api/districts")
def api_districts():
    """Returns list of districts, optionally filtered by state."""
    state = request.args.get("state")
    if state:
        filtered = DF_MASTER[DF_MASTER["state"] == state]
    else:
        filtered = DF_MASTER

    results = [{
        "district": r["district"],
        "state": r["state"],
        "score": round(float(r["bharatlens_score_current"]), 1),
        "trend": r["final_trend_class"],
        "tier": r.get("cluster_name", "Tier 2")
    } for _, r in filtered.sort_values("district").iterrows()]
    return jsonify(results)


@app.route("/api/district/<state>/<district>")
def api_district_detail(state, district):
    """Returns complete granular intelligence profile for a specific district."""
    mask = (DF_MASTER["state"].str.lower() == state.lower()) & (DF_MASTER["district"].str.lower() == district.lower())
    if not mask.any():
        return jsonify({"error": "District not found"}), 404

    r = DF_MASTER[mask].iloc[0]

    # Peer group districts (same cluster_id, excluding current)
    cid = r.get("cluster_id")
    peer_mask = (DF_MASTER["cluster_id"] == cid) & (DF_MASTER["district"] != r["district"])
    peers_df = DF_MASTER[peer_mask].nlargest(8, "bharatlens_score_current")
    peers = [{
        "district": p["district"],
        "state": p["state"],
        "score": round(float(p["bharatlens_score_current"]), 1),
        "delta": round(float(p["score_change"]), 1),
        "trend": p["final_trend_class"]
    } for _, p in peers_df.iterrows()]

    # SHAP explanations lookup
    d_key = f"{r['state']}___{r['district']}"
    shap_data = SHAP_META.get("district_explanations", {}).get(d_key, {})

    narrative = ""
    top_pos = []
    top_neg = []
    if shap_data:
        from src.explainability.shap_analysis import format_natural_language_explanation
        narrative = format_natural_language_explanation(shap_data)
        top_pos = shap_data.get("top_positive_contributors", [])
        top_neg = shap_data.get("top_negative_contributors", [])

    data = {
        "state": r["state"],
        "district": r["district"],
        "population": int(r.get("census_population", 0)) if pd.notna(r.get("census_population")) else None,
        "urban_pct": round(float(r.get("census_urban_pct", 0.0)), 1),
        "bharatlens_score": round(float(r["bharatlens_score_current"]), 1),
        "score_previous": round(float(r["bharatlens_score_previous"]), 1),
        "score_change": round(float(r["score_change"]), 1),
        "historical_average": round(float(r.get("historical_average_score", 0.0)), 1),
        "state_average": round(float(r.get("state_average_score", 0.0)), 1),
        "state_relative_deviation": round(float(r.get("state_relative_deviation", 0.0)), 1),
        "national_average": round(float(r.get("national_average_score", 56.0)), 1),
        "national_relative_deviation": round(float(r.get("national_relative_deviation", 0.0)), 1),
        "relative_growth_zscore": round(float(r.get("relative_growth_zscore", 0.0)), 2),
        "trend": r["final_trend_class"],
        "cluster_id": int(r.get("cluster_id", 1)),
        "cluster_name": r.get("cluster_name", "Tier 2"),
        "is_anomaly": int(r.get("is_anomaly", 0)),
        "anomaly_risk": round(float(r.get("anomaly_risk_score", 0.0)), 1),
        "dimensions": {
            "health_nutrition": round(float(r.get("dim_health_nutrition_current", 50.0)), 1),
            "women_empowerment": round(float(r.get("dim_women_empowerment_current", 50.0)), 1),
            "infrastructure": round(float(r.get("dim_infrastructure_amenities_current", 50.0)), 1),
            "economic_employment": round(float(r.get("dim_economic_employment_current", 50.0)), 1),
            "demographics_education": round(float(r.get("dim_demographic_education_current", 50.0)), 1),
        },
        "historical_tracking": {
            "institutional_births_prev": round(float(r.get("institutional_births_nfhs4", 0.0)), 1),
            "institutional_births_curr": round(float(r.get("institutional_births_nfhs5", 0.0)), 1),
            "children_vaccinated_prev": round(float(r.get("children_vaccinated_nfhs4", 0.0)), 1),
            "children_vaccinated_curr": round(float(r.get("children_vaccinated_nfhs5", 0.0)), 1),
            "female_literacy_prev": round(float(r.get("female_literacy_nfhs4", 0.0)), 1),
            "female_literacy_curr": round(float(r.get("female_literacy_nfhs5", 0.0)), 1),
            "electricity_access_prev": round(float(r.get("electricity_access_nfhs4", 0.0)), 1),
            "electricity_access_curr": round(float(r.get("electricity_access_nfhs5", 0.0)), 1),
            "improved_sanitation_prev": round(float(r.get("improved_sanitation_nfhs4", 0.0)), 1),
            "improved_sanitation_curr": round(float(r.get("improved_sanitation_nfhs5", 0.0)), 1),
            "clean_fuel_cooking_prev": round(float(r.get("clean_fuel_cooking_nfhs4", 0.0)), 1),
            "clean_fuel_cooking_curr": round(float(r.get("clean_fuel_cooking_nfhs5", 0.0)), 1),
        },
        "ml_forecast": {
            "projected_next_score": round(float(r.get("predicted_next_bharatlens_score", r["bharatlens_score_current"])), 1),
            "predicted_delta": round(float(r.get("predicted_score_change", 0.0)), 2),
            "predicted_trend": str(r.get("ml_predicted_trend", r["final_trend_class"])),
            "confidence_pct": round(float(r.get("ml_prediction_confidence", 85.0)), 1)
        },
        "shap_analysis": {
            "narrative": narrative,
            "top_positive": top_pos,
            "top_negative": top_neg
        },
        "peer_districts": peers
    }
    return jsonify(data)


@app.route("/api/state/<state>")
def api_state_detail(state):
    """Returns state summary, rankings, and sub-regional distributions."""
    st_df = DF_MASTER[DF_MASTER["state"].str.lower() == state.lower()]
    if st_df.empty:
        return jsonify({"error": "State not found"}), 404

    count = len(st_df)
    mean_score = float(st_df["bharatlens_score_current"].mean())
    nat_mean = float(DF_MASTER["bharatlens_score_current"].mean())

    lead_row = st_df.loc[st_df["bharatlens_score_current"].idxmax()]
    lag_row = st_df.loc[st_df["bharatlens_score_current"].idxmin()]

    trend_counts = st_df["final_trend_class"].value_counts().to_dict()
    tier_counts = st_df["cluster_name"].value_counts().to_dict()

    ranked = st_df.sort_values("bharatlens_score_current", ascending=False)
    districts = [{
        "district": r["district"],
        "score": round(float(r["bharatlens_score_current"]), 1),
        "delta": round(float(r["score_change"]), 1),
        "trend": r["final_trend_class"],
        "tier": r.get("cluster_name", "Tier 2"),
        "is_anomaly": int(r.get("is_anomaly", 0))
    } for _, r in ranked.iterrows()]

    return jsonify({
        "state": state,
        "district_count": count,
        "state_mean_score": round(mean_score, 1),
        "state_vs_national": round(mean_score - nat_mean, 1),
        "leading_district": {
            "district": lead_row["district"],
            "score": round(float(lead_row["bharatlens_score_current"]), 1)
        },
        "lagging_district": {
            "district": lag_row["district"],
            "score": round(float(lag_row["bharatlens_score_current"]), 1)
        },
        "trend_breakdown": trend_counts,
        "tier_breakdown": tier_counts,
        "districts": districts
    })


@app.route("/api/compare")
def api_compare():
    """Returns side-by-side comparative analysis of two districts."""
    d1_key = request.args.get("d1", "")
    d2_key = request.args.get("d2", "")

    if "___" not in d1_key or "___" not in d2_key:
        return jsonify({"error": "Invalid district keys"}), 400

    s1, dist1 = d1_key.split("___", 1)
    s2, dist2 = d2_key.split("___", 1)

    r1_df = DF_MASTER[(DF_MASTER["state"] == s1) & (DF_MASTER["district"] == dist1)]
    r2_df = DF_MASTER[(DF_MASTER["state"] == s2) & (DF_MASTER["district"] == dist2)]

    if r1_df.empty or r2_df.empty:
        return jsonify({"error": "One or both districts not found"}), 404

    r1 = r1_df.iloc[0]
    r2 = r2_df.iloc[0]

    def extract_dim(row):
        return {
            "health_nutrition": round(float(row.get("dim_health_nutrition_current", 50.0)), 1),
            "women_empowerment": round(float(row.get("dim_women_empowerment_current", 50.0)), 1),
            "infrastructure": round(float(row.get("dim_infrastructure_amenities_current", 50.0)), 1),
            "economic_employment": round(float(row.get("dim_economic_employment_current", 50.0)), 1),
            "demographics_education": round(float(row.get("dim_demographic_education_current", 50.0)), 1),
        }

    return jsonify({
        "district_1": {
            "state": r1["state"],
            "district": r1["district"],
            "score": round(float(r1["bharatlens_score_current"]), 1),
            "delta": round(float(r1["score_change"]), 1),
            "trend": r1["final_trend_class"],
            "tier": r1.get("cluster_name", "Tier 2"),
            "dimensions": extract_dim(r1)
        },
        "district_2": {
            "state": r2["state"],
            "district": r2["district"],
            "score": round(float(r2["bharatlens_score_current"]), 1),
            "delta": round(float(r2["score_change"]), 1),
            "trend": r2["final_trend_class"],
            "tier": r2.get("cluster_name", "Tier 2"),
            "dimensions": extract_dim(r2)
        }
    })


@app.route("/api/ml-insights")
def api_ml_insights():
    """Returns model benchmark comparisons, confusion matrix, global SHAP drivers, and cluster profiles."""
    return jsonify({
        "regression_champion": REG_META.get("champion_model_name", "RidgeRegression"),
        "regression_benchmarks": REG_META.get("benchmark_table", []),
        "classification_champion": CLF_META.get("champion_model_name", "LogisticRegression"),
        "classification_benchmarks": CLF_META.get("benchmark_table", []),
        "confusion_matrix": CLF_META.get("confusion_matrix", []),
        "global_shap_importance": SHAP_META.get("global_importance", []),
        "cluster_profiles": CLU_META.get("profiles", {}),
        "anomaly_diagnostics": ANO_META.get("diagnostics", [])
    })


@app.route("/api/data-sources")
def api_data_sources():
    """Returns public data source catalog metadata."""
    return jsonify(DATA_SOURCES_CFG)


@app.route("/api/search")
def api_search():
    """Instant search for autocomplete across districts and states."""
    q = request.args.get("q", "").strip().lower()
    if len(q) < 2:
        return jsonify([])

    matches = DF_MASTER[
        DF_MASTER["district"].str.lower().str.contains(q) | DF_MASTER["state"].str.lower().str.contains(q)
    ].head(15)

    results = [{
        "district": r["district"],
        "state": r["state"],
        "score": round(float(r["bharatlens_score_current"]), 1),
        "trend": r["final_trend_class"]
    } for _, r in matches.iterrows()]

    return jsonify(results)


@app.route("/api/export-csv")
def api_export_csv():
    """Downloads the master results CSV dataset."""
    if os.path.exists(DATA_PATH):
        return send_file(DATA_PATH, as_attachment=True, download_name="BharatLens_ML_Districts_Master.csv")
    return jsonify({"error": "Data file not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==========================================================================")
    print(f"  BHARATLENS ML - FLASK WEB APP ONLINE")
    print(f"  India.gov.in National Portal Inspired UI")
    print(f"  Access locally at: http://localhost:{port}")
    print(f"==========================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
