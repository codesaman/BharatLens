import os
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import shap

from src.utils.logger import get_logger
from src.utils.io import save_json, load_json, load_model, load_yaml

logger = get_logger("ShapAnalysis")

MODELS_DIR = os.path.join("models", "saved")
METADATA_DIR = os.path.join("models", "metadata")

FEATURE_FRIENDLY_NAMES = {
    "feat_institutional_births_abs_change": "Change in Institutional Deliveries (%)",
    "feat_children_vaccinated_abs_change": "Change in Child Full Immunization (%)",
    "feat_stunting_under5_abs_change": "Change in Child Stunting Rate (%)",
    "feat_wasting_under5_abs_change": "Change in Child Wasting Rate (%)",
    "feat_anaemia_women_abs_change": "Change in Female Anaemia Prevalence (%)",
    "feat_mothers_anc_visits_abs_change": "Change in Antenatal Care Coverage (%)",
    "feat_female_literacy_abs_change": "Change in Female Literacy (%)",
    "feat_women_schooling_10yrs_abs_change": "Change in Women with 10+ Years Schooling (%)",
    "feat_electricity_access_abs_change": "Change in Household Electricity Access (%)",
    "feat_improved_sanitation_abs_change": "Change in Improved Sanitation Coverage (%)",
    "feat_clean_fuel_cooking_abs_change": "Change in Clean Fuel Cooking Adoption (%)",
    "feat_mgnrega_daily_wage_rs_state_diff": "MGNREGA Daily Wage Differential vs State (₹)",
    "feat_mgnrega_avg_days_per_hh_state_diff": "Rural Workdays per Household vs State",
    "feat_institutional_births_nfhs5_state_zscore": "Healthcare Delivery Intensity (State Z-score)",
    "feat_stunting_under5_nfhs5_state_zscore": "Child Malnutrition Burden (State Z-score)",
    "feat_economic_dynamism_index": "Economic Dynamism & Non-Farm Employment Index",
    "feat_clean_living_environment_index": "Clean Living Environment Index",
    "feat_female_literacy_parity_gap": "Female Literacy Parity Gap",
}

def clean_feature_display_name(feature_key: str) -> str:
    """Translates raw code feature keys into plain human-readable descriptions."""
    if feature_key in FEATURE_FRIENDLY_NAMES:
        return FEATURE_FRIENDLY_NAMES[feature_key]
    cleaned = feature_key.replace("feat_", "").replace("_", " ").title()
    return cleaned

def compute_and_cache_shap_explanations(
    model: Any,
    scaler: Any,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """Computes model-agnostic SHAP values (TreeExplainer or LinearExplainer) and district-level narrative drivers."""
    logger.info(f"Computing SHAP explanations for model type: {type(model).__name__}...")

    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    X_scaled = scaler.transform(X)

    # Determine optimal SHAP explainer
    model_type_str = type(model).__name__.lower()
    if "ridge" in model_type_str or "linear" in model_type_str or "lasso" in model_type_str or "logistic" in model_type_str:
        logger.info("Using SHAP LinearExplainer for linear model architecture.")
        explainer = shap.LinearExplainer(model, X_scaled)
    elif "forest" in model_type_str or "tree" in model_type_str or "gradient" in model_type_str or "xgb" in model_type_str:
        logger.info("Using SHAP TreeExplainer for tree ensemble architecture.")
        explainer = shap.TreeExplainer(model)
    else:
        logger.info("Using general SHAP Explainer.")
        explainer = shap.Explainer(model, X_scaled)

    shap_values = explainer.shap_values(X_scaled)

    # Handle multi-output or list formats
    if isinstance(shap_values, list):
        shap_matrix = np.array(shap_values[0])
    elif hasattr(shap_values, "values"):
        shap_matrix = np.array(shap_values.values)
    else:
        shap_matrix = np.array(shap_values)

    # Global feature importance (mean absolute SHAP)
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    top_global_indices = np.argsort(mean_abs_shap)[::-1]

    global_importance = []
    for idx in top_global_indices[:20]:
        f_name = feature_cols[idx]
        global_importance.append({
            "feature": f_name,
            "display_name": clean_feature_display_name(f_name),
            "mean_abs_shap": round(float(mean_abs_shap[idx]), 4)
        })

    # Expected base value
    exp_val = getattr(explainer, "expected_value", 0.0)
    if isinstance(exp_val, (list, np.ndarray)):
        base_val = float(exp_val[0])
    else:
        base_val = float(exp_val)

    # District-level explanations cache
    district_explanations = {}

    for i in range(len(df)):
        row = df.iloc[i]
        key = f"{row['state']}___{row['district']}"
        row_shap = shap_matrix[i]

        sorted_feat_indices = np.argsort(np.abs(row_shap))[::-1]

        positive_drivers = []
        negative_drivers = []

        for f_idx in sorted_feat_indices[:15]:
            val = float(row_shap[f_idx])
            feat_name = feature_cols[f_idx]
            disp_name = clean_feature_display_name(feat_name)
            raw_val = float(X.iloc[i, f_idx])

            item = {
                "feature": feat_name,
                "display_name": disp_name,
                "shap_value": round(val, 4),
                "actual_value": round(raw_val, 2)
            }

            if val > 0:
                positive_drivers.append(item)
            elif val < 0:
                negative_drivers.append(item)

        district_explanations[key] = {
            "state": row["state"],
            "district": row["district"],
            "base_value": round(base_val, 3),
            "predicted_score_change": round(float(row.get("predicted_score_change", 0.0)), 2),
            "top_positive_contributors": positive_drivers[:4],
            "top_negative_contributors": negative_drivers[:4],
        }

    shap_metadata = {
        "global_importance": global_importance,
        "base_value": round(base_val, 3),
        "district_explanations": district_explanations
    }

    os.makedirs(METADATA_DIR, exist_ok=True)
    save_json(shap_metadata, os.path.join(METADATA_DIR, "shap_explanations.json"))
    logger.info(f"SHAP explanation matrix generated and saved for {len(district_explanations)} districts.")

    return shap_metadata

def format_natural_language_explanation(district_shap: Dict[str, Any]) -> str:
    """Formats honest, disciplined natural language explanation based on top SHAP drivers."""
    pos = district_shap.get("top_positive_contributors", [])
    neg = district_shap.get("top_negative_contributors", [])
    pred_change = district_shap.get("predicted_score_change", 0.0)

    lines = []
    if pred_change >= 0:
        lines.append(f"The model projects an upward growth momentum of **+{pred_change:.2f} points** in overall development index.")
    else:
        lines.append(f"The model projects a decelerating trajectory of **{pred_change:.2f} points** relative to peer momentum.")

    if pos:
        lines.append("\n**Key factors associated with positive contributions to the prediction:**")
        for p in pos[:3]:
            lines.append(f"- **{p['display_name']}** (observed value: {p['actual_value']}), contributing +{p['shap_value']:.2f} to model score.")

    if neg:
        lines.append("\n**Key factors associated with downward contributions to the prediction:**")
        for n in neg[:3]:
            lines.append(f"- **{n['display_name']}** (observed value: {n['actual_value']}), pulling score down by {abs(n['shap_value']):.2f}.")

    lines.append("\n*Note: These factors reflect statistical feature attributions from the trained ML model and indicate strong association, not direct deterministic causation.*")
    return "\n".join(lines)
