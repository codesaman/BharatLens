import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger
from src.utils.io import save_dataframe, save_model, save_json

logger = get_logger("FeatureEngineering")

PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = os.path.join("models", "saved")

def engineer_district_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Engineers rich econometric, demographic, baseline delta, and state-relative features."""
    logger.info("Starting advanced feature engineering...")
    fe_df = df.copy()

    tracked_indicators = [
        "institutional_births", "children_vaccinated", "stunting_under5",
        "wasting_under5", "anaemia_women", "mothers_anc_visits",
        "female_literacy", "women_schooling_10yrs", "electricity_access",
        "improved_sanitation", "clean_fuel_cooking"
    ]

    # 1. Delta and Percentage Change features
    for ind in tracked_indicators:
        curr_col = f"{ind}_nfhs5"
        prev_col = f"{ind}_nfhs4"
        if curr_col in fe_df.columns and prev_col in fe_df.columns:
            # Absolute change
            delta_col = f"feat_{ind}_abs_change"
            fe_df[delta_col] = (fe_df[curr_col] - fe_df[prev_col]).round(3)

            # Percentage change relative to baseline
            pct_col = f"feat_{ind}_pct_growth"
            denom = fe_df[prev_col].abs().replace(0, np.nan)
            fe_df[pct_col] = (((fe_df[curr_col] - fe_df[prev_col]) / denom) * 100).fillna(0.0).clip(-200, 200).round(2)

            # Ratio to historical baseline
            fe_df[f"feat_{ind}_ratio_to_baseline"] = (fe_df[curr_col] / (fe_df[prev_col] + 1e-4)).clip(0.1, 10.0).round(3)

    # 2. State-relative deviations and Z-scores
    state_benchmark_cols = [
        "institutional_births_nfhs5", "children_vaccinated_nfhs5", "stunting_under5_nfhs5",
        "female_literacy_nfhs5", "electricity_access_nfhs5", "mgnrega_daily_wage_rs",
        "mgnrega_avg_days_per_hh", "census_literacy_rate"
    ]

    for col in state_benchmark_cols:
        if col in fe_df.columns:
            state_mean = fe_df.groupby("state")[col].transform("mean")
            state_std = fe_df.groupby("state")[col].transform("std").fillna(1.0).replace(0, 1.0)

            fe_df[f"feat_{col}_state_diff"] = (fe_df[col] - state_mean).round(3)
            fe_df[f"feat_{col}_state_zscore"] = ((fe_df[col] - state_mean) / state_std).clip(-4.0, 4.0).round(3)

    # 3. Socioeconomic Structural Ratios & Composite Proxies
    if "census_worker_rate" in fe_df.columns and "census_agri_worker_share" in fe_df.columns:
        fe_df["feat_non_agri_worker_pct"] = (100.0 - fe_df["census_agri_worker_share"]).clip(0, 100).round(2)
        fe_df["feat_economic_dynamism_index"] = (
            (fe_df["census_worker_rate"] * 0.4) +
            ((100.0 - fe_df["census_agri_worker_share"]) * 0.3) +
            (fe_df["census_urban_pct"] * 0.3)
        ).round(2)

    if "female_literacy_nfhs5" in fe_df.columns and "census_literacy_rate" in fe_df.columns:
        fe_df["feat_female_literacy_parity_gap"] = (fe_df["census_literacy_rate"] - fe_df["female_literacy_nfhs5"]).round(2)

    if "improved_sanitation_nfhs5" in fe_df.columns and "clean_fuel_cooking_nfhs5" in fe_df.columns:
        fe_df["feat_clean_living_environment_index"] = (
            (fe_df["improved_sanitation_nfhs5"] * 0.5) +
            (fe_df["clean_fuel_cooking_nfhs5"] * 0.5)
        ).round(2)

    # Nutrition vulnerability index
    if "stunting_under5_nfhs5" in fe_df.columns and "wasting_under5_nfhs5" in fe_df.columns:
        fe_df["feat_child_malnutrition_burden"] = (
            (fe_df["stunting_under5_nfhs5"] * 0.6) +
            (fe_df["wasting_under5_nfhs5"] * 0.4)
        ).round(2)

    # 4. Extract ML feature names
    feature_cols = [c for c in fe_df.columns if c.startswith("feat_")]
    logger.info(f"Feature engineering generated {len(feature_cols)} rich model features.")

    return fe_df, feature_cols

def build_and_save_feature_store(input_path: str = os.path.join(PROCESSED_DATA_DIR, "district_master_processed.csv")) -> pd.DataFrame:
    """Executes end-to-end feature pipeline, fits standardizer, and saves feature matrix."""
    from src.scoring.bharatlens_score import calculate_bharatlens_score

    logger.info(f"Loading processed master dataset from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Calculate BharatLens Scores & baseline trends
    scored_df = calculate_bharatlens_score(df)

    # 2. Engineer features
    fe_df, feature_cols = engineer_district_features(scored_df)

    # 3. Fit and persist StandardScaler pipeline for model reproducibility
    scaler = StandardScaler()
    X = fe_df[feature_cols].copy()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_model(scaler, os.path.join(MODELS_DIR, "feature_scaler.joblib"))
    save_json(feature_cols, os.path.join(MODELS_DIR, "feature_names.json"))

    # Save feature matrix
    out_path = os.path.join(PROCESSED_DATA_DIR, "district_features_dataset.csv")
    save_dataframe(fe_df, out_path)
    logger.info(f"Feature store successfully saved to {out_path} ({fe_df.shape[0]} districts, {fe_df.shape[1]} columns).")
    return fe_df

if __name__ == "__main__":
    build_and_save_feature_store()
