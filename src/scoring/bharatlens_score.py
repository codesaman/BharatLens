import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.io import load_yaml, save_dataframe

logger = get_logger("BharatLensScore")

CONFIG_PATH = os.path.join("config", "config.yaml")

def min_max_normalize(series: pd.Series, direction: str = "positive") -> pd.Series:
    """Normalizes a feature to [0, 100] scale honoring directional polarity."""
    s_min = series.min()
    s_max = series.max()
    if s_max == s_min:
        return pd.Series(50.0, index=series.index)
    if direction == "positive":
        return ((series - s_min) / (s_max - s_min) * 100).clip(0, 100)
    else:  # negative (e.g. stunting, anaemia)
        return ((s_max - series) / (s_max - s_min) * 100).clip(0, 100)

def compute_dimension_scores(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
    """Computes dimension-level sub-scores (0-100) for both current (NFHS-5) and previous (NFHS-4) rounds."""
    config = load_yaml(CONFIG_PATH)
    dim_defs = config["scoring"]["dimensions"]

    current_dim_scores: Dict[str, pd.Series] = {}
    previous_dim_scores: Dict[str, pd.Series] = {}

    col_mapping_curr = {
        "institutional_births_pct": "institutional_births_nfhs5",
        "children_vaccinated_pct": "children_vaccinated_nfhs5",
        "stunting_under5_pct": "stunting_under5_nfhs5",
        "wasting_under5_pct": "wasting_under5_nfhs5",
        "anaemia_women_pct": "anaemia_women_nfhs5",
        "mothers_anc_visits_pct": "mothers_anc_visits_nfhs5",
        "female_literacy_pct": "female_literacy_nfhs5",
        "women_schooling_10yrs_pct": "women_schooling_10yrs_nfhs5",
        "women_bank_account_pct": "mgnrega_women_persondays_pct",
        "women_household_decisions_pct": "female_literacy_nfhs5",
        "women_mgnrega_participation_pct": "mgnrega_women_persondays_pct",
        "electricity_access_pct": "electricity_access_nfhs5",
        "improved_sanitation_pct": "improved_sanitation_nfhs5",
        "clean_fuel_cooking_pct": "clean_fuel_cooking_nfhs5",
        "improved_drinking_water_pct": "electricity_access_nfhs5",
        "internet_access_pct": "census_internet_pct",
        "mgnrega_daily_wage_rs": "mgnrega_daily_wage_rs",
        "mgnrega_avg_days_per_hh": "mgnrega_avg_days_per_hh",
        "worker_participation_rate": "census_worker_rate",
        "agri_works_expenditure_pct": "mgnrega_agri_works_pct",
        "census_literacy_rate": "census_literacy_rate",
        "sex_ratio": "census_sex_ratio",
        "urban_population_pct": "census_urban_pct",
    }

    col_mapping_prev = {
        "institutional_births_pct": "institutional_births_nfhs4",
        "children_vaccinated_pct": "children_vaccinated_nfhs4",
        "stunting_under5_pct": "stunting_under5_nfhs4",
        "wasting_under5_pct": "wasting_under5_nfhs4",
        "anaemia_women_pct": "anaemia_women_nfhs4",
        "mothers_anc_visits_pct": "mothers_anc_visits_nfhs4",
        "female_literacy_pct": "female_literacy_nfhs4",
        "women_schooling_10yrs_pct": "women_schooling_10yrs_nfhs4",
        "women_bank_account_pct": "mgnrega_women_persondays_pct",
        "women_household_decisions_pct": "female_literacy_nfhs4",
        "women_mgnrega_participation_pct": "mgnrega_women_persondays_pct",
        "electricity_access_pct": "electricity_access_nfhs4",
        "improved_sanitation_pct": "improved_sanitation_nfhs4",
        "clean_fuel_cooking_pct": "clean_fuel_cooking_nfhs4",
        "improved_drinking_water_pct": "electricity_access_nfhs4",
        "internet_access_pct": "census_internet_pct",
        "mgnrega_daily_wage_rs": "mgnrega_daily_wage_rs",
        "mgnrega_avg_days_per_hh": "mgnrega_avg_days_per_hh",
        "worker_participation_rate": "census_worker_rate",
        "agri_works_expenditure_pct": "mgnrega_agri_works_pct",
        "census_literacy_rate": "census_literacy_rate",
        "sex_ratio": "census_sex_ratio",
        "urban_population_pct": "census_urban_pct",
    }

    for dim_name, dim_cfg in dim_defs.items():
        indicators = dim_cfg["indicators"]
        curr_norm_list = []
        prev_norm_list = []

        for ind in indicators:
            iname = ind["name"]
            direction = ind["direction"]
            curr_col = col_mapping_curr.get(iname)
            prev_col = col_mapping_prev.get(iname)

            if curr_col in df.columns:
                curr_norm_list.append(min_max_normalize(df[curr_col], direction))
            if prev_col in df.columns:
                prev_norm_list.append(min_max_normalize(df[prev_col], direction))

        if curr_norm_list:
            current_dim_scores[dim_name] = pd.concat(curr_norm_list, axis=1).mean(axis=1)
        else:
            current_dim_scores[dim_name] = pd.Series(50.0, index=df.index)

        if prev_norm_list:
            previous_dim_scores[dim_name] = pd.concat(prev_norm_list, axis=1).mean(axis=1)
        else:
            previous_dim_scores[dim_name] = pd.Series(50.0, index=df.index)

    return df, current_dim_scores, previous_dim_scores

def calculate_bharatlens_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates composite 0-100 BharatLens scores, dimension breakdowns, baselines, and peer relative trends."""
    logger.info("Computing BharatLens composite score and dimension indices...")
    config = load_yaml(CONFIG_PATH)
    dim_defs = config["scoring"]["dimensions"]
    weights = {k: v["weight"] for k, v in dim_defs.items()}

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    df_out, current_dims, prev_dims = compute_dimension_scores(df)

    # Attach dimension scores to dataframe
    for dim_name in dim_defs.keys():
        df_out[f"dim_{dim_name}_current"] = current_dims[dim_name].round(2)
        df_out[f"dim_{dim_name}_prev"] = prev_dims[dim_name].round(2)
        df_out[f"dim_{dim_name}_delta"] = (current_dims[dim_name] - prev_dims[dim_name]).round(2)

    # Compute overall weighted scores
    curr_overall = pd.Series(0.0, index=df.index)
    prev_overall = pd.Series(0.0, index=df.index)

    for dim_name, w in weights.items():
        curr_overall += current_dims[dim_name] * w
        prev_overall += prev_dims[dim_name] * w

    df_out["bharatlens_score_current"] = curr_overall.clip(0, 100).round(2)
    df_out["bharatlens_score_previous"] = prev_overall.clip(0, 100).round(2)
    df_out["score_change"] = (df_out["bharatlens_score_current"] - df_out["bharatlens_score_previous"]).round(2)
    df_out["historical_average_score"] = ((df_out["bharatlens_score_current"] + df_out["bharatlens_score_previous"]) / 2).round(2)

    # State Benchmark & Relative Deviation
    state_avg = df_out.groupby("state")["bharatlens_score_current"].transform("mean").round(2)
    state_delta_avg = df_out.groupby("state")["score_change"].transform("mean").round(2)
    df_out["state_average_score"] = state_avg
    df_out["state_relative_deviation"] = (df_out["bharatlens_score_current"] - state_avg).round(2)
    df_out["state_relative_growth_delta"] = (df_out["score_change"] - state_delta_avg).round(2)

    # National Benchmark & Deviation
    national_avg = round(float(df_out["bharatlens_score_current"].mean()), 2)
    national_delta_mean = float(df_out["score_change"].mean())
    national_delta_std = float(df_out["score_change"].std()) + 1e-5
    df_out["national_average_score"] = national_avg
    df_out["national_relative_deviation"] = (df_out["bharatlens_score_current"] - national_avg).round(2)

    # Relative Progress Index (Z-score of growth rate)
    df_out["relative_growth_zscore"] = ((df_out["score_change"] - national_delta_mean) / national_delta_std).round(3)

    # Composite Statistical Trend Assignment
    # Improving: Above-average momentum (relative z-score >= +0.35 OR state-relative delta >= +1.5)
    # Declining: Lagging momentum (relative z-score <= -0.35 OR state-relative delta <= -1.5)
    # Stable: Pacing national and state momentum closely
    cond_improving = (df_out["relative_growth_zscore"] >= 0.35) | (df_out["state_relative_growth_delta"] >= 1.5)
    cond_declining = (df_out["relative_growth_zscore"] <= -0.35) | (df_out["state_relative_growth_delta"] <= -1.5)

    df_out["trend_label"] = "Stable"
    df_out.loc[cond_declining, "trend_label"] = "Declining"
    df_out.loc[cond_improving, "trend_label"] = "Improving"

    logger.info(f"BharatLens Score calculation complete: Mean current score = {national_avg:.2f}")
    trend_counts = df_out["trend_label"].value_counts().to_dict()
    logger.info(f"Trend distribution: {trend_counts}")
    return df_out

if __name__ == "__main__":
    master_path = os.path.join("data", "processed", "district_master_processed.csv")
    if os.path.exists(master_path):
        df = pd.read_csv(master_path)
        scored = calculate_bharatlens_score(df)
        save_dataframe(scored, os.path.join("data", "processed", "district_scored.csv"))
