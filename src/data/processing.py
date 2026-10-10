import os
import re
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.utils.logger import get_logger
from src.utils.geo import normalize_state_name, normalize_district_name
from src.utils.io import save_dataframe, load_yaml
from src.data.validation import run_data_quality_audit, generate_markdown_quality_report

logger = get_logger("DataProcessing")

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

NFHS_INDICATOR_MAPPING = {
    "institutional_births": "Institutional births (%)",
    "children_vaccinated": "Children age 12-23 months fully vaccinated based on information from either vaccination card or mother's recall (%)",
    "stunting_under5": "Children under 5 years who are stunted (height for age) (%)",
    "wasting_under5": "Children under 5 years who are wasted (weight for height) (%)",
    "anaemia_women": "All women age 15-49 years who are anaemic (%)",
    "mothers_anc_visits": "Mothers who had at least 4 antenatal care visits (%)",
    "female_literacy": "Women who are literate (%)",
    "women_schooling_10yrs": "Women with 10 or more years of schooling (%)",
    "electricity_access": "Population living in households with electricity (%)",
    "improved_sanitation": "Population living in households that use an improved sanitation facility (%)",
    "clean_fuel_cooking": "Households using clean fuel for cooking (%)",
}

def clean_numeric_col(series: pd.Series) -> pd.Series:
    """Safely coerces series to numeric, replacing strings, symbols and NaNs."""
    return pd.to_numeric(
        series.astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors="coerce"
    )

def process_nfhs_data(raw_path: str = os.path.join(RAW_DATA_DIR, "nfhs_district_change.csv")) -> pd.DataFrame:
    """Pivots and standardizes NFHS district observations across survey rounds."""
    logger.info(f"Processing NFHS dataset from {raw_path}...")
    df = pd.read_csv(raw_path, low_memory=False)

    # Standardize column headers
    df["State"] = df["State"].apply(normalize_state_name)
    df["District Name"] = df["District Name"].apply(normalize_district_name)

    # Clean numeric values
    df["NFHS 4"] = clean_numeric_col(df["NFHS 4"])
    df["NFHS 5"] = clean_numeric_col(df["NFHS 5"])
    df["Change"] = clean_numeric_col(df["Change"])

    # If Change is missing or NaN, compute NFHS 5 - NFHS 4
    mask = df["Change"].isna() & df["NFHS 5"].notna() & df["NFHS 4"].notna()
    df.loc[mask, "Change"] = df.loc[mask, "NFHS 5"] - df.loc[mask, "NFHS 4"]

    # Filter to target indicators
    indicator_inv_map = {v: k for k, v in NFHS_INDICATOR_MAPPING.items()}
    df_filtered = df[df["Indicator"].isin(NFHS_INDICATOR_MAPPING.values())].copy()
    df_filtered["indicator_key"] = df_filtered["Indicator"].map(indicator_inv_map)

    # Group by district and pivot
    records = []
    grouped = df_filtered.groupby(["State", "District Name"])

    for (state, district), group in grouped:
        row = {"state": state, "district": district}
        for _, item in group.iterrows():
            key = item["indicator_key"]
            if pd.notna(key):
                row[f"{key}_nfhs4"] = item["NFHS 4"]
                row[f"{key}_nfhs5"] = item["NFHS 5"]
                row[f"{key}_delta"] = item["Change"]
        records.append(row)

    df_pivoted = pd.DataFrame(records)
    logger.info(f"Pivoted NFHS district table: {df_pivoted.shape[0]} districts, {df_pivoted.shape[1]} columns.")
    return df_pivoted

def process_census_data(raw_path: str = os.path.join(RAW_DATA_DIR, "census_district_pca.csv")) -> pd.DataFrame:
    """Processes Census 2011 Primary Census Abstract into structural per-capita indicators."""
    logger.info(f"Processing Census dataset from {raw_path}...")
    df = pd.read_csv(raw_path)

    df["state"] = df["State name"].apply(normalize_state_name)
    df["district"] = df["District name"].apply(normalize_district_name)

    # Population & Sex Ratio
    pop = clean_numeric_col(df["Population"])
    male = clean_numeric_col(df["Male"])
    female = clean_numeric_col(df["Female"])
    literate = clean_numeric_col(df["Literate"])
    workers = clean_numeric_col(df["Workers"])
    households = clean_numeric_col(df["Households"]).replace(0, np.nan)

    res = pd.DataFrame({
        "state": df["state"],
        "district": df["district"],
        "census_population": pop,
        "census_sex_ratio": (female / male.replace(0, np.nan) * 1000).clip(500, 1500),
        "census_literacy_rate": (literate / pop.replace(0, np.nan) * 100).clip(0, 100),
        "census_worker_rate": (workers / pop.replace(0, np.nan) * 100).clip(0, 100),
        "census_agri_worker_share": ((clean_numeric_col(df["Agricultural_Workers"]) + clean_numeric_col(df["Cultivator_Workers"])) / workers.replace(0, np.nan) * 100).clip(0, 100),
        "census_urban_pct": (clean_numeric_col(df["Urban_Households"]) / households * 100).clip(0, 100),
        "census_electricity_pct": (clean_numeric_col(df["Housholds_with_Electric_Lighting"]) / households * 100).clip(0, 100),
        "census_lpg_pct": (clean_numeric_col(df["LPG_or_PNG_Households"]) / households * 100).clip(0, 100),
        "census_internet_pct": (clean_numeric_col(df["Households_with_Internet"]) / households * 100).clip(0, 100),
    })

    # Drop potential duplicate district entries in census
    res = res.drop_duplicates(subset=["state", "district"])
    logger.info(f"Census processing complete: {res.shape[0]} districts.")
    return res

def process_mgnrega_data(raw_path: str = os.path.join(RAW_DATA_DIR, "mgnrega_district_employment.csv")) -> pd.DataFrame:
    """Processes MGNREGA rural employment & wage floor dataset."""
    logger.info(f"Processing MGNREGA dataset from {raw_path}...")
    df = pd.read_csv(raw_path, sep=";")

    state_col = "state_name"
    dist_col = "district_name"

    df["state"] = df[state_col].apply(normalize_state_name)
    df["district"] = df[dist_col].apply(normalize_district_name)

    wage = clean_numeric_col(df["Average Wage rate per day per person(Rs.)"]).clip(50, 600)
    days = clean_numeric_col(df["Average days of employment provided per Household"]).clip(0, 120)
    agri_exp = clean_numeric_col(df["% of Expenditure on Agriculture & Agriculture Allied Works"]).clip(0, 100)
    active_workers = clean_numeric_col(df["Total No. of Active Workers"])
    total_workers = clean_numeric_col(df["Total No. of Workers"]).replace(0, np.nan)
    women_persondays = clean_numeric_col(df["Women Persondays"])
    total_liability = clean_numeric_col(df["Persondays of Central Liability so far"]).replace(0, np.nan)

    res = pd.DataFrame({
        "state": df["state"],
        "district": df["district"],
        "mgnrega_daily_wage_rs": wage,
        "mgnrega_avg_days_per_hh": days,
        "mgnrega_agri_works_pct": agri_exp,
        "mgnrega_active_worker_ratio": (active_workers / total_workers * 100).clip(0, 100),
        "mgnrega_women_persondays_pct": (women_persondays / total_liability * 100).clip(0, 100),
    })

    res = res.drop_duplicates(subset=["state", "district"])
    logger.info(f"MGNREGA processing complete: {res.shape[0]} districts.")
    return res

def process_subdivision_rainfall(raw_path: str = os.path.join(RAW_DATA_DIR, "subdivision_rainfall.csv")) -> pd.DataFrame:
    """Computes long-term normal rainfall and climate variability per subdivision."""
    logger.info(f"Processing rainfall dataset from {raw_path}...")
    df = pd.read_csv(raw_path)

    # Compute mean annual rainfall and monsoon share over last 30 years (1986-2015)
    recent = df[df["YEAR"] >= 1985].copy()
    stats = recent.groupby("DIVISION").agg(
        normal_annual_rainfall_mm=("ANNUAL", "mean"),
        monsoon_rainfall_mm=("Jun-Sep", "mean"),
        rainfall_cv_pct=("ANNUAL", lambda x: (x.std() / (x.mean() + 1e-5)) * 100)
    ).reset_index()

    # Map subdivisions to state keywords
    stats["division_clean"] = stats["DIVISION"].str.lower()
    return stats

def merge_and_harmonize_datasets() -> pd.DataFrame:
    """Merges all domain datasets into an integrated panel dataset and imputes missing values."""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df_nfhs = process_nfhs_data()
    df_census = process_census_data()
    df_mgnrega = process_mgnrega_data()

    # Base merge on NFHS districts
    logger.info("Merging NFHS, Census, and MGNREGA datasets...")
    merged = pd.merge(df_nfhs, df_census, on=["state", "district"], how="left")
    merged = pd.merge(merged, df_mgnrega, on=["state", "district"], how="left")

    logger.info(f"Merged dataset shape before imputation: {merged.shape}")

    # Hierarchical State-Aware Median Imputation
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()

    # Pass 1: State median
    for col in numeric_cols:
        state_medians = merged.groupby("state")[col].transform("median")
        merged[col] = merged[col].fillna(state_medians)

    # Pass 2: National median
    for col in numeric_cols:
        national_median = merged[col].median()
        if pd.isna(national_median):
            national_median = 0.0
        merged[col] = merged[col].fillna(national_median)

    # Validation audit
    audit = run_data_quality_audit(merged, "Integrated_District_Master")
    generate_markdown_quality_report([audit])

    out_path = os.path.join(PROCESSED_DATA_DIR, "district_master_processed.csv")
    save_dataframe(merged, out_path)
    logger.info(f"Integrated district master dataset saved to {out_path} ({merged.shape[0]} districts, {merged.shape[1]} features).")
    return merged

if __name__ == "__main__":
    merge_and_harmonize_datasets()
