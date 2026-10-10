import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.io import save_model, save_json, load_yaml

logger = get_logger("AnomalyDetection")

CONFIG_PATH = os.path.join("config", "config.yaml")
MODELS_DIR = os.path.join("models", "saved")
METADATA_DIR = os.path.join("models", "metadata")

ANOMALY_FEATURE_CANDIDATES = [
    "score_change",
    "state_relative_growth_delta",
    "relative_growth_zscore",
    "feat_institutional_births_abs_change",
    "feat_stunting_under5_abs_change",
    "feat_anaemia_women_abs_change",
    "feat_female_literacy_abs_change",
    "feat_clean_fuel_cooking_abs_change",
    "feat_stunting_under5_nfhs5_state_zscore",
    "feat_mgnrega_daily_wage_rs_state_zscore",
]

def train_anomaly_detectors(
    df: pd.DataFrame,
    contamination: float = 0.05
) -> Tuple[IsolationForest, LocalOutlierFactor, pd.DataFrame, Dict[str, Any]]:
    """Trains Isolation Forest and Local Outlier Factor to detect districts with divergent or irregular trajectories."""
    logger.info(f"Training Anomaly Detectors (Contamination = {contamination})...")

    # Select available features
    features = [f for f in ANOMALY_FEATURE_CANDIDATES if f in df.columns]
    X_raw = df[features].fillna(df[features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 1. Primary Model: Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_preds = iso_forest.fit_predict(X_scaled)  # -1 for anomaly, 1 for normal
    # Raw decision function: lower is more anomalous. Convert to [0, 100] anomaly risk score
    raw_scores = iso_forest.decision_function(X_scaled)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    # Invert so higher score = higher risk/anomaly
    anomaly_risk_score = ((max_s - raw_scores) / (max_s - min_s + 1e-6) * 100).round(2)

    # 2. Validation Model: Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=False
    )
    lof_preds = lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_  # ~1 is normal, >>1 is outlier

    is_anomaly_iso = (iso_preds == -1).astype(int)
    is_anomaly_lof = (lof_preds == -1).astype(int)

    # Combined consensus anomaly flag (agreed or strong isolation score)
    consensus_anomaly = (is_anomaly_iso == 1) | (anomaly_risk_score >= 80.0)

    results_df = pd.DataFrame({
        "anomaly_risk_score": anomaly_risk_score,
        "is_anomaly_isolation_forest": is_anomaly_iso,
        "is_anomaly_lof": is_anomaly_lof,
        "is_anomaly_consensus": consensus_anomaly.astype(int),
        "lof_score": np.round(lof_scores, 3)
    }, index=df.index)

    # Identify primary drivers of anomalies for flagged districts
    anomalous_indices = df[consensus_anomaly].index
    anomaly_diagnostics = []

    for idx in anomalous_indices:
        district_row = df.loc[idx]
        district_features = X_raw.loc[idx]
        feature_zscores = ((district_features - X_raw.mean()) / (X_raw.std() + 1e-5)).abs()
        top_deviations = feature_zscores.nlargest(3).to_dict()

        anomaly_diagnostics.append({
            "district": district_row.get("district", "Unknown"),
            "state": district_row.get("state", "Unknown"),
            "anomaly_risk_score": float(anomaly_risk_score[idx]),
            "score_change": float(district_row.get("score_change", 0.0)),
            "top_deviating_features": {k: round(float(v), 2) for k, v in top_deviations.items()}
        })

    anomaly_count = int(consensus_anomaly.sum())
    logger.info(f"Anomaly detection complete: {anomaly_count} districts flagged ({anomaly_count / len(df) * 100:.1f}%).")

    metadata = {
        "primary_model": "Isolation Forest",
        "validation_model": "Local Outlier Factor (LOF)",
        "features_used": features,
        "contamination": contamination,
        "anomalous_districts_count": anomaly_count,
        "diagnostics": anomaly_diagnostics[:15]  # Top diagnostics
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    save_model(iso_forest, os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    save_model(scaler, os.path.join(MODELS_DIR, "anomaly_scaler.joblib"))
    save_json(metadata, os.path.join(METADATA_DIR, "anomaly_metadata.json"))

    return iso_forest, lof, results_df, metadata

if __name__ == "__main__":
    features_path = os.path.join("data", "processed", "district_features_dataset.csv")
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        train_anomaly_detectors(df)
