import os
import sys
from typing import Dict, Any
import pandas as pd

from src.utils.logger import get_logger
from src.utils.io import save_dataframe, load_yaml
from src.ml.clustering import train_clustering_model
from src.ml.anomaly import train_anomaly_detectors
from src.ml.prediction import train_and_compare_regressors
from src.ml.classification import train_and_compare_classifiers
from src.explainability.shap_analysis import compute_and_cache_shap_explanations
from src.ml.evaluation import generate_model_comparison_report

logger = get_logger("TrainPipeline")

PROCESSED_DATA_DIR = os.path.join("data", "processed")
FEATURES_PATH = os.path.join(PROCESSED_DATA_DIR, "district_features_dataset.csv")

def train_entire_ml_pipeline() -> pd.DataFrame:
    """Orchestrates end-to-end ML training: clustering, anomaly detection, regression, classification, SHAP, and report generation."""
    logger.info("=================================================================")
    logger.info("Initiating BharatLens ML Autonomous Model Training Pipeline")
    logger.info("=================================================================")

    if not os.path.exists(FEATURES_PATH):
        logger.error(f"Features dataset missing at {FEATURES_PATH}. Run feature engineering first.")
        raise FileNotFoundError(f"Missing {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)
    logger.info(f"Loaded feature dataset: {df.shape[0]} districts, {df.shape[1]} columns.")

    # 1. Socioeconomic Clustering
    logger.info("\n--- STEP 1: Socioeconomic Tier Clustering ---")
    kmeans, cluster_labels, clu_meta = train_clustering_model(df, n_clusters=4)
    df["cluster_id"] = cluster_labels
    tier_map = clu_meta.get("tier_names", {})
    df["cluster_name"] = df["cluster_id"].map(lambda x: tier_map.get(x, tier_map.get(str(x), f"Tier {x}")))

    # 2. Anomaly Detection
    logger.info("\n--- STEP 2: Anomaly Detection (Isolation Forest & LOF) ---")
    iso_model, lof_model, anomaly_df, ano_meta = train_anomaly_detectors(df, contamination=0.05)
    df["anomaly_risk_score"] = anomaly_df["anomaly_risk_score"]
    df["is_anomaly"] = anomaly_df["is_anomaly_consensus"]
    df["lof_score"] = anomaly_df["lof_score"]

    # 3. Construct Unified Multi-Class Target
    # If consensus anomaly -> 'Anomalous', otherwise use trend_label (Improving, Stable, Declining)
    df["final_trend_class"] = df["trend_label"]
    df.loc[df["is_anomaly"] == 1, "final_trend_class"] = "Anomalous"
    logger.info(f"Final 4-Class Target Distribution:\n{df['final_trend_class'].value_counts().to_dict()}")

    # 4. Trend Forecasting Regression Comparison
    logger.info("\n--- STEP 3: Trend Forecasting Regressors Benchmark ---")
    champ_regressor, reg_meta, reg_preds = train_and_compare_regressors(df, target_col="score_change")
    df["predicted_score_change"] = reg_preds["predicted_score_change"]
    df["predicted_next_bharatlens_score"] = reg_preds["predicted_next_bharatlens_score"]

    # 5. Multi-Class Trend Classification Comparison
    logger.info("\n--- STEP 4: Trend Classifiers Benchmark ---")
    champ_classifier, clf_meta, clf_preds = train_and_compare_classifiers(df, target_col="final_trend_class")
    for col in clf_preds.columns:
        df[col] = clf_preds[col]

    # 6. Explainable AI (SHAP TreeExplainer)
    logger.info("\n--- STEP 5: Explainable AI & SHAP Computations ---")
    import joblib
    reg_scaler = joblib.load(os.path.join("models", "saved", "regression_scaler.joblib"))
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    shap_meta = compute_and_cache_shap_explanations(champ_regressor, reg_scaler, df, feature_cols)

    # 7. Generate Evaluation and Comparison Report
    logger.info("\n--- STEP 6: Generating Model Evaluation Report ---")
    generate_model_comparison_report()

    # Save final enriched master results
    out_master_path = os.path.join(PROCESSED_DATA_DIR, "district_master_ml_results.csv")
    save_dataframe(df, out_master_path)

    logger.info("=================================================================")
    logger.info(f"BharatLens ML Training Pipeline Finished Successfully.")
    logger.info(f"Master ML Results persisted to {out_master_path}")
    logger.info("=================================================================")
    return df

if __name__ == "__main__":
    train_entire_ml_pipeline()
