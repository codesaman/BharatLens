import pytest
import os
import joblib
import pandas as pd
import numpy as np
from src.utils.io import load_model, load_json

def test_serialized_models_loadable():
    models_dir = os.path.join("models", "saved")
    assert os.path.exists(os.path.join(models_dir, "clustering_kmeans.joblib"))
    assert os.path.exists(os.path.join(models_dir, "isolation_forest.joblib"))
    assert os.path.exists(os.path.join(models_dir, "trend_regressor_champion.joblib"))
    assert os.path.exists(os.path.join(models_dir, "trend_classifier_champion.joblib"))

    kmeans = load_model(os.path.join(models_dir, "clustering_kmeans.joblib"))
    iso = load_model(os.path.join(models_dir, "isolation_forest.joblib"))
    reg = load_model(os.path.join(models_dir, "trend_regressor_champion.joblib"))
    clf = load_model(os.path.join(models_dir, "trend_classifier_champion.joblib"))

    assert kmeans is not None
    assert iso is not None
    assert reg is not None
    assert clf is not None

def test_inference_pipeline_predictions():
    df_results = pd.read_csv(os.path.join("data", "processed", "district_master_ml_results.csv"))
    assert "cluster_id" in df_results.columns
    assert "anomaly_risk_score" in df_results.columns
    assert "predicted_score_change" in df_results.columns
    assert "ml_predicted_trend" in df_results.columns

    # Check cluster values
    assert df_results["cluster_id"].isin([0, 1, 2, 3]).all()

    # Check anomaly scores in [0, 100]
    assert (df_results["anomaly_risk_score"] >= 0.0).all()
    assert (df_results["anomaly_risk_score"] <= 100.0).all()

    # Check trend predictions are valid classes
    valid_classes = {"Improving", "Stable", "Declining", "Anomalous"}
    assert set(df_results["ml_predicted_trend"].unique()).issubset(valid_classes)

def test_shap_explanations_cache():
    shap_meta = load_json(os.path.join("models", "metadata", "shap_explanations.json"))
    assert shap_meta is not None
    assert "global_importance" in shap_meta
    assert "district_explanations" in shap_meta
    assert len(shap_meta["district_explanations"]) >= 600
