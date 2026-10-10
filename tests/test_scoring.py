import pytest
import pandas as pd
import numpy as np
import os
from src.features.engineering import engineer_district_features
from src.scoring.bharatlens_score import min_max_normalize, calculate_bharatlens_score

def test_min_max_normalize_directions():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    # Positive direction: higher value -> higher normalized score
    norm_pos = min_max_normalize(s, direction="positive")
    assert norm_pos.iloc[0] == 0.0
    assert norm_pos.iloc[-1] == 100.0

    # Negative direction: higher burden -> lower normalized score
    norm_neg = min_max_normalize(s, direction="negative")
    assert norm_neg.iloc[0] == 100.0
    assert norm_neg.iloc[-1] == 0.0

def test_bharatlens_score_bounds_and_trends():
    df_features = pd.read_csv(os.path.join("data", "processed", "district_features_dataset.csv"))
    assert "bharatlens_score_current" in df_features.columns
    assert "bharatlens_score_previous" in df_features.columns

    scores = df_features["bharatlens_score_current"]
    assert (scores >= 0.0).all() and (scores <= 100.0).all(), "Scores must be strictly bounded in [0, 100]"

    valid_trends = {"Improving", "Stable", "Declining"}
    observed_trends = set(df_features["trend_label"].unique())
    assert observed_trends.issubset(valid_trends), f"Observed trends {observed_trends} must be in {valid_trends}"

def test_feature_engineering_outputs():
    df_features = pd.read_csv(os.path.join("data", "processed", "district_features_dataset.csv"))
    feat_cols = [c for c in df_features.columns if c.startswith("feat_")]
    assert len(feat_cols) >= 30, "Should generate at least 30 rich econometric/demographic features"
    # Verify no infinite or NaN values in feature matrix
    for c in feat_cols:
        assert not np.isinf(df_features[c]).any(), f"Column {c} contains inf"
        assert not df_features[c].isna().any(), f"Column {c} contains NaN"
