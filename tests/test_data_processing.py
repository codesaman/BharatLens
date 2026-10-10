import pytest
import pandas as pd
import numpy as np
import os
from src.utils.geo import normalize_state_name, normalize_district_name, clean_text
from src.data.processing import clean_numeric_col

def test_geo_normalization():
    assert normalize_state_name("orissa") == "Odisha"
    assert normalize_state_name("UTTARANCHAL") == "Uttarakhand"
    assert normalize_state_name("Pondicherry") == "Puducherry"
    assert normalize_district_name("Allahabad") == "Prayagraj"
    assert normalize_district_name("Bangalore Urban") == "Bengaluru Urban"
    assert normalize_district_name("Gurgaon") == "Gurugram"

def test_clean_numeric_col():
    sample = pd.Series(["12.5%", "₹ 240.50", "N/A", "1,200", None, 45.2])
    cleaned = clean_numeric_col(sample)
    assert cleaned.iloc[0] == 12.5
    assert cleaned.iloc[1] == 240.50
    assert np.isnan(cleaned.iloc[2])
    assert cleaned.iloc[3] == 1200.0
    assert np.isnan(cleaned.iloc[4])
    assert cleaned.iloc[5] == 45.2

def test_processed_master_dataset_integrity():
    master_path = os.path.join("data", "processed", "district_master_processed.csv")
    assert os.path.exists(master_path), "Master dataset must exist after pipeline execution"
    df = pd.read_csv(master_path)
    assert len(df) >= 600, "Should contain at least 600 Indian districts"
    assert "state" in df.columns
    assert "district" in df.columns
    assert df.isnull().sum().sum() == 0, "No missing values allowed after median imputation"
