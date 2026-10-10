# BharatLens ML - Empirical Model Evaluation and Comparison Report

> Rigorous comparative evaluation of Machine Learning models trained on Indian district panel datasets.

## 1. Executive Summary & Champion Models

- **Trend Regression Champion**: `RidgeRegression` (Selected by minimum test RMSE = `0.1863`)
- **Trend Classification Champion**: `LogisticRegression` (Selected by maximum Macro-F1 = `0.802`)
- **Socioeconomic Clustering**: `K-Means (k=4)` with Silhouette Score = `0.2061` and Davies-Bouldin Index = `1.5415`
- **Anomaly Detection**: `Isolation Forest (150 trees, contamination=0.05)` + `LOF Validation`

## 2. Regression Model Benchmarks (District Score Delta & Trajectory)

| Model Architecture | Test MAE | Test RMSE | Test R² | 5-Fold CV R² (Mean ± Std) | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RandomForestRegressor** | 0.8894 | 1.1511 | 0.9299 | 0.9326 ± 0.0146 | Evaluated |
| **GradientBoostingRegressor** | 0.6723 | 0.9061 | 0.9565 | 0.9596 ± 0.0104 | Evaluated |
| **RidgeRegression** | 0.1355 | 0.1863 | 0.9982 | 0.9961 ± 0.0024 | **CHAMPION (Selected)** |
| **XGBoostRegressor** | 0.6689 | 0.9284 | 0.9544 | 0.9581 ± 0.0147 | Evaluated |

## 3. Multi-Class Classification Benchmarks (Improving / Stable / Declining / Anomalous)

| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | 5-Fold CV Macro-F1 | Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForestClassifier** | 0.7519 | 0.7889 | 0.7076 | 0.7383 | 0.637 ± 0.0542 | Evaluated |
| **GradientBoostingClassifier** | 0.7442 | 0.7766 | 0.6792 | 0.7127 | 0.6194 ± 0.0303 | Evaluated |
| **LogisticRegression** | 0.8372 | 0.8224 | 0.7867 | 0.802 | 0.6855 ± 0.0883 | **CHAMPION (Selected)** |
| **XGBoostClassifier** | 0.7752 | 0.8053 | 0.7324 | 0.7615 | 0.613 ± 0.056 | Evaluated |

## 4. Socioeconomic Cluster Tiers

| Cluster ID | Tier Description | Districts | Share | Mean BharatLens Score | Mean Score Delta |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | **Tier 1: High-Priority / Aspirational** | 212 | 32.97% | 49.44 | 11.44 |
| `1` | **Tier 2: Developing / Rural-Agricultural** | 153 | 23.79% | 53.37 | 10.81 |
| `2` | **Tier 3: Progressing / Semi-Urban Transition** | 194 | 30.17% | 61.03 | 10.82 |
| `3` | **Tier 4: Front-Runner / Advanced Socioeconomic** | 84 | 13.06% | 66.06 | 12.22 |

## 5. Anomaly Detection Diagnostics

- **Contamination Factor**: `0.05`
- **Total Flagged Districts**: `33`
- **Top Diagnosed Divergent Districts**:
  - **Tirap (Arunachal Pradesh)**: Anomaly Risk Score = `79.16`, Extreme Drivers = [feat_stunting_under5_abs_change: 2.92σ, relative_growth_zscore: 2.47σ, score_change: 2.47σ]
  - **Dhubri (Assam)**: Anomaly Risk Score = `79.76`, Extreme Drivers = [feat_stunting_under5_abs_change: 3.69σ, feat_anaemia_women_abs_change: 2.92σ, feat_institutional_births_abs_change: 2.03σ]
  - **Jorhat (Assam)**: Anomaly Risk Score = `79.08`, Extreme Drivers = [feat_institutional_births_abs_change: 3.4σ, feat_anaemia_women_abs_change: 3.38σ, state_relative_growth_delta: 3.28σ]
  - **Sivasagar (Assam)**: Anomaly Risk Score = `85.79`, Extreme Drivers = [state_relative_growth_delta: 3.79σ, feat_institutional_births_abs_change: 3.34σ, feat_anaemia_women_abs_change: 3.15σ]
  - **Bastar (Chhattisgarh)**: Anomaly Risk Score = `85.77`, Extreme Drivers = [feat_anaemia_women_abs_change: 3.67σ, feat_stunting_under5_abs_change: 3.66σ, feat_institutional_births_abs_change: 2.0σ]

## 6. Top Global Model Drivers (Explainable AI / SHAP)

- **Change in Female Literacy (%)**: Mean |SHAP| Impact = `0.7846`
- **Change in Household Electricity Access (%)**: Mean |SHAP| Impact = `0.7432`
- **Change in Antenatal Care Coverage (%)**: Mean |SHAP| Impact = `0.6804`
- **Change in Child Full Immunization (%)**: Mean |SHAP| Impact = `0.6341`
- **Change in Institutional Deliveries (%)**: Mean |SHAP| Impact = `0.6153`
- **Change in Child Wasting Rate (%)**: Mean |SHAP| Impact = `0.5871`
- **Change in Improved Sanitation Coverage (%)**: Mean |SHAP| Impact = `0.5276`
- **Change in Clean Fuel Cooking Adoption (%)**: Mean |SHAP| Impact = `0.5169`