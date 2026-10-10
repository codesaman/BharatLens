import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.utils.logger import get_logger
from src.utils.io import save_model, save_json, load_yaml

logger = get_logger("RegressionPrediction")

CONFIG_PATH = os.path.join("config", "config.yaml")
MODELS_DIR = os.path.join("models", "saved")
METADATA_DIR = os.path.join("models", "metadata")

def train_and_compare_regressors(
    df: pd.DataFrame,
    target_col: str = "score_change",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Trains multiple regression models, benchmarks them, and serializes the champion model."""
    logger.info(f"Starting Regression Model Comparison. Target: '{target_col}'...")

    # Filter feature columns
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feature_cols:
        raise ValueError("No feature columns starting with 'feat_' found in dataset.")

    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df[target_col].copy().fillna(0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize candidate algorithms
    candidates: Dict[str, Any] = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=150, max_depth=8, min_samples_split=4, random_state=random_state, n_jobs=-1
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=random_state
        ),
        "RidgeRegression": Ridge(alpha=10.0, random_state=random_state),
    }

    if HAS_XGB:
        candidates["XGBoostRegressor"] = XGBRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=random_state, n_jobs=-1, verbosity=0
        )

    comparison_records = []
    trained_models = {}
    best_model_name = ""
    best_rmse = float("inf")
    champion_model = None

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in candidates.items():
        logger.info(f"Training and evaluating {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        mae = float(mean_absolute_error(y_test, preds))
        mse = float(mean_squared_error(y_test, preds))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, preds))

        # 5-Fold Cross Validation R2
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring="r2")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        record = {
            "model_name": name,
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "r2_score": round(r2, 4),
            "cv_r2_mean": round(cv_mean, 4),
            "cv_r2_std": round(cv_std, 4),
        }
        comparison_records.append(record)
        trained_models[name] = model

        logger.info(f"{name} Results -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, 5-Fold CV R²: {cv_mean:.4f} (±{cv_std:.4f})")

        # Select champion based on lowest RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            champion_model = model

    comparison_df = pd.DataFrame(comparison_records).sort_values("rmse", ascending=True).reset_index(drop=True)
    logger.info(f"Champion Regression Model: {best_model_name} (RMSE: {best_rmse:.4f})")

    # Generate full dataset predictions with Champion
    full_X_scaled = scaler.transform(X)
    full_preds = champion_model.predict(full_X_scaled)
    preds_df = pd.DataFrame({
        "predicted_score_change": np.round(full_preds, 2),
        "predicted_next_bharatlens_score": np.round(df["bharatlens_score_current"] + full_preds, 2).clip(0, 100)
    }, index=df.index)

    metadata = {
        "champion_model_name": best_model_name,
        "best_rmse": round(best_rmse, 4),
        "target_variable": target_col,
        "feature_count": len(feature_cols),
        "test_size": test_size,
        "benchmark_table": comparison_records,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    save_model(champion_model, os.path.join(MODELS_DIR, "trend_regressor_champion.joblib"))
    save_model(scaler, os.path.join(MODELS_DIR, "regression_scaler.joblib"))
    save_json(metadata, os.path.join(METADATA_DIR, "regression_metadata.json"))

    return champion_model, metadata, preds_df

if __name__ == "__main__":
    features_path = os.path.join("data", "processed", "district_features_dataset.csv")
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        train_and_compare_regressors(df)
