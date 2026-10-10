import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from src.utils.logger import get_logger
from src.utils.io import save_model, save_json, load_yaml

logger = get_logger("TrendClassification")

CONFIG_PATH = os.path.join("config", "config.yaml")
MODELS_DIR = os.path.join("models", "saved")
METADATA_DIR = os.path.join("models", "metadata")

CLASS_ORDER = ["Improving", "Stable", "Declining", "Anomalous"]

def train_and_compare_classifiers(
    df: pd.DataFrame,
    target_col: str = "final_trend_class",
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[Any, Dict[str, Any], pd.DataFrame]:
    """Trains multi-class classification models (Improving, Stable, Declining, Anomalous), compares them and serializes champion."""
    logger.info(f"Starting Multi-Class Classification comparison for target '{target_col}'...")

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols].copy().fillna(df[feature_cols].median())
    y = df[target_col].copy()

    # Map class labels to integers for algorithms like XGBoost
    label_to_int = {lbl: i for i, lbl in enumerate(CLASS_ORDER)}
    int_to_label = {i: lbl for lbl, i in label_to_int.items()}
    y_encoded = y.map(label_to_int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    candidates: Dict[str, Any] = {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=150, max_depth=8, min_samples_split=4, random_state=random_state, n_jobs=-1
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=random_state
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000, multi_class="multinomial", random_state=random_state
        ),
    }

    if HAS_XGB:
        candidates["XGBoostClassifier"] = XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=random_state,
            eval_metric="mlogloss", n_jobs=-1, verbosity=0
        )

    comparison_records = []
    best_f1 = -1.0
    best_model_name = ""
    champion_model = None
    champion_cm = []
    champion_report = {}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in candidates.items():
        logger.info(f"Training and evaluating {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, average="macro", zero_division=0))
        rec = float(recall_score(y_test, preds, average="macro", zero_division=0))
        f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=skf, scoring="f1_macro")
        cv_mean = float(cv_scores.mean())
        cv_std = float(cv_scores.std())

        record = {
            "model_name": name,
            "accuracy": round(acc, 4),
            "macro_precision": round(prec, 4),
            "macro_recall": round(rec, 4),
            "macro_f1": round(f1, 4),
            "cv_f1_macro_mean": round(cv_mean, 4),
            "cv_f1_macro_std": round(cv_std, 4),
        }
        comparison_records.append(record)

        logger.info(f"{name} Results -> Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}, 5-Fold CV Macro-F1: {cv_mean:.4f} (±{cv_std:.4f})")

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            champion_model = model
            champion_cm = confusion_matrix(y_test, preds, labels=list(range(len(CLASS_ORDER)))).tolist()
            champion_report = classification_report(
                y_test, preds, target_names=CLASS_ORDER, output_dict=True, zero_division=0
            )

    logger.info(f"Champion Classification Model: {best_model_name} (Macro-F1: {best_f1:.4f})")

    # Full dataset predictions and class probabilities
    full_X_scaled = scaler.transform(X)
    full_preds_encoded = champion_model.predict(full_X_scaled)
    full_preds_labels = [int_to_label[i] for i in full_preds_encoded]

    preds_df = pd.DataFrame({
        "ml_predicted_trend": full_preds_labels,
    }, index=df.index)

    if hasattr(champion_model, "predict_proba"):
        probs = champion_model.predict_proba(full_X_scaled)
        for i, class_name in enumerate(CLASS_ORDER):
            preds_df[f"ml_confidence_{class_name.lower()}"] = np.round(probs[:, i], 3)
        preds_df["ml_prediction_confidence"] = np.round(np.max(probs, axis=1) * 100, 1)

    metadata = {
        "champion_model_name": best_model_name,
        "best_macro_f1": round(best_f1, 4),
        "class_order": CLASS_ORDER,
        "confusion_matrix": champion_cm,
        "detailed_classification_report": champion_report,
        "benchmark_table": comparison_records,
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    save_model(champion_model, os.path.join(MODELS_DIR, "trend_classifier_champion.joblib"))
    save_model(scaler, os.path.join(MODELS_DIR, "classifier_scaler.joblib"))
    save_json(metadata, os.path.join(METADATA_DIR, "classification_metadata.json"))

    return champion_model, metadata, preds_df

if __name__ == "__main__":
    features_path = os.path.join("data", "processed", "district_features_dataset.csv")
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        # Placeholder call
        if "final_trend_class" in df.columns:
            train_and_compare_classifiers(df)
