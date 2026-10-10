import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.io import save_model, save_json, load_yaml

logger = get_logger("Clustering")

CONFIG_PATH = os.path.join("config", "config.yaml")
MODELS_DIR = os.path.join("models", "saved")
METADATA_DIR = os.path.join("models", "metadata")

CLUSTER_FEATURES = [
    "dim_health_nutrition_current",
    "dim_women_empowerment_current",
    "dim_infrastructure_amenities_current",
    "dim_economic_employment_current",
    "dim_demographic_education_current",
    "census_worker_rate",
    "census_urban_pct",
    "mgnrega_daily_wage_rs",
]

def find_optimal_clusters(X: np.ndarray, candidate_k: List[int] = [3, 4, 5, 6]) -> Dict[str, Any]:
    """Evaluates candidate cluster counts using Silhouette and Davies-Bouldin metrics."""
    results = {}
    best_k = candidate_k[0]
    best_sil = -1.0

    for k in candidate_k:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        db = float(davies_bouldin_score(X, labels))
        results[k] = {
            "silhouette_score": round(sil, 4),
            "davies_bouldin_index": round(db, 4),
            "inertia": round(float(kmeans.inertia_), 2)
        }
        if sil > best_sil:
            best_sil = sil
            best_k = k

    logger.info(f"Cluster evaluation across candidate K {candidate_k}: Best K by silhouette = {best_k} (score: {best_sil:.4f})")
    return {"candidate_evaluations": results, "optimal_k": best_k}

def train_clustering_model(df: pd.DataFrame, n_clusters: int = 4) -> Tuple[KMeans, pd.Series, Dict[str, Any]]:
    """Trains K-Means model, assigns intuitive ordered cluster labels, and generates statistical profiles."""
    logger.info(f"Training K-Means clustering with k={n_clusters}...")

    # Ensure all required features are present
    avail_features = [f for f in CLUSTER_FEATURES if f in df.columns]
    X_raw = df[avail_features].fillna(df[avail_features].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    eval_results = find_optimal_clusters(X_scaled, candidate_k=[3, 4, 5, 6])

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    raw_labels = kmeans.fit_predict(X_scaled)

    # Order clusters monotonically by mean BharatLens current score
    cluster_score_means = df.groupby(raw_labels)["bharatlens_score_current"].mean()
    sorted_clusters = cluster_score_means.sort_values().index.tolist()
    label_remap = {old_label: new_label for new_label, old_label in enumerate(sorted_clusters)}

    ordered_labels = pd.Series([label_remap[lbl] for lbl in raw_labels], index=df.index, name="cluster_id")

    # Defined meaningful socioeconomic tier labels
    cluster_tier_names = {
        0: "Tier 1: High-Priority / Aspirational",
        1: "Tier 2: Developing / Rural-Agricultural",
        2: "Tier 3: Progressing / Semi-Urban Transition",
        3: "Tier 4: Front-Runner / Advanced Socioeconomic"
    }

    # Generate Statistical Cluster Profiles
    profiles = {}
    for cid in range(n_clusters):
        c_mask = (ordered_labels == cid)
        c_df = df[c_mask]

        tier_title = cluster_tier_names.get(cid, f"Cluster {cid}")
        profile_stats = {
            "cluster_id": cid,
            "cluster_name": tier_title,
            "district_count": int(c_mask.sum()),
            "pct_of_all_districts": round(float(c_mask.sum() / len(df) * 100), 2),
            "mean_bharatlens_score": round(float(c_df["bharatlens_score_current"].mean()), 2),
            "mean_score_change": round(float(c_df["score_change"].mean()), 2),
            "mean_health_score": round(float(c_df["dim_health_nutrition_current"].mean()), 2),
            "mean_women_empowerment": round(float(c_df["dim_women_empowerment_current"].mean()), 2),
            "mean_infrastructure": round(float(c_df["dim_infrastructure_amenities_current"].mean()), 2),
            "mean_economic_score": round(float(c_df["dim_economic_employment_current"].mean()), 2),
            "mean_education_score": round(float(c_df["dim_demographic_education_current"].mean()), 2),
            "mean_urban_pct": round(float(c_df["census_urban_pct"].mean()), 2),
            "representative_districts": c_df[["district", "state"]].head(6).to_dict(orient="records")
        }
        profiles[str(cid)] = profile_stats

    # Package model metadata
    model_metadata = {
        "algorithm": "K-Means",
        "n_clusters": n_clusters,
        "features_used": avail_features,
        "silhouette_score": round(float(silhouette_score(X_scaled, ordered_labels)), 4),
        "davies_bouldin_index": round(float(davies_bouldin_score(X_scaled, ordered_labels)), 4),
        "tier_names": cluster_tier_names,
        "evaluation_summary": eval_results,
        "profiles": profiles
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    save_model(kmeans, os.path.join(MODELS_DIR, "clustering_kmeans.joblib"))
    save_model(scaler, os.path.join(MODELS_DIR, "clustering_scaler.joblib"))
    save_json(model_metadata, os.path.join(METADATA_DIR, "clustering_metadata.json"))

    logger.info(f"Clustering complete. Silhouette: {model_metadata['silhouette_score']}, DB Index: {model_metadata['davies_bouldin_index']}")
    return kmeans, ordered_labels, model_metadata

if __name__ == "__main__":
    features_path = os.path.join("data", "processed", "district_features_dataset.csv")
    if os.path.exists(features_path):
        df = pd.read_csv(features_path)
        train_clustering_model(df, n_clusters=4)
