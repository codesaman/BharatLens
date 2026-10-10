import os
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.io import load_json, save_json

logger = get_logger("ModelEvaluation")

FIGURES_DIR = os.path.join("reports", "figures")
REPORTS_DIR = "reports"
METADATA_DIR = os.path.join("models", "metadata")

def plot_regression_comparison(regression_meta: Dict[str, Any], output_path: str) -> None:
    """Generates comparison bar chart for regression candidates."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bench = regression_meta.get("benchmark_table", [])
    if not bench:
        return
    df = pd.DataFrame(bench)

    plt.figure(figsize=(10, 5))
    bar_width = 0.35
    x = np.arange(len(df))

    plt.bar(x - bar_width/2, df["rmse"], width=bar_width, label="Test RMSE (Lower is better)", color="#EF476F")
    plt.bar(x + bar_width/2, df["r2_score"], width=bar_width, label="Test R² Score (Higher is better)", color="#10B981")

    plt.xticks(x, df["model_name"], rotation=15, ha="right", fontsize=10)
    plt.ylabel("Metric Value")
    plt.title("BharatLens ML - Regression Model Benchmark Comparison", fontsize=12, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Regression comparison figure saved to {output_path}")

def plot_confusion_matrix(classification_meta: Dict[str, Any], output_path: str) -> None:
    """Plots multi-class confusion matrix heatmap."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = classification_meta.get("confusion_matrix", [])
    classes = classification_meta.get("class_order", [])
    if not cm:
        return

    cm_array = np.array(cm)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_array, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title("BharatLens ML - Trend Classification Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Confusion matrix figure saved to {output_path}")

def plot_global_feature_importance(shap_meta: Dict[str, Any], output_path: str) -> None:
    """Plots top 12 global SHAP importance drivers."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    importance = shap_meta.get("global_importance", [])
    if not importance:
        return

    df = pd.DataFrame(importance[:12])
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="mean_abs_shap", y="display_name", palette="viridis")
    plt.title("BharatLens ML - Top Global Prediction Drivers (Mean |SHAP| Value)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Absolute SHAP Impact")
    plt.ylabel("Indicator / Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Global SHAP feature importance figure saved to {output_path}")

def generate_model_comparison_report(output_path: str = os.path.join(REPORTS_DIR, "model_comparison_report.md")) -> str:
    """Generates an exhaustive Markdown Model Comparison Report backed by empirical results."""
    reg_meta = load_json(os.path.join(METADATA_DIR, "regression_metadata.json")) or {}
    clf_meta = load_json(os.path.join(METADATA_DIR, "classification_metadata.json")) or {}
    clu_meta = load_json(os.path.join(METADATA_DIR, "clustering_metadata.json")) or {}
    ano_meta = load_json(os.path.join(METADATA_DIR, "anomaly_metadata.json")) or {}
    shap_meta = load_json(os.path.join(METADATA_DIR, "shap_explanations.json")) or {}

    # Render figures
    plot_regression_comparison(reg_meta, os.path.join(FIGURES_DIR, "regression_benchmark.png"))
    plot_confusion_matrix(clf_meta, os.path.join(FIGURES_DIR, "classification_confusion_matrix.png"))
    plot_global_feature_importance(shap_meta, os.path.join(FIGURES_DIR, "global_shap_importance.png"))

    lines = [
        "# BharatLens ML - Empirical Model Evaluation and Comparison Report",
        "",
        "> Rigorous comparative evaluation of Machine Learning models trained on Indian district panel datasets.",
        "",
        "## 1. Executive Summary & Champion Models",
        "",
        f"- **Trend Regression Champion**: `{reg_meta.get('champion_model_name', 'N/A')}` (Selected by minimum test RMSE = `{reg_meta.get('best_rmse', 'N/A')}`)",
        f"- **Trend Classification Champion**: `{clf_meta.get('champion_model_name', 'N/A')}` (Selected by maximum Macro-F1 = `{clf_meta.get('best_macro_f1', 'N/A')}`)",
        f"- **Socioeconomic Clustering**: `K-Means (k={clu_meta.get('n_clusters', 4)})` with Silhouette Score = `{clu_meta.get('silhouette_score', 'N/A')}` and Davies-Bouldin Index = `{clu_meta.get('davies_bouldin_index', 'N/A')}`",
        f"- **Anomaly Detection**: `Isolation Forest (150 trees, contamination={ano_meta.get('contamination', 0.05)})` + `LOF Validation`",
        "",
        "## 2. Regression Model Benchmarks (District Score Delta & Trajectory)",
        "",
        "| Model Architecture | Test MAE | Test RMSE | Test R² | 5-Fold CV R² (Mean ± Std) | Decision |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    ]

    for row in reg_meta.get("benchmark_table", []):
        is_champ = (row["model_name"] == reg_meta.get("champion_model_name"))
        decision = "**CHAMPION (Selected)**" if is_champ else "Evaluated"
        lines.append(
            f"| **{row['model_name']}** | {row['mae']} | {row['rmse']} | {row['r2_score']} | "
            f"{row['cv_r2_mean']} ± {row['cv_r2_std']} | {decision} |"
        )

    lines.append("")
    lines.append("## 3. Multi-Class Classification Benchmarks (Improving / Stable / Declining / Anomalous)",)
    lines.append("")
    lines.append("| Model Architecture | Accuracy | Macro Precision | Macro Recall | Macro F1 | 5-Fold CV Macro-F1 | Decision |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for row in clf_meta.get("benchmark_table", []):
        is_champ = (row["model_name"] == clf_meta.get("champion_model_name"))
        decision = "**CHAMPION (Selected)**" if is_champ else "Evaluated"
        lines.append(
            f"| **{row['model_name']}** | {row['accuracy']} | {row['macro_precision']} | {row['macro_recall']} | "
            f"{row['macro_f1']} | {row['cv_f1_macro_mean']} ± {row['cv_f1_macro_std']} | {decision} |"
        )

    lines.append("")
    lines.append("## 4. Socioeconomic Cluster Tiers")
    lines.append("")
    lines.append("| Cluster ID | Tier Description | Districts | Share | Mean BharatLens Score | Mean Score Delta |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: |")

    for cid, prof in clu_meta.get("profiles", {}).items():
        lines.append(
            f"| `{cid}` | **{prof['cluster_name']}** | {prof['district_count']} | "
            f"{prof['pct_of_all_districts']}% | {prof['mean_bharatlens_score']} | {prof['mean_score_change']} |"
        )

    lines.append("")
    lines.append("## 5. Anomaly Detection Diagnostics")
    lines.append("")
    lines.append(f"- **Contamination Factor**: `{ano_meta.get('contamination', 0.05)}`")
    lines.append(f"- **Total Flagged Districts**: `{ano_meta.get('anomalous_districts_count', 0)}`")
    lines.append("- **Top Diagnosed Divergent Districts**:")
    for diag in ano_meta.get("diagnostics", [])[:5]:
        top_feats = ", ".join([f"{k}: {v}σ" for k, v in diag.get("top_deviating_features", {}).items()])
        lines.append(f"  - **{diag['district']} ({diag['state']})**: Anomaly Risk Score = `{diag['anomaly_risk_score']}`, Extreme Drivers = [{top_feats}]")

    lines.append("")
    lines.append("## 6. Top Global Model Drivers (Explainable AI / SHAP)")
    lines.append("")
    for item in shap_meta.get("global_importance", [])[:8]:
        lines.append(f"- **{item['display_name']}**: Mean |SHAP| Impact = `{item['mean_abs_shap']}`")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Model comparison report written to {output_path}")
    return content

if __name__ == "__main__":
    generate_model_comparison_report()
