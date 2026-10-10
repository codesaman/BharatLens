import os
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.io import save_json

logger = get_logger("DataValidation")

def validate_dataframe_schema(df: pd.DataFrame, required_columns: List[str], dataset_name: str) -> bool:
    """Validates that all required columns exist in the DataFrame."""
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        logger.error(f"[{dataset_name}] Missing required columns: {missing_cols}")
        return False
    logger.info(f"[{dataset_name}] Schema check passed. All {len(required_columns)} required columns present.")
    return True

def run_data_quality_audit(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """Performs deep data quality and sanity checks on a dataset."""
    total_rows = len(df)
    total_cols = len(df.columns)

    # 1. Duplicate check
    duplicate_rows = int(df.duplicated().sum())

    # 2. Missing value audit
    missing_per_col = df.isnull().sum()
    missing_pct_per_col = (missing_per_col / total_rows * 100).round(2).to_dict()
    total_missing_cells = int(missing_per_col.sum())
    overall_completeness_pct = round(100.0 - (total_missing_cells / (total_rows * total_cols) * 100), 2)

    # 3. Numeric range & anomaly checks
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    range_anomalies: Dict[str, Any] = {}
    for col in numeric_cols:
        min_val = float(df[col].min()) if not df[col].empty else 0.0
        max_val = float(df[col].max()) if not df[col].empty else 0.0
        # If column name ends with '_pct' or '%'
        if "pct" in col.lower() or "%" in col.lower() or "rate" in col.lower():
            if min_val < 0.0 or max_val > 105.0:  # slight tolerance for rounding/rates
                range_anomalies[col] = {
                    "min": min_val,
                    "max": max_val,
                    "issue": "Values outside expected [0, 100] percentage bounds"
                }

    # 4. District and state identification check
    state_col = next((c for c in ["state", "state_name", "State"] if c in df.columns), None)
    dist_col = next((c for c in ["district", "district_name", "District", "District Name"] if c in df.columns), None)

    num_states = int(df[state_col].nunique()) if state_col else 0
    num_districts = int(df[dist_col].nunique()) if dist_col else 0

    audit_result = {
        "dataset_name": dataset_name,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "duplicate_rows": duplicate_rows,
        "num_states_covered": num_states,
        "num_districts_covered": num_districts,
        "overall_completeness_pct": overall_completeness_pct,
        "total_missing_cells": total_missing_cells,
        "range_anomalies": range_anomalies,
        "columns_with_high_missingness": {k: v for k, v in missing_pct_per_col.items() if v > 20.0},
        "status": "PASSED" if duplicate_rows == 0 and len(range_anomalies) == 0 else "WARNINGS_DETECTED"
    }

    logger.info(f"[{dataset_name}] Audit: {total_rows} rows, {num_districts} districts, Completeness: {overall_completeness_pct}%")
    return audit_result

def generate_markdown_quality_report(audit_results: List[Dict[str, Any]], output_path: str = "reports/data_quality_report.md") -> str:
    """Generates an exhaustive Markdown data quality audit report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    lines = [
        "# BharatLens ML - Data Quality and Validation Report",
        "",
        "> Automated telemetry generated during data validation and pipeline execution.",
        "",
        "## 1. Summary of Data Assets Audited",
        "",
        "| Dataset Name | Records | Columns | States | Districts | Completeness | Status |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    for audit in audit_results:
        lines.append(
            f"| **{audit['dataset_name']}** | {audit['total_rows']:,} | {audit['total_columns']} | "
            f"{audit['num_states_covered']} | {audit['num_districts_covered']} | "
            f"{audit['overall_completeness_pct']}% | `{audit['status']}` |"
        )

    lines.append("")
    lines.append("## 2. Integrity and Anomaly Checks")
    lines.append("")

    for audit in audit_results:
        lines.append(f"### {audit['dataset_name']}")
        lines.append(f"- **Duplicate Rows**: `{audit['duplicate_rows']}`")
        if audit['range_anomalies']:
            lines.append("- **Range Boundary Flags**:")
            for col, err in audit['range_anomalies'].items():
                lines.append(f"  - `{col}`: min={err['min']}, max={err['max']} ({err['issue']})")
        else:
            lines.append("- **Range Boundary Flags**: None. All bounded variables strictly satisfy domain constraints.")

        if audit['columns_with_high_missingness']:
            lines.append("- **Columns with Missingness > 20%**:")
            for col, pct in audit['columns_with_high_missingness'].items():
                lines.append(f"  - `{col}`: {pct}% missing")
        else:
            lines.append("- **High Missingness Columns**: None. Feature density exceeds 80% across all indicators.")
        lines.append("")

    report_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Data quality report successfully written to {output_path}")
    return report_content
