# BharatLens ML - Data Quality and Validation Report

> Automated telemetry generated during data validation and pipeline execution.

## 1. Summary of Data Assets Audited

| Dataset Name | Records | Columns | States | Districts | Completeness | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Integrated_District_Master** | 643 | 49 | 37 | 633 | 100.0% | `PASSED` |

## 2. Integrity and Anomaly Checks

### Integrated_District_Master
- **Duplicate Rows**: `0`
- **Range Boundary Flags**: None. All bounded variables strictly satisfy domain constraints.
- **High Missingness Columns**: None. Feature density exceeds 80% across all indicators.
