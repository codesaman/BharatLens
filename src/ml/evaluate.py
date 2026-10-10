import os
import sys
from src.utils.logger import get_logger
from src.ml.evaluation import generate_model_comparison_report

logger = get_logger("EvaluatePipeline")

def main():
    logger.info("Executing BharatLens ML standalone evaluation runner...")
    report_path = os.path.join("reports", "model_comparison_report.md")
    report = generate_model_comparison_report(report_path)
    print("\n" + "="*80)
    print("BharatLens ML - Model Evaluation Summary")
    print("="*80)
    # Print first 35 lines of report
    for line in report.split("\n")[:40]:
        print(line)
    print("...\n[Full report available at: " + report_path + "]")

if __name__ == "__main__":
    main()
