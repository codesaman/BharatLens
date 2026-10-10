import os
import sys
import time
from src.utils.logger import get_logger
from src.data.download import download_all_datasets
from src.data.processing import merge_and_harmonize_datasets
from src.features.engineering import build_and_save_feature_store
from src.ml.train import train_entire_ml_pipeline
from src.ml.evaluation import generate_model_comparison_report

logger = get_logger("EndToEndPipeline")

def run_complete_pipeline(force_download: bool = False):
    start_time = time.time()
    logger.info("==========================================================================")
    logger.info("  STARTING BHARATLENS ML END-TO-END AUTONOMOUS PIPELINE")
    logger.info("==========================================================================")

    # 1. Download
    logger.info("\n>>> STAGE 1: Dataset Acquisition & Validation <<<")
    download_all_datasets(force=force_download)

    # 2. Harmonization
    logger.info("\n>>> STAGE 2: Multi-Domain Harmonization & Data Cleaning <<<")
    merge_and_harmonize_datasets()

    # 3. Feature Engineering & Composite Scoring
    logger.info("\n>>> STAGE 3: BharatLens Index Scoring & Feature Engineering <<<")
    build_and_save_feature_store()

    # 4. Machine Learning Training & Evaluation
    logger.info("\n>>> STAGE 4: Machine Learning Modeling, Benchmarking & Explainability <<<")
    train_entire_ml_pipeline()

    # 5. Final Evaluation Reports
    logger.info("\n>>> STAGE 5: Synthesis of Governance & Model Evaluation Reports <<<")
    generate_model_comparison_report()

    elapsed = time.time() - start_time
    logger.info("==========================================================================")
    logger.info(f"  BHARATLENS ML PIPELINE FINISHED IN {elapsed:.1f} SECONDS")
    logger.info("  All artifacts, serialized models, and processed feature stores are ready.")
    logger.info("  Launch Streamlit via: streamlit run app.py")
    logger.info("==========================================================================")

if __name__ == "__main__":
    force = "--force" in sys.argv
    run_complete_pipeline(force_download=force)
