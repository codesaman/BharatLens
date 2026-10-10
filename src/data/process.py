from src.data.processing import merge_and_harmonize_datasets
from src.features.engineering import build_and_save_feature_store
from src.utils.logger import get_logger

logger = get_logger("DataProcessCLI")

def main():
    logger.info("Executing Data Harmonization and Feature Pipeline...")
    merge_and_harmonize_datasets()
    build_and_save_feature_store()
    logger.info("Data processing and feature engineering complete.")

if __name__ == "__main__":
    main()
