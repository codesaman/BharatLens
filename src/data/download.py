import argparse
import os
import sys
import time
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

from src.utils.logger import get_logger
from src.utils.io import load_yaml

load_dotenv()
logger = get_logger("DataDownloader")

CONFIG_PATH = os.path.join("config", "data_sources.yaml")
RAW_DATA_DIR = os.getenv("CACHE_DIR", os.path.join("data", "raw"))
EXTERNAL_DATA_DIR = os.path.join("data", "external")

FILE_DESTINATIONS = {
    "nfhs_district_change": os.path.join(RAW_DATA_DIR, "nfhs_district_change.csv"),
    "census_district_pca": os.path.join(RAW_DATA_DIR, "census_district_pca.csv"),
    "mgnrega_district_employment": os.path.join(RAW_DATA_DIR, "mgnrega_district_employment.csv"),
    "subdivision_rainfall": os.path.join(RAW_DATA_DIR, "subdivision_rainfall.csv"),
    "district_boundaries_geojson": os.path.join(EXTERNAL_DATA_DIR, "districts.geojson"),
}

def download_file(url: str, destination: str, max_retries: int = 3, timeout: int = 30) -> bool:
    """Downloads a file with streaming, retries, and caching validation."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)

    if os.path.exists(destination) and os.path.getsize(destination) > 1024:
        logger.info(f"Using cached file: {destination} ({os.path.getsize(destination) / 1024:.1f} KB)")
        return True

    headers = {"User-Agent": "BharatLens-ML/1.0 (Public Research Data Ingestion)"}
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Downloading from {url} (Attempt {attempt}/{max_retries})...")
            response = requests.get(url, headers=headers, stream=True, timeout=timeout)
            response.raise_for_status()

            temp_dest = destination + ".tmp"
            with open(temp_dest, "wb") as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

            if os.path.exists(destination):
                os.remove(destination)
            os.rename(temp_dest, destination)
            logger.info(f"Successfully downloaded to {destination} ({os.path.getsize(destination) / 1024:.1f} KB)")
            return True
        except Exception as e:
            logger.warning(f"Download attempt {attempt} failed for {url}: {e}")
            if attempt < max_retries:
                time.sleep(2 * attempt)
            else:
                logger.error(f"Exhausted retries for {url}.")
                return False
    return False

def check_data_gov_in_api() -> Dict[str, Any]:
    """Inspects optional data.gov.in API key configuration."""
    api_key = os.getenv("DATA_GOV_IN_API_KEY", "").strip()
    if not api_key:
        logger.info("No DATA_GOV_IN_API_KEY detected in environment. Using robust public mirrors & open releases.")
        return {"status": "unconfigured", "key_present": False, "mode": "mirror_fallback"}
    
    logger.info("DATA_GOV_IN_API_KEY detected in environment. Live API synchronization enabled.")
    return {"status": "configured", "key_present": True, "mode": "live_api"}

def download_all_datasets(force: bool = False) -> Dict[str, bool]:
    """Downloads all configured public datasets."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)

    catalog = load_yaml(CONFIG_PATH)
    sources = catalog.get("sources", {})
    results: Dict[str, bool] = {}

    check_data_gov_in_api()

    for key, dest in FILE_DESTINATIONS.items():
        if key not in sources:
            logger.warning(f"Key '{key}' not found in data_sources.yaml configuration.")
            continue

        source_info = sources[key]
        raw_url = source_info.get("raw_url")
        if not raw_url:
            continue

        if force and os.path.exists(dest):
            logger.info(f"Force re-download requested. Removing {dest}")
            os.remove(dest)

        success = download_file(raw_url, dest)
        results[key] = success

    success_count = sum(1 for v in results.values() if v)
    logger.info(f"Data download finished: {success_count}/{len(results)} sources ready.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download public datasets for BharatLens ML")
    parser.add_argument("--force", action="store_true", help="Force re-download of existing files")
    args = parser.parse_args()

    results = download_all_datasets(force=args.force)
    if not all(results.values()):
        logger.warning("Some datasets could not be downloaded. Check logs for details.")
    else:
        logger.info("All datasets are fully available locally.")
