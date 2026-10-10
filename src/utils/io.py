import json
import os
from typing import Any, Dict, Optional
import joblib
import pandas as pd
import yaml
from src.utils.logger import get_logger

logger = get_logger("IOUtils")

def load_yaml(filepath: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(filepath):
        logger.error(f"Configuration file not found: {filepath}")
        raise FileNotFoundError(f"Missing config file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Saves data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)
    logger.debug(f"Saved JSON to {filepath}")

def load_json(filepath: str) -> Any:
    """Loads a JSON file."""
    if not os.path.exists(filepath):
        logger.warning(f"JSON file not found: {filepath}")
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_model(model: Any, filepath: str) -> None:
    """Serializes a model or pipeline using Joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"Model saved successfully to {filepath}")

def load_model(filepath: str) -> Optional[Any]:
    """Loads a serialized model using Joblib."""
    if not os.path.exists(filepath):
        logger.warning(f"Model file not found: {filepath}")
        return None
    return joblib.load(filepath)

def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """Saves DataFrame as CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index)
    logger.info(f"Saved DataFrame with shape {df.shape} to {filepath}")

def load_dataframe(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """Loads DataFrame from CSV if it exists."""
    if not os.path.exists(filepath):
        logger.warning(f"CSV file not found: {filepath}")
        return None
    return pd.read_csv(filepath, **kwargs)
