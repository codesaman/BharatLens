import re
import unicodedata
from typing import Dict, Optional

STATE_ALIASES: Dict[str, str] = {
    "orissa": "odisha",
    "uttaranchal": "uttarakhand",
    "pondicherry": "puducherry",
    "nct of delhi": "delhi",
    "jammu and kashmir": "jammu & kashmir",
    "andaman and nicobar islands": "andaman & nicobar island",
    "andaman & nicobar islands": "andaman & nicobar island",
    "dadra and nagar haveli and daman and diu": "dadra & nagar haveli and daman & diu",
    "daman and diu": "dadra & nagar haveli and daman & diu",
    "dadra and nagar haveli": "dadra & nagar haveli and daman & diu",
}

DISTRICT_ALIASES: Dict[str, str] = {
    "allahabad": "prayagraj",
    "faizabad": "ayodhya",
    "bangalore urban": "bengaluru urban",
    "bangalore rural": "bengaluru rural",
    "gurgaon": "gurugram",
    "firozpur": "ferozepur",
    "kanpur": "kanpur nagar",
    "pashchim champaran": "west champaran",
    "purba champaran": "east champaran",
    "purbi singhbhum": "east singhbhum",
    "pashchimi singhbhum": "west singhbhum",
    "paschim medinipur": "west medinipur",
    "purba medinipur": "east medinipur",
    "sant ravidas nagar": "bhadohi",
    "jyotiba phule nagar": "amroha",
    "kanshiram nagar": "kasganj",
    "chhatrapati shahuji maharaj nagar": "amethi",
    "panchsheel nagar": "hapur",
    "prabuddhanagar": "shamli",
    "sambhal (bhim nagar)": "sambhal",
    "lehdakh": "leh",
    "badgam": "budgam",
}

def clean_text(text: Optional[str]) -> str:
    """Normalizes string casing, diacritics, and symbols."""
    if not text or not isinstance(text, str):
        return ""
    # NFKD normalization
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = text.lower().strip()
    text = re.sub(r"[\*#\(\)\[\]\{\}\.,_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_state_name(state: Optional[str]) -> str:
    """Standardizes state name to canonical form."""
    cleaned = clean_text(state)
    return STATE_ALIASES.get(cleaned, cleaned).title()

def normalize_district_name(district: Optional[str]) -> str:
    """Standardizes district name to canonical form."""
    cleaned = clean_text(district)
    # Strip suffixes like '_1', 'district', 'dist'
    cleaned = re.sub(r"\b(district|dist)\b", "", cleaned).strip()
    cleaned = DISTRICT_ALIASES.get(cleaned, cleaned)
    return cleaned.title()
