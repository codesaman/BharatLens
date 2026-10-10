import pytest
import os
from src.data.download import check_data_gov_in_api, download_file

def test_missing_api_key_graceful_handling():
    # If DATA_GOV_IN_API_KEY is not set or empty, check_data_gov_in_api returns fallback status
    status = check_data_gov_in_api()
    assert "status" in status
    assert "key_present" in status
    assert "mode" in status

def test_cached_download_skips_network():
    # Calling download_file on existing valid file should immediately return True without downloading
    dest = os.path.join("data", "raw", "census_district_pca.csv")
    if os.path.exists(dest):
        res = download_file("https://example.com/invalid_url_that_would_fail", dest)
        assert res is True, "Should succeed via cached file validation"
