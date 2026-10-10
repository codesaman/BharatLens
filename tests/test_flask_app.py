import pytest
import json
from app import app, DF_MASTER

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"BHARATLENS ML" in res.data
    assert b"National District" in res.data

def test_api_summary(client):
    res = client.get("/api/summary")
    assert res.status_code == 200
    data = res.get_json()
    assert data["total_districts"] == len(DF_MASTER)
    assert "national_avg_score" in data
    assert "top_10_leading" in data
    assert len(data["top_10_leading"]) == 10
    assert len(data["map_points"]) == len(DF_MASTER)

def test_api_states(client):
    res = client.get("/api/states")
    assert res.status_code == 200
    states = res.get_json()
    assert isinstance(states, list)
    assert "Maharashtra" in states
    assert "Uttar Pradesh" in states

def test_api_districts(client):
    res = client.get("/api/districts?state=Maharashtra")
    assert res.status_code == 200
    districts = res.get_json()
    assert len(districts) > 0
    assert any(d["district"] == "Pune" for d in districts)

def test_api_district_detail(client):
    res = client.get("/api/district/Maharashtra/Pune")
    assert res.status_code == 200
    data = res.get_json()
    assert data["district"] == "Pune"
    assert data["state"] == "Maharashtra"
    assert "bharatlens_score" in data
    assert "dimensions" in data
    assert "historical_tracking" in data
    assert "ml_forecast" in data
    assert "peer_districts" in data

def test_api_state_detail(client):
    res = client.get("/api/state/Uttar Pradesh")
    assert res.status_code == 200
    data = res.get_json()
    assert data["state"] == "Uttar Pradesh"
    assert data["district_count"] > 0
    assert "leading_district" in data
    assert "districts" in data

def test_api_compare(client):
    res = client.get("/api/compare?d1=Maharashtra___Pune&d2=Karnataka___Bangalore")
    assert res.status_code == 200
    data = res.get_json()
    assert "district_1" in data
    assert "district_2" in data
    assert data["district_1"]["district"] == "Pune"

def test_api_ml_insights(client):
    res = client.get("/api/ml-insights")
    assert res.status_code == 200
    data = res.get_json()
    assert "regression_benchmarks" in data
    assert "classification_benchmarks" in data
    assert "global_shap_importance" in data

def test_api_data_sources(client):
    res = client.get("/api/data-sources")
    assert res.status_code == 200
    data = res.get_json()
    assert "nfhs_district_change" in data
    assert "census_district_pca" in data

def test_api_search(client):
    res = client.get("/api/search?q=pune")
    assert res.status_code == 200
    results = res.get_json()
    assert len(results) > 0
    assert any(r["district"] == "Pune" for r in results)

def test_api_export_csv(client):
    res = client.get("/api/export-csv")
    assert res.status_code == 200
    assert "csv" in res.headers.get("Content-Type", "").lower() or "excel" in res.headers.get("Content-Type", "").lower()
