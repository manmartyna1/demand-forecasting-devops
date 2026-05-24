from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert "model_name" in data
    assert "artifacts" in data


def test_predict_validation():
    response = client.post("/predict", json={
        "Product_Code": "Product_0979",
        "Warehouse": "Whse_J",
        "Product_Category": "Category_028",
        "Year": 2017,
        "Month": 20,
        "WeekOfYear": 1,
        "lag_1": 500.0,
        "lag_2": 500.0,
        "lag_4": 500.0,
        "rolling_mean_4": 500.0,
        "rolling_std_4": 0.0
    })

    assert response.status_code == 422


def test_predict_known_example():
    if not Path("rf_week.pkl").exists():
        pytest.skip("Model file rf_week.pkl is not available in this environment.")

    response = client.post("/predict", json={
        "Product_Code": "Product_0314",
        "Warehouse": "Whse_J",
        "Product_Category": "Category_011",
        "Year": 2016,
        "Month": 4,
        "WeekOfYear": 14,
        "lag_1": 16.0,
        "lag_2": 10.0,
        "lag_4": 100.0,
        "rolling_mean_4": 36.75,
        "rolling_std_4": 42.41
    })

    assert response.status_code == 200

    data = response.json()

    assert "predicted_weekly_demand" in data
    assert data["predicted_weekly_demand"] == pytest.approx(22.36, abs=0.1)