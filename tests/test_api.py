from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.api import main as api


class FakeModel:
    def predict(self, features):
        assert "MonthlyChargesPerTenure" in features.columns
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.2, 0.8]])


def test_health_returns_service_status() -> None:
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_churn_response_with_mocked_model(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_model", lambda: FakeModel())
    client = TestClient(api.app)

    response = client.post("/predict", json={})

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "churn_label": "Yes",
        "churn_probability": 0.8,
        "model_uri": "models:/churn360-champion@champion",
    }
