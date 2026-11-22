from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

from api.main import app, model_state


VALID_INVOICE = {
    "amount": 1000.0,
    "payment_terms_days": 30,
    "line_item_count": 2,
    "invoice_date": "2025-01-01",
    "due_date": "2025-01-31",
}


def test_health_and_startup():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["model_version"] == "1.0.0"
        assert body["last_training_date"] == "unknown"
        assert body["threshold"] == 0.10


def test_successful_fraud_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", json=VALID_INVOICE)
        assert response.status_code == 200
        body = response.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert body["threshold"] == 0.10
        assert body["fraud_flag"] == (body["fraud_probability"] >= 0.10)
        assert body["prediction"] == ("Fraudulent" if body["fraud_flag"] else "Legitimate")


def test_legitimate_prediction_from_mock_model():
    pipeline = Mock()
    pipeline.classes_ = np.array([0, 1])
    pipeline.predict_proba.return_value = np.array([[0.92, 0.08]])
    old_pipeline, old_threshold = model_state.pipeline, model_state.threshold
    try:
        with TestClient(app) as client:
            model_state.pipeline, model_state.threshold = pipeline, 0.10
            response = client.post("/predict", json=VALID_INVOICE)
            assert response.status_code == 200
            body = response.json()
            assert body["fraud_probability"] == 0.08
            assert body["fraud_flag"] is False
            assert body["prediction"] == "Legitimate"
    finally:
        model_state.pipeline, model_state.threshold = old_pipeline, old_threshold


def test_invalid_json():
    with TestClient(app) as client:
        response = client.post("/predict", content="{", headers={"content-type": "application/json"})
        assert response.status_code == 422


def test_missing_required_field():
    with TestClient(app) as client:
        payload = dict(VALID_INVOICE)
        del payload["amount"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_invalid_numeric_values_and_extra_field():
    with TestClient(app) as client:
        payload = dict(VALID_INVOICE, amount=-1)
        assert client.post("/predict", json=payload).status_code == 422
        payload = dict(VALID_INVOICE, line_item_count=2.5)
        assert client.post("/predict", json=payload).status_code == 422
        payload = dict(VALID_INVOICE, unexpected=1)
        assert client.post("/predict", json=payload).status_code == 422


def test_invalid_dates_and_terms():
    with TestClient(app) as client:
        assert client.post("/predict", json=dict(VALID_INVOICE, invoice_date="not-a-date")).status_code == 422
        assert client.post("/predict", json=dict(VALID_INVOICE, payment_terms_days=20)).status_code == 422
        assert client.post("/predict", json=dict(VALID_INVOICE, due_date="2025-01-30")).status_code == 422


def test_missing_model_artifact_returns_unhealthy():
    with patch("api.dependencies.load_config", side_effect=FileNotFoundError("missing")):
        model_state.load()
        assert model_state.pipeline is None
        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 503


def test_invalid_model_configuration():
    state = model_state
    with patch("api.dependencies.load_config", side_effect=ValueError("bad config")):
        state.load()
        assert state.pipeline is None
        assert state.error == "ValueError"


def test_model_prediction_failure():
    pipeline = Mock()
    pipeline.classes_ = np.array([0, 1])
    pipeline.predict_proba.side_effect = RuntimeError("failure")
    old_pipeline, old_threshold = model_state.pipeline, model_state.threshold
    try:
        with TestClient(app) as client:
            model_state.pipeline, model_state.threshold = pipeline, 0.10
            response = client.post("/predict", json=VALID_INVOICE)
            assert response.status_code == 503
    finally:
        model_state.pipeline, model_state.threshold = old_pipeline, old_threshold


def test_positive_class_validation():
    pipeline = Mock()
    pipeline.classes_ = np.array([1, 0])
    with patch("api.dependencies.joblib.load", return_value=pipeline):
        with patch("api.dependencies.load_config", return_value={"threshold": 0.10, "artifact": "models/threshold_optimized_pipeline.joblib"}):
            with patch("api.dependencies.resolve_model_artifact", return_value=object()):
                state = model_state
                state.load()
                assert state.pipeline is None
