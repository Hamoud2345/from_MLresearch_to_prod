"""Tests end-to-end de l'API FastAPI.

On entraine un petit modele dans un dossier temporaire puis on tape l'API
via le TestClient (request -> features -> model -> response, drift compris).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from power_forecaster.config import settings
from power_forecaster.features import build_default_pipeline
from power_forecaster.features.base import TARGET
from power_forecaster.models import create_model


@pytest.fixture(scope="module")
def client(tmp_path_factory, raw_data):
    # artifacts dans un dossier temp + petit modele entraine
    settings.artifacts_dir = tmp_path_factory.mktemp("artifacts")
    feats = build_default_pipeline().transform(raw_data).dropna().reset_index(drop=True)
    cols = [c for c in feats.columns if c not in {"timestamp", TARGET}]
    model = create_model("lightgbm", n_estimators=60).fit(feats[cols], feats[TARGET])
    model.save(settings.model_path())
    (settings.artifacts_dir / "reference.json").write_text(
        json.dumps({"price": feats[TARGET].tail(200).round(3).tolist()})
    )

    from power_forecaster.serving.api import app, predictor

    predictor._model = None  # force le rechargement depuis l'artifact temp
    predictor._reference = None
    return TestClient(app), feats


def _payload(feats, hours=24):
    tail = feats.tail(hours)
    return {
        "observations": [
            {
                "timestamp": ts.isoformat(),
                "load": float(r.load),
                "wind": float(r.wind),
                "solar": float(r.solar),
                "price_lag_24": float(r.price_lag_24),
                "price_lag_48": float(r.price_lag_48),
                "price_lag_168": float(r.price_lag_168),
            }
            for ts, r in zip(tail["timestamp"], tail.itertuples(), strict=True)
        ]
    }


def test_health_and_ready(client):
    c, _ = client
    assert c.get("/health").status_code == 200
    assert c.get("/ready").json()["model_loaded"] is True


def test_predict_returns_ordered_intervals(client):
    c, feats = client
    resp = c.post("/predict", json=_payload(feats))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 24
    for p in body["predictions"]:
        assert p["lower"] <= p["median"] <= p["upper"]
    assert "drift_psi" in body and "drift_alert" in body


def test_predict_rejects_empty_request(client):
    c, _ = client
    assert c.post("/predict", json={"observations": []}).status_code == 422


def test_metrics_endpoint_exposes_prometheus(client):
    c, feats = client
    c.post("/predict", json=_payload(feats))
    text = c.get("/metrics").text
    assert "ppf_predictions_total" in text
