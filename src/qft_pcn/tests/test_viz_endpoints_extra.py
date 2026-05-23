"""Tests for /presets, /params/schema, /runs/{id}/{pause,resume,step},
and /export/run/{id} JSONL export.
"""

import json

import pytest
from fastapi.testclient import TestClient

from src.qft_pcn.viz.server import app
from src.qft_pcn.viz.presets import PRESETS, PARAM_SCHEMA


@pytest.fixture
def client():
    return TestClient(app)


def test_presets_endpoint_returns_catalog(client):
    res = client.get("/presets")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) == len(PRESETS)
    keys = {"id", "layer", "label", "description", "spec_overrides"}
    for entry in body:
        assert keys.issubset(entry.keys())


def test_params_schema_endpoint_returns_per_layer_schema(client):
    res = client.get("/params/schema")
    assert res.status_code == 200
    body = res.json()
    for layer in PARAM_SCHEMA:
        assert layer in body
        assert body[layer]["type"] == "object"
