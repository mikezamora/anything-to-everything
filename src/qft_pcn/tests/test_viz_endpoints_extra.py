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


def test_pause_resume_step_404_when_not_streaming(client):
    """Lifecycle endpoints require an active WS connection for the run."""
    for verb in ("pause", "resume", "step"):
        res = client.post(f"/runs/nonexistent/{verb}")
        assert res.status_code == 404


def test_jsonl_export_streams_one_object_per_frame(client):
    run = client.post("/run", json={"layers": ["manifold"], "steps": 3,
                                     "grid": 8}).json()
    run_id = run["run_id"]
    res = client.get(f"/export/run/{run_id}")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("application/x-ndjson")
    lines = [ln for ln in res.text.splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "step" in obj
        assert "layer_states" in obj


def test_jsonl_export_404_for_unknown_run(client):
    res = client.get("/export/run/nope")
    assert res.status_code == 404


def test_set_params_404_when_run_not_streaming(client):
    """`/runs/{id}/params` requires an active WS connection for the run."""
    res = client.post("/runs/nonexistent/params", json={"qpcn": {"mass": 1.0}})
    assert res.status_code == 404


def test_set_params_400_when_no_qpcn_substrate(client, monkeypatch):
    """When the active controller has no qpcn substrate, the endpoint 400s."""
    from src.qft_pcn.viz import server as srv
    from src.qft_pcn.viz.runs import RunSpec
    from src.qft_pcn.viz.controller import RunController

    ctrl = RunController(RunSpec(layers=["manifold"], steps=1))
    ctrl._substrates = {"qpcn": None}
    srv._controllers["live"] = ctrl
    try:
        res = client.post("/runs/live/params", json={"qpcn": {"mass": 1.0}})
        assert res.status_code == 400
        assert "qpcn" in res.json()["detail"].lower()
    finally:
        srv._controllers.pop("live", None)
