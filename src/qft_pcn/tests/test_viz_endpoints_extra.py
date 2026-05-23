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


from unittest.mock import patch


def test_dsl_schema_endpoint(client):
    r = client.get("/dsl/schema")
    assert r.status_code == 200
    assert r.json()["$id"].endswith("dsl/v1.json")


def test_dsl_models_endpoint_503_when_ollama_down(client):
    # When httpx.get raises, the endpoint should surface 503, not 500.
    with patch("src.qft_pcn.viz.server.llm.list_models",
               side_effect=Exception("connection refused")):
        r = client.get("/dsl/models")
        assert r.status_code == 503


def test_dsl_translate_returns_dsl(client):
    fake = {"dsl": {"fields": [{"name": "A", "cutoff": 2}],
                    "hamiltonian": {"terms": [
                        {"kind": "mass", "species": "A", "coefficient": 1.0}]},
                    "observables": [
                        {"operator": "n", "site": 0,
                         "species": "A", "target": 0.25}]}}
    with patch("src.qft_pcn.viz.server.llm.generate_dsl",
               return_value=fake):
        r = client.post("/dsl/translate",
                        json={"prompt": "go", "model": "gemma3:4b"})
        assert r.status_code == 200
        assert r.json()["dsl"]["fields"][0]["name"] == "A"


def test_dsl_run_registers_runspec(client):
    dsl = {"fields": [{"name": "A", "cutoff": 2}],
           "hamiltonian": {"terms": [
               {"kind": "mass", "species": "A", "coefficient": 1.0},
               {"kind": "kinetic", "species": "A", "coefficient": 0.5}]},
           "observables": [
               {"operator": "n", "site": 0,
                "species": "A", "target": 0.25}]}
    r = client.post("/dsl/run", json={"dsl": dsl})
    assert r.status_code == 200
    assert "run_id" in r.json()


def test_dsl_run_validates(client):
    r = client.post("/dsl/run", json={"dsl": {"fields": []}})
    assert r.status_code == 400


def test_dsl_export_returns_dsl_for_known_run(client):
    run = client.post("/run", json={"layers": ["qpcn"], "steps": 2,
                                     "grid": 8}).json()
    r = client.get(f"/dsl/export/{run['run_id']}")
    assert r.status_code == 200
    body = r.json()
    assert "fields" in body and "hamiltonian" in body
