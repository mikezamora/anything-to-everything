"""Tests for the viz FastAPI server: health, run registry, WS stream, export.

Run from repo root with --noconftest (the repo conftest imports the Unix-only
`resource` module and crashes on Windows)::

    python -m pytest src/qft_pcn/tests/test_viz_server.py -v --noconftest
"""

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.qft_pcn.viz.server import app


def test_health():
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"


def test_run_returns_run_id():
    c = TestClient(app)
    r = c.post("/run", json={"layers": ["manifold"], "steps": 5, "grid": 8})
    assert r.status_code == 200
    assert "run_id" in r.json()


def test_ws_streams_frames():
    c = TestClient(app)
    run_id = c.post("/run", json={"layers": ["manifold"],
                                  "steps": 3, "grid": 8}).json()["run_id"]
    with c.websocket_connect(f"/ws/{run_id}") as ws:
        first = ws.receive_json()
        assert first["step"] == 0
        assert "manifold" in first["layer_states"]


def test_ws_unknown_run_closes():
    c = TestClient(app)
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect("/ws/deadbeef") as ws:
            ws.receive_json()


def test_ws_emits_done_sentinel():
    c = TestClient(app)
    run_id = c.post("/run", json={"layers": ["manifold"],
                                  "steps": 2, "grid": 8}).json()["run_id"]
    with c.websocket_connect(f"/ws/{run_id}") as ws:
        frames = []
        while True:
            msg = ws.receive_json()
            if msg.get("done"):
                break
            frames.append(msg)
        assert len(frames) == 2
        assert [f["step"] for f in frames] == [0, 1]


def test_ws_streams_qpcn_layer():
    c = TestClient(app)
    run_id = c.post("/run", json={"layers": ["qpcn"],
                                  "steps": 2, "grid": 8}).json()["run_id"]
    with c.websocket_connect(f"/ws/{run_id}") as ws:
        first = ws.receive_json()
        assert "qpcn" in first["layer_states"]


def test_ws_streams_mera_layer():
    c = TestClient(app)
    run_id = c.post("/run", json={"layers": ["mera"],
                                  "steps": 2, "grid": 8}).json()["run_id"]
    with c.websocket_connect(f"/ws/{run_id}") as ws:
        first = ws.receive_json()
        assert "mera" in first["layer_states"]


def test_export_returns_job_id():
    c = TestClient(app)
    run_id = c.post("/run", json={"layers": ["manifold"],
                                  "steps": 2, "grid": 8}).json()["run_id"]
    r = c.post("/export", json={"run_id": run_id})
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "queued"


def test_export_status_lookup():
    c = TestClient(app)
    run_id = c.post("/run", json={"layers": ["manifold"],
                                  "steps": 2, "grid": 8}).json()["run_id"]
    job_id = c.post("/export", json={"run_id": run_id}).json()["job_id"]
    r = c.get(f"/export/{job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
