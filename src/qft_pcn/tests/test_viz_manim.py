"""Tests for the Manim export pipeline and the wired /export endpoint.

Run from repo root with --noconftest (the repo conftest imports the Unix-only
`resource` module and crashes on Windows)::

    python -m pytest src/qft_pcn/tests/test_viz_manim.py -v --noconftest

The render tests SKIP when `manim` is not installed -- that is an acceptable
outcome. The no-manim test (`test_server_imports_without_manim`) always runs
and proves the server stays manim-independent.
"""

import importlib.util
import time

import pytest
from fastapi.testclient import TestClient

# Per-test skip marker: skips only the manim-dependent tests, leaving the
# no-manim guard test to always run.
requires_manim = pytest.mark.skipif(
    importlib.util.find_spec("manim") is None,
    reason="manim is not installed",
)


def test_server_imports_without_manim():
    """Importing the server and hitting /health must work without manim.

    This guards the CRITICAL lazy-import requirement: `render_layer` must be
    imported lazily inside the background task, never at server module scope.
    """
    from src.qft_pcn.viz.server import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"


def test_render_module_importable_without_manim():
    """`render.py` must be importable even when manim is absent."""
    import src.qft_pcn.viz.manim.render as render_mod

    assert hasattr(render_mod, "render_layer")


@requires_manim
def test_render_produces_mp4(tmp_path):
    from src.qft_pcn.viz.manim.render import render_layer

    frames = [{"step": i, "layer_states": {"qpcn": {"energy": 1.0 - i * 0.1}}}
              for i in range(3)]
    out = render_layer("qpcn", frames, tmp_path, quality="low")
    assert out.exists() and out.suffix == ".mp4"


@requires_manim
def test_export_endpoint_renders():
    """POST /export, poll the job to `done`, assert the MP4 downloads."""
    from src.qft_pcn.viz.server import app

    c = TestClient(app)
    run_id = c.post("/run", json={"layers": ["qpcn"],
                                  "steps": 3, "grid": 8}).json()["run_id"]
    job_id = c.post("/export", json={"run_id": run_id,
                                     "layer": "qpcn"}).json()["job_id"]

    deadline = time.time() + 180
    status = "queued"
    while time.time() < deadline:
        status = c.get(f"/export/{job_id}").json()["status"]
        if status in ("done", "error"):
            break
        time.sleep(0.5)

    assert status == "done", f"export job ended as {status}"
    dl = c.get(f"/export/{job_id}/download")
    assert dl.status_code == 200
