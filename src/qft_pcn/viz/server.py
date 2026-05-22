"""FastAPI app streaming QFT-PCN layer state to the browser.

`POST /run` registers a `RunSpec` and returns an opaque run id. Connecting a
WebSocket to `/ws/{run_id}` replays that run, streaming one JSON `Frame` per
simulation step followed by a ``{"done": true}`` sentinel. `POST /export`
schedules a background Manim render job that re-runs the simulation and
renders the requested layer to an MP4; poll ``GET /export/{job_id}`` for
progress and download the result from ``GET /export/{job_id}/download``.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import (BackgroundTasks, FastAPI, HTTPException, WebSocket,
                      WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .runs import RunSpec, RunRegistry, run_simulation

app = FastAPI(title="QFT-PCN Visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_registry = RunRegistry()
# job_id -> {"job_id", "run_id", "layer", "status", ...}. In-memory: a render
# job transitions queued -> running -> done|error, gaining an "output" path on
# success or an "error" message on failure. Bounded to the most recent
# `_MAX_EXPORT_JOBS` entries (oldest evicted) so the dict cannot grow without
# limit over the server's lifetime.
_export_jobs: dict[str, dict] = {}
_MAX_EXPORT_JOBS = 50

# Persistent location for rendered MP4s. Each job renders into a throwaway
# temp dir, then copies the final MP4 here so the temp dir can be removed
# while `/download` still serves the result.
_EXPORTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "viz_exports"


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/run")
def start_run(spec: dict):
    """Register a simulation run and return its id."""
    run_id = uuid.uuid4().hex
    _registry.add(run_id, RunSpec.from_dict(spec))
    return {"run_id": run_id}


@app.websocket("/ws/{run_id}")
async def stream(ws: WebSocket, run_id: str):
    """Stream one JSON Frame per step, then a {"done": true} sentinel."""
    await ws.accept()
    spec = _registry.get(run_id)
    if spec is None:
        await ws.close(code=4004)
        return
    try:
        for frame in run_simulation(spec):
            await ws.send_text(frame.to_json())
        await ws.send_text('{"done": true}')
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        # A mid-stream failure would otherwise close the socket abruptly with
        # no application-level signal; send a terminal error frame first.
        await ws.send_text(json.dumps({"error": str(exc)}))
        await ws.close(code=1011)


def _run_export(job_id: str, spec: RunSpec, layer: str) -> None:
    """Background task: re-run the simulation and render `layer` to an MP4.

    `manim` (via `render_layer`) is imported LAZILY here, never at module
    scope, so the server starts and serves every non-export endpoint even
    when manim is absent. A missing manim simply marks the job `error`.
    """
    job = _export_jobs[job_id]
    job["status"] = "running"
    out_dir = None
    try:
        # Lazy import: keeps `import server` manim-independent.
        from .manim.render import render_layer

        frames = [f.to_dict() for f in run_simulation(spec)]
        # Render into a throwaway temp dir, then copy the final MP4 into the
        # persistent exports dir so the temp dir can be cleaned up.
        out_dir = tempfile.mkdtemp(prefix=f"qftpcn_export_{job_id}_")
        rendered = render_layer(layer, frames, out_dir, quality="low")
        _EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        final = _EXPORTS_DIR / f"{job_id}_{layer}.mp4"
        shutil.copyfile(rendered, final)
        # Publish-after-fill: set every result field BEFORE flipping status to
        # "done" so a concurrent /download poll never sees done without output.
        job["output"] = str(final)
        job["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        # Publish-after-fill: set the error message before the status.
        job["error"] = str(exc)
        job["status"] = "error"
    finally:
        # Always remove the temp render dir; the MP4 was copied out above.
        if out_dir is not None:
            shutil.rmtree(out_dir, ignore_errors=True)


@app.post("/export")
def start_export(req: dict, background: BackgroundTasks):
    """Schedule a background Manim render job for a registered run.

    Body: ``{"run_id": <id>, "layer": <optional layer>}``. `layer` defaults to
    the first layer in the run's spec. Returns the job dict (status
    ``queued``); poll ``GET /export/{job_id}`` for progress.
    """
    run_id = req.get("run_id")
    spec = _registry.get(run_id) if run_id else None
    if spec is None:
        raise HTTPException(status_code=404, detail="run not found")

    layer = req.get("layer") or (spec.layers[0] if spec.layers else "manifold")
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "run_id": run_id,
        "layer": layer,
        "status": "queued",
    }
    _export_jobs[job_id] = job
    # Bound the in-memory job table: evict oldest entries (dicts preserve
    # insertion order) once the cap is exceeded.
    while len(_export_jobs) > _MAX_EXPORT_JOBS:
        _export_jobs.pop(next(iter(_export_jobs)))
    background.add_task(_run_export, job_id, spec, layer)
    return job


@app.get("/export/{job_id}")
def export_status(job_id: str):
    """Return the status of an export job."""
    job = _export_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/export/{job_id}/download")
def export_download(job_id: str):
    """Serve the rendered MP4 for a finished export job."""
    job = _export_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.get("status") != "done" or not job.get("output"):
        raise HTTPException(status_code=404, detail="export not ready")
    return FileResponse(job["output"], media_type="video/mp4",
                        filename=f"{job.get('layer', 'export')}.mp4")
