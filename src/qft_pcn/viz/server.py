"""FastAPI app streaming QFT-PCN layer state to the browser.

`POST /run` registers a `RunSpec` and returns an opaque run id. Connecting a
WebSocket to `/ws/{run_id}` replays that run, streaming one JSON `Frame` per
simulation step followed by a ``{"done": true}`` sentinel. `POST /export`
queues a (stub) Manim render job; the real render is wired in a later task.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .runs import RunSpec, RunRegistry, run_simulation

app = FastAPI(title="QFT-PCN Visualizer")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

_registry = RunRegistry()
# job_id -> {"job_id", "run_id", "status"}. In-memory; the real render
# pipeline (Task 5) replaces the stub status transitions.
_export_jobs: dict[str, dict] = {}


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


@app.post("/export")
def start_export(req: dict):
    """Queue a (stub) Manim render job for a previously registered run."""
    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "run_id": req.get("run_id"),
        "status": "queued",
    }
    _export_jobs[job_id] = job
    return job


@app.get("/export/{job_id}")
def export_status(job_id: str):
    """Return the status of a queued export job."""
    job = _export_jobs.get(job_id)
    if job is None:
        return {"job_id": job_id, "status": "not_found"}
    return job
