"""Synchronous JSON-RPC 2.0 server over stdio for the QPCN bridge (spec §6).

Methods: problem.run, problem.diagnose, dsl.validate, health.check.

Stdout is the protocol channel; all logs go to stderr.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .dsl.pipeline import validate_dsl
from .errors import BridgeError, InternalError
from .runtime import run_problem, diagnose_problem


log = logging.getLogger("qft_pcn.bridge")


_VERSION = "0.1.0"


def _subproject_status() -> dict[str, str]:
    status: dict[str, str] = {}
    try:
        import src.qft_pcn.logic  # noqa: F401
        status["A"] = "ok"
    except Exception:
        status["A"] = "absent"
    for name in ("B", "C", "D", "E", "F"):
        status[name] = "stub"
    status["G"] = "ok"
    return status


def _ok(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str,
         data: dict[str, Any] | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _err_from_bridge(req_id: Any, exc: BridgeError) -> dict[str, Any]:
    return _err(req_id, -32001,
                f"{exc.code}: {exc.message}",
                data={"code": exc.code, "message": exc.message,
                      "details": exc.details})


def handle_request(req: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one JSON-RPC request and produce one response."""
    req_id = req.get("id")
    if req.get("jsonrpc") != "2.0":
        return _err(req_id, -32600, "jsonrpc field must be '2.0'")
    method = req.get("method")
    params = req.get("params") or {}
    if not isinstance(method, str):
        return _err(req_id, -32600, "method must be a string")
    if not isinstance(params, dict):
        return _err(req_id, -32602, "params must be an object")

    try:
        if method == "health.check":
            return _ok(req_id, {"ok": True, "version": _VERSION,
                                "subprojects": _subproject_status()})
        if method == "dsl.validate":
            if "dsl" not in params:
                return _err(req_id, -32602, "missing params.dsl")
            return _ok(req_id, validate_dsl(params["dsl"]))
        if method == "problem.run":
            if "dsl" not in params:
                return _err(req_id, -32602, "missing params.dsl")
            res = run_problem(params["dsl"])
            return _ok(req_id, res.to_dict())
        if method == "problem.diagnose":
            if "dsl" not in params:
                return _err(req_id, -32602, "missing params.dsl")
            res = diagnose_problem(params["dsl"])
            return _ok(req_id, res.to_dict())
        return _err(req_id, -32601, f"method not found: {method!r}")
    except BridgeError as e:
        log.warning("BridgeError on %s: %s", method, e)
        return _err_from_bridge(req_id, e)
    except Exception as e:          # noqa: BLE001
        log.exception("internal error on %s", method)
        return _err_from_bridge(req_id, InternalError(
            message=f"unexpected: {type(e).__name__}: {e}",
            details={"exc_type": type(e).__name__},
        ))


def serve_stdio(stream_in=None, stream_out=None) -> None:
    """Read newline-delimited JSON requests; write newline-delimited
    JSON responses. Stdout is reserved for protocol; logs go to stderr."""
    if stream_in is None:
        stream_in = sys.stdin
    if stream_out is None:
        stream_out = sys.stdout
    for line in stream_in:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = _err(None, -32700, f"JSON parse error: {exc.msg}")
        else:
            resp = handle_request(req)
        stream_out.write(json.dumps(resp) + "\n")
        stream_out.flush()


def serve_once(payload: str, stream_out=None) -> None:
    """Handle exactly one request from a CLI string."""
    if stream_out is None:
        stream_out = sys.stdout
    try:
        req = json.loads(payload)
    except json.JSONDecodeError as exc:
        resp = _err(None, -32700, f"JSON parse error: {exc.msg}")
    else:
        resp = handle_request(req)
    stream_out.write(json.dumps(resp) + "\n")
    stream_out.flush()
