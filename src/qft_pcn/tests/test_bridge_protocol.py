"""Tests for the JSON-RPC protocol layer (spec §6, §11.5)."""

from __future__ import annotations

import json
import subprocess
import sys

from src.qft_pcn.bridge.api import handle_request


CANONICAL = {
    "fields": [{"name": "x", "cutoff": 4}],
    "sites": 2,
    "constraints": [],
    "observables": [{"site": 0, "field": "x", "op": "n"}],
    "search": {"method": "imag_time", "steps": 5, "chi_max": 4},
}


def test_handle_health_check_in_proc():
    req = {"jsonrpc": "2.0", "id": 1, "method": "health.check", "params": {}}
    resp = handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["ok"] is True


def test_handle_dsl_validate_in_proc():
    req = {"jsonrpc": "2.0", "id": 7, "method": "dsl.validate",
           "params": {"dsl": CANONICAL}}
    resp = handle_request(req)
    assert resp["id"] == 7
    assert resp["result"]["valid"] is True


def test_handle_problem_run_in_proc():
    req = {"jsonrpc": "2.0", "id": 2, "method": "problem.run",
           "params": {"dsl": CANONICAL}}
    resp = handle_request(req)
    assert resp["id"] == 2
    assert "result" in resp
    assert "observables" in resp["result"]


def test_handle_method_not_found_in_proc():
    req = {"jsonrpc": "2.0", "id": 3, "method": "nope", "params": {}}
    resp = handle_request(req)
    assert resp["id"] == 3
    assert resp["error"]["code"] == -32601


def test_handle_invalid_jsonrpc_version():
    req = {"jsonrpc": "1.0", "id": 4, "method": "health.check", "params": {}}
    resp = handle_request(req)
    assert resp["error"]["code"] == -32600


def test_handle_bridge_error_carries_typed_code():
    bad = {**CANONICAL, "sites": -3}
    req = {"jsonrpc": "2.0", "id": 5, "method": "problem.run",
           "params": {"dsl": bad}}
    resp = handle_request(req)
    assert resp["error"]["code"] == -32001
    assert resp["error"]["data"]["code"] == "BRIDGE_E_BAD_SCHEMA"


# --- subprocess tests ---


def _spawn_once(request_obj: dict) -> dict:
    payload = json.dumps(request_obj)
    proc = subprocess.run(
        [sys.executable, "-m", "src.qft_pcn.bridge", "--once", payload],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_subprocess_health_check():
    resp = _spawn_once({"jsonrpc": "2.0", "id": 9, "method": "health.check",
                        "params": {}})
    assert resp["result"]["ok"] is True


def test_subprocess_problem_run():
    resp = _spawn_once({"jsonrpc": "2.0", "id": 10, "method": "problem.run",
                        "params": {"dsl": CANONICAL}})
    assert "result" in resp
    assert len(resp["result"]["observables"]) == 1
