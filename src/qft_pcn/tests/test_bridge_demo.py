"""End-to-end demo tests (spec §8, §11.8)."""

from __future__ import annotations

import subprocess
import sys


def _run_module(module: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_demo_mock_path_runs_to_completion():
    rc, out, err = _run_module("src.qft_pcn.bridge.demo")
    assert rc == 0, f"stderr: {err}"
    for label in ("PROMPT", "DSL", "OBSERVABLES", "ENERGY", "VERBALIZATION"):
        assert label in out, f"missing {label} section:\n{out}"


def test_demo_protocol_path_runs_to_completion():
    rc, out, err = _run_module("src.qft_pcn.bridge.demo_protocol")
    assert rc == 0, f"stderr: {err}"
    for label in ("PROMPT", "DSL", "OBSERVABLES", "ENERGY", "VERBALIZATION"):
        assert label in out, f"missing {label} section:\n{out}"
    assert "jsonrpc" in out
