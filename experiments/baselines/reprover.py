"""ReProver baseline adapter (spec §14.3).

ReProver / LeanDojo is open-source (https://github.com/lean-dojo/ReProver)
but requires a Lean toolchain + model checkpoint. The adapter invokes
the published CLI when ``$REPROVER_CMD`` is set; otherwise raises
``BaselineUnavailable``.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

from ..schema import ProblemSpec, ProofAttempt
from ._base import Baseline, BaselineUnavailable

_EXTENSIONS_ANCHOR = "ReProver / LeanDojo binary not bundled"


class ReProverBaseline(Baseline):
    name = "reprover"

    def __init__(self, cmd: str | None = None):
        self.cmd = cmd or os.environ.get("REPROVER_CMD")

    def solve(self, problem: ProblemSpec) -> ProofAttempt:
        if not self.cmd:
            raise BaselineUnavailable(
                self.name,
                "REPROVER_CMD not set; no Lean toolchain present",
                _EXTENSIONS_ANCHOR,
            )
        t0 = time.time()
        try:
            proc = subprocess.run(
                [self.cmd, "--problem-id", problem.problem_id],
                capture_output=True, text=True, timeout=600,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise BaselineUnavailable(
                self.name, f"invocation failed: {e!r}", _EXTENSIONS_ANCHOR,
            ) from e
        wall = time.time() - t0
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise BaselineUnavailable(
                self.name,
                f"output not JSON-parseable: {e!r}",
                _EXTENSIONS_ANCHOR,
            ) from e
        return ProofAttempt(
            solver=self.name,
            problem_id=problem.problem_id,
            solved=bool(data.get("proved")),
            well_typed=bool(data.get("proved")),
            residual_energy=None,
            candidates=tuple(data.get("tactics", [])),
            wall_time_s=wall,
            error=data.get("error"),
            diagnostics={"raw": data},
        )
