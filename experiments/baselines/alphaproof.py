"""AlphaProof baseline adapter (spec §14.3).

AlphaProof (DeepMind 2024 silver-medal at IMO) is not open-source; the
spec §14.3 row simply names it as the comparison target on miniF2F.

Adapter contract:
* If ``$ALPHAPROOF_CMD`` points at a runnable binary (e.g. an
  organisation-internal wrapper), invoke it with ``--problem-file`` and
  parse the standard ``alphaproof_result.json`` output schema (timestamp,
  proof, status).
* Otherwise raise ``BaselineUnavailable`` -- the gap is recorded in
  EXTENSIONS.md as "AlphaProof binary not available; comparison row is
  a structured no-attempt".
"""
from __future__ import annotations

import json
import os
import subprocess
import time

from ..schema import ProblemSpec, ProofAttempt
from ._base import Baseline, BaselineUnavailable


_EXTENSIONS_ANCHOR = "AlphaProof binary not bundled"


class AlphaProofBaseline(Baseline):
    name = "alphaproof"

    def __init__(self, cmd: str | None = None):
        self.cmd = cmd or os.environ.get("ALPHAPROOF_CMD")

    def solve(self, problem: ProblemSpec) -> ProofAttempt:
        if not self.cmd:
            raise BaselineUnavailable(
                self.name,
                "ALPHAPROOF_CMD not set and no bundled binary",
                _EXTENSIONS_ANCHOR,
            )
        t0 = time.time()
        try:
            proc = subprocess.run(
                [self.cmd, "--problem", problem.problem_id,
                 "--statement", problem.statement],
                capture_output=True, text=True, timeout=300,
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
                f"output not JSON-parseable: {e!r}; stderr={proc.stderr[:200]}",
                _EXTENSIONS_ANCHOR,
            ) from e
        return ProofAttempt(
            solver=self.name,
            problem_id=problem.problem_id,
            solved=bool(data.get("status") == "proved"),
            well_typed=bool(data.get("status") == "proved"),
            residual_energy=None,
            candidates=tuple(data.get("proofs", [])),
            wall_time_s=wall,
            error=data.get("error"),
            diagnostics={"raw": data},
        )
