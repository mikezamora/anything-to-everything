"""Synquid baseline adapter (spec §14.3).

Synquid (Polikarpova, Kuraj & Solar-Lezama 2016) is open-source
(https://github.com/nadia-polikarpova/synquid) but requires a Haskell
toolchain. The adapter expects ``$SYNQUID_CMD`` pointing at the
``synquid`` binary; the input format is a Synquid spec file. Without
that binary, the adapter raises ``BaselineUnavailable``.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

from ..schema import ProblemSpec, ProofAttempt
from ._base import Baseline, BaselineUnavailable


_EXTENSIONS_ANCHOR = "Synquid binary not bundled"


class SynquidBaseline(Baseline):
    name = "synquid"

    def __init__(self, cmd: str | None = None):
        self.cmd = cmd or os.environ.get("SYNQUID_CMD")

    def solve(self, problem: ProblemSpec) -> ProofAttempt:
        if not self.cmd:
            raise BaselineUnavailable(
                self.name,
                "SYNQUID_CMD not set; no Haskell toolchain present",
                _EXTENSIONS_ANCHOR,
            )
        t0 = time.time()
        # Synquid reads a `.sq` file. Construct one from the payload.
        # The signature/sketch lives in the payload for Hazel/Myth-style
        # problems; HumanEval lives in the prompt.
        signature = (
            problem.payload.get("signature")
            or problem.payload.get("sketch")
            or problem.statement
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sq", delete=False) as f:
            f.write(signature)
            sq_path = f.name
        try:
            proc = subprocess.run(
                [self.cmd, sq_path],
                capture_output=True, text=True, timeout=300,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            Path(sq_path).unlink(missing_ok=True)
            raise BaselineUnavailable(
                self.name, f"invocation failed: {e!r}", _EXTENSIONS_ANCHOR,
            ) from e
        Path(sq_path).unlink(missing_ok=True)
        wall = time.time() - t0
        # Synquid prints the synthesised term on stdout; non-empty means
        # success in the upstream CLI contract.
        candidate = proc.stdout.strip()
        solved = bool(candidate) and proc.returncode == 0
        return ProofAttempt(
            solver=self.name,
            problem_id=problem.problem_id,
            solved=solved,
            well_typed=solved,  # synquid only emits well-typed terms
            residual_energy=None,
            candidates=(candidate,) if candidate else (),
            wall_time_s=wall,
            error=proc.stderr[:200] if not solved else None,
            diagnostics={"returncode": proc.returncode},
        )
