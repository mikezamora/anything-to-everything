"""Baseline-system adapters (spec §14.3).

Each adapter exposes::

    class XBaseline:
        name: str
        def solve(self, problem: ProblemSpec) -> ProofAttempt: ...

When the upstream binary IS available on PATH (env var per adapter), it
is invoked for real and the stdout/stderr is parsed. When it is NOT
available, the adapter raises ``BaselineUnavailable`` and the runner
records an honest "no-attempt" row with that exact reason; it does NOT
fabricate a result. The per-baseline deferral is mirrored in
EXTENSIONS.md so the gap is tracked.
"""

from .alphaproof import AlphaProofBaseline, BaselineUnavailable
from .reprover import ReProverBaseline
from .synquid import SynquidBaseline
from .qpcn import QPCNBaseline

__all__ = [
    "AlphaProofBaseline", "ReProverBaseline", "SynquidBaseline",
    "QPCNBaseline", "BaselineUnavailable",
]
