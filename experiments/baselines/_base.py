"""Shared base + sentinel for baseline adapters."""
from __future__ import annotations

from ..schema import ProblemSpec, ProofAttempt


class BaselineUnavailable(RuntimeError):
    """Raised by an adapter when its upstream binary / model / service
    is not available on this host. The runner catches this and records
    an honest no-attempt row -- per memory/no-placeholders, the gap is
    surfaced loudly rather than papered over with a fake output.
    """

    def __init__(self, baseline: str, reason: str, extensions_anchor: str):
        super().__init__(
            f"{baseline} unavailable: {reason}. See EXTENSIONS.md anchor "
            f"{extensions_anchor!r} for the deferral note."
        )
        self.baseline = baseline
        self.reason = reason
        self.extensions_anchor = extensions_anchor


class Baseline:
    """Adapter protocol. Subclasses set ``name`` and implement ``solve``."""

    name: str = ""

    def solve(self, problem: ProblemSpec) -> ProofAttempt:  # pragma: no cover - protocol
        raise NotImplementedError
