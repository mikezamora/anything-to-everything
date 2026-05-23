"""Exception family for sub-project I (lemma library and promotion). Spec §6."""
from __future__ import annotations


class CompositionError(Exception):
    """Base of every sub-project I error."""


class LemmaHashCollision(CompositionError):
    """save() of a content hash that already maps to different content."""


class LemmaNotFound(CompositionError):
    """load/materialize/compile_constraint of an unknown lemma_id."""


class LemmaLeafCountMismatch(CompositionError):
    """use_lemma 'leaves' length != lemma n_leaves_L."""


class LemmaSpeciesMismatch(CompositionError):
    """Re-indexing onto a host window with a misaligned species pattern."""


class ConditionalLemmaRefused(CompositionError):
    """compile_constraint of a conditional lemma without allow_conditional."""


class LemmaValidationError(CompositionError):
    """Registration validation pass failed (surfaced via RegistrationResult)."""


class CompressionError(CompositionError):
    """Bond compression could not reach eps_compress without exceeding it."""


class GoalGraphError(CompositionError):
    """A back-edge slipped past cycle detection. A bug, not recoverable."""


class DispatchTimeout(CompositionError):
    """Child goal exceeded its dispatch timeout (carried in ChildResult, spec §5.4)."""

    def __init__(self, goal_id: str, timeout_s: float):
        super().__init__(f"child {goal_id!r} exceeded timeout {timeout_s}s")
        self.goal_id = goal_id
        self.timeout_s = timeout_s


class IntegrationRefused(CompositionError):
    """Integration residual exceeded the configured gate (spec §6.3)."""

    def __init__(self, residual: float, gate: float):
        super().__init__(f"integration refused: residual {residual!r} exceeds gate {gate!r}")
        self.residual = residual
        self.gate = gate


class RevisionExhausted(CompositionError):
    """Goal revision attempts exhausted without convergence (spec §6.5)."""

    def __init__(self, goal_id: str, attempts: int):
        super().__init__(f"revision exhausted for {goal_id!r} after {attempts} attempts")
        self.goal_id = goal_id
        self.attempts = attempts
