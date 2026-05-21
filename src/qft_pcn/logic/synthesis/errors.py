"""Exception types for the synthesis runner (spec §9)."""

from __future__ import annotations


class SynthesisError(Exception):
    """Base class for synthesis-runner errors."""


class SynthesisProblemError(SynthesisError):
    """The SynthesisProblem violates spec §3.1 invariants
    (no holes, malformed examples, undefined hole candidate, etc.)."""


class SynthesisRuntimeError(SynthesisError):
    """Evolution diverged, NaN energies, or sampling produced no decodable
    completions — distinct from a problem that 'just has no good completion'
    which is reported via SynthesisResult.failure_mode rather than raised."""
