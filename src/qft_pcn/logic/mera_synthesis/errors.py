"""Synthesis error model (spec §10)."""
from __future__ import annotations


class SynthesisError(Exception):
    """Base class for MERA synthesis runner errors."""


class SynthesisProblemError(SynthesisError):
    """Malformed SynthesisProblem (spec §6.1 invariants violated)."""


class SynthesisRuntimeError(SynthesisError):
    """Evolution diverged, NaN energies, or zero decodable samples.
    A problem with no good completion is NOT raised -- it is reported
    via SynthesisResult.failure_mode (spec §1.10)."""
