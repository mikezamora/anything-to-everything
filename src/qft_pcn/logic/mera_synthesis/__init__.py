"""MERA-native STLC synthesis (migration sub-project M3, spec §6-§7).

`synthesize` is imported lazily via ``__getattr__`` to break a circular
import: ``mera_encoder`` imports ``mera_synthesis.encode_ext`` at module
load, and the runner depends on ``mera_encoder``. Eagerly re-exporting
``synthesize`` here would force the cycle on package init.
"""
from .problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)
from .errors import (
    SynthesisError, SynthesisProblemError, SynthesisRuntimeError,
)

__all__ = [
    "SynthesisProblem", "IOExample", "Completion", "SynthesisResult",
    "HamiltonianWeights", "synthesize",
    "SynthesisError", "SynthesisProblemError", "SynthesisRuntimeError",
]


def __getattr__(name):
    if name == "synthesize":
        from .runner import synthesize
        return synthesize
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")
