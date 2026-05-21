"""Sub-project E: STLC synthesis pipeline.

Public API:

    synthesize(problem) -> SynthesisResult       (runner.py)
    SynthesisProblem, IOExample, Completion,
        SynthesisResult, HamiltonianWeights      (problem.py)
    SynthesisError, SynthesisProblemError,
        SynthesisRuntimeError                    (errors.py)

The synthesis Hamiltonian is built in `hamiltonian.py`; encoder
extensions live in `encode_ext.py`; sample/dedupe/rank in `ranking.py`.
"""

from __future__ import annotations

from .problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)
from .errors import (
    SynthesisError, SynthesisProblemError, SynthesisRuntimeError,
)

# runner imports happen lazily where needed to avoid pulling heavy deps
# when only the data types are required.


def synthesize(problem, rng=None, verbose: bool = False):
    """Entry point — see runner.synthesize for details."""
    from .runner import synthesize as _impl
    return _impl(problem, rng=rng, verbose=verbose)


__all__ = [
    "SynthesisProblem", "IOExample", "Completion", "SynthesisResult",
    "HamiltonianWeights",
    "SynthesisError", "SynthesisProblemError", "SynthesisRuntimeError",
    "synthesize",
]
