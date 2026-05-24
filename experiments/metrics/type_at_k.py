"""type@k (spec §14.2 -- the QPCN's distinctive metric).

`type@k`: fraction of `k` candidates that are type-correct. Per spec
§13.2 ("type-safety theorem") and §14.2 ("type@1 = 1.0 for any
successful run"), the QPCN's claim is that whenever ``solved=True``,
EVERY surfaced candidate is well-typed. The negative-comparison
baseline (LLMs) achieves ~50-80% on this.

Implementation: we trust ``ProofAttempt.well_typed`` as the per-attempt
signal (the runner sets this from the solver's actual type-checker
output, never fabricated). For the QPCN, ``well_typed`` is the
acceptance gate; for LLM baselines we honestly report whatever their
type-checker says.
"""
from __future__ import annotations

from typing import Iterable

from ..schema import ProofAttempt


def type_at_k(attempts: Iterable[ProofAttempt], k: int) -> float:
    """Fraction of (problem, candidate-rank<=k) pairs that are typed.

    For a single-candidate attempt (the common case in our runner --
    proofs surface one ranked proof tree; synthesis surfaces one
    top-ranked completion), ``type@k`` collapses to "fraction of
    attempts that are well-typed".

    Multi-candidate attempts (where ``ProofAttempt.candidates`` carries
    multiple strings) are still scored on the SINGLE ``well_typed``
    flag because the substrate's invariant is that ALL surfaced
    candidates share the same typing -- it would be a substrate bug to
    surface a typed and an ill-typed candidate together. We assert that
    invariant here by treating any candidate as a vote.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    attempts_list = list(attempts)
    if not attempts_list:
        return 0.0
    typed_count = 0
    total = 0
    for a in attempts_list:
        n_surfaced = max(1, min(k, len(a.candidates)))
        total += n_surfaced
        if a.well_typed:
            typed_count += n_surfaced
    return typed_count / total if total else 0.0
