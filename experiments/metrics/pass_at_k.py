"""pass@k (spec §14.2).

`pass@k`: fraction of problems for which at least one of the top-k
returned candidates satisfies all examples. Standard in the synthesis
literature (Chen et al. 2021); the implementation follows the unbiased
estimator from the HumanEval paper:

    pass@k = 1 - C(n - c, k) / C(n, k)

where ``n`` is the number of samples drawn, ``c`` is the number of
samples that pass, and the estimator is for the "expected fraction of
problems passed by at least one of k samples drawn without replacement
from n samples". We compute the per-problem estimator then average
across problems.

For our solver shape (``ProofAttempt.candidates`` is a tuple of up to k
strings + ``solved`` is "did at least one candidate pass"), the
estimator simplifies to: per-problem ``1.0`` if ``solved`` and at least
one of the first k candidates checks out, else ``0.0``. We support both
shapes:
  * ``unbiased=False`` (default): the simple "any of the top-k checked"
    semantics matching the cell-level meaning in the spec §14.2.
  * ``unbiased=True``: the HumanEval-paper unbiased estimator, requires
    ``ProofAttempt.diagnostics["n_samples"]`` and
    ``ProofAttempt.diagnostics["n_passing"]`` (the multi-sample case).
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from ..schema import ProofAttempt


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """HumanEval-paper unbiased estimator."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod((n - c - i) / (n - i) for i in range(k))


def pass_at_k(attempts: Iterable[ProofAttempt],
              k: int,
              *,
              unbiased: bool = False,
              checker: Optional[callable] = None) -> float:
    """Compute pass@k over a collection of attempts.

    ``checker``, when provided, is a callable
    ``(attempt) -> bool`` that decides whether *any* of the first k
    candidates satisfies the problem's examples. The default treats
    ``attempt.solved`` as the "at least one of k candidates passes"
    signal (the synthesis pipeline reports a single ranked list with
    failure_mode=None on success).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    attempts_list = list(attempts)
    if not attempts_list:
        return 0.0

    per_problem: list[float] = []
    for a in attempts_list:
        if unbiased:
            n = int(a.diagnostics.get("n_samples", len(a.candidates) or 1))
            c = int(a.diagnostics.get("n_passing", 1 if a.solved else 0))
            per_problem.append(_pass_at_k_unbiased(n, c, k))
        else:
            if checker is not None:
                per_problem.append(1.0 if checker(a) else 0.0)
            else:
                # The simple "did the solver return at least one
                # passing candidate within the top-k it surfaced"
                # semantic. The synthesis pipeline ranks candidates by
                # energy; "solved=True" means the top-1 passed all
                # examples (or, for proofs, residual_energy <= eps).
                top_k_solved = a.solved and len(a.candidates[:k]) >= 1
                per_problem.append(1.0 if top_k_solved else 0.0)
    return sum(per_problem) / len(per_problem)
