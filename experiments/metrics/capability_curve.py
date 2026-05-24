"""capability_curve C(t) and capability-vs-difficulty (spec §14.2).

Two related curves, both surfaced by this module:

* ``capability_curve``: fraction of held-out problems solved at
  wake-sleep cycle ``t`` (spec §14.2 / §11.7). Drives the §13.8
  scaling-law fit.
* ``capability_vs_difficulty``: per-difficulty-bucket pass rate. Spec
  §14.2 "capability vs problem-difficulty curve". The runner uses this
  to visualise where the substrate's competence cliff sits.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, NamedTuple, Sequence

from ..schema import ProblemSpec, ProofAttempt


class CapabilityPoint(NamedTuple):
    bucket: float       # difficulty bucket midpoint
    pass_rate: float    # fraction solved in that bucket
    n: int              # bucket population


def capability_curve(
    attempts_per_cycle: Sequence[Sequence[ProofAttempt]],
) -> tuple[float, ...]:
    """C(t): pass-rate per wake-sleep cycle.

    ``attempts_per_cycle[t]`` is the list of attempts collected at
    cycle ``t``. The returned tuple is the per-cycle solved-fraction.
    Spec §13.8's saturating-exponential fit lives in the stats package
    (``stats.protocol.fit_capability_law``); this module returns the
    raw points.
    """
    out: list[float] = []
    for cycle_attempts in attempts_per_cycle:
        attempts_list = list(cycle_attempts)
        if not attempts_list:
            out.append(0.0)
            continue
        out.append(sum(1 for a in attempts_list if a.solved) / len(attempts_list))
    return tuple(out)


def capability_vs_difficulty(
    problems: Iterable[ProblemSpec],
    attempts: Iterable[ProofAttempt],
    *,
    n_buckets: int = 5,
) -> tuple[CapabilityPoint, ...]:
    """Bucket problems by difficulty, report pass-rate per bucket.

    Problems and attempts are joined by ``problem_id``. Problems with
    no matching attempt are EXCLUDED (the metric is "of the problems
    we tried, what fraction passed"). Difficulty in [0, 1] is mapped
    to ``n_buckets`` equal-width bins.
    """
    by_id: dict[str, ProblemSpec] = {p.problem_id: p for p in problems}
    by_id_att: Mapping[str, list[ProofAttempt]] = defaultdict(list)
    for a in attempts:
        by_id_att[a.problem_id].append(a)

    buckets: dict[int, list[bool]] = defaultdict(list)
    for pid, atts in by_id_att.items():
        if pid not in by_id:
            continue
        d = by_id[pid].difficulty
        idx = min(n_buckets - 1, max(0, int(d * n_buckets)))
        for a in atts:
            buckets[idx].append(a.solved)

    out: list[CapabilityPoint] = []
    for idx in range(n_buckets):
        votes = buckets.get(idx, [])
        midpoint = (idx + 0.5) / n_buckets
        rate = sum(1 for v in votes if v) / len(votes) if votes else 0.0
        out.append(CapabilityPoint(bucket=midpoint, pass_rate=rate, n=len(votes)))
    return tuple(out)
