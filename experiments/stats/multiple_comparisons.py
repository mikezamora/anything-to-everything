"""Multiple-comparisons correction (spec §14.5).

Spec §14.5: "For multiple comparisons within a benchmark, Bonferroni
correction". We implement Holm-Bonferroni (uniformly more powerful
than plain Bonferroni, controls FWER) and expose plain Bonferroni for
spec-literal interpretations.
"""
from __future__ import annotations

from typing import NamedTuple, Sequence


class CorrectedPValue(NamedTuple):
    label: str
    raw_p: float
    adjusted_p: float
    significant: bool


def holm_bonferroni(
    p_values: Sequence[tuple[str, float]],
    *,
    alpha: float = 0.05,
) -> list[CorrectedPValue]:
    """Holm-Bonferroni step-down. Input: ``[(label, p), ...]``.

    Returns the same labels with adjusted p-values and a
    ``significant`` flag against the corrected threshold.
    """
    indexed = sorted(enumerate(p_values), key=lambda t: t[1][1])
    m = len(indexed)
    out: list[CorrectedPValue | None] = [None] * m
    for rank, (orig_idx, (label, raw_p)) in enumerate(indexed):
        # Holm: multiply by (m - rank); cap at 1.0.
        adj = min(1.0, raw_p * (m - rank))
        # Ensure monotonicity: an earlier (smaller) raw_p cannot have a
        # larger adjusted_p than a later one.
        if rank > 0:
            prev = out[indexed[rank - 1][0]]
            assert prev is not None
            adj = max(adj, prev.adjusted_p)
        out[orig_idx] = CorrectedPValue(
            label=label, raw_p=raw_p,
            adjusted_p=adj, significant=(adj <= alpha),
        )
    # All entries are filled.
    return [cp for cp in out if cp is not None]


def bonferroni(
    p_values: Sequence[tuple[str, float]],
    *,
    alpha: float = 0.05,
) -> list[CorrectedPValue]:
    """Plain Bonferroni (spec-literal interpretation of §14.5)."""
    m = len(p_values)
    return [
        CorrectedPValue(
            label=label,
            raw_p=raw_p,
            adjusted_p=min(1.0, raw_p * m),
            significant=(raw_p * m <= alpha),
        )
        for label, raw_p in p_values
    ]
