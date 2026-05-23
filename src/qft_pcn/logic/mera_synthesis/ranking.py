"""Synthesis ranking: dedupe, rerank, classify failure mode (spec §6.3)."""
from __future__ import annotations

from ..decoder import ast_alpha_eq


def dedupe_by_alpha_eq(samples, residual_max: float = 1e-3):
    """Group decoded samples by alpha-equivalence (spec §6.3).

    Drops samples whose decode residual exceeds `residual_max`. Returns a
    list of (representative_ast, multiplicity) for each unique group.
    """
    groups = []   # list of [ast, count]
    for s in samples:
        if s.residual_norm > residual_max:
            continue
        for g in groups:
            if ast_alpha_eq(g[0], s.ast):
                g[1] += 1
                break
        else:
            groups.append([s.ast, 1])
    return [(g[0], g[1]) for g in groups]


def classify_failure_mode(completions, tolerance_correct: float,
                          tolerance_ambiguous: float = 1e-3) -> str | None:
    """Classify a sorted-ascending completion list (spec §6.4)."""
    if not completions:
        return "no_valid_completion"
    top = completions[0]
    if len(completions) >= 2:
        gap = completions[1].energy - top.energy
        if gap < tolerance_ambiguous and top.energy < tolerance_correct:
            return "ambiguous_top1"
    if top.energy < tolerance_correct:
        return None
    return "imag_time_did_not_converge"
