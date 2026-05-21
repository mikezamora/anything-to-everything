"""Sampling, deduplication, and ranking for synthesis (spec §6).

Given a relaxed MPS state and a (composed) Hamiltonian, draw samples,
group by alpha-equivalence, re-encode the unique completions, compute
⟨H_total⟩ per completion, and return a sorted list of Completion
records.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np

from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.logic.ast import Node
from src.qft_pcn.logic.decoder import sample, ast_alpha_eq
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.encoding import EncodingMeta
from .problem import Completion


def _ast_size(node: Node) -> int:
    from src.qft_pcn.logic.ast import (
        Lam, App, If, Bin, Var, IntLit, BoolLit, HoleVar,
    )
    if isinstance(node, (Var, IntLit, BoolLit, HoleVar)):
        return 1
    if isinstance(node, Lam):
        return 1 + _ast_size(node.body)
    if isinstance(node, App):
        return 1 + _ast_size(node.fn) + _ast_size(node.arg)
    if isinstance(node, If):
        return (1 + _ast_size(node.cond) + _ast_size(node.then_b)
                + _ast_size(node.else_b))
    if isinstance(node, Bin):
        return 1 + _ast_size(node.lhs) + _ast_size(node.rhs)
    return 1


def dedupe_by_alpha_eq(asts: list[Node]) -> list[tuple[Node, int]]:
    """Group asts by ast_alpha_eq; return list of (representative, count).

    Returns groups sorted by count descending (most common first); ties
    broken arbitrarily.
    """
    groups: list[list[Node]] = []
    for a in asts:
        placed = False
        for g in groups:
            if ast_alpha_eq(g[0], a):
                g.append(a)
                placed = True
                break
        if not placed:
            groups.append([a])
    # Sort groups by count descending.
    groups.sort(key=lambda g: -len(g))
    return [(g[0], len(g)) for g in groups]


def rank_completions(
    asts: list[Node],
    N: int,
    chi_max: int,
    hamiltonian_blocks: dict,    # {block_name: Hamiltonian}
) -> list[Completion]:
    """Given a list of sampled ASTs, group by alpha-eq, re-encode each
    unique AST, compute composed ⟨H⟩ per block, return sorted ascending
    by total energy.

    hamiltonian_blocks: dict mapping block name ("typing", "eval",
    "examples", "target_type", "size") to a Hamiltonian object exposing
    total_energy(state). The composed total energy is the sum.
    """
    groups = dedupe_by_alpha_eq(asts)
    completions: list[Completion] = []
    for rep_ast, multiplicity in groups:
        try:
            state_c, meta_c = encode(rep_ast, N=N, chi_max=chi_max)
        except Exception as ex:
            # Skip un-encodable representatives.
            continue
        breakdown: dict[str, float] = {}
        total = 0.0
        for name, H in hamiltonian_blocks.items():
            e = float(H.total_energy(state_c))
            breakdown[name] = e
            total += e
        completions.append(Completion(
            ast=rep_ast,
            energy=total,
            energy_breakdown=breakdown,
            diagnostics={},
            multiplicity=multiplicity,
        ))
    completions.sort(
        key=lambda c: (c.energy, -c.multiplicity, _ast_size(c.ast))
    )
    return completions


def classify_failure_mode(
    completions: list[Completion],
    weights,
    tolerance_correct: float = 1e-3,
) -> Optional[str]:
    """Return None on success, or a failure-mode string per spec §6.4."""
    if not completions:
        return "no_valid_completion"
    top1 = completions[0]
    # Tolerance: dominant constraints (T, E, X, Y) should be ~ 0; size may remain.
    dom_weight_sum = (weights.w_typing + weights.w_eval
                      + weights.w_examples + weights.w_target_type)
    threshold = tolerance_correct * dom_weight_sum
    # Compute "non-size" energy.
    non_size = sum(
        v for k, v in top1.energy_breakdown.items() if k != "size"
    )
    if non_size > threshold + 0.5 * weights.w_typing:
        return "no_valid_completion"
    if len(completions) >= 2:
        gap = completions[1].energy - top1.energy
        if gap < 1e-3 and non_size <= threshold:
            return "ambiguous_top1"
    return None
