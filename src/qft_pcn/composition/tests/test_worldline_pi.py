"""Tests for spec §12.16 worldline path integral / Bayesian proof ranking.

These tests rank *real* :class:`ProofTree` instances (the §4.6 / §10.10
dataclass output of :func:`extract_proof_tree`) -- no mock tree class is
involved. The residual_energy slots carry numbers that would, in a full
substrate run, come from the converged ``<H>`` of imaginary-time
evolution (the §10.10 K-8 acceptance produces these directly); for unit
tests we construct ProofTrees with hand-chosen residuals so the relative
ordering of candidate proofs is deterministic and verifiable.

The §1.1 anti-shortcut bites: the action functional under test reads
:attr:`ProofTreeNode.residual_energy` (the substrate measurement) and the
tree's topology (depth, complexity) -- it never tokenises the AST.
"""
from __future__ import annotations

import math

import pytest

from src.qft_pcn.composition.goal_graph import ProofTree, ProofTreeNode
from src.qft_pcn.composition.worldline_pi import (
    ProofRanking,
    bayesian_rank_proofs,
    compute_action,
)


# ---------------------------------------------------------------------------
# Helpers: build real ProofTree dataclasses (NOT mocks) with the residuals
# that an actual §10.10 substrate run would deposit at each node.
# ---------------------------------------------------------------------------


def _leaf(prop: str, residual: float,
          bond_entanglement: float = 0.0) -> ProofTreeNode:
    return ProofTreeNode(
        goal_prop=prop, solved_ast=None,
        residual_energy=float(residual), children=(),
        bond_entanglement=float(bond_entanglement),
    )


def _node(prop: str, residual: float,
          children: tuple[ProofTreeNode, ...],
          bond_entanglement: float = 0.0) -> ProofTreeNode:
    return ProofTreeNode(
        goal_prop=prop, solved_ast=None,
        residual_energy=float(residual), children=children,
        bond_entanglement=float(bond_entanglement),
    )


def _tree(root: ProofTreeNode) -> ProofTree:
    total = 0.0

    def walk(n: ProofTreeNode) -> None:
        nonlocal total
        total += n.residual_energy
        for c in n.children:
            walk(c)
    walk(root)
    return ProofTree(root=root, total_residual=total)


def _short_low_residual_proof() -> ProofTree:
    """A 2-node proof with low substrate residual -- the "natural" proof."""
    return _tree(_node("theorem", 1e-7, (_leaf("axiom", 1e-7),)))


def _long_high_residual_proof() -> ProofTree:
    """A 4-node proof with higher substrate residual -- the "unusual" one."""
    return _tree(_node("theorem", 1e-3, (
        _node("lemma_L1", 1e-3, (_leaf("axiom_A1", 1e-3),)),
        _leaf("axiom_A2", 1e-3),
    )))


# ---------------------------------------------------------------------------
# §12.16 tests
# ---------------------------------------------------------------------------


def test_compute_action_is_nonnegative_and_uses_residuals():
    """Action is sum-of-residuals + alpha*complexity + beta*depth; with
    nonnegative residuals and unit weights every term is nonnegative, and
    larger residuals strictly raise the action (the substrate measurement
    is the load-bearing contribution -- anti-shortcut §1.1)."""
    low = _short_low_residual_proof()
    high = _long_high_residual_proof()
    s_low = compute_action(low)
    s_high = compute_action(high)
    assert s_low >= 0.0
    assert s_high >= 0.0
    # Residual contribution is the load-bearing differentiator.
    assert s_high > s_low


def test_lower_action_higher_weight():
    """The shorter / lower-residual proof receives the higher Boltzmann
    weight (spec §12.16: "the most probable hypothesis is the saddle
    point")."""
    short = _short_low_residual_proof()
    long_ = _long_high_residual_proof()

    ranked = bayesian_rank_proofs([long_, short], T=1.0)

    # Highest weight is at index 0 by construction (sort descending).
    assert ranked[0].tree is short, (
        f"shorter / lower-residual proof should rank first; got "
        f"goal_prop={ranked[0].tree.root.goal_prop!r}, "
        f"action={ranked[0].action}, weight={ranked[0].weight}"
    )
    assert ranked[0].weight > ranked[1].weight
    # And lower action implies higher weight under the Boltzmann map.
    assert ranked[0].action < ranked[1].action


def test_softmax_weights_sum_to_one():
    """``sum_c exp(-S_c/T) / Z = 1`` by construction; the numerical
    residue must be below machine epsilon."""
    candidates = [
        _short_low_residual_proof(),
        _long_high_residual_proof(),
        _tree(_node("alt", 5e-4, (_leaf("alt_axiom", 5e-4),))),
    ]
    ranked = bayesian_rank_proofs(candidates, T=0.5)

    assert len(ranked) == len(candidates)
    total = sum(r.weight for r in ranked)
    assert math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12), (
        f"softmax weights must sum to 1.0; got {total!r}"
    )
    # And every weight is in [0, 1].
    for r in ranked:
        assert 0.0 <= r.weight <= 1.0


def test_temperature_zero_is_max_a_posteriori():
    """T -> 0 collapses the path integral to the saddle point: the
    minimum-action proof receives weight 1.0 and every other receives
    weight 0.0 (spec §12.16 saddle-point limit)."""
    short = _short_low_residual_proof()
    long_ = _long_high_residual_proof()
    alt = _tree(_node("alt", 5e-4, (_leaf("alt_axiom", 5e-4),)))

    ranked = bayesian_rank_proofs([long_, alt, short], T=0.0)

    # MAP: the single minimum-action proof gets all the mass.
    assert ranked[0].tree is short
    assert ranked[0].weight == 1.0
    for r in ranked[1:]:
        assert r.weight == 0.0
    # Sum still 1 (the MAP branch must not lose normalisation).
    assert sum(r.weight for r in ranked) == 1.0


def test_temperature_zero_splits_ties_uniformly():
    """When two candidates tie for minimum action, T=0 splits the mass
    uniformly across the tied set -- the path integral does not pick a
    favourite by tiebreak, and weights still sum to 1."""
    t1 = _tree(_node("a", 1e-6, (_leaf("a_leaf", 1e-6),)))
    t2 = _tree(_node("b", 1e-6, (_leaf("b_leaf", 1e-6),)))
    bigger = _long_high_residual_proof()

    ranked = bayesian_rank_proofs([t1, bigger, t2], T=0.0)

    tied = [r for r in ranked if r.weight > 0.0]
    assert len(tied) == 2
    for r in tied:
        assert r.weight == 0.5
    assert sum(r.weight for r in ranked) == 1.0


def test_higher_temperature_flattens_distribution():
    """Large ``T`` exposes ambiguity: the gap between best and worst
    weights narrows toward the uniform distribution (spec §12.16:
    "whether the most likely proof is robust or fragile")."""
    short = _short_low_residual_proof()
    long_ = _long_high_residual_proof()

    cold = bayesian_rank_proofs([short, long_], T=0.01)
    warm = bayesian_rank_proofs([short, long_], T=100.0)

    gap_cold = cold[0].weight - cold[1].weight
    gap_warm = warm[0].weight - warm[1].weight
    assert gap_warm < gap_cold, (
        f"raising T should flatten the Boltzmann distribution; "
        f"gap_cold={gap_cold}, gap_warm={gap_warm}"
    )


def test_action_uses_substrate_residual_not_ast_symbols():
    """§1.1 anti-shortcut probe: hold topology fixed and vary ONLY the
    residual_energy field. The action must change strictly with the
    substrate measurement (residual), proving the functional reads the
    operator-algebraic certificate and not an AST-symbol count."""
    base = _tree(_node("t", 1e-7, (_leaf("a", 1e-7),)))
    perturbed = _tree(_node("t", 1e-2, (_leaf("a", 1e-2),)))
    # Same topology (2 nodes, depth 1) -> alpha*complexity + beta*depth
    # contributions are identical. The only difference is residual.
    s_base = compute_action(base)
    s_pert = compute_action(perturbed)
    assert s_pert > s_base
    # And the delta is exactly the residual delta (within fp tolerance):
    assert math.isclose(
        (s_pert - s_base),
        (perturbed.total_residual - base.total_residual),
        rel_tol=1e-9, abs_tol=1e-12,
    )


def test_empty_candidates_rejected():
    """The partition function over an empty set is undefined; the API
    surfaces a ValueError rather than returning a meaningless [] (the
    architecture never silently fabricates a normalisation)."""
    with pytest.raises(ValueError):
        bayesian_rank_proofs([], T=1.0)


def test_negative_temperature_rejected():
    """``T`` is a Boltzmann temperature, not a free parameter; negative
    values invert the distribution and would be nonsense as a posterior."""
    short = _short_low_residual_proof()
    with pytest.raises(ValueError):
        bayesian_rank_proofs([short], T=-0.5)


def test_compute_action_type_guarded():
    """A bare ProofTreeNode is NOT a ProofTree (spec §4.6 wraps the root
    in a ProofTree to carry total_residual); the API enforces the wrapper
    so callers cannot accidentally rank node objects."""
    bare = _node("t", 0.0, ())
    with pytest.raises(TypeError):
        compute_action(bare)


def test_compute_action_includes_bond_entanglement():
    """Spec §12.16: the action functional carries a bond-entanglement
    contribution gated by ``gamma``. Holding topology + residuals fixed,
    a proof tree with larger summed bond_entanglement must have action
    raised by EXACTLY ``gamma * delta_S`` (the Schmidt-spectrum readout
    is the load-bearing path-fitness signal, anti-shortcut §1.1)."""
    # Two trees with identical topology, residuals, depth, and complexity;
    # only the bond_entanglement fields differ.
    base = _tree(_node("t", 1e-7, (_leaf("a", 1e-7, bond_entanglement=0.0),),
                       bond_entanglement=0.0))
    entangled = _tree(_node("t", 1e-7,
                            (_leaf("a", 1e-7, bond_entanglement=0.6),),
                            bond_entanglement=0.3))
    delta_S_bond = 0.6 + 0.3   # root + leaf

    # gamma = 1.0
    diff_unit = compute_action(entangled, gamma=1.0) - compute_action(base, gamma=1.0)
    assert math.isclose(diff_unit, 1.0 * delta_S_bond,
                        rel_tol=1e-9, abs_tol=1e-12), (
        f"gamma=1 contribution should be {delta_S_bond}; got {diff_unit}"
    )

    # gamma = 2.5
    diff_scaled = (
        compute_action(entangled, gamma=2.5)
        - compute_action(base, gamma=2.5)
    )
    assert math.isclose(diff_scaled, 2.5 * delta_S_bond,
                        rel_tol=1e-9, abs_tol=1e-12), (
        f"gamma=2.5 contribution should be {2.5 * delta_S_bond}; "
        f"got {diff_scaled}"
    )

    # gamma = 0 -> term drops out entirely (action identical).
    s_off_base = compute_action(base, gamma=0.0)
    s_off_ent = compute_action(entangled, gamma=0.0)
    assert math.isclose(s_off_base, s_off_ent,
                        rel_tol=0.0, abs_tol=1e-12), (
        f"gamma=0 should null the bond-entanglement term; got "
        f"{s_off_base} vs {s_off_ent}"
    )


def test_ranking_dataclass_carries_action_and_tree():
    """The ProofRanking record exposes (tree, action, weight) so callers
    can downstream-rank or audit without recomputing the action."""
    short = _short_low_residual_proof()
    ranked = bayesian_rank_proofs([short], T=1.0)
    assert len(ranked) == 1
    r = ranked[0]
    assert isinstance(r, ProofRanking)
    assert r.tree is short
    assert r.weight == 1.0   # single candidate gets all the mass
    assert r.action == compute_action(short)
