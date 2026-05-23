"""Worldline path integral for Bayesian proof ranking (spec §12.16).

Feynman's path-integral formulation assigns each candidate path an amplitude
``exp(-S/T)`` where ``S`` is the action along the path and ``T`` is a
"temperature" that governs exploration. The Boltzmann distribution

    P(path) = exp(-S[path]/T) / Z,   Z = sum_paths exp(-S/T)

is the saddle-point posterior over candidate proofs (spec §12.16). The
sharply-peaked regime (low ``T``) collapses onto the maximum-a-posteriori
(MAP) proof; the broad regime (large ``T``) exposes genuine ambiguity.

The action functional ``S[path]`` is *not* an AST-symbol count (§1.1
anti-shortcut). It is a sum of operator-algebraic measurements on the
solved :class:`ProofTree` produced by the §10.10 orchestrator:

    S[tree] = sum_n residual_energy(n)        # substrate measurement
            + alpha * complexity(tree)        # lemma / proof-step count
            + beta  * depth(tree)             # path length

``residual_energy`` is the converged ``<H>`` from imaginary-time evolution
at each node (the substrate's quantitative proof certificate). The
complexity and depth contributions are weighted at unit scale by default,
mirroring the architecture's free-energy decomposition (§9.5).

Public surface:

* :func:`compute_action` -- evaluate ``S[tree]`` for a single ProofTree.
* :func:`bayesian_rank_proofs` -- softmax-normalise ``-S/T`` across a batch
  of candidate :class:`ProofTree` instances; return ordered
  :class:`ProofRanking` records.

The temperature limit ``T -> 0`` is handled explicitly: it collapses to a
hard argmax on ``-S`` (MAP estimate) and never divides by zero.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .goal_graph import ProofTree, ProofTreeNode


# ---------------------------------------------------------------------------
# Action functional (spec §12.16)
# ---------------------------------------------------------------------------


def _walk(node: ProofTreeNode) -> Iterable[ProofTreeNode]:
    """Yield every node in the proof tree in pre-order (root first)."""
    yield node
    for child in node.children:
        yield from _walk(child)


def _max_depth(node: ProofTreeNode) -> int:
    """Length of the longest root-to-leaf path (root has depth 0)."""
    if not node.children:
        return 0
    return 1 + max(_max_depth(c) for c in node.children)


def compute_action(
    tree: ProofTree,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> float:
    """Compute the worldline action ``S[tree]`` (spec §12.16).

    The action is a sum of local terms over the SOLVED proof tree:

    * **residual energy** -- the converged ``<H>`` at every node, taken
      directly from the substrate's imaginary-time measurement (the §10.10
      proof certificate). This is the operator-algebraic contribution
      that anchors the action to physical measurement (anti-shortcut §1.1:
      we do not count AST symbols).
    * **lemma complexity** -- the number of proof steps, i.e. the total
      node count of the tree. Each lemma introduced into a derivation
      contributes one unit of complexity, weighted by ``alpha``.
    * **path length** -- the depth of the deepest root-to-leaf chain,
      weighted by ``beta``. A deeper hierarchy traverses more state-space
      transitions and so accumulates more action.

    Parameters
    ----------
    tree
        A :class:`ProofTree` -- the §4.6 / §10.10 output of
        :func:`composition.goal_graph.extract_proof_tree`.
    alpha
        Weight on the lemma-complexity term (default ``1.0``).
    beta
        Weight on the path-length term (default ``1.0``).

    Returns
    -------
    float
        The action ``S[tree]`` -- nonnegative whenever residuals,
        ``alpha``, and ``beta`` are nonnegative.
    """
    if not isinstance(tree, ProofTree):
        raise TypeError(
            f"compute_action expects a ProofTree (spec §4.6); got "
            f"{type(tree).__name__}"
        )
    residual_sum = sum(n.residual_energy for n in _walk(tree.root))
    complexity = sum(1 for _ in _walk(tree.root))
    depth = _max_depth(tree.root)
    return float(residual_sum) + alpha * float(complexity) + beta * float(depth)


# ---------------------------------------------------------------------------
# Bayesian ranking (softmax over -S/T)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofRanking:
    """One row of the Bayesian-ranked output.

    ``weight`` is the Boltzmann probability ``exp(-S/T) / Z``. Weights
    across a single :func:`bayesian_rank_proofs` call sum to 1.0 by
    construction (numerical residue aside).
    """
    tree: ProofTree
    action: float
    weight: float


def bayesian_rank_proofs(
    candidates: Sequence[ProofTree],
    *,
    T: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> list[ProofRanking]:
    """Rank candidate proofs by ``exp(-S/T)`` (spec §12.16).

    The Boltzmann distribution

        P(tree) = exp(-S[tree]/T) / Z,
        Z       = sum_c exp(-S[c]/T)

    is computed in a numerically stable way (subtract the min action
    before exponentiation, equivalent to shifting the partition function
    by a constant).

    Special case ``T -> 0`` (MAP estimate): the softmax becomes a hard
    argmax on ``-S`` (== argmin on ``S``). The lowest-action candidate
    receives weight ``1.0``; ties are split uniformly across the tied set;
    every other candidate receives weight ``0.0``. This branch is the
    saddle-point limit of the path integral and is what naive proof
    search returns (spec §12.16 "saddle-point approximation").

    Parameters
    ----------
    candidates
        A nonempty sequence of :class:`ProofTree` instances -- typically
        the top-``k`` proofs returned by :func:`solve_goal_graph` across
        alternative decomposition strategies.
    T
        Temperature. Must be ``>= 0``. ``T == 0`` triggers the MAP branch.
    alpha, beta
        Forwarded to :func:`compute_action`.

    Returns
    -------
    list[ProofRanking]
        Sorted by ``weight`` descending (highest-probability proof first).
    """
    if T < 0.0:
        raise ValueError(
            f"bayesian_rank_proofs requires T >= 0 (T is a Boltzmann "
            f"temperature, not a free parameter); got T={T!r}"
        )
    if not candidates:
        raise ValueError(
            "bayesian_rank_proofs requires at least one candidate proof "
            "(the partition function over an empty set is undefined)"
        )

    actions = [compute_action(t, alpha=alpha, beta=beta) for t in candidates]

    if T == 0.0:
        # T -> 0 collapses to argmax on -S; equivalent to argmin on S.
        # Ties split uniformly so weights still sum to 1 (spec §12.16
        # "Multiple distinct proof strategies have similar probability").
        s_min = min(actions)
        tied = [i for i, s in enumerate(actions) if s == s_min]
        share = 1.0 / float(len(tied))
        weights = [share if i in set(tied) else 0.0
                   for i in range(len(actions))]
    else:
        # Numerically stable softmax on (-S / T).
        scaled = [-(s / T) for s in actions]
        m = max(scaled)
        exps = [math.exp(x - m) for x in scaled]
        Z = sum(exps)
        weights = [e / Z for e in exps]

    rankings = [
        ProofRanking(tree=t, action=s, weight=w)
        for t, s, w in zip(candidates, actions, weights)
    ]
    rankings.sort(key=lambda r: r.weight, reverse=True)
    return rankings
