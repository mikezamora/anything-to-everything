"""Result integrator: bottom-up clamping (spec §6).

A solved child is integrated by clamping its sub-MERA into the parent's
MERA via sub-project I's lemma-promotion machinery -- never by writing the
child AST into a Python dict. The residual-energy gate is strict: only a
true ground state is clamped. The clamp strength is the precision of the
bottom-up free-energy message: high-residual children clamp weakly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .goal_graph import Node, Status
from .lemma_library import (
    LemmaLibrary, DerivationMetadata, register_lemma,
)
from .promoter import Promoter
from ..logic.mera_encoding import LEAVES_PER_NODE

RESIDUAL_GATE = 1e-6        # absolute residual ceiling for a true ground state
CONJECTURE_CEILING = 1e-3   # above the gate, below this: integrate as conjecture
GROUND_STATE_GAP = 1e-4     # min spectral gap to the first excited state
RESIDUAL_SCALE = 1e-4       # precision-weighting scale
STRENGTH_MAX = 1.0          # clamp strength at zero residual


def precision_weight(residual: float) -> float:
    """Precision of the bottom-up message (spec §6.2).

    1 / (1 + residual/RESIDUAL_SCALE): a low-residual child is a
    high-precision message and clamps strongly; a higher-residual child
    clamps proportionally weaker -- the free-energy precision term.
    """
    if residual == float("inf"):
        return 0.0
    return 1.0 / (1.0 + residual / RESIDUAL_SCALE)


@dataclass(frozen=True)
class IntegrationOutcome:
    integrated: bool
    provisional: bool
    clamp_strength: float
    reason: str


def _spectral_gap(child_result) -> float:
    """Read the child's reported gap. Missing-diagnostic => 0.0 (refuse).

    Returning ``0.0`` makes the gap guard a *strict* gate: a child whose
    runner did not surface a spectral_gap is treated as near-degenerate
    and refused. The previous fallback (``GROUND_STATE_GAP`` exactly)
    silently accepted such children because the comparison was
    ``gap < GROUND_STATE_GAP`` -- equality is not less-than. The
    safer-by-default behaviour here forces a runner to publish the gap
    explicitly to clear the gate.
    """
    return float(child_result.run_diagnostic.get("spectral_gap", 0.0))


def _resolve_host_leaves(node: Node, child_meta) -> tuple[int, ...]:
    """Return the explicit host-leaf footprint the lemma occupies on the
    parent MERA (spec §5.2a).

    ``SubGoal.parent_leaves`` is the canonical tuple form: it carries the
    full host-leaf window directly, so non-contiguous or species-permuted
    layouts are first-class. The integrator does not extend, guess, or
    rebuild a window from a single base int -- the decomposer is
    responsible for publishing the correct tuple. The contiguous case is
    available via :func:`goal_graph.make_contiguous_sub_goal`.

    ANTI-SHORTCUT (§1.1 binding-as-entanglement /
    memory:anti-shortcut-directive): the host-leaf window is NOT a
    classical site->index dict lookup. The tuple IS the entanglement
    footprint -- the indices into the host MERA's leaves[] that the
    clamp will write into, threaded through the species-pattern check
    in ``Promoter._check_species`` so the §1.3 locality contract holds
    bitwise (sentinel leaves outside the window are bitwise unchanged).
    A future implementer must NOT replace this with
    ``meta.site_to_leaves_map[base]`` or any classical dispatch keyed
    by site index -- the binding IS the leaf tuple, not a lookup of
    one.

    ``child_meta`` is accepted for API compatibility (callers pass it),
    but no longer participates in window resolution.
    """
    _ = child_meta  # see docstring; preserved for caller signature parity
    return tuple(int(i) for i in node.goal.parent_leaves)


def integrate_child(parent_state: Any, parent_meta: Any, node: Node,
                    child_result, lemma_library) -> IntegrationOutcome:
    """Clamp a solved child's sub-MERA into the parent (spec §6.1).

    Strict gate (spec §6.3):
      converged AND residual <= RESIDUAL_GATE AND gap >= GROUND_STATE_GAP
        -> SOLVED, full-strength clamp.
      converged AND RESIDUAL_GATE < residual <= CONJECTURE_CEILING AND gap ok
        -> SOLVED provisional, reduced-strength clamp.
      otherwise -> node FAILED, parent PENDING_REVISION, no clamp.
    """
    residual = child_result.residual_energy
    gap = _spectral_gap(child_result)

    # --- refusal paths -----------------------------------------------------
    if not child_result.converged:
        return _refuse(node, "child did not converge")
    if gap < GROUND_STATE_GAP:
        return _refuse(node, "near-degenerate: not a true ground state")
    if residual > CONJECTURE_CEILING:
        return _refuse(
            node,
            f"residual {residual} exceeds conjecture ceiling "
            f"{CONJECTURE_CEILING}",
        )

    # --- integration: promote + clamp via sub-project I --------------------
    provisional = residual > RESIDUAL_GATE
    strength = STRENGTH_MAX * precision_weight(residual)

    child_meta = getattr(child_result, "meta", None)
    if child_meta is None:
        return _refuse(
            node, "child_result.meta missing: cannot register lemma")
    if child_result.ground_state is None:
        return _refuse(
            node, "child_result.ground_state missing: cannot register lemma")

    # 1. Register the child's converged state as a lemma (real I-Task-7
    #    surface). ``register_lemma`` is total: a bad candidate surfaces
    #    via ``accepted=False`` rather than raising.
    deriv = DerivationMetadata(
        hamiltonian_id=str(node.goal.goal_id),
        residual_energy=float(residual),
        energy_gap=float(gap),
        trotter_steps=int(getattr(child_result, "trotter_steps", 0) or 0),
        assumptions=(),
        lemma_deps=(),
        conditional=provisional,  # above-gate residual: a conjecture
        source_run_id=str(node.goal.goal_id),
    )
    reg = register_lemma(
        lemma_library,
        child_result.ground_state,
        child_meta,
        hamiltonian=getattr(child_result, "hamiltonian", None),
        derivation=deriv,
        eps_register=CONJECTURE_CEILING,
    )
    if not reg.accepted:
        return _refuse(node, f"lemma registration failed: {reg.reason}")

    # 2. Clamp the registered lemma onto the parent MERA (real I-Task-8
    #    surface). Strength carries the precision-weighting from §6.2.
    #
    # When ``parent_state`` / ``parent_meta`` is None (the root-leaf case,
    # or any caller that drives the integrator purely for lemma
    # registration), there is nothing to clamp into -- the lemma is
    # registered, the node is marked SOLVED, and the clamp is skipped.
    # Skipping is principled here: spec §6.1's clamp targets a *parent*
    # MERA; without one, registration alone is the integration step.
    if parent_state is not None and parent_meta is not None:
        # --- caller-discipline preconditions (loud failures, not coerce) --
        # A non-root node MUST publish a non-empty parent_leaves tuple: the
        # integrator is about to clamp a lemma footprint, and an empty
        # footprint on a non-root child is a decomposer / dispatcher bug,
        # not a runtime input to be silently tolerated. Gated on
        # ``parent_state is not None`` so root-leaf integrations (no parent
        # workspace) still flow through register-only path.
        if node.parent is not None and len(node.goal.parent_leaves) == 0:
            raise ValueError(
                "non-root SubGoal must have non-empty parent_leaves; "
                "got empty tuple -- caller discipline failure"
            )
        # Range-check the footprint against the parent MERA. Host capacity
        # is ``parent_meta.n_leaves`` (layout-padded leaf count); a stub
        # meta carrying only ``n_nodes`` falls back to
        # ``n_nodes * LEAVES_PER_NODE`` (the raw, non-PAD count). A stub
        # without either skips the check -- the structural contract is
        # only binding when the parent meta surfaces its host capacity.
        n_host: int | None = None
        n_leaves_attr = getattr(parent_meta, "n_leaves", None)
        if n_leaves_attr is not None:
            n_host = int(n_leaves_attr)
        else:
            n_nodes_attr = getattr(parent_meta, "n_nodes", None)
            if n_nodes_attr is not None:
                n_host = int(n_nodes_attr) * LEAVES_PER_NODE
        if n_host is not None and node.goal.parent_leaves:
            if not all(0 <= leaf < n_host
                       for leaf in node.goal.parent_leaves):
                raise ValueError(
                    f"SubGoal.parent_leaves out of range for parent MERA "
                    f"(n_host={n_host}): got {node.goal.parent_leaves!r}"
                )
        host_leaves = _resolve_host_leaves(node, child_meta)
        promoter = Promoter(lemma_library, mode="init_clamp")
        try:
            promoted = promoter.compile_constraint({
                "kind": "use_lemma",
                "lemma_id": reg.lemma_id,
                "leaves": list(host_leaves),
                # A provisional (conjectural) lemma is registered with
                # conditional=True; explicitly opt in so the clamp does
                # not get refused by the conditional-lemma guard.
                "allow_conditional": provisional,
            })
            promoter.apply_init_clamp(parent_state, parent_meta, promoted,
                                      strength=strength)
        except Exception as exc:            # noqa: BLE001 -- one-shot guard
            return _refuse(node, f"clamp failed: {exc}")

    node.status = Status.SOLVED
    node.result = child_result
    return IntegrationOutcome(
        integrated=True, provisional=provisional,
        clamp_strength=strength,
        reason="conjecture" if provisional else "ground state",
    )


def _refuse(node: Node, reason: str) -> IntegrationOutcome:
    node.status = Status.FAILED
    if node.parent is not None:
        node.parent.status = Status.PENDING_REVISION
    return IntegrationOutcome(integrated=False, provisional=False,
                              clamp_strength=0.0, reason=reason)
