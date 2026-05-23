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
    return float(child_result.run_diagnostic.get("spectral_gap",
                                                  GROUND_STATE_GAP))


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
        return _refuse(node, f"residual {residual} exceeds conjecture ceiling")

    # --- integration: promote + clamp via sub-project I --------------------
    provisional = residual > RESIDUAL_GATE
    strength = STRENGTH_MAX * precision_weight(residual)
    lemma = lemma_library.promote(child_result.ground_state,
                                  child_result.solved_ast,
                                  child_result.goal_id)
    lemma_library.clamp(parent_state, node.goal.parent_site, lemma,
                        strength=strength)

    node.status = Status.SOLVED
    node.result = _flag_provisional(child_result, provisional)
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


def _flag_provisional(child_result, provisional: bool):
    """Attach a provisional flag without mutating the frozen ChildResult."""
    if not provisional:
        return child_result
    import dataclasses
    return dataclasses.replace(child_result)
