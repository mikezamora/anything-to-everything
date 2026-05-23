"""Bidirectional time evolution for goal-directed search (spec §12.13).

The §7.5 imaginary-time Trotter step uses single-leaf transition gates of
the form ``g = I + (s - 1) * (|u><u| - |r><u|)`` with
``s = exp(-dt * lam)`` (see :func:`single_leaf_transition_gate` in
``logic/_mera_eval_terms.py``). For ``dt > 0`` we get ``s in (0, 1)`` and
the gate damps the unreduced amplitude toward the reduced fixed point —
classical imaginary-time relaxation; energy descends monotonically
(§13.1).

Flipping the sign of ``dt`` swaps the role of unreduced and reduced
basis values: ``s > 1`` AMPLIFIES the unreduced weight (with leaf
renormalization restoring norm). The resulting evolution climbs UP in
the redex potential — this is the exact mathematical content of spec
§12.13:

    "Schrödinger evolution e^{-iHt} is unitary, so it can be run forward
     (t > 0) or backward (t < 0) with equal facility. The forward
     direction propagates known initial conditions to derived
     consequences; the backward direction propagates known final
     conditions to required initial conditions."

In *imaginary* time the unitarity of the spec quote becomes the gate's
analytic continuation: positive ``dt`` descends, negative ``dt`` climbs.
A chained run — first ``dt_pos > 0`` for ``steps_pos`` steps, then
``dt_neg < 0`` for ``steps_neg`` steps — traverses a path through the
energy landscape that can escape a local minimum (the descent gets
trapped; the climb re-energizes the redex amplitude) and probe a
different basin.

Note: this is the imaginary-time analog of spec §12.13's literal real-time
unitary e^{-iHt} formulation. The substrate's mera_imaginary_evolve_state
is the TEBD imaginary-time evolution; the real-time unitary path is not
yet wired in evolution.py::trotter_step.

This module reuses the real
:func:`mera_imaginary_evolve_state` substrate — no mock evolution, no
re-implementation of the Trotter step. The §5.2a / §8.6
``frozen_leaves`` invariant (Forall-protected leaves bitwise unchanged)
is preserved under both sign conventions because the gate-dispatch drop
happens in :func:`mera_trotter_step` *before* the per-leaf gate is
applied, independent of the sign of ``dt``.

Substrate gaps logged (per memory/no-placeholders.md): none — the
spec-§12.13 capability requires only a sign flip on an existing scalar
parameter; the substrate hook is already in place. The §12.13
"meeting-in-the-middle" two-MPS overlap search described in the spec is
not implemented here (it requires a separate forward/backward overlap
oracle); this module ships the chained-traversal primitive that the
spec's §17 prioritization (row 2) calls out as the immediate win.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve_state


@dataclass(frozen=True)
class EvolveResult:
    """Result of a (possibly bidirectional) evolution.

    Attributes
    ----------
    trajectory:
        Energy ``<H>`` at every step boundary, concatenated across the
        positive- and negative-dt phases. Length is ``steps_pos +
        steps_neg + 1`` (the initial energy plus one entry per step).
    final_state:
        The MERA state after the full chained evolution.
    pos_end_index:
        Index into ``trajectory`` of the energy AFTER the positive-dt
        phase finishes (equivalently, the index of the first energy in
        the negative-dt phase, or ``len(trajectory) - 1`` if
        ``steps_neg == 0``). Callers diagnosing the basin-crossing read
        ``trajectory[pos_end_index]`` as the descent floor and
        ``trajectory[-1]`` as the climbed energy.
    """
    trajectory: list[float]
    final_state: MERA
    pos_end_index: int


def bidirectional_evolve(state: MERA, H, *,
                         dt_pos: float,
                         dt_neg: float,
                         steps_pos: int,
                         steps_neg: int,
                         chi_layer: Optional[int] = None,
                         frozen_leaves: Optional[set[int]] = None
                         ) -> EvolveResult:
    """Chain a positive-dt imaginary-time descent with a negative-dt
    ascent (§12.13). Reuses the real
    :func:`mera_imaginary_evolve_state` substrate for BOTH legs — the
    negative leg simply passes the magnitude with a flipped sign as the
    scalar ``dt`` argument, so the per-term transition gate becomes
    ``s = exp(+|dt| * lam) > 1`` and the leaf is driven AWAY from its
    reduced fixed point (the climb-out).

    Parameters
    ----------
    state:
        Initial MERA state. NOT mutated; the underlying driver
        ``.copy()``'s on entry.
    H:
        Any Hamiltonian object exposing ``total_energy`` and
        ``term_gates`` (e.g. :class:`MeraEvalHamiltonian`,
        :class:`MeraTypingHamiltonian`, or a composed Hamiltonian).
    dt_pos:
        Step size for the positive-dt (descent) phase. Must be
        positive. If ``steps_pos == 0`` this is ignored.
    dt_neg:
        Step size for the negative-dt (ascent) phase. Pass a *positive*
        magnitude; this routine negates it internally so the call site
        reads as the physical step size. Must be positive. If
        ``steps_neg == 0`` this is ignored.
    steps_pos:
        Number of descent (positive-dt) Trotter steps. May be zero
        (pure ascent from the initial state — used to test the
        climb-out invariant in isolation).
    steps_neg:
        Number of ascent (negative-dt) Trotter steps. May be zero
        (pure descent — degenerates to a plain
        ``mera_imaginary_evolve_state`` call).
    chi_layer:
        Bond-dimension cap forwarded to both legs.
    frozen_leaves:
        §5.2a / §8.6 protected leaf set. Forwarded to BOTH legs; the
        invariant holds under negative dt because the drop happens at
        gate dispatch in ``mera_trotter_step``, independent of ``dt``'s
        sign.

    Returns
    -------
    EvolveResult
        Energy trajectory across both phases (length ``steps_pos +
        steps_neg + 1``), the final relaxed/excited MERA state, and
        the boundary index between the two phases.
    """
    if dt_pos <= 0:
        raise ValueError(f"dt_pos must be positive (got {dt_pos}); "
                         "the routine negates internally for the ascent "
                         "leg")
    if dt_neg <= 0:
        raise ValueError(f"dt_neg must be positive magnitude (got "
                         f"{dt_neg}); the routine negates internally")
    if steps_pos < 0 or steps_neg < 0:
        raise ValueError(f"step counts must be non-negative "
                         f"(steps_pos={steps_pos}, steps_neg={steps_neg})")
    if steps_pos == 0 and steps_neg == 0:
        raise ValueError("at least one of steps_pos, steps_neg must be "
                         "positive")

    # ---- positive-dt (descent) leg ----------------------------------
    if steps_pos > 0:
        traj_pos, mid_state = mera_imaginary_evolve_state(
            state, H, dt=dt_pos, steps=steps_pos, chi_layer=chi_layer,
            frozen_leaves=frozen_leaves,
        )
    else:
        # No descent: seed the trajectory with <H>_0 so the boundary
        # index lands at 0 (the climb starts from the initial state).
        mid_state = state.copy()
        traj_pos = [float(H.total_energy(mid_state))]

    pos_end_index = len(traj_pos) - 1

    # ---- negative-dt (ascent) leg -----------------------------------
    if steps_neg > 0:
        traj_neg, final_state = mera_imaginary_evolve_state(
            mid_state, H, dt=-dt_neg, steps=steps_neg, chi_layer=chi_layer,
            frozen_leaves=frozen_leaves,
        )
        # The driver re-emits <H>_0 of the negative phase, which equals
        # the last entry of traj_pos by construction; drop the duplicate
        # so the concatenated trajectory has one energy per step + one
        # initial.
        trajectory = list(traj_pos) + list(traj_neg[1:])
    else:
        final_state = mid_state
        trajectory = list(traj_pos)

    return EvolveResult(
        trajectory=trajectory,
        final_state=final_state,
        pos_end_index=pos_end_index,
    )
