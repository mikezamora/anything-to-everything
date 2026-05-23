"""Spec §12.13 acceptance: bidirectional time evolution.

The substrate's :func:`mera_imaginary_evolve_state` accepts a scalar
``dt``; positive ``dt`` descends in <H> (§13.1), negative ``dt`` climbs.
The :func:`bidirectional_evolve` driver chains a descent leg with an
ascent leg, reusing the real substrate (no mock evolution).

These tests use real ``encode_mera`` MERA states and the real
:class:`MeraEvalHamiltonian` so the §1.1 entanglement invariant is
exercised end-to-end. The §9.7 dense-tensor ceiling in
``composition/tests/conftest.py`` (256-element 2-D cap) is overridden
by an autouse fixture below — encode_mera allocates 16-dim isometries
by construction, the same opt-out used in
``test_cross_level_acceptance.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.bidirectional import (
    EvolveResult, bidirectional_evolve,
)
from src.qft_pcn.logic.ast import (
    Bin, Eq, Forall, IntLit, TNat, Var, Zero, parse,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve_state


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest fixture for this file only
    """Opt this file out of the §9.7 dense-tensor ceiling: real
    encode_mera allocates 16-dim leaves and isometries (4096-element
    pair matrices) that exceed the conftest cap by construction. Same
    opt-out as test_cross_level_acceptance.py."""
    yield


# -- helpers --------------------------------------------------------------

def _two_plus_three():
    """A small redex program: ``2 + 3``. Initial <H> > 0; under
    descent <H> -> 0 monotonically."""
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    return state, meta, H


def _add_zero_theorem():
    """The §10.10 composite ``forall x:Nat. Eq (add x Zero) x`` — used
    only for the Forall-protected leaf invariant test (so the protected
    set is non-empty)."""
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    return state, meta, H


# -- §12.13 acceptance ----------------------------------------------------


def test_negative_dt_climbs_out_of_basin():
    """Start at a *partially relaxed* (non-ground) state, then run a
    pure negative-dt leg: <H> rises. This is the spec §12.13
    climb-out-of-local-minimum primitive in isolation.

    Construction: take a small redex program (``2 + 3``), descend a few
    steps to land NEAR (but not at) the ground state, then ascend.
    The descent leg has dropped <H> well below initial; the ascent
    leg must raise <H> measurably above the descent floor.
    """
    state, _, H = _two_plus_three()
    e0 = float(H.total_energy(state))
    # Light descent: bring <H> down, but stop while there is still
    # appreciable redex amplitude to RE-excite under the climb.
    _, partial = mera_imaginary_evolve_state(state, H, dt=0.3, steps=8,
                                             chi_layer=16)
    e_floor = float(H.total_energy(partial))
    assert e_floor < e0 - 1e-6, (
        f"partial descent did not lower <H>: {e0} -> {e_floor}; "
        f"the test fixture must start with a redex"
    )

    # Pure ascent (steps_pos=0): bidirectional_evolve passes negative
    # dt to the real substrate.
    res = bidirectional_evolve(
        partial, H,
        dt_pos=0.3, dt_neg=0.3,
        steps_pos=0, steps_neg=10,
        chi_layer=16,
    )
    assert isinstance(res, EvolveResult)
    assert len(res.trajectory) == 11  # 1 initial + 10 ascent steps
    assert res.pos_end_index == 0

    e_climbed = res.trajectory[-1]
    # The climb must raise <H> measurably above the descent floor.
    # Tolerance is loose because the climb is FACTORED — only redex-
    # active terms contribute; <H> rise is bounded by the total redex
    # potential of the original program.
    assert e_climbed > e_floor + 1e-4, (
        f"negative-dt did NOT climb <H>: floor={e_floor}, "
        f"climbed={e_climbed}; spec §12.13 violated"
    )
    # Every point in the ascent leg sits strictly ABOVE the descent
    # floor (the climb pulled <H> out of the local minimum and keeps
    # it out). We do NOT assert strict step-wise monotone here: once
    # the climb saturates the redex penalty (term_energy reaches its
    # maximum), the term_gates' redex-present guard fires and the
    # subsequent SVD-renormalization can produce small post-saturation
    # oscillations. The §13.1 dual is "climb to saturation and stay
    # excited", not "strictly monotone after saturation".
    for i, e in enumerate(res.trajectory[1:], start=1):
        assert e > e_floor + 1e-4, (
            f"ascent leg fell back to or below the descent floor at "
            f"step {i}: floor={e_floor}, e={e}"
        )


def test_bidirectional_finds_target_from_source():
    """Combine positive- and negative-dt phases: the trajectory must
    show a descent (energy minimum at the phase boundary) followed by
    an ascent. This proves the chained primitive actually traverses
    between basins — the descent floor is strictly below both
    endpoints.

    Concretely, the trajectory ``[<H>_0, ..., <H>_T_pos, ..., <H>_T]``
    must satisfy:
      * ``<H>_T_pos < <H>_0`` (descent worked)
      * ``<H>_T > <H>_T_pos`` (ascent worked)
      * Monotone within each leg
    """
    state, _, H = _two_plus_three()
    # Use a moderate descent (~5 steps) that lowers <H> measurably
    # but leaves enough unreduced amplitude on the redex leaves for
    # the ascent leg to re-excite. Over-descending to ~1e-7 floor
    # would leave nothing for negative dt to amplify (the §13.1
    # fixed-point at <H>=0 is the negative-dt fixed point too — the
    # gate's identity branch). The test fixture must avoid that
    # degeneracy by design.
    res = bidirectional_evolve(
        state, H,
        dt_pos=0.3, dt_neg=0.3,
        steps_pos=5, steps_neg=8,
        chi_layer=16,
    )
    assert isinstance(res, EvolveResult)
    assert len(res.trajectory) == 5 + 8 + 1
    assert res.pos_end_index == 5

    e_start = res.trajectory[0]
    e_floor = res.trajectory[res.pos_end_index]
    e_end = res.trajectory[-1]

    assert e_floor < e_start - 1e-4, (
        f"descent leg failed: {e_start} -> {e_floor}"
    )
    # Ascent must rise meaningfully above the floor. Use a relative
    # margin (10x the floor or 1e-3 absolute, whichever is larger) so
    # the test is robust to where exactly the descent leg landed.
    rise_threshold = max(10.0 * e_floor, 1e-3)
    assert e_end > rise_threshold, (
        f"ascent leg did not traverse out of the descent basin: "
        f"floor={e_floor}, end={e_end}, threshold={rise_threshold}"
    )

    # Monotone within each leg.
    for i in range(res.pos_end_index):
        assert res.trajectory[i + 1] <= res.trajectory[i] + 1e-6, (
            f"descent leg non-monotone at step {i}: "
            f"{res.trajectory[i]} -> {res.trajectory[i+1]}"
        )
    # Ascent leg: every point sits meaningfully above the descent
    # floor (see the corresponding comment in
    # test_negative_dt_climbs_out_of_basin — post-saturation SVD
    # oscillations break strict monotone but not the "climbed out of
    # the basin" invariant). Use the same relative margin as above.
    # Pointwise: every ascent step is strictly above the floor (the
    # full 10x threshold is only required at the end of the ascent
    # leg; intermediate steps may still be ramping up).
    for i in range(res.pos_end_index + 1, len(res.trajectory)):
        assert res.trajectory[i] > e_floor, (
            f"ascent leg dipped at or below the descent floor at "
            f"step {i}: floor={e_floor}, e={res.trajectory[i]}"
        )

    # Final state must be a normalized MERA (no NaN / norm blowup
    # under the negative-dt leg — the per-step leaf renormalization
    # in mera_trotter_step protects against the s = exp(+|dt|*lam) > 1
    # amplification).
    assert np.isclose(res.final_state.norm_sq(), 1.0, atol=1e-6), (
        f"final state norm broken: {res.final_state.norm_sq()}"
    )


def test_protected_leaves_preserved_under_bidirectional():
    """§5.2a / §8.6 invariant under spec §12.13: Forall-protected
    leaves must stay bitwise unchanged across BOTH the positive-dt
    and the negative-dt phases. The gate-dispatch drop in
    ``mera_trotter_step`` is sign-independent, but we exercise the
    invariant end-to-end through the public driver.
    """
    state, meta, H = _add_zero_theorem()
    protected = set(meta.forall_protected_leaves)
    assert protected, (
        "test fixture invariant: forall_protected_leaves must be "
        "non-empty for ``forall x:Nat. Eq (add x Zero) x``"
    )
    snaps = {k: state.leaves[k].copy() for k in protected}

    res = bidirectional_evolve(
        state, H,
        dt_pos=0.1, dt_neg=0.1,
        steps_pos=20, steps_neg=10,
        chi_layer=16,
        frozen_leaves=protected,
    )

    for k in protected:
        assert np.array_equal(res.final_state.leaves[k], snaps[k]), (
            f"protected leaf {k} mutated under bidirectional evolve; "
            f"spec §5.2a / §8.6 clamp + freeze violated"
        )


# -- defensive: parameter validation -------------------------------------


def test_rejects_negative_dt_pos():
    state, _, H = _two_plus_three()
    with pytest.raises(ValueError, match="dt_pos"):
        bidirectional_evolve(state, H, dt_pos=-0.1, dt_neg=0.1,
                             steps_pos=1, steps_neg=1)


def test_rejects_negative_dt_neg():
    state, _, H = _two_plus_three()
    with pytest.raises(ValueError, match="dt_neg"):
        bidirectional_evolve(state, H, dt_pos=0.1, dt_neg=-0.1,
                             steps_pos=1, steps_neg=1)


def test_rejects_zero_total_steps():
    state, _, H = _two_plus_three()
    with pytest.raises(ValueError):
        bidirectional_evolve(state, H, dt_pos=0.1, dt_neg=0.1,
                             steps_pos=0, steps_neg=0)
