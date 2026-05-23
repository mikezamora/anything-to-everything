"""Tests for MERA imaginary-time evolution (spec §7.5)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_compose import compose_mera_hamiltonians
from src.qft_pcn.logic.ast import Bin, IntLit
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_trotter_step, mera_imaginary_evolve, mera_imaginary_evolve_state,
)


def test_trotter_step_preserves_norm():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    state = mera_trotter_step(state, H, dt=0.1, imaginary=True)
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-8)


def test_imaginary_evolve_returns_energy_trajectory():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    traj = mera_imaginary_evolve(state, H, dt=0.1, steps=20)
    assert len(traj) == 21


def test_imaginary_evolve_monotone_decrease():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    traj = mera_imaginary_evolve(state, H, dt=0.3, steps=30)
    for i in range(len(traj) - 1):
        assert traj[i + 1] <= traj[i] + 1e-6, (
            f"energy rose at step {i}: {traj[i]} -> {traj[i+1]}")
    assert traj[-1] < traj[0] * 0.5


def test_evolve_accepts_composed_hamiltonian():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
                                  MeraEvalHamiltonian(meta))
    traj = mera_imaginary_evolve(state, H, dt=0.3, steps=10)
    assert traj[-1] <= traj[0] + 1e-6


# ----------------------------------------------------------------------
# I-Task-10 blocker #6: frozen_leaves= in MERA imag-time evolution
# (spec §5.2a / §8.6 "clamp + freeze"). The Promoter's apply_init_clamp
# writes lemma leaf tensors into the host MERA's lemma window and returns
# the set of clamped leaves; subsequent relaxation must NOT deform those
# leaves. The hook is operator-algebraic (per-leaf gate-dispatch skip),
# never an AST rewrite (§1.5).
# ----------------------------------------------------------------------


def _bin_2_plus_3():
    src = Bin(op="+", lhs=IntLit(val=2), rhs=IntLit(val=3))
    state, meta = encode_mera(src)
    return state, meta, MeraEvalHamiltonian(meta)


def test_trotter_step_with_all_leaves_frozen_is_identity():
    """All-leaves-frozen: every gate is dropped at dispatch, so each
    leaf vector is bitwise unchanged through one Trotter step."""
    state, _, H = _bin_2_plus_3()
    frozen = set(range(state.N))
    snaps = [state.leaves[k].copy() for k in range(state.N)]
    out = mera_trotter_step(state, H, dt=0.05, imaginary=True,
                            chi_layer=16, frozen_leaves=frozen)
    for k in range(state.N):
        assert np.array_equal(out.leaves[k], snaps[k]), (
            f"leaf {k} mutated under all-frozen evolution")


def test_imaginary_evolve_many_steps_preserves_frozen_leaves():
    """Subset frozen across many steps: those leaves stay bitwise
    unchanged after the full evolution — no accumulated drift."""
    state, meta, H = _bin_2_plus_3()
    # Freeze the BIN node's kind+value leaves at site 0. These are
    # exactly the leaves an active reduction rule would otherwise drive.
    bin_kind_leaf = meta.layout.leaf_of(0, "kind")
    bin_val_leaf = meta.layout.leaf_of(0, "value")
    frozen = {bin_kind_leaf, bin_val_leaf}
    snap_kind = state.leaves[bin_kind_leaf].copy()
    snap_val = state.leaves[bin_val_leaf].copy()
    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=40, chi_layer=16,
        frozen_leaves=frozen)
    assert np.array_equal(final.leaves[bin_kind_leaf], snap_kind), (
        "frozen BIN kind leaf drifted across many steps")
    assert np.array_equal(final.leaves[bin_val_leaf], snap_val), (
        "frozen BIN value leaf drifted across many steps")


def test_imaginary_evolve_descends_on_non_frozen_leaves():
    """Freezing a non-redex leaf (PAD) does not turn off relaxation
    of the unfrozen subsystem: energy still descends. The contrapositive
    of the previous test — bit-stability is purely scoped to the set."""
    state, meta, H = _bin_2_plus_3()
    e0 = H.total_energy(state)
    pad_leaves = [i for i, sp in enumerate(meta.species_of_leaf)
                  if sp == "PAD"]
    frozen = {pad_leaves[0]} if pad_leaves else set()
    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.3, steps=30, chi_layer=16,
        frozen_leaves=frozen)
    e_final = H.total_energy(final)
    assert e_final < e0 - 1e-6, (
        f"energy did not descend on unfrozen subsystem: "
        f"{e0} -> {e_final}")
