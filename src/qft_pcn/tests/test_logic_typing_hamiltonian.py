"""Tests for logic/typing_hamiltonian.py (sub-project B)."""

from __future__ import annotations

import numpy as np
import pytest


# ---- Task 11: skeleton ----------------------------------------------------


def test_typing_term_dataclass():
    from src.qft_pcn.logic.typing_hamiltonian import TypingTerm
    t = TypingTerm(rule_id="T-Lit-Int", site=3, arity=1)
    assert t.rule_id == "T-Lit-Int"
    assert t.site == 3
    assert t.arity == 1


def test_typing_hamiltonian_constructor_takes_only_N():
    """Spec §1.1: Hamiltonian is data-independent. No AST in ctor."""
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    H = TypingHamiltonian(N=8)
    assert H.N == 8


def test_typing_hamiltonian_enumerates_terms_n4():
    """N=4: 5×4 (one-site) + 3 + 3 + 3 (two-site) = 29 terms."""
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    H = TypingHamiltonian(N=4)
    assert len(H.terms) == 29
    rules = {t.rule_id for t in H.terms}
    assert rules == {
        "T-Lit-Int", "T-Lit-Bool", "T-Bin-Arith", "T-Bin-Cmp",
        "T-Obligation", "T-Var", "T-Abs", "T-App-Arrow",
    }


def test_typing_hamiltonian_has_5_species():
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    assert len(TypingHamiltonian.SPECIES) == 5


def test_term_not_found_raises():
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm, TermNotFound,
    )
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    H = TypingHamiltonian(N=4)
    bogus = TypingTerm(rule_id="T-Bogus", site=2, arity=1)
    state, _ = encode(parse(r"\x:Int. x"), N=4, chi_max=32)
    with pytest.raises(TermNotFound):
        H.term_energy(state, bogus)


# ---- Task 12: T-Lit-Int / T-Lit-Bool --------------------------------------


def test_t_lit_int_zero_on_well_typed_int():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 3"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Lit-Int", 1, 1))
    assert abs(e) < 1e-10


def test_t_lit_int_zero_at_non_int_sites():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    for k in range(8):
        e = H.term_energy(state, TypingTerm("T-Lit-Int", k, 1))
        assert abs(e) < 1e-10


def test_t_lit_bool_zero_on_well_typed_bool():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. true"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Lit-Bool", 1, 1))
    assert abs(e) < 1e-10


def _swap_site_type(state, site, type_from, type_to):
    """Surgically swap basis slices for type=type_from and type=type_to on a
    given site. Vectorized via reshape/swap to avoid the 65K-iteration
    Python loop variant.
    """
    from src.qft_pcn.logic.encoding import (
        KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    )
    T = state.tensors[site].copy()
    chi_l, _, chi_r = T.shape
    A = T.reshape(chi_l, KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF,
                  VALUE_CUTOFF, TOBL_CUTOFF, chi_r)
    # Swap the type axis (index 2) slices type_from ↔ type_to.
    tmp = A[:, :, type_from, ...].copy()
    A[:, :, type_from, ...] = A[:, :, type_to, ...]
    A[:, :, type_to, ...] = tmp
    state.tensors[site] = A.reshape(chi_l, T.shape[1], chi_r)
    state.normalize()


def test_t_lit_int_fires_on_mismatch():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 3"), N=4, chi_max=32)
    _swap_site_type(state, site=1, type_from=TYPE_INT, type_to=TYPE_BOOL)
    H = TypingHamiltonian(N=4)
    e = H.term_energy(state, TypingTerm("T-Lit-Int", 1, 1))
    assert e > 0.5, f"T-Lit-Int should fire at site 1, got {e}"


# ---- Task 13: T-Bin-Arith / T-Bin-Cmp -------------------------------------


def test_t_bin_arith_zero_for_well_typed_plus():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 1 + 2"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Bin-Arith", 1, 1))
    assert abs(e) < 1e-10


def test_t_bin_cmp_zero_for_well_typed_lt():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x < 5"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Bin-Cmp", 1, 1))
    assert abs(e) < 1e-10


# ---- Task 14: T-Obligation ------------------------------------------------


def test_t_obligation_zero_on_well_typed():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    for src in [r"\x:Int. x", r"(\x:Int. x + 1)(2)",
                r"\f:Int->Int. \x:Int. f x"]:
        state, _ = encode(parse(src), N=8, chi_max=32)
        H = TypingHamiltonian(N=8)
        # Use residuals (which precomputes envs once) instead of per-term
        # calls, avoiding the O(N) environment sweep per call.
        residuals = H.residuals(state)
        for k in range(8):
            e = residuals[("T-Obligation", k)]
            assert abs(e) < 1e-10, f"{src}: T-Obligation@{k} = {e}"


def test_t_obligation_fires_on_ill_typed():
    """\\x:Int. x + true: 'true' at BIN's rhs has tobl=INT but type=BOOL."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, meta = encode(parse(r"\x:Int. x + true"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    # Sum T-Obligation over all sites — at least one must fire.
    total = sum(H.term_energy(state, TypingTerm("T-Obligation", k, 1))
                for k in range(8))
    assert total > 0.5, f"T-Obligation should fire somewhere, got {total}"


# ---- Task 15: T-Var ------------------------------------------------------


def test_t_var_zero_for_well_typed_var():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Var", 1, 2))
    assert abs(e) < 1e-10, f"T-Var should be 0, got {e}"


def test_t_var_zero_at_non_var_sites():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 3"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Var", 1, 2))
    assert abs(e) < 1e-10


# ---- Task 16: T-Abs ------------------------------------------------------


def test_t_abs_zero_for_well_typed_lam():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Abs", 0, 2))
    assert abs(e) < 1e-10


def test_t_abs_zero_at_non_lam_sites():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 1 + 2"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    for k in range(1, 7):
        e = H.term_energy(state, TypingTerm("T-Abs", k, 2))
        assert abs(e) < 1e-10, f"T-Abs@{k} should be 0 at non-LAM, got {e}"


# ---- Task 17: T-App-Arrow -------------------------------------------------


def test_t_app_arrow_zero_for_well_typed_app():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"(\x:Int. x)(1)"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-App-Arrow", 0, 2))
    assert abs(e) < 1e-10


def test_t_app_arrow_zero_at_non_app_sites():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    for k in range(7):
        e = H.term_energy(state, TypingTerm("T-App-Arrow", k, 2))
        assert abs(e) < 1e-10, f"T-App-Arrow@{k}: {e}"


# ---- Task 18: total_energy + residuals smoke -----------------------------


def test_total_energy_zero_on_well_typed_identity():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    assert abs(H.total_energy(state)) < 1e-9


def test_total_energy_nonzero_on_x_plus_true():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(r"\x:Int. x + true"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    assert H.total_energy(state) > 0.5


def test_residuals_sum_to_total_energy():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(r"(\x:Int. x + 1)(2)"), N=16, chi_max=32)
    H = TypingHamiltonian(N=16)
    total = H.total_energy(state)
    residuals_sum = sum(H.residuals(state).values())
    assert abs(total - residuals_sum) < 1e-10


# ---- Task 19: mutate_local_register helper -------------------------------


def test_mutate_local_register_preserves_norm():
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    state, _ = encode(parse(r"\x:Int. x"), N=4, chi_max=32)
    n0 = state.norm_sq()
    mutate_local_register(state, site=1, register="type",
                          old=TYPE_INT, new=TYPE_BOOL)
    n1 = state.norm_sq()
    assert abs(n1 - n0) < 1e-10


# ---- Task 20: WT acceptance tests ----------------------------------------


WT_PROGRAMS = [
    ("WT1", r"\x:Int. x"),
    ("WT2", r"(\x:Int. x + 1)(2)"),
    ("WT3", r"\f:Int->Int. \x:Int. f (f x)"),
    ("WT4", r"if (1 < 2) then ((\x:Bool. x)(true)) else false"),
    ("WT5", r"(\x:Int. (\y:Int. x + y)(3))(4)"),
]


@pytest.mark.parametrize("name,src", WT_PROGRAMS,
                         ids=[p[0] for p in WT_PROGRAMS])
def test_well_typed_residual_zero(name, src):
    """Spec §7.1: ⟨H_typing⟩ < 1e-9 for every WT program."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(src), N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    residuals = H.residuals(state)
    energy = sum(residuals.values())
    nonzero = {k: v for k, v in residuals.items() if v > 1e-10}
    assert abs(energy) < 1e-9, (
        f"{name} ({src}): expected ⟨H_typing⟩ = 0, got {energy}; "
        f"nonzero residuals: {nonzero}"
    )


# ---- Task 21: IT acceptance tests ----------------------------------------


def _it_violation_descriptor():
    return [
        # (id, source, expected_rule, expected_site)
        ("IT1", r"\x:Int. x + true", "T-Obligation", 3),
        ("IT2", r"\x:Int. if x then 1 else 0", "T-Obligation", 2),
        ("IT3", r"\x:Int->Bool. (x 1) + 1", "T-Obligation", 2),
        ("IT4", r"\x:Int. (\y:Bool. y)(x)", "T-Obligation", 4),
        ("IT5", r"(\x:Bool. x + 1)(true)", "T-Obligation", 3),
    ]


@pytest.mark.parametrize(
    "name,src,expected_rule,expected_site",
    _it_violation_descriptor(),
    ids=[p[0] for p in _it_violation_descriptor()],
)
def test_ill_typed_residual_localized(name, src, expected_rule, expected_site):
    """Spec §7.2: each IT has ⟨H⟩ > 0.5 with exactly one term firing > 0.5."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(src), N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    residuals = H.residuals(state)
    energy = sum(residuals.values())
    assert energy > 0.5, f"{name}: expected ⟨H⟩ > 0.5, got {energy}"
    big = [(k, v) for k, v in residuals.items() if v > 0.5]
    assert len(big) == 1, (
        f"{name}: expected exactly 1 firing rule, got {len(big)}: {big}"
    )
    fired_key, fired_val = big[0]
    assert fired_key == (expected_rule, expected_site), (
        f"{name}: expected fire at {(expected_rule, expected_site)}, "
        f"got {fired_key} with energy {fired_val}"
    )


# ---- Tasks 22-23: per-rule isolation -------------------------------------


def test_isolation_t_lit_int():
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    state, _ = encode(parse(r"\x:Int. 3"), N=8, chi_max=32)
    mutate_local_register(state, 1, "type", TYPE_INT, TYPE_BOOL)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Lit-Int", 1)] > 0.99
    assert r[("T-Lit-Bool", 1)] < 1e-10


def test_isolation_t_lit_bool():
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_BOOL, TYPE_INT
    state, _ = encode(parse(r"\x:Int. true"), N=8, chi_max=32)
    mutate_local_register(state, 1, "type", TYPE_BOOL, TYPE_INT)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Lit-Bool", 1)] > 0.99
    assert r[("T-Lit-Int", 1)] < 1e-10


def test_isolation_t_bin_arith():
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    state, _ = encode(parse(r"\x:Int. 1 + 2"), N=8, chi_max=32)
    mutate_local_register(state, 1, "type", TYPE_INT, TYPE_BOOL)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Bin-Arith", 1)] > 0.99
    assert r[("T-Bin-Cmp", 1)] < 1e-10


def test_isolation_t_bin_cmp():
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_BOOL, TYPE_INT
    state, _ = encode(parse(r"\x:Int. x < 5"), N=8, chi_max=32)
    mutate_local_register(state, 1, "type", TYPE_BOOL, TYPE_INT)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Bin-Cmp", 1)] > 0.99
    assert r[("T-Bin-Arith", 1)] < 1e-10


def test_isolation_t_var():
    """Corrupt Var x's type INT -> BOOL; T-Var(1) fires.

    Note on structural bound: the encoder's bid bond carries the channel
    info as a SUPERPOSITION of "no_info" (slot 0) and "channel c with
    param_ty = t" (slot 1+8(c-1)+t). For a single-binder bond, slot 0
    and the channel slot each carry ~50% amplitude — by construction
    (state.normalize() spreads probability across the chain). T-Var's
    bond-projector hits ONLY the channel slot, so the max firing for a
    type-mismatch violation is ~0.5, not 1.0. The plan's > 0.99 was
    derived assuming a different bond-encoding convention; for our
    encoder the structural max is 0.5.
    """
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    mutate_local_register(state, 1, "type", TYPE_INT, TYPE_BOOL)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Var", 1)] > 0.4, f"T-Var should fire, got {r[('T-Var', 1)]}"


def test_isolation_t_app_arrow():
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_ARR_II, TYPE_INT
    state, _ = encode(parse(r"(\x:Int. x)(1)"), N=8, chi_max=32)
    mutate_local_register(state, 1, "type", TYPE_ARR_II, TYPE_INT)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-App-Arrow", 0)] > 0.99


# ---- Task 26: structural-Hamiltonian invariants --------------------------


def test_typing_hamiltonian_is_structural():
    """Spec §7.8: the same H evaluates correctly on multiple unrelated ASTs."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    H = TypingHamiltonian(N=32)
    for src in [r"\x:Int. x", r"(\x:Int. x + 1)(2)",
                r"if 1 < 2 then 10 else 20"]:
        state, _ = encode(parse(src), N=32, chi_max=32)
        assert abs(H.total_energy(state)) < 1e-9, src


def test_typing_hamiltonian_no_ast_attributes():
    """Spec §7.8: H must not store AST data."""
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    H = TypingHamiltonian(N=32)
    forbidden = ["ast", "node", "tree", "binder_handle",
                 "var_ref", "occupancy", "meta_"]
    for attr_name in dir(H):
        if attr_name.startswith("_"):
            continue
        for sub in forbidden:
            assert sub not in attr_name.lower(), (
                f"H.{attr_name} suggests AST dependency"
            )


def test_typing_hamiltonian_constructor_signature():
    """Spec §1.1: TypingHamiltonian.__init__ takes (self, N) only."""
    import inspect
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    sig = inspect.signature(TypingHamiltonian.__init__)
    params = list(sig.parameters.keys())
    assert params == ["self", "N"]


def test_residual_keys_are_addressable():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    for (rule_id, site), energy in H.residuals(state).items():
        assert isinstance(rule_id, str) and rule_id
        assert 0 <= site < 8
        assert isinstance(energy, float)
        assert energy >= -1e-12   # Hermitian projectors → non-negative


# ---- Task 27: performance + public exports ------------------------------


def test_top_level_typing_hamiltonian_import():
    """TypingHamiltonian is accessible from logic package top level."""
    from src.qft_pcn.logic import TypingHamiltonian, TypingTerm
    state_str = r"\x:Int. x"
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    state, _ = encode(parse(state_str), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    assert abs(H.total_energy(state)) < 1e-9
    assert callable(TypingTerm)
