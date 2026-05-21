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
