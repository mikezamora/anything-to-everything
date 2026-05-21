"""Tests for EvalHamiltonian (sub-project C of the §10 roadmap).

Per spec docs/superpowers/specs/2026-05-21-eval-hamiltonian-design.md and the
controller resolutions: EvalHamiltonian mirrors TypingHamiltonian's shape —
factored, no dense d_local^2 op, no AST inspection during construction.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

from src.qft_pcn.logic import encode, parse


# ---- Constructor + structural shape --------------------------------------


def test_eval_hamiltonian_constructor_takes_only_N():
    from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
    H = EvalHamiltonian(N=8)
    assert H.N == 8


def test_eval_hamiltonian_has_5_species():
    from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
    assert len(EvalHamiltonian.SPECIES) == 5


def test_eval_hamiltonian_enumerates_terms_n4():
    from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
    H = EvalHamiltonian(N=4)
    # At minimum: beta and if redex bonds = (N-1) each; arith pre and post
    # bonds = (N-1) and (N-2) each (per spec §3.3).
    # We don't lock the exact count here, only that .terms is non-empty
    # and every term has a known rule_id / site.
    assert len(H.terms) > 0
    for t in H.terms:
        assert isinstance(t.rule_id, str)
        assert 0 <= t.site < H.N


def test_eval_term_not_found_raises():
    from src.qft_pcn.logic.evaluation_hamiltonian import (
        EvalHamiltonian, EvalTerm, EvalTermNotFound,
    )
    H = EvalHamiltonian(N=4)
    bogus = EvalTerm(rule_id="nonsense", site=0, arity=1)
    state, _ = encode(parse(r"\x:Int. x"), N=4, chi_max=32)
    with pytest.raises(EvalTermNotFound):
        H.term_energy(state, bogus)


# ---- Normal-form has zero H_eval energy (spec §7.2) ----------------------


@pytest.mark.parametrize("src", [
    r"\x:Int. x",
    r"\x:Int. 3",
    r"\x:Bool. true",
    r"\x:Int. \y:Int. y",
])
def test_normal_form_has_zero_eval_energy(src):
    from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
    state, _ = encode(parse(src), N=8, chi_max=32)
    H = EvalHamiltonian(N=8)
    e = H.total_energy(state)
    assert abs(e) < 1e-9, f"normal form {src!r} has eval energy {e}"


# ---- Unreduced programs light up H_eval (spec §7.3) ----------------------


@pytest.mark.parametrize("src", [
    "2 + 3",
    r"(\x:Int. x + 1) 2",
    "if (1 < 2) then 7 else 8",
])
def test_unreduced_has_positive_eval_energy(src):
    from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
    state, _ = encode(parse(src), N=12, chi_max=32)
    H = EvalHamiltonian(N=12)
    e = H.total_energy(state)
    assert e > 0.5, f"unreduced {src!r} has eval energy {e} <= 0.5"


def test_beta_redex_fires_on_app_lam():
    """`(\\x:Int. x) 2` has APP@0, LAM@1 — beta term must be > 0."""
    from src.qft_pcn.logic.evaluation_hamiltonian import (
        EvalHamiltonian, RULE_R_BETA,
    )
    state, _ = encode(parse(r"(\x:Int. x) 2"), N=8, chi_max=32)
    H = EvalHamiltonian(N=8)
    r = H.residuals(state)
    # The (RULE_R_BETA, 0) term must be > 0.5.
    e = r[(RULE_R_BETA, 0)]
    assert e > 0.5, f"beta redex did not fire: {e}"


def test_arith_redex_fires_on_bin_intlit():
    """`2 + 3` has BIN@0, INT@1, INT@2 — arith pre and post terms must be > 0."""
    from src.qft_pcn.logic.evaluation_hamiltonian import (
        EvalHamiltonian, RULE_R_ARITH_PRE, RULE_R_ARITH_POST,
    )
    state, _ = encode(parse("2 + 3"), N=6, chi_max=32)
    H = EvalHamiltonian(N=6)
    r = H.residuals(state)
    pre = r[(RULE_R_ARITH_PRE, 0)]
    post = r[(RULE_R_ARITH_POST, 1)]
    assert pre > 0.5, f"arith pre did not fire: {pre}"
    assert post > 0.5, f"arith post did not fire: {post}"


def test_if_redex_fires_on_if_bool():
    """`if true then 1 else 2` has IF@0, BOOL@1 — if-redex must be > 0."""
    from src.qft_pcn.logic.evaluation_hamiltonian import (
        EvalHamiltonian, RULE_R_IF,
    )
    state, _ = encode(parse("if true then 1 else 2"), N=8, chi_max=32)
    H = EvalHamiltonian(N=8)
    r = H.residuals(state)
    e = r[(RULE_R_IF, 0)]
    assert e > 0.5, f"if redex did not fire: {e}"


# ---- Composer (controller resolution #4) ---------------------------------


def test_compose_hamiltonians_is_sum():
    """⟨ψ|H_total|ψ⟩ = ⟨ψ|H_typing|ψ⟩ + ⟨ψ|H_eval|ψ⟩."""
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
    from src.qft_pcn.logic.compose import compose_hamiltonians
    state, _ = encode(parse("2 + 3"), N=8, chi_max=32)
    H_t = TypingHamiltonian(N=8)
    H_e = EvalHamiltonian(N=8)
    H = compose_hamiltonians(H_t, H_e)
    e_total = H.total_energy(state)
    e_t = H_t.total_energy(state)
    e_e = H_e.total_energy(state)
    assert abs(e_total - (e_t + e_e)) < 1e-9


def test_compose_terms_dispatches_by_identity():
    """compose's term_energy delegates to the originating sub-Hamiltonian."""
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
    from src.qft_pcn.logic.compose import compose_hamiltonians
    state, _ = encode(parse("2 + 3"), N=8, chi_max=32)
    H_t = TypingHamiltonian(N=8)
    H_e = EvalHamiltonian(N=8)
    H = compose_hamiltonians(H_t, H_e)
    # Each typing term should match its T-energy from H_t.
    for t in H_t.terms[:3]:
        v_compose = H.term_energy(state, t)
        v_direct = H_t.term_energy(state, t)
        assert abs(v_compose - v_direct) < 1e-12
    # Each eval term should match its E-energy from H_e.
    for t in H_e.terms[:3]:
        v_compose = H.term_energy(state, t)
        v_direct = H_e.term_energy(state, t)
        assert abs(v_compose - v_direct) < 1e-12


# ---- Static guard: no classical interpreter (spec §7.10) -----------------


def test_no_classical_evaluator_imported():
    """H_eval construction must not depend on a Python AST evaluator."""
    import src.qft_pcn.logic.evaluation_hamiltonian as eh
    src_text = open(eh.__file__).read()
    lowered = src_text.lower()
    # Spec's banned substrings — comments are checked too (per §7.10).
    # Allow "evaluate" only inside docstrings/comments that explicitly disclaim
    # it; we keep the test strict.
    assert "def evaluate" not in lowered, (
        "evaluation_hamiltonian.py defines an evaluator — manifesto violation"
    )
    assert "substitute" not in lowered, (
        "evaluation_hamiltonian.py mentions substitute — possible interpreter"
    )
    assert "beta_reduce" not in lowered, "beta_reduce found"
    assert "from .ast" not in src_text, "imports AST classes — should not need to"


def test_no_classical_evaluator_in_eval_terms():
    import src.qft_pcn.logic._eval_terms as et
    src_text = open(et.__file__).read()
    lowered = src_text.lower()
    assert "def evaluate" not in lowered
    assert "substitute" not in lowered
    assert "beta_reduce" not in lowered
    assert "from .ast" not in src_text
    assert re.search(r"\bimport ast\b", src_text) is None
