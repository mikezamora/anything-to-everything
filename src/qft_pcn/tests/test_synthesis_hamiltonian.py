"""Tests for synthesis/hamiltonian.py: H_examples, H_target_type, H_size."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    Lam, Var, App, IntLit, BoolLit, Bin, HoleVar, TInt, TBool, TArrow,
)
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.compose import compose_hamiltonians
from src.qft_pcn.logic.synthesis.problem import (
    IOExample, SynthesisProblem, HamiltonianWeights,
)
from src.qft_pcn.logic.synthesis.hamiltonian import (
    ExamplesHamiltonian, TargetTypeHamiltonian, SizeHamiltonian,
    build_synthesis_hamiltonians,
)


N = 8


@pytest.mark.timeout(30)
def test_examples_ham_zero_when_root_matches_output():
    """root = IntLit(3); IOExample output = IntLit(3) ⇒ energy ~ 0."""
    sketch = IntLit(val=3)
    state, meta = encode(sketch, N=N, chi_max=8)
    H = ExamplesHamiltonian(
        N=N,
        examples=(IOExample(inputs=(), output=IntLit(val=3)),),
        weight=3.0,
    )
    e = H.total_energy(state)
    assert e < 1e-6, f"expected ~0, got {e}"


@pytest.mark.timeout(30)
def test_examples_ham_positive_when_root_does_not_match():
    sketch = IntLit(val=3)
    state, meta = encode(sketch, N=N, chi_max=8)
    H = ExamplesHamiltonian(
        N=N,
        examples=(IOExample(inputs=(), output=IntLit(val=5)),),
        weight=3.0,
    )
    e = H.total_energy(state)
    assert e > 2.0, f"expected ~3 (full weight), got {e}"


@pytest.mark.timeout(30)
def test_target_type_ham_zero_when_root_matches_type():
    sketch = IntLit(val=3)
    state, meta = encode(sketch, N=N, chi_max=8)
    H = TargetTypeHamiltonian(N=N, target_type=TInt(), weight=2.0)
    e = H.total_energy(state)
    assert e < 1e-6


@pytest.mark.timeout(30)
def test_target_type_ham_positive_on_type_mismatch():
    sketch = IntLit(val=3)
    state, meta = encode(sketch, N=N, chi_max=8)
    H = TargetTypeHamiltonian(N=N, target_type=TBool(), weight=2.0)
    e = H.total_energy(state)
    assert e > 1.5


@pytest.mark.timeout(30)
def test_target_type_ham_none_is_inactive():
    state, meta = encode(IntLit(val=3), N=N, chi_max=8)
    H = TargetTypeHamiltonian(N=N, target_type=None, weight=2.0)
    assert H.terms == []
    assert H.total_energy(state) == 0.0


@pytest.mark.timeout(30)
def test_size_ham_counts_nonpad_sites():
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))  # 2 sites
    state, meta = encode(sketch, N=N, chi_max=8)
    H = SizeHamiltonian(N=N, weight=0.1)
    e = H.total_energy(state)
    # 2 non-PAD sites * 0.1 weight = ~0.2
    assert abs(e - 0.2) < 1e-3, f"expected 0.2, got {e}"


@pytest.mark.timeout(30)
def test_compose_with_typing_eval_runs():
    """Smoke: compose ExamplesHamiltonian with a placeholder; total_energy works."""
    from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
    sketch = IntLit(val=3)
    state, meta = encode(sketch, N=N, chi_max=8)
    H_eval = EvalHamiltonian(N=N)
    H_ex = ExamplesHamiltonian(
        N=N,
        examples=(IOExample(inputs=(), output=IntLit(val=3)),),
        weight=3.0,
    )
    H_total = compose_hamiltonians(H_eval, H_ex)
    e = H_total.total_energy(state)
    assert np.isfinite(e)
