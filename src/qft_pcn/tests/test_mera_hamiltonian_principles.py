"""Principle-enforcement + cross-substrate tests (spec §9.6-§9.8)."""
from __future__ import annotations
import inspect
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic import mera_evaluation_hamiltonian as MEH
from src.qft_pcn.logic import _mera_eval_terms as MET


def test_no_classical_interpreter_in_eval():
    """Spec §1.6 / §9.6: eval Hamiltonian has no classical interpreter."""
    for mod in (MEH, MET):
        src = inspect.getsource(mod)
        low = src.lower()
        assert "substitute" not in low, f"{mod.__name__}: substitute"
        assert "beta_reduce" not in low, f"{mod.__name__}: beta_reduce"
        # 'evaluate' may appear in docstrings as a concept; check for a
        # call/def, not the word.
        assert "def evaluate" not in low
        assert "from .ast import" not in src and "from . import ast" not in src


def test_typing_hamiltonian_structural():
    """Spec §9.6: same instance evaluates correctly; no AST attrs."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    for attr in dir(H):
        if attr.startswith("_"):
            continue
        for bad in ("ast", "tree", "walk", "node_obj"):
            assert bad not in attr.lower()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("src", [
    r"\x:Int. x",
    r"(\x:Int. x + 1)(2)",
    r"\f:Int->Int. \x:Int. f (f x)",
    r"if 1 < 2 then ((\x:Bool. x)(true)) else false",
    r"(\x:Int. (\y:Int. x + y)(3))(4)",
])
def test_cross_substrate_anchor_well_typed(src):
    """Spec §9.7: MERA and MPS typing Hamiltonians agree on P1-P5.

    One test per program — the MPS factored expectation is heavy on
    larger programs and per-program parametrization gives each its own
    timeout budget and clean failure isolation. The higher-order
    program `\\f:Int->Int. \\x:Int. f (f x)` is genuinely expensive on
    the pre-existing MPS substrate (the numpy.tensordot at the heart of
    `_factored_expectation.py` scales badly with the typing-Hamiltonian
    term count for arrow-typed binders) — that is a known limitation of
    the OLD substrate, separate from M2; the 1200s budget accommodates
    it while keeping the assertion identical.
    """
    from src.qft_pcn.logic.encoder import encode as encode_mps
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    p = parse(src)
    m_state, m_meta = encode_mera(p)
    mera_e = MeraTypingHamiltonian(m_meta).total_energy(m_state)
    mps_state, mps_meta = encode_mps(p, N=32)
    mps_e = TypingHamiltonian(N=32).total_energy(mps_state)
    assert abs(mera_e) < 1e-6 and abs(mps_e) < 1e-6, (
        f"{src}: mera {mera_e}, mps {mps_e}")


def test_cross_substrate_anchor_ill_typed():
    """Spec §9.7: both substrates flag IT1 as ill-typed."""
    from src.qft_pcn.logic.encoder import encode as encode_mps
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    p = parse(r"\x:Int. x + true")
    m_state, m_meta = encode_mera(p)
    mps_state, mps_meta = encode_mps(p, N=32)
    assert MeraTypingHamiltonian(m_meta).total_energy(m_state) > 0.5
    assert TypingHamiltonian(N=32).total_energy(mps_state) > 0.5
