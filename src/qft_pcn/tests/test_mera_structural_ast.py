"""Tests for the structural-HoleVar AST upgrade (spec §5.1)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import HoleVar, Var, App, Bin, IntLit, Node


def test_holevar_accepts_str_candidates():
    h = HoleVar(candidates=("x", "y"))
    assert h.candidates == ("x", "y")
    assert h.candidate_kind() == "var"


def test_holevar_accepts_node_candidates():
    h = HoleVar(candidates=(Var(name="x"), App(fn=Var(name="f"), arg=Var(name="x"))))
    assert h.candidate_kind() == "structural"
    assert all(isinstance(c, Node) for c in h.candidates)


def test_holevar_empty_candidates_is_var():
    assert HoleVar(candidates=()).candidate_kind() == "var"


def test_holevar_rejects_mixed_candidates():
    with pytest.raises(ValueError, match="must not mix"):
        HoleVar(candidates=("x", Var(name="y")))


def test_holevar_is_a_node():
    assert isinstance(HoleVar(candidates=("x",)), Node)


def test_structural_hole_with_var_referencing_candidate_resolves():
    """Fix 1 regression: encode_mera on `\\f. \\x. ?HOLE` with a structural
    candidate `App(f, x)` containing Vars referencing the enclosing lambdas
    must NOT raise IllScopedVar — the resolver recurses into candidate
    sub-trees against the surrounding lexical scope.
    """
    from src.qft_pcn.logic.ast import Lam, TInt, TArrow
    from src.qft_pcn.logic.mera_encoder import encode_mera
    sketch = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(
            param="x", param_ty=TInt(),
            body=HoleVar(candidates=(
                Var(name="x"),
                App(fn=Var(name="f"), arg=Var(name="x")),
            )),
        ),
    )
    state, meta = encode_mera(sketch, chi_layer=32)
    assert state.norm_sq() > 0


def test_empty_candidate_holevar_resolves_and_encodes():
    """Fix 3 regression: `\\x:Int. ?HoleVar(candidates=())` resolves the
    empty candidate list to all in-scope binders (the lone `x`) and
    encodes cleanly.
    """
    from src.qft_pcn.logic.ast import Lam, TInt
    from src.qft_pcn.logic.mera_encoder import encode_mera
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=()))
    state, meta = encode_mera(sketch, chi_layer=32)
    assert state.norm_sq() > 0
