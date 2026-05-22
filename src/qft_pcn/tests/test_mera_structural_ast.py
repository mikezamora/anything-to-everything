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
