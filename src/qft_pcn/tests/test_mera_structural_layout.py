"""Tests for structural-hole layout expansion (spec §5.2, §5.6)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import Lam, TInt, TArrow, Var, App, HoleVar
from src.qft_pcn.logic.mera_synthesis.encode_ext import (
    _expand_structural_holes, HoleRegion,
)


def _p3_sketch():
    """\\f:Int->Int. \\x:Int. ?HOLE  with structural candidates."""
    hole = HoleVar(candidates=(
        Var(name="x"),
        App(fn=Var(name="f"), arg=Var(name="x")),
        App(fn=Var(name="f"), arg=App(fn=Var(name="f"), arg=Var(name="x"))),
    ))
    return Lam(param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
               body=Lam(param="x", param_ty=TInt(), body=hole))


def test_expand_reports_one_region():
    _skel, regions = _expand_structural_holes(_p3_sketch())
    assert len(regions) == 1
    assert isinstance(regions[0], HoleRegion)


def test_region_n_max_is_largest_candidate_node_count():
    _skel, regions = _expand_structural_holes(_p3_sketch())
    # candidates have 1, 2, 3 nodes -> n_max = 3
    assert regions[0].n_max == 3


def test_region_has_k_branches():
    _skel, regions = _expand_structural_holes(_p3_sketch())
    assert len(regions[0].candidate_branches) == 3


def test_no_structural_hole_yields_no_regions():
    plain = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    _skel, regions = _expand_structural_holes(plain)
    assert regions == []
