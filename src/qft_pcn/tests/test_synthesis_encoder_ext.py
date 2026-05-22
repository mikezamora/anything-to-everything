"""Tests for synthesis/encode_ext.py: TypeHole superposition encoder."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    Var, Lam, App, IntLit, HoleVar, TypeHole, TInt, TBool,
)
from src.qft_pcn.logic.synthesis.encode_ext import (
    encode_synthesis, witness_augmented_sketch,
)
from src.qft_pcn.logic.synthesis.problem import IOExample


@pytest.mark.timeout(30)
def test_encode_synthesis_returns_state_and_meta():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=["x"]))
    state, meta = encode_synthesis(sketch, N=32, chi_max=32)
    assert state.N == 32
    assert meta.N == 32
    assert meta.chi_max == 32


@pytest.mark.timeout(30)
def test_encode_synthesis_records_type_hole():
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    state, meta = encode_synthesis(sketch, N=32, chi_max=32)
    assert len(meta.type_holes) == 1
    handle = next(iter(meta.type_holes.values()))
    # Lam-site type tag candidates: TArrow(TInt,TInt)=3 and TArrow(TBool,TBool)=6.
    assert sorted(handle.candidate_tags) == [3, 6]


@pytest.mark.timeout(60)
def test_encode_synthesis_typehole_produces_type_register_entropy():
    """The principled superposition test (analogous to A.§7.4)."""
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    state, meta = encode_synthesis(sketch, N=12, chi_max=32)
    # Bond between site 0 (Lam) and site 1 (Var) should carry entanglement
    # because the Lam's type-register superposition is correlated with the
    # body's type.
    S = state.entanglement_entropy(0)
    assert S > 0.3, (
        f"bond 0 entropy = {S}; type-register superposition not engaging"
    )


@pytest.mark.timeout(30)
def test_encode_synthesis_no_typehole_no_entry():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=["x"]))
    state, meta = encode_synthesis(sketch, N=32, chi_max=32)
    assert meta.type_holes == {}


@pytest.mark.timeout(30)
def test_encode_synthesis_norm_preserved():
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    state, meta = encode_synthesis(sketch, N=12, chi_max=32)
    assert abs(state.norm_sq() - 1.0) < 1e-6


@pytest.mark.timeout(20)
def test_witness_augmented_no_examples_returns_sketch_unchanged():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=["x"]))
    out, regions = witness_augmented_sketch(sketch, ())
    assert out is sketch
    assert regions == []


@pytest.mark.timeout(20)
def test_witness_augmented_single_example_returns_extended_ast():
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    ex = IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3))
    out, regions = witness_augmented_sketch(sketch, (ex,))
    # Augmented AST is bigger than the sketch.
    from src.qft_pcn.logic._serialize import serialize_preorder
    sketch_sites = sum(1 for s in serialize_preorder(sketch, N=32)
                       if s.kind != 0)
    aug_sites = sum(1 for s in serialize_preorder(out, N=32)
                    if s.kind != 0)
    assert aug_sites > sketch_sites


@pytest.mark.timeout(20)
def test_witness_augmented_multiple_examples():
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    examples = (
        IOExample(inputs=(IntLit(val=2),), output=IntLit(val=2)),
        IOExample(inputs=(IntLit(val=5),), output=IntLit(val=5)),
    )
    out, regions = witness_augmented_sketch(sketch, examples)
    assert out is not sketch


@pytest.mark.timeout(20)
def test_witness_augmented_reports_witness_regions():
    """Spec §5.5: witness_regions identifies each witness's site range.

    For a single-example, single-input witness ``App(sketch_copy, in_0)``,
    the witness occupies ``size(sketch)+2`` sites (the App parent + the
    input literal + the copied sketch). The regions list must be non-
    empty and each region must be a non-trivial site range.
    """
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    examples = (
        IOExample(inputs=(IntLit(val=2),), output=IntLit(val=2)),
        IOExample(inputs=(IntLit(val=5),), output=IntLit(val=5)),
    )
    out, regions = witness_augmented_sketch(sketch, examples)
    assert len(regions) == len(examples)
    for start, end in regions:
        assert 0 <= start < end
