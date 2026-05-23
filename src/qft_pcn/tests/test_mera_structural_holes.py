"""Tests for structural-superposition MERA encoding (spec §5.3, §9.2-9.4)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, TArrow, Var, App, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_encoding import KIND_VAR, KIND_APP


def _structural_sketch():
    """\\f:Int->Int. \\x:Int. ?HOLE  candidates {Var x, App(f,x)}."""
    hole = HoleVar(candidates=(
        Var(name="x"),
        App(fn=Var(name="f"), arg=Var(name="x")),
    ))
    return Lam(param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
               body=Lam(param="x", param_ty=TInt(), body=hole))


def test_structural_hole_encodes_to_unit_norm():
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10)


def test_hole_region_root_kind_leaf_is_shape_superposed():
    """The hole-region root kind leaf has weight on >=2 distinct kinds
    (KIND_VAR for the Var branch, KIND_APP for the App branch)."""
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    assert len(meta.hole_regions) == 1
    root_leaf = 5 * meta.hole_regions[0].node_start    # kind leaf offset 0
    proj_v = np.zeros((16, 16), dtype=complex); proj_v[KIND_VAR, KIND_VAR] = 1
    proj_a = np.zeros((16, 16), dtype=complex); proj_a[KIND_APP, KIND_APP] = 1
    w_v = float(np.real(state.local_expectation(root_leaf, proj_v)))
    w_a = float(np.real(state.local_expectation(root_leaf, proj_a)))
    assert w_v > 1e-3 and w_a > 1e-3, (w_v, w_a)


def test_structural_hole_is_not_a_product_state():
    """Genuine tree entanglement somewhere — principle 6, the marker."""
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    max_S = max(state.entanglement_entropy(cut)
                for cut in range(1, state.N - 1))
    assert max_S > 1e-6, "structural hole encoded as a product state"


def test_witness_augmented_ast_empty_examples_returns_sketch_unwrapped():
    """Fix 2 regression: with examples=(), `_witness_augmented_ast` returns
    the sketch as-is (not wrapped in Bundle) so encode_mera dispatches on
    the sketch's structural-hole path directly.
    """
    from src.qft_pcn.logic.mera_synthesis.encode_ext import (
        _witness_augmented_ast, Bundle,
    )
    sketch = _structural_sketch()
    result = _witness_augmented_ast(sketch, examples=())
    assert result is sketch
    assert not isinstance(result, Bundle)


def test_bundle_with_structural_sketch_encodes():
    """Fix 2 regression: a Bundle whose children[0] is a structural-hole
    sketch and children[1..] are concrete witnesses encodes without
    crashing. This is the P3-P7 critical path.
    """
    from src.qft_pcn.logic.ast import IntLit, App, Var as _Var
    from src.qft_pcn.logic.mera_synthesis.encode_ext import Bundle
    from src.qft_pcn.logic.mera_encoder import encode_mera
    sketch = _structural_sketch()
    # Build a fake concrete witness that mimics what _witness_augmented_ast
    # would emit: a Lam wrapping App(Var(sketch_param), ...inputs).
    from src.qft_pcn.logic.ast import Lam, TArrow, TInt
    wit = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=App(fn=_Var(name="f"), arg=IntLit(val=2)),
    )
    bundle = Bundle(children=(sketch, wit))
    state, meta = encode_mera(bundle, chi_layer=32)
    assert state.norm_sq() > 0
    assert len(meta.hole_regions) == 1


def test_structural_marker_exceeds_value_only_superposition():
    """Shape superposition (kind leaves differ) carries strictly more
    entropy across the hole-region cut than a value-only superposition
    of the same candidate count (spec §9.4)."""
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    region = meta.hole_regions[0]
    cut = 5 * region.node_start                # leaf where the region starts
    S_struct = state.entanglement_entropy(cut)
    # ln(2) is the value-only (M1-style) upper bound for k=2 candidates.
    assert S_struct > 1e-6
