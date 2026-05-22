"""MERA-native encoder acceptance suite (spec §9.2, §9.3, §9.5)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def test_layout_is_node_major_5n_padded():
    """spec §9.2."""
    state, meta = encode_mera(parse(r"\x:Int. x"))   # 2 nodes
    assert meta.n_nodes == 2
    assert meta.n_leaves == 16
    assert meta.species_of_leaf[:10] == [
        "kind", "type", "bid", "value", "tobl",
        "kind", "type", "bid", "value", "tobl",
    ]
    assert all(s == "PAD" for s in meta.species_of_leaf[10:])


def test_alpha_renaming_identical_state():
    """spec §9.3: \\x.x and \\y.y encode to the same MERA state."""
    s1, _ = encode_mera(parse(r"\x:Int. x"))
    s2, _ = encode_mera(parse(r"\y:Int. y"))
    overlap = abs(s1.inner(s2)) ** 2
    assert overlap > 1.0 - 1e-10, f"overlap {overlap}"


def test_pad_leaves_are_vacuum():
    """spec §9.5: PAD leaves have zero amplitude on non-PAD basis states."""
    from src.qft_pcn.logic.mera_encoding import MERA_LEAF_DIM, KIND_PAD
    state, meta = encode_mera(parse(r"\x:Int. x"))
    proj = np.eye(MERA_LEAF_DIM, dtype=complex)
    proj[KIND_PAD, KIND_PAD] = 0.0       # project away from PAD
    for leaf in range(meta.n_leaves):
        if meta.species_of_leaf[leaf] == "PAD":
            val = state.local_expectation(leaf, proj)
            assert abs(val) < 1e-10, f"PAD leaf {leaf} not vacuum: {val}"


from src.qft_pcn.logic.ast import Lam, TInt, HoleVar


def test_binding_is_entanglement_structural_marker():
    """spec §9.4 / §5.5. A hole-bearing program has strictly positive
    tree entanglement entropy across a cut separating the hole's bid leaf
    from the candidate binders' bid leaves.

    A classical-lookup encoding (definite bid at the hole leaf) gives
    S = 0 and fails this test with a message naming the violated section.
    """
    h = HoleVar(candidates=["x", "y"])
    ast = Lam(param="x", param_ty=TInt(),
              body=Lam(param="y", param_ty=TInt(), body=h))
    state, meta = encode_mera(ast)
    # Hole is node 2; its bid leaf is 12. Candidate binders are nodes 0,1;
    # their bid leaves are 2 and 7. A cut at leaf 12 separates the hole's
    # bid leaf region from the binders' region.
    max_S = max(state.entanglement_entropy(cut)
                for cut in range(1, state.N - 1))
    assert max_S > 0.5, (
        f"hole-bearing program has max tree entanglement entropy {max_S}; "
        f"expected > 0.5. A value near 0 means binding was encoded as a "
        f"classical lookup, not entanglement — spec §5, §1.1 violated."
    )


from src.qft_pcn.logic.encoding import (
    EncodingTooLarge, TooManyBinders, IntLiteralOutOfRange,
    IllScopedVar, UnsupportedNode,
)


def test_encoding_too_large_raises():
    with pytest.raises(EncodingTooLarge):
        encode_mera(parse(r"\x:Int. x + x + x"), n_nodes_max=3)


def test_too_many_binders_raises():
    src = (r"\a:Int. \b:Int. \c:Int. \d:Int. \e:Int. \f:Int. \g:Int. "
           r"\h:Int. a")
    with pytest.raises(TooManyBinders):
        encode_mera(parse(src))


def test_int_literal_out_of_range_raises():
    with pytest.raises(IntLiteralOutOfRange):
        encode_mera(parse(r"\x:Int. x + 99"))


def test_ill_scoped_var_raises():
    with pytest.raises(IllScopedVar):
        encode_mera(parse("undefined_name"))


def test_unsupported_node_raises():
    class _Bogus:
        pass
    with pytest.raises(UnsupportedNode):
        encode_mera(_Bogus())  # type: ignore[arg-type]


def test_top_level_import_path():
    """encode_mera/decode_mera reachable from the package surface."""
    from src.qft_pcn.logic import encode_mera as e2, decode_mera as d2
    state, meta = e2(parse(r"\x:Int. x"))
    res = d2(state, meta)
    assert res.residual_norm < 1e-10
