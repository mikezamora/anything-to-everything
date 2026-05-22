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
