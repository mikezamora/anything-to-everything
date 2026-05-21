from __future__ import annotations

import numpy as np

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._types import compute_site_types
from src.qft_pcn.logic._channels import compute_live_binders
from src.qft_pcn.logic._tensors import build_site_tensors
from src.qft_pcn.logic.encoding import (
    D_LOCAL, KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF,
    KIND_LAM, KIND_VAR, BID_0, BID_NONE,
)


def _basis_index(kind_idx: int, type_idx: int, bid_idx: int,
                 value_idx: int) -> int:
    """Linear index of (kind, type, bid, value) in the local basis.

    Per embed_op convention in fock.py: leftmost species changes slowest.
    """
    return (((kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF + bid_idx)
            * VALUE_CUTOFF + value_idx)


def test_site_tensor_shapes():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    # All tensors are rank-3.
    assert len(tensors) == 4
    for t in tensors:
        assert t.ndim == 3
        assert t.shape[1] == D_LOCAL


def test_site_tensor_boundary_bonds_are_dim_1():
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    # Left boundary of site 0 and right boundary of site N-1 are dim 1.
    assert tensors[0].shape[0] == 1
    assert tensors[-1].shape[2] == 1


def test_p1_lam_site_has_growing_right_bond():
    """For \\x.x, bond 0 between LAM and VAR carries Lam_x as a channel.

    Bond dim = len(live[0]) + 1 = 1 + 1 = 2.
    """
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    # LAM site: left bond = 1 (boundary), right bond = 2.
    assert tensors[0].shape == (1, D_LOCAL, 2)
    # VAR site: left bond = 2, right bond = 1 (no live binders after).
    assert tensors[1].shape == (2, D_LOCAL, 1)
    # PAD sites: dim 1 on both sides.
    assert tensors[2].shape == (1, D_LOCAL, 1)
    assert tensors[3].shape == (1, D_LOCAL, 1)


def test_p1_lam_site_nonzero_amplitude_at_correct_basis_state():
    """The LAM site tensor at \\x.x should have amplitude 1 at the basis
    state (KIND_LAM, TYPE_ARR_II, BID_0, VALUE_NONE) on the "no-info-in,
    channel-x-out" bond direction.
    """
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    from src.qft_pcn.logic.encoding import KIND_LAM, TYPE_ARR_II, BID_0, VALUE_NONE
    idx = _basis_index(KIND_LAM, TYPE_ARR_II, BID_0, VALUE_NONE)
    T = tensors[0]  # shape (1, 8192, 2)
    # The bond goes from "no_info" channel (index 0) in to "Lam_x" channel
    # (index 1) out.
    assert abs(T[0, idx, 1] - 1.0) < 1e-12, (
        "LAM site tensor should activate the Lam_x channel on the right bond"
    )
    # All other entries at this local basis state should be zero.
    assert abs(T[0, idx, 0]) < 1e-12  # no-info-out: zero
    # Other local basis states should be zero too.
    other_idx = _basis_index(0, 0, 0, 0)  # PAD/none/none/none
    assert abs(T[0, other_idx, 0]) < 1e-12
    assert abs(T[0, other_idx, 1]) < 1e-12


def test_p1_var_site_reads_channel():
    """The VAR site tensor at \\x.x should read the Lam_x channel from its
    left bond and emit no_info on its right bond, with local bid = BID_0
    (Var(x)'s binder is innermost from its perspective)."""
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    from src.qft_pcn.logic.encoding import KIND_VAR, TYPE_INT, BID_0, VALUE_NONE
    idx = _basis_index(KIND_VAR, TYPE_INT, BID_0, VALUE_NONE)
    T = tensors[1]   # shape (2, 8192, 1)
    # Reading channel 1 (Lam_x) -> no_info out (index 0).
    assert abs(T[1, idx, 0] - 1.0) < 1e-12


def test_pad_site_is_vacuum_amplitude():
    """PAD sites have amplitude 1 at (KIND_PAD, TYPE_NONE, BID_NONE,
    VALUE_NONE) with no_info channels through, and 0 elsewhere."""
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    pad_idx = _basis_index(0, 0, 0, 0)
    for k in (2, 3):
        T = tensors[k]
        assert T.shape == (1, D_LOCAL, 1)
        assert abs(T[0, pad_idx, 0] - 1.0) < 1e-12
        # Some other index must be zero.
        from src.qft_pcn.logic.encoding import KIND_VAR
        other = _basis_index(KIND_VAR, 0, 0, 0)
        assert abs(T[0, other, 0]) < 1e-12


def test_unused_binder_does_not_crash():
    """Lam_y is introduced but never used; must not crash, and bond dim
    immediately after Lam_y must include Lam_y as a live channel."""
    ast = parse(r"\x:Int. \y:Int. x")
    sites = serialize_preorder(ast, N=8)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    # Layout: LAM_x@0, LAM_y@1, VAR_x@2, PAD@3, ...
    # Bond 0 (between LAM_x and LAM_y): only Lam_x live (1 channel + no_info = dim 2)
    assert tensors[0].shape[2] == 2
    # Bond 1 (between LAM_y and VAR_x): both Lam_x AND Lam_y live (2 channels + no_info = dim 3)
    assert tensors[1].shape[2] == 3
    # Bond 2 (between VAR_x and PAD): nothing live
    assert tensors[2].shape[2] == 1


def test_nested_binder_channel_mapping():
    """For \\x. \\y. x, Var x must read from channel 1 (Lam_x) under BID_1
    (depth-from-innermost = 1), and Var y would read from channel 2 (Lam_y)
    under BID_0 — exercises the depth->local_bid mapping for non-innermost
    binders."""
    from src.qft_pcn.logic.encoding import (
        KIND_VAR, TYPE_INT, BID_0, BID_1, VALUE_NONE,
    )
    ast = parse(r"\x:Int. \y:Int. x")
    sites = serialize_preorder(ast, N=8)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    # Var_x is at site 2. Its left_live should be [Lam_x, Lam_y].
    # Lam_x is channel 1 (declared first); Lam_y is channel 2.
    # Var_x reads channel 1 under BID_1 (depth=1 from innermost which is Lam_y).
    var_x = tensors[2]   # shape: (3, 8192, 1) — 2 binders left, both consumed (Lam_y is unused so it gets dropped too at this bond)
    idx = _basis_index(KIND_VAR, TYPE_INT, BID_1, VALUE_NONE)
    # The "consumed" path emits to NO_INFO_OUT (channel index 0 on the right).
    assert abs(var_x[1, idx, 0] - 1.0) < 1e-12, (
        f"VAR_x should read channel 1 (Lam_x) under BID_1, got "
        f"{var_x[1, idx, 0]}"
    )


def test_bond_dim_at_least_n_live_plus_one():
    """Spec §7.4 test (structural marker): for every bond, the total bond
    dim must be at least |L_i| + 1. This proves the principled channel
    construction was used; if a subagent collapsed bid to a classical
    field, the bond would be 1 even with binders crossing.
    """
    ast = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    sites = serialize_preorder(ast, N=32)
    types = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, types, live)
    for i in range(len(live)):
        # Right bond of site i has dimension = tensors[i].shape[2].
        n_live = len(live[i])
        assert tensors[i].shape[2] >= n_live + 1, (
            f"bond {i} right-bond dim {tensors[i].shape[2]} < "
            f"|L_i|+1 = {n_live + 1}; spec §1.1 violation"
        )
