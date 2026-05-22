"""Site-tensor construction for the AST <-> MPS encoder.

The per-site MPS tensor has shape (chi_left, d_local, chi_right) with
d_local = KIND_CUTOFF * TYPE_CUTOFF * BID_CUTOFF * VALUE_CUTOFF = 8192.

This is the architectural-soul module: the bid register's bond carries
the live-binder channels. Bond dim on bid = |L_i| + 1 (one channel per
live binder + a "no_info" channel). The kind/type/value registers are
product on every bond (bond dim 1 on those registers).

The full bond dim equals the product over registers of register-bond-dims.
Since kind/type/value contribute 1 each, the full bond dim equals the
bid bond dim.

Basis ordering inside the local 8192-dim register follows embed_op in
fock.py: leftmost species (kind) changes slowest. The flat index of
(s_kind, s_type, s_bid, s_value) is:

    s_kind * (T * B * V) + s_type * (B * V) + s_bid * V + s_value

where T = TYPE_CUTOFF, B = BID_CUTOFF, V = VALUE_CUTOFF.

DO NOT short-circuit the channel mechanism with a classical bid lookup at
Var sites — that is the spec §1.1 shortcut, and the bond-dim test in
test_logic_tensors.py will fail with a message naming the violation.
"""

from __future__ import annotations

import numpy as np

from .encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN, KIND_CUTOFF,
    TYPE_NONE, TYPE_INT, TYPE_BOOL, TYPE_CUTOFF,
    BID_NONE, BID_0, BID_CUTOFF,
    VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
    BIN_VALUE_FROM_OP, INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX,
    VALUE_CUTOFF, D_LOCAL,
    TOBL_CUTOFF, TOBL_NONE,
    BinderHandle, IntLiteralOutOfRange,
)
from ._serialize import NodeOccupancy


def _basis_index(kind_idx: int, type_idx: int, bid_idx: int,
                 value_idx: int, tobl_idx: int) -> int:
    """Linear index of (kind, type, bid, value, tobl) in the local
    65536-dim basis. Leftmost species changes slowest.
    """
    return ((((kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF + bid_idx)
             * VALUE_CUTOFF + value_idx) * TOBL_CUTOFF + tobl_idx)


def _local_kind_type_value(occ: NodeOccupancy, type_tag: int
                           ) -> tuple[int, int, int]:
    """Return (kind_idx, type_idx, value_idx) for this site.

    The bid_idx is computed separately because it depends on the bond
    structure (it's read from / written to a channel).
    """
    kind = occ.kind
    if kind == KIND_PAD:
        return (KIND_PAD, TYPE_NONE, VALUE_NONE)
    if kind == KIND_INT:
        if occ.int_val is None:
            raise ValueError("KIND_INT site missing int_val")
        if not INT_LIT_MIN <= occ.int_val <= INT_LIT_MAX:
            raise IntLiteralOutOfRange(n=occ.int_val)
        return (KIND_INT, type_tag, occ.int_val + INT_LIT_OFFSET)
    if kind == KIND_BOOL:
        return (KIND_BOOL, type_tag,
                VALUE_TRUE if occ.bool_val else VALUE_FALSE)
    if kind == KIND_BIN:
        return (KIND_BIN, type_tag, BIN_VALUE_FROM_OP[occ.bin_op])
    from .mera_encoding import KIND_NATLIT
    if kind == KIND_NATLIT:
        # A Peano-natural literal stores its value directly (no offset:
        # Peano nats are >= 0, within the 16-slot value register).
        if occ.int_val is None:
            raise ValueError("KIND_NATLIT site missing int_val")
        if not 0 <= occ.int_val < VALUE_CUTOFF:
            raise IntLiteralOutOfRange(n=occ.int_val)
        return (kind, type_tag, occ.int_val)
    # VAR / LAM / APP / IF and the structural extended nodes
    # (Zero / Succ / Nil / Cons / Eq): value is NONE.
    return (kind, type_tag, VALUE_NONE)


def _local_bid_for_kind(kind: int, occ: NodeOccupancy) -> int:
    """Return the local bid register value for this site.

    LAM: BID_0 (the binder is innermost from its own perspective).
    VAR: BID_k where k = depth_from_innermost at the use site.
    Everything else: BID_NONE.
    """
    from .mera_encoding import KIND_FORALL, KIND_FIX
    if kind in (KIND_LAM, KIND_FORALL, KIND_FIX):
        return BID_0   # "I am introducing a new binder; it is my innermost"
    if kind == KIND_VAR:
        from .encoding import TooManyBinders, MAX_BINDER_DEPTH
        depth = occ.var_ref.depth_from_innermost
        if depth >= MAX_BINDER_DEPTH:
            raise TooManyBinders(depth=depth + 1, cutoff=MAX_BINDER_DEPTH)
        # BID_0 .. BID_6 occupy indices 1..7. BID_k = k + 1.
        return BID_0 + depth
    return BID_NONE


def _bid_channel_slot(
    channel_index_1_based: int,   # 1-based (1..L)
    param_ty_tag: int,            # 0..TOBL_CUTOFF-1
) -> int:
    """Bond bid-register slot index for a binder's (channel, param_ty) pair.

    Channel indexing:
      slot 0          : no_info
      slot 1 + 8*(i-1) + t : channel i (1-based) with param_ty tag t.
    """
    assert 1 <= channel_index_1_based, (
        f"channel index must be >= 1, got {channel_index_1_based}"
    )
    assert 0 <= param_ty_tag < TOBL_CUTOFF, (
        f"param_ty_tag out of range: {param_ty_tag}"
    )
    return 1 + 8 * (channel_index_1_based - 1) + param_ty_tag


def _bid_bond_dim(L: int) -> int:
    """Bond dim on the bid register = 1 + 8*L (one no_info slot + 8 slots per
    live binder)."""
    return 1 + 8 * L


def _bid_bond_tensor_at_site(
    site_idx: int,
    occ: NodeOccupancy,
    local_bid_value: int,
    left_live: list[BinderHandle],
    right_live: list[BinderHandle],
    left_param_ty: list[int],     # per-channel param_ty for left bond
    right_param_ty: list[int],    # per-channel param_ty for right bond
) -> np.ndarray:
    """Construct the bid-register-only sub-tensor for one site.

    Shape: (1 + 8*L_in, BID_CUTOFF, 1 + 8*L_out).

    Per spec §5.2 of B (and §5.4 of A), each binder channel carries an
    extra param_ty tag. The bond's bid-register basis is:
        index 0:        |no_info>
        index 1 + 8(i-1) + t : |channel_i, param_ty=t> for i=1..L, t=0..7

    For each binder b on a bond, ONLY the slot with t = b.param_ty is nonzero;
    the other 7 slots in b's orbit are zero.
    """
    L_in = len(left_live)
    L_out = len(right_live)
    bond_dim_in = _bid_bond_dim(L_in)
    bond_dim_out = _bid_bond_dim(L_out)
    T = np.zeros((bond_dim_in, BID_CUTOFF, bond_dim_out), dtype=complex)

    NO_INFO_IN = 0
    NO_INFO_OUT = 0

    # Map binder handle -> (channel_index_1_based, param_ty_tag).
    left_ch_map: dict[BinderHandle, tuple[int, int]] = {}
    for i, bh in enumerate(left_live):
        left_ch_map[bh] = (i + 1, left_param_ty[i])
    right_ch_map: dict[BinderHandle, tuple[int, int]] = {}
    for i, bh in enumerate(right_live):
        right_ch_map[bh] = (i + 1, right_param_ty[i])

    def L_slot(bh: BinderHandle) -> int:
        c, t = left_ch_map[bh]
        return _bid_channel_slot(c, t)

    def R_slot(bh: BinderHandle) -> int:
        c, t = right_ch_map[bh]
        return _bid_channel_slot(c, t)

    # --- Non-binder, non-var, non-PAD passthrough sites (APP / IF / BIN / INT / BOOL / PAD) ---
    if occ.kind not in (KIND_VAR, KIND_LAM):
        # local bid = BID_NONE. no_info passes through; each live binder
        # passes through under its assigned slot.
        T[NO_INFO_IN, BID_NONE, NO_INFO_OUT] = 1.0
        for bh in left_live:
            if bh in right_ch_map:
                T[L_slot(bh), BID_NONE, R_slot(bh)] = 1.0
            else:
                # Shouldn't happen — non-Var sites don't drop binders.
                T[L_slot(bh), BID_NONE, NO_INFO_OUT] = 1.0
        return T

    if occ.kind == KIND_LAM:
        # The new binder is the rightmost entry in right_live (declaration order).
        new_binder = None
        for bh in right_live:
            if bh.lam_site == site_idx:
                new_binder = bh
                break
        if new_binder is None:
            raise RuntimeError(
                f"LAM site {site_idx}: no matching binder in right_live; "
                f"encoder bug — channels misaligned"
            )
        c_new_out_slot = R_slot(new_binder)
        # At a LAM the local bid is BID_0 (its own innermost-binder index).
        # Existing binders pass through under BID_0; the new binder is created
        # by routing no_info_in -> c_new_out_slot.
        T[NO_INFO_IN, local_bid_value, NO_INFO_OUT] = 1.0   # passthrough no_info
        T[NO_INFO_IN, local_bid_value, c_new_out_slot] = 1.0  # create new binder
        for bh in left_live:
            if bh in right_ch_map:
                T[L_slot(bh), local_bid_value, R_slot(bh)] = 1.0
            else:
                T[L_slot(bh), local_bid_value, NO_INFO_OUT] = 1.0
        return T

    # --- VAR site ---
    # The referenced binder's channel is consumed (if last use) or passed through.
    assert occ.var_ref is not None
    cands = (occ.var_ref.candidates
             if occ.var_ref.candidates
             else [(occ.var_ref.binder_site,
                    occ.var_ref.depth_from_innermost)])
    amp = 1.0 / np.sqrt(len(cands))

    from .encoding import MAX_BINDER_DEPTH, TooManyBinders
    ref_handles_used: set[BinderHandle] = set()
    for cand_lam_site, cand_depth in cands:
        ref_handle = None
        for bh in left_live:
            if bh.lam_site == cand_lam_site:
                ref_handle = bh
                break
        if ref_handle is None:
            raise RuntimeError(
                f"VAR site {site_idx}: candidate binder at lam_site="
                f"{cand_lam_site} not in left_live (bookkeeping bug)"
            )
        ref_handles_used.add(ref_handle)
        if cand_depth >= MAX_BINDER_DEPTH:
            raise TooManyBinders(depth=cand_depth + 1,
                                 cutoff=MAX_BINDER_DEPTH)
        local_bid = BID_0 + cand_depth
        c_in = L_slot(ref_handle)
        if ref_handle not in right_ch_map:
            T[c_in, local_bid, NO_INFO_OUT] = amp
        else:
            T[c_in, local_bid, R_slot(ref_handle)] = amp

    # Pass-through for binders NOT referenced by this use, located on the
    # primary candidate's bid slice.
    primary_depth = cands[0][1]
    primary_local_bid = BID_0 + primary_depth
    T[NO_INFO_IN, primary_local_bid, NO_INFO_OUT] = 1.0
    for bh in left_live:
        if bh in ref_handles_used:
            continue
        c_in_other = L_slot(bh)
        if bh in right_ch_map:
            T[c_in_other, primary_local_bid, R_slot(bh)] = 1.0
        else:
            T[c_in_other, primary_local_bid, NO_INFO_OUT] = 1.0
    return T


def _combine_factored_site(
    kind_idx: int, type_idx: int, value_idx: int, tobl_idx: int,
    bid_tensor: np.ndarray
) -> np.ndarray:
    """Combine the (kind, type, value, tobl) deterministic indices and the
    (chi_l, BID, chi_r) bid sub-tensor into the full
    (chi_l, D_LOCAL, chi_r) site tensor.

    The kind/type/value/tobl registers are *product*: one basis index has
    full amplitude. So the output tensor is nonzero only on the slice
    flat_basis_index = base + bid_idx * (VALUE_CUTOFF * TOBL_CUTOFF) + ...
    for varying bid_idx.

    Specifically, for each bid index b in [0, BID_CUTOFF):
        flat = ((kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF + b)
                * VALUE_CUTOFF * TOBL_CUTOFF
              + value_idx * TOBL_CUTOFF
              + tobl_idx
        out[:, flat, :] = bid_tensor[:, b, :]
    """
    chi_l, B, chi_r = bid_tensor.shape
    assert B == BID_CUTOFF
    out = np.zeros((chi_l, D_LOCAL, chi_r), dtype=complex)
    inner = value_idx * TOBL_CUTOFF + tobl_idx
    bid_stride = VALUE_CUTOFF * TOBL_CUTOFF
    base_no_bid = (kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF * bid_stride
    for b in range(BID_CUTOFF):
        flat = base_no_bid + b * bid_stride + inner
        out[:, flat, :] = bid_tensor[:, b, :]
    return out


def build_site_tensors(
    sites: list[NodeOccupancy],
    type_tags: list[int],
    live_binders_per_bond: list[list[BinderHandle]],
) -> list[np.ndarray]:
    """Top-level builder. Produces the list of N rank-3 site tensors.

    Bond i has dimension 1 + 8 * |live_binders_per_bond[i]| (one no_info
    slot plus 8 slots per live binder, encoding the binder's param_ty as
    a real bond DOF, per B's spec §5.2). Boundary bonds (left of site 0,
    right of site N-1) are dimension 1.

    Returns a list of complex numpy arrays.
    """
    from ._channels import compute_channel_param_ty_per_bond
    pt_per_bond = compute_channel_param_ty_per_bond(sites, live_binders_per_bond)

    N = len(sites)
    bounded_live = [[]] + list(live_binders_per_bond) + [[]]
    bounded_pt = [[]] + pt_per_bond + [[]]

    tensors: list[np.ndarray] = []
    for k, occ in enumerate(sites):
        left_live = bounded_live[k]
        right_live = bounded_live[k + 1]
        left_pt = bounded_pt[k]
        right_pt = bounded_pt[k + 1]
        type_tag = type_tags[k]
        kind_idx, type_idx, value_idx = _local_kind_type_value(occ, type_tag)
        local_bid = _local_bid_for_kind(occ.kind, occ)
        bid_T = _bid_bond_tensor_at_site(
            site_idx=k, occ=occ, local_bid_value=local_bid,
            left_live=left_live, right_live=right_live,
            left_param_ty=left_pt, right_param_ty=right_pt,
        )
        tobl_idx = occ.tobl_tag
        site_T = _combine_factored_site(
            kind_idx=kind_idx, type_idx=type_idx, value_idx=value_idx,
            tobl_idx=tobl_idx, bid_tensor=bid_T,
        )
        tensors.append(site_T)
    return tensors
