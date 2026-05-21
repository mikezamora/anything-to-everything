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
    BinderHandle, IntLiteralOutOfRange,
)
from ._serialize import NodeOccupancy


def _basis_index(kind_idx: int, type_idx: int, bid_idx: int,
                 value_idx: int) -> int:
    """Linear index of (kind, type, bid, value) in the local 8192-dim basis."""
    return (((kind_idx * TYPE_CUTOFF + type_idx) * BID_CUTOFF + bid_idx)
            * VALUE_CUTOFF + value_idx)


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
    # VAR / LAM / APP / IF: value is NONE.
    return (kind, type_tag, VALUE_NONE)


def _local_bid_for_kind(kind: int, occ: NodeOccupancy) -> int:
    """Return the local bid register value for this site.

    LAM: BID_0 (the binder is innermost from its own perspective).
    VAR: BID_k where k = depth_from_innermost at the use site.
    Everything else: BID_NONE.
    """
    if kind == KIND_LAM:
        return BID_0   # "I am introducing a new binder; it is my innermost"
    if kind == KIND_VAR:
        depth = occ.var_ref.depth_from_innermost
        # BID_0 .. BID_6 occupy indices 1..7. BID_k = k + 1.
        return BID_0 + depth
    return BID_NONE


def _bid_bond_tensor_at_site(
    site_idx: int,
    occ: NodeOccupancy,
    local_bid_value: int,
    left_live: list[BinderHandle],
    right_live: list[BinderHandle],
) -> np.ndarray:
    """Construct the bid-register-only sub-tensor for one site.

    Shape: (|left_live| + 1, BID_CUTOFF, |right_live| + 1).

    Channel ordering at each bond:
      - index 0: "no_info" channel
      - index 1: first live binder
      - index 2: second live binder
      - ...

    Tensor entries (all complex, real-valued):

      For a PAD or non-binder, non-var site (just passing channels through):
        T[ch, BID_NONE, ch'] = 1  iff ch and ch' refer to the same binder
                                  or both are no_info
        T[*, BID_NONE, *] = 0    elsewhere
        T[*, BID_!= NONE, *] = 0  (site has no local bid value)

      For a LAM site introducing binder b which is at right channel c_b:
        - The new binder enters via the "no_info" left channel and goes
          out on its own channel.
        - Pre-existing binders pass through (channel preserved).
        T[ch, BID_0, ch']
            = 1 if ch == no_info and ch' == c_b (new binder created)
            = 1 if ch != no_info and ch' is the same binder (passthrough)
        all other T entries = 0

      For a VAR site referring to binder b at left channel c_b:
        - If this is the binder's LAST use, the channel is removed from
          the right bond (right_live drops it).
        - If it's not the last use, the channel passes through.
        T[c_b, BID_local, ch']
            = 1 if ch' is the same binder b passed through, OR
            = 1 if ch' = no_info and b is no longer in right_live
        Other channels pass through normally:
        T[ch, BID_NONE, ch']
            = 1 if ch and ch' refer to the same binder, where ch != c_b
            = 1 if ch == no_info and ch' == no_info
    """
    L_in = len(left_live)
    L_out = len(right_live)
    T = np.zeros((L_in + 1, BID_CUTOFF, L_out + 1), dtype=complex)

    # Pre-compute the mapping from binder handle to channel index on each
    # side. Channel 0 is no_info, 1..L is binders in order.
    left_ch = {bh: i + 1 for i, bh in enumerate(left_live)}
    right_ch = {bh: i + 1 for i, bh in enumerate(right_live)}
    NO_INFO_IN = 0
    NO_INFO_OUT = 0

    # All sites have "no_info passthrough" on the no_info channel by default.
    if occ.kind != KIND_VAR:
        T[NO_INFO_IN, BID_NONE, NO_INFO_OUT] = 1.0

    # Common channel passthrough for binders that survive across this bond.
    # For LAM sites, the new binder goes from no_info_in to its channel_out
    # under BID_0 — but other binders still pass through under BID_NONE.
    # For VAR sites, the referenced binder's channel may be consumed.
    consumed_binder: BinderHandle | None = None
    if occ.kind == KIND_VAR:
        # Identify the binder being referenced. We can recover it from the
        # left_live channels by matching against the var_ref's binder_site.
        target_lam_site = occ.var_ref.binder_site
        ref_handle = None
        for bh in left_live:
            if bh.lam_site == target_lam_site:
                ref_handle = bh
                break
        if ref_handle is None:
            raise RuntimeError(
                f"VAR site {site_idx} references binder at lam_site="
                f"{target_lam_site} but it is not in left_live; "
                f"this is an encoder bug — channel bookkeeping is wrong."
            )
        # Is this the binder's last use? It's consumed iff the binder is
        # not in right_live.
        if ref_handle not in right_ch:
            consumed_binder = ref_handle
        c_in = left_ch[ref_handle]
        # Read the channel: BID_local is the local bid value at this site.
        if consumed_binder is not None:
            # The channel is dropped at this bond — emit to no_info_out.
            T[c_in, local_bid_value, NO_INFO_OUT] = 1.0
        else:
            # Pass the channel through.
            c_out = right_ch[ref_handle]
            T[c_in, local_bid_value, c_out] = 1.0

        # Other channels passthrough as identity under BID_NONE
        # (these binders are not being used at this site).
        for bh in left_live:
            if bh == ref_handle:
                continue
            c_in_other = left_ch[bh]
            # If still live on right, passthrough.
            if bh in right_ch:
                c_out_other = right_ch[bh]
                T[c_in_other, BID_NONE, c_out_other] = 1.0
            # Otherwise the binder is being dropped at this bond despite
            # not being referenced here — that shouldn't happen if our
            # liveness analysis is correct (last_use_site logic).
            else:
                # Defensive: drop to no_info_out under BID_NONE.
                T[c_in_other, BID_NONE, NO_INFO_OUT] = 1.0
        return T

    if occ.kind == KIND_LAM:
        # The new binder is the rightmost entry of right_live (declaration
        # order). Find its channel.
        new_binder = None
        for bh in right_live:
            if bh.lam_site == site_idx:
                new_binder = bh
                break
        if new_binder is None:
            raise RuntimeError(
                f"LAM site {site_idx}: no matching binder in right_live; "
                f"this is an encoder bug — channels misaligned"
            )
        c_new_out = right_ch[new_binder]
        # The new binder enters from no_info_in with the LAM's local bid (BID_0).
        T[NO_INFO_IN, local_bid_value, c_new_out] = 1.0
        # Existing binders pass through under BID_NONE.
        for bh in left_live:
            c_in = left_ch[bh]
            if bh in right_ch:
                c_out = right_ch[bh]
                T[c_in, BID_NONE, c_out] = 1.0
            else:
                # Shouldn't happen at a LAM site (LAM doesn't drop binders).
                T[c_in, BID_NONE, NO_INFO_OUT] = 1.0
        # Remove the no_info passthrough we added at the top, since at a
        # LAM site the no_info channel is being consumed by the new binder
        # creation — it does NOT also passthrough.
        T[NO_INFO_IN, BID_NONE, NO_INFO_OUT] = 0.0
        return T

    # PAD / APP / IF / INT / BOOL / BIN: pure passthrough.
    # All left channels go to the matching right channel under BID_NONE.
    for bh in left_live:
        c_in = left_ch[bh]
        if bh in right_ch:
            c_out = right_ch[bh]
            T[c_in, BID_NONE, c_out] = 1.0
        else:
            # Binder dropping at a non-VAR site shouldn't happen.
            T[c_in, BID_NONE, NO_INFO_OUT] = 1.0
    return T


def _combine_factored_site(
    kind_idx: int, type_idx: int, value_idx: int,
    bid_tensor: np.ndarray
) -> np.ndarray:
    """Combine the (kind, type, value) deterministic indices and the
    (chi_l, BID, chi_r) bid sub-tensor into the full
    (chi_l, D_LOCAL, chi_r) site tensor.

    The kind/type/value registers are *product*: one basis index has full
    amplitude. So the output tensor is nonzero only on the slice
    flat_basis_index = base + bid_idx * VALUE_CUTOFF for varying bid_idx.

    Specifically, T[ch_in, base + b * V, ch_out] = bid_tensor[ch_in, b, ch_out]
    for each bid index b in [0, BID_CUTOFF).
    """
    chi_l, B, chi_r = bid_tensor.shape
    assert B == BID_CUTOFF
    out = np.zeros((chi_l, D_LOCAL, chi_r), dtype=complex)
    # Compute the flat index for (kind_idx, type_idx, *, value_idx) for each
    # possible bid index.
    base_no_bid = kind_idx * (TYPE_CUTOFF * BID_CUTOFF * VALUE_CUTOFF) \
                  + type_idx * (BID_CUTOFF * VALUE_CUTOFF)
    for b in range(BID_CUTOFF):
        flat = base_no_bid + b * VALUE_CUTOFF + value_idx
        out[:, flat, :] = bid_tensor[:, b, :]
    return out


def build_site_tensors(
    sites: list[NodeOccupancy],
    type_tags: list[int],
    live_binders_per_bond: list[list[BinderHandle]],
) -> list[np.ndarray]:
    """Top-level builder. Produces the list of N rank-3 site tensors.

    Bond i has dimension |live_binders_per_bond[i]| + 1 (one channel per
    live binder + no_info). The full site tensor has shape
    (chi_left, D_LOCAL, chi_right) where chi values come from adjacent
    bond live-binder counts. Boundary bonds (left of site 0, right of
    site N-1) are dimension 1.

    Returns a list of complex numpy arrays.
    """
    N = len(sites)
    # Boundary-aware live lists: prepend [] for left boundary, append []
    # for right boundary so live_at_bond[i] = live across bond between
    # sites i-1 and i.
    bounded_live = [[]] + list(live_binders_per_bond) + [[]]
    # bounded_live has length N + 1. left_live[k] = bounded_live[k];
    # right_live[k] = bounded_live[k + 1].

    tensors: list[np.ndarray] = []
    for k, occ in enumerate(sites):
        left_live = bounded_live[k]
        right_live = bounded_live[k + 1]
        type_tag = type_tags[k]
        kind_idx, type_idx, value_idx = _local_kind_type_value(occ, type_tag)
        local_bid = _local_bid_for_kind(occ.kind, occ)
        bid_T = _bid_bond_tensor_at_site(
            site_idx=k, occ=occ, local_bid_value=local_bid,
            left_live=left_live, right_live=right_live,
        )
        site_T = _combine_factored_site(
            kind_idx=kind_idx, type_idx=type_idx, value_idx=value_idx,
            bid_tensor=bid_T,
        )
        tensors.append(site_T)
    return tensors
