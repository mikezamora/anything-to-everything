"""Per-leaf basis vectors for one AST node (spec §6.2 step 4).

Reuses the MPS encoder's per-species index computation so the MERA and
MPS encoders agree on what each species register holds. Concrete
(hole-free) nodes only; HoleVar superpositions are handled in Part 2.
"""
from __future__ import annotations

import numpy as np

from ._serialize import NodeOccupancy
from ._tensors import _local_kind_type_value, _local_bid_for_kind
from .mera_encoding import (
    MERA_LEAF_DIM, KIND_NIL, KIND_CONS, KIND_FORALL, KIND_FIX,
)
from ._types import ty_to_tag


def nested_binder_ty(occ: NodeOccupancy):
    """For Forall / Fix / Nil / Cons sites, return the *full* ``param_ty``
    (for binders) or ``elem`` (for Nil/Cons) when it is non-flat — i.e.
    when ``ty_to_tag`` would collapse it to a single tag and lose
    structural information.

    The encoder records the returned Ty in ``nested_type_index[site_idx]``
    so the decoder can reconstruct, e.g., ``forall xs:List Bool`` rather
    than defaulting to ``TList(elem=TNat())``. Returns ``None`` for flat
    types (TNat / TInt / TBool / TProp / TList(elem=TNat()) / etc.) where
    the leaf encoding alone suffices.
    """
    from .ast import TList as _TList, TArrow as _TArrow, TNat as _TNat
    if occ.kind in (KIND_FORALL, KIND_FIX):
        if occ.binder_ref is None:
            return None
        pty = occ.binder_ref.lam_node.param_ty
    elif occ.kind in (KIND_NIL, KIND_CONS):
        pty = getattr(occ.ty, "elem", None)
    else:
        return None
    if pty is None:
        return None
    # A nested TArrow always needs the side table (matches the existing
    # TYPE_ARR_NESTED mechanism).
    if isinstance(pty, _TArrow):
        return pty
    # A TList whose elem is anything other than TNat (the legacy default
    # baked into ``_extended_type_from_tag``) needs the side table.
    if isinstance(pty, _TList) and not isinstance(pty.elem, _TNat):
        return pty
    return None


def _one_hot(index: int) -> np.ndarray:
    v = np.zeros(MERA_LEAF_DIM, dtype=complex)
    v[index] = 1.0
    return v


def node_leaf_vectors(occ: NodeOccupancy, type_tag: int) -> list[np.ndarray]:
    """The five 16-dim one-hot leaf vectors for one concrete AST node.

    Order: [kind, type, bid, value, tobl].

    The kind/type/value indices come from the MPS encoder's
    ``_local_kind_type_value`` and the bid index from ``_local_bid_for_kind``
    so the MERA and MPS substrates write identical species registers
    (cross-substrate anchor, spec §9.7). The tobl index is the site's
    ``tobl_tag`` attribute, written by ``compute_tobl_tags``; it defaults to
    TOBL_NONE (0) when obligations have not been computed.
    """
    kind_idx, type_idx, value_idx = _local_kind_type_value(occ, type_tag)
    bid_idx = _local_bid_for_kind(occ.kind, occ)
    tobl_idx = occ.tobl_tag

    # Extended-calculus value-leaf contract (spec §6.4-§6.8):
    #   Nil / Cons : the `value` leaf carries the list element-type tag.
    #   Forall / Fix : the `value` leaf carries the binder parameter-type tag.
    # M2's typing rules (T-Cons, T-Fix, T-Var-against-Forall) read these.
    if occ.kind in (KIND_NIL, KIND_CONS):
        # occ.ty is a TList; its `elem` field is the element type.
        elem_ty = getattr(occ.ty, "elem", None)
        if elem_ty is not None:
            value_idx, _ = ty_to_tag(elem_ty)
    elif occ.kind in (KIND_FORALL, KIND_FIX):
        if occ.binder_ref is not None:
            param_ty = occ.binder_ref.lam_node.param_ty
            value_idx, _ = ty_to_tag(param_ty)
    for name, idx in (("kind", kind_idx), ("type", type_idx),
                      ("bid", bid_idx), ("value", value_idx),
                      ("tobl", tobl_idx)):
        if not 0 <= idx < MERA_LEAF_DIM:
            raise ValueError(
                f"{name} index {idx} out of leaf range [0, {MERA_LEAF_DIM})")
    return [_one_hot(kind_idx), _one_hot(type_idx), _one_hot(bid_idx),
            _one_hot(value_idx), _one_hot(tobl_idx)]
