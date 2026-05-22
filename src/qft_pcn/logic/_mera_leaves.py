"""Per-leaf basis vectors for one AST node (spec §6.2 step 4).

Reuses the MPS encoder's per-species index computation so the MERA and
MPS encoders agree on what each species register holds. Concrete
(hole-free) nodes only; HoleVar superpositions are handled in Part 2.
"""
from __future__ import annotations

import numpy as np

from ._serialize import NodeOccupancy
from ._tensors import _local_kind_type_value, _local_bid_for_kind
from .mera_encoding import MERA_LEAF_DIM


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
    for name, idx in (("kind", kind_idx), ("type", type_idx),
                      ("bid", bid_idx), ("value", value_idx),
                      ("tobl", tobl_idx)):
        if not 0 <= idx < MERA_LEAF_DIM:
            raise ValueError(
                f"{name} index {idx} out of leaf range [0, {MERA_LEAF_DIM})")
    return [_one_hot(kind_idx), _one_hot(type_idx), _one_hot(bid_idx),
            _one_hot(value_idx), _one_hot(tobl_idx)]
