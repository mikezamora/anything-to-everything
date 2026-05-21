"""Gate-based construction of the AST encoding — alternative to the analytic
path in _tensors.py.

Builds the same MPS state from the all-PAD vacuum by applying single-site
'write' gates that set each non-PAD site's complete local state (kind,
type, bid, value) in one shot.

For concrete (no-hole) inputs the result equals the analytic encoder's
state to fidelity > 1 - 1e-8. The cross-check test catches bugs in either
path.

The gate path is test-only:
- It does NOT exercise the channel bond mechanism (the result is a pure
  product state on every register).
- HoleVar superposition is rejected (raises NotImplementedError).
"""

from __future__ import annotations

import numpy as np

from .ast import Node
from .encoding import (
    SPECIES, EncodingMeta, BinderHandle,
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, D_LOCAL,
    KIND_PAD, KIND_VAR, KIND_LAM,
    TYPE_NONE, TYPE_ARR_NESTED,
    BID_NONE, BID_0, VALUE_NONE,
)
from ._serialize import serialize_preorder
from ._types import compute_site_types
from ._channels import compute_live_binders
from ._tensors import _basis_index, _local_kind_type_value
from src.qft_pcn.qft.mps import MPS


def _single_site_write_gate(target_kind: int, target_type: int,
                            target_bid: int, target_value: int
                            ) -> np.ndarray:
    """An 8192x8192 gate mapping |PAD,NONE,NONE,NONE> -> |target>.

    Used to write a definite local state on top of the vacuum at one site.
    Gate G[a, b] = delta_{a, target_idx} * delta_{b, PAD_idx}.
    """
    G = np.zeros((D_LOCAL, D_LOCAL), dtype=complex)
    target_idx = _basis_index(target_kind, target_type, target_bid,
                              target_value)
    pad_idx = _basis_index(KIND_PAD, TYPE_NONE, BID_NONE, VALUE_NONE)
    G[target_idx, pad_idx] = 1.0
    return G


def encode_gate(ast: Node, N: int = 32, chi_max: int = 16
                ) -> tuple[MPS, EncodingMeta]:
    """Gate-based equivalent of encode(). Starts from the all-PAD vacuum
    MPS and writes each non-PAD site's local state via single-site gates.

    Each site's local state is set as (kind, type, bid_for_kind, value):
      - LAM sites: bid = BID_0 (the LAM introducing its own binder).
      - VAR sites (single-candidate): bid = BID_0 + depth_from_innermost.
      - All others: bid = BID_NONE.
    HoleVar (multi-candidate VAR) raises NotImplementedError.

    Returns (state, EncodingMeta).
    """
    sites = serialize_preorder(ast, N=N)
    type_tags = compute_site_types(ast, sites)
    live = compute_live_binders(sites)

    # Start from the all-PAD vacuum MPS.
    pad_idx = _basis_index(KIND_PAD, TYPE_NONE, BID_NONE, VALUE_NONE)
    initial_states = []
    for _ in range(N):
        v = np.zeros(D_LOCAL, dtype=complex)
        v[pad_idx] = 1.0
        initial_states.append(v)
    state = MPS.from_product(initial_states)

    # Apply per-site write gates.
    for k, occ in enumerate(sites):
        if occ.kind == KIND_PAD:
            continue
        kind_idx, type_idx, value_idx = _local_kind_type_value(occ, type_tags[k])
        # Determine local bid.
        if occ.kind == KIND_LAM:
            bid_idx = BID_0
        elif occ.kind == KIND_VAR:
            cands = occ.var_ref.candidates
            if cands and len(cands) > 1:
                raise NotImplementedError(
                    "gate construction does not handle HoleVar superposition; "
                    "use the analytic encoder for hole-bearing inputs."
                )
            # Plain Var or single-candidate hole: use the (single) depth.
            if cands:
                d = cands[0][1]
            else:
                d = occ.var_ref.depth_from_innermost
            bid_idx = BID_0 + d
        else:
            bid_idx = BID_NONE

        G = _single_site_write_gate(kind_idx, type_idx, bid_idx, value_idx)
        state.apply_local_gate(k, G)

    state.normalize()

    # Build EncodingMeta mirroring what encode() returns.
    nested = {}
    for k, occ in enumerate(sites):
        if type_tags[k] == TYPE_ARR_NESTED and occ.ty is not None:
            nested[k] = occ.ty
    site_to_path = {k: occ.ast_path for k, occ in enumerate(sites)}
    meta = EncodingMeta(
        N=N, chi_max=chi_max,
        field_dims={"kind": KIND_CUTOFF, "type": TYPE_CUTOFF,
                    "bid": BID_CUTOFF, "value": VALUE_CUTOFF},
        species=list(SPECIES),
        nested_type_index=nested,
        site_to_ast_path=site_to_path,
        live_binders_per_bond=live,
    )
    return state, meta
