"""Top-level encoder: AST -> (MPS, EncodingMeta)."""

from __future__ import annotations

from typing import Optional

from .ast import Node, Ty
from .encoding import (
    SPECIES, EncodingMeta, KIND_CUTOFF, TYPE_CUTOFF,
    BID_CUTOFF, VALUE_CUTOFF, TYPE_ARR_NESTED,
)
from ._serialize import serialize_preorder
from ._types import compute_site_types
from ._channels import compute_live_binders
from ._tensors import build_site_tensors
from src.qft_pcn.qft.mps import MPS


def encode(ast: Node, N: int = 32, chi_max: int = 16
           ) -> tuple[MPS, EncodingMeta]:
    """Encode an AST into a unit-norm MPS of length N."""
    sites = serialize_preorder(ast, N=N)
    type_tags = compute_site_types(ast, sites)
    live = compute_live_binders(sites)
    tensors = build_site_tensors(sites, type_tags, live)
    state = MPS(tensors=tensors)
    state.normalize()

    nested: dict[int, Ty] = {}
    for k, occ in enumerate(sites):
        if type_tags[k] == TYPE_ARR_NESTED and occ.ty is not None:
            nested[k] = occ.ty

    site_to_path: dict[int, tuple[int, ...]] = {k: occ.ast_path
                                                for k, occ in enumerate(sites)}

    meta = EncodingMeta(
        N=N,
        chi_max=chi_max,
        field_dims={
            "kind": KIND_CUTOFF, "type": TYPE_CUTOFF,
            "bid": BID_CUTOFF, "value": VALUE_CUTOFF,
        },
        species=list(SPECIES),
        nested_type_index=nested,
        site_to_ast_path=site_to_path,
        live_binders_per_bond=live,
    )
    return state, meta
