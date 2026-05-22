"""MERA -> AST decoder (spec §7).

Measures each leaf (argmax of its per-leaf marginal), groups leaves into
nodes (5 per node, node-major), recovers per-node (kind,type,bid,value,
tobl), then reuses the MPS decoder's structural parse to rebuild the AST.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ast import Node
from .mera_encoder import MeraEncodingMeta
from .mera_encoding import MERA_LEAF_DIM, LEAVES_PER_NODE
from .decoder import parse_kind_stream
from src.qft_pcn.qft.mera import MERA


@dataclass
class DecodeResult:
    ast: Node
    residual_norm: float


def _leaf_marginal(state: MERA, leaf: int) -> np.ndarray:
    """The (MERA_LEAF_DIM,) probability vector for one leaf.

    Computed via MERA.local_expectation against each basis projector.
    The operator is 16x16 — trivially cheap (spec §9.5 note).
    """
    p = np.empty(MERA_LEAF_DIM, dtype=float)
    for b in range(MERA_LEAF_DIM):
        proj = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
        proj[b, b] = 1.0
        p[b] = float(np.real(state.local_expectation(leaf, proj)))
    total = p.sum()
    if total > 1e-15:
        p = p / total
    return p


def decode_mera(state: MERA, meta: MeraEncodingMeta) -> DecodeResult:
    """Deterministic argmax decode of a (concrete-program) MERA state."""
    # Measure every non-PAD leaf; group into per-node 5-tuples.
    per_node: list[tuple[int, int, int, int, int]] = []
    residual = 0.0
    for node_idx in range(meta.n_nodes):
        idxs = []
        for offset in range(LEAVES_PER_NODE):
            leaf = LEAVES_PER_NODE * node_idx + offset
            p = _leaf_marginal(state, leaf)
            b = int(np.argmax(p))
            idxs.append(b)
            residual = max(residual, 1.0 - float(p[b]))
        per_node.append(tuple(idxs))   # (kind, type, bid, value, tobl)

    # The shared structural parse consumes (kind,type,bid,value[,tobl])
    # tuples and ignores the trailing tobl entry (a typing-obligation tag,
    # not structural). Var->Lam wiring is done by parse_kind_stream's
    # binder stack, not duplicated here.
    ast = parse_kind_stream(per_node, meta.nested_type_index)
    return DecodeResult(ast=ast, residual_norm=residual)
