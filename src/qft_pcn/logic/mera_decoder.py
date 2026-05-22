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


def _project_leaf(state: MERA, leaf: int, b: int) -> None:
    """In-place: project ``state``'s ``leaf`` onto basis state ``b`` and
    renormalize.

    For a term-superposition state (hole-bearing program) the projection
    acts on the explicit branch decomposition: each branch is reweighted by
    the amplitude its leaf-``leaf`` vector places on ``b``, branches with
    zero weight are dropped, and the survivors are renormalized. This is
    exact conditional measurement — the MERA conditional-sampling step.

    For a product (concrete) state the marginal is already a delta, so the
    projection is a no-op on the encoded wavefunction; we still apply the
    local projector so the leaf vector reflects the measured value.
    """
    if state._superposition_terms is not None:
        kept: list[tuple[complex, list[np.ndarray]]] = []
        for coeff, sites in state._superposition_terms:
            amp = complex(sites[leaf][b])
            if abs(amp) < 1e-12:
                continue
            new_sites = [s.copy() for s in sites]
            proj = np.zeros(MERA_LEAF_DIM, dtype=complex)
            proj[b] = amp
            new_sites[leaf] = proj
            kept.append((coeff, new_sites))
        if not kept:
            raise ValueError(
                f"projecting leaf {leaf} onto basis {b} annihilated the "
                "state (zero-probability outcome)")
        state._superposition_terms = kept
        state.normalize()
        return
    proj = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
    proj[b, b] = 1.0
    state.apply_local_gate(leaf, proj)
    nrm = state.norm_sq()
    if nrm > 1e-15:
        state.normalize()


def sample_mera(state: MERA, meta: MeraEncodingMeta,
                n_samples: int = 1, rng=None) -> list[DecodeResult]:
    """Sample ``n_samples`` ASTs from the MERA distribution.

    Left-to-right leaf-by-leaf conditional sampling: measure each leaf
    from its marginal conditioned on the prior measurements, project the
    state onto the outcome, advance. For a product (concrete) state every
    marginal is a delta and every sample equals ``decode_mera``. For a
    hole-bearing state each call draws a fresh completion — the §1.1
    structural superposition collapsed to one resolved program.

    Sub-project M3's synthesis samples hole completions through this
    entry point.
    """
    if rng is None:
        rng = np.random.default_rng()
    results: list[DecodeResult] = []
    for _ in range(n_samples):
        ket = state.copy()
        per_node: list[tuple[int, int, int, int, int]] = []
        node_idxs: list[int] = []
        for leaf in range(LEAVES_PER_NODE * meta.n_nodes):
            p = _leaf_marginal(ket, leaf)
            b = int(rng.choice(MERA_LEAF_DIM, p=p))
            _project_leaf(ket, leaf, b)
            node_idxs.append(b)
            if len(node_idxs) == LEAVES_PER_NODE:
                per_node.append(tuple(node_idxs))
                node_idxs = []
        ast = parse_kind_stream(per_node, meta.nested_type_index)
        results.append(DecodeResult(ast=ast, residual_norm=0.0))
    return results
