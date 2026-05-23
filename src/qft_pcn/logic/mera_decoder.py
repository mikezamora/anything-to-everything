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
from src.qft_pcn.qft._backend import contract, to_device, to_host, xp as _xp, GPU_ACTIVE


@dataclass
class DecodeResult:
    ast: Node
    residual_norm: float


def _ascend_one_layer_batched(state: MERA, op: np.ndarray, ell: int,
                              pos: int) -> np.ndarray:
    """Batched analogue of MERA._ascend_one_layer.

    Input ``op`` has shape ``(B, d_ell, d_ell)`` where ``B`` is a batch
    dim independent of the ascent; output is ``(B, d_{ell+1}, d_{ell+1})``.
    Mathematically equivalent to mapping ``_ascend_one_layer`` over the
    leading axis; physically: ascent is linear in the operator, so
    batching changes nothing.
    """
    j = pos // 2
    d_ell = state.layer_dims[ell]
    I = _xp.eye(d_ell, dtype=complex)
    op_dev = to_device(op)
    if pos % 2 == 0:
        # op acts on left slot of pair j; I on right slot.
        # op_pair[X, a, b, c, d] = op[X, a, c] * I[b, d]
        op_pair = contract('Xac,bd->Xabcd', op_dev, I)
    else:
        # op acts on right slot; I on left slot.
        # op_pair[X, a, b, c, d] = I[a, c] * op[X, b, d]
        op_pair = contract('ac,Xbd->Xabcd', I, op_dev)
    u = to_device(state.disentanglers[ell][j])
    # u_{A,B,a,b} . op_pair_{batch,a,b,c,d} -> tmp_{batch,A,B,c,d}
    tmp = contract('ABab,Xabcd->XABcd', u, op_pair)
    # tmp_{X,A,B,c,d} . conj(u)_{C,D,c,d} -> op_pair_conj_{X,A,B,C,D}
    op_pair_conj = contract('XABcd,CDcd->XABCD', tmp, u.conj())
    w = to_device(state.isometries[ell][j])
    # w_{A,a,b} . op_pair_conj_{X,a,b,c,d} . conj(w)_{B,c,d} -> op_up_{X,A,B}
    op_up = contract('Aab,Xabcd,Bcd->XAB', w, op_pair_conj, w.conj())
    return op_up


def _leaf_marginal(state: MERA, leaf: int) -> np.ndarray:
    """The (MERA_LEAF_DIM,) probability vector for one leaf.

    Optimization: instead of calling ``state.local_expectation`` once per
    basis state (16 full ascents through L-1 layers), exploit linearity of
    the ascent in the operator: ascend all 16 basis projectors
    ``|s><s|`` simultaneously via a batched ascent. Returns the same
    marginal as the loop-of-local_expectation form, up to floating-point
    contraction order.

    Falls back to the term-superposition path for hole-bearing states.
    """
    d = state.d_local
    if state._superposition_terms is not None:
        # Superposition path: leaf-only inner products are cheap (the leaf
        # vector is rank-1), so use the existing per-projector formula.
        p = np.empty(MERA_LEAF_DIM, dtype=float)
        for b in range(MERA_LEAF_DIM):
            proj = np.zeros((d, d), dtype=complex)
            proj[b, b] = 1.0
            p[b] = float(np.real(state.local_expectation(leaf, proj)))
        total = p.sum()
        if total > 1e-15:
            p = p / total
        return p

    if not 0 <= leaf < state.N:
        raise IndexError(f"leaf {leaf} out of range [0, {state.N})")

    # Build the batched operator: op0[s, a, b] = delta(s, a) * delta(s, b)
    # i.e. op0[s] = |s><s|. Shape (d, d, d). Allocate via backend module
    # so the ascent stays on-device end-to-end when GPU is active.
    op_layer = _xp.zeros((d, d, d), dtype=complex)
    idx = _xp.arange(d)
    op_layer[idx, idx, idx] = 1.0

    pos = leaf
    for ell in range(state.L - 1):
        op_layer = _ascend_one_layer_batched(state, op_layer, ell, pos)
        pos //= 2

    T = to_device(state.top[..., 0])   # (d_top, d_top)
    if pos == 0:
        # <O_s> = sum_{a,A,b} T.conj()[a,b] * op_layer[s,a,A] * T[A,b]
        vals = contract('ab,SaA,Ab->S', T.conj(), op_layer, T)
    else:
        # <O_s> = sum_{a,b,B} T.conj()[a,b] * op_layer[s,b,B] * T[a,B]
        vals = contract('ab,SbB,aB->S', T.conj(), op_layer, T)
    # Bring back to host for the rest of the function (probabilities,
    # clip, normalization happen as plain NumPy floats).
    vals = to_host(vals)
    p = np.real(vals).astype(float)
    # Clamp tiny negatives from floating-point noise.
    np.clip(p, 0.0, None, out=p)
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

    Optimization: for non-superposition MERAs the per-leaf marginal
    computed by ``_leaf_marginal`` depends only on the (top, isometries,
    disentanglers) fields — these are NOT touched by ``_project_leaf``,
    which only mutates the per-leaf vector. The marginals are therefore
    invariant across the sample loop and across samples, so they are
    computed once and shared. Distribution identical to the pre-optimization
    code (verified by chi-squared test). For superposition (hole-bearing)
    states the leaf marginal is non-stationary across measurements
    (branches drop), so we keep the original conditional loop.
    """
    if rng is None:
        rng = np.random.default_rng()
    n_leaves = LEAVES_PER_NODE * meta.n_nodes
    results: list[DecodeResult] = []

    # Fast path: non-superposition state.  Marginals are stationary; cache
    # them once and draw each sample independently.
    if state._superposition_terms is None:
        marginals = [_leaf_marginal(state, leaf) for leaf in range(n_leaves)]
        for _ in range(n_samples):
            per_node: list[tuple[int, int, int, int, int]] = []
            node_idxs: list[int] = []
            for leaf in range(n_leaves):
                b = int(rng.choice(MERA_LEAF_DIM, p=marginals[leaf]))
                node_idxs.append(b)
                if len(node_idxs) == LEAVES_PER_NODE:
                    per_node.append(tuple(node_idxs))
                    node_idxs = []
            ast = parse_kind_stream(per_node, meta.nested_type_index)
            results.append(DecodeResult(ast=ast, residual_norm=0.0))
        return results

    # Superposition path: branches drop as leaves project, so the
    # conditional loop is genuinely needed.
    for _ in range(n_samples):
        ket = state.copy()
        per_node: list[tuple[int, int, int, int, int]] = []
        node_idxs: list[int] = []
        for leaf in range(n_leaves):
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
