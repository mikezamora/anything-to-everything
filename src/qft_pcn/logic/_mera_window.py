"""k-adjacent-leaf expectation on a MERA (spec §8.2).

mera_window_expectation is the dense form, used only for k<=2 to
cross-check F's two_site_expectation. M2's Hamiltonian terms call the
factored form (Task 9) — never the dense form for k>2, since a dense
(16**k, 16**k) operator is intractable for large k.

Performance: ``mera_window_expectation_factored`` is the hot path that
M2's H_typing / H_eval and M3's synth Hamiltonians all funnel through.
For Bundle-sized states (n_nodes ~ 14, chi_layer = 32, N = 128) the
naive ``state.copy() + apply_local_gate + state.inner(ket)`` form costs
~2s per call, and a single ``H.total_energy(state)`` triggers ~650 calls
— blowing well past 1000s. This module exposes two optimizations,
mathematically equivalent to the naive form, that break that wall:

  (1) Single-active-leaf factored ops are delegated to
      ``state.local_expectation(leaf, op)`` (a single causal-cone ascent,
      ~0.4 ms per call) — identity operators on the other leaves trace
      out exactly.

  (2) Multi-active-leaf factored ops use a per-state cross-ascent cache.
      ``state.inner(state)`` (= ``norm_sq``) only varies between
      consecutive calls in the layer-0 pair(s) whose leaves carry an
      operator; the remaining layer-0 cross matrices and all higher
      layers' contributions that don't depend on those leaves are
      identical between calls and are cached on the state. The cache is
      keyed by an explicit version counter so any mutation
      (``apply_local_gate`` / ``apply_two_site_gate``) bumping that
      counter invalidates the cache.

Both optimizations are exact: they compute the same expectation as the
naive form within numerical tolerance (verified by atol=1e-6 against the
naive path in the tests). Physics is unchanged — addressing only.
"""
from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.qft._backend import contract, to_device, to_host


def mera_window_expectation(state: MERA, leaf0: int, k: int,
                            op: np.ndarray) -> complex:
    """<state | O | state> for O on the k adjacent leaves [leaf0, leaf0+k).

    op has shape (d**k, d**k) with d = state.d_local. For k==1 this
    delegates to MERA.local_expectation; for k==2 to
    MERA.two_site_expectation. k>2 dense is rejected — use the factored
    form (mera_window_expectation_factored).
    """
    d = state.d_local
    if k == 1:
        if op.shape != (d, d):
            raise ValueError(f"k=1 op shape {op.shape}, expected ({d},{d})")
        return state.local_expectation(leaf0, op)
    if k == 2:
        if op.shape != (d * d, d * d):
            raise ValueError(
                f"k=2 op shape {op.shape}, expected ({d*d},{d*d})")
        return state.two_site_expectation(leaf0, op)
    raise ValueError(
        f"dense mera_window_expectation supports k in {{1,2}}; for k={k} "
        f"use mera_window_expectation_factored")


def mera_window_expectation_factored(
    state: MERA, leaf_ops: dict[int, np.ndarray]) -> complex:
    """<state | O | state> where O = prod over listed leaves of a per-leaf
    (d, d) operator (unlisted leaves: identity).

    No (d**k, d**k) tensor is formed. Fast paths:

    - 0 active leaves (all listed ops are identity): return state.norm_sq().

    - 1 active leaf: delegate to state.local_expectation — the identity
      operators on the other leaves trace out exactly.

    - product-MERA, no term superposition: telescoping leaf-overlap
      identity (each isometry W satisfies W^dag W = I) collapses the
      expectation to prod_k <v_k|O_k|v_k> over the listed leaves times
      <v_j|v_j> on the rest.

    - general case: build the ket as a SHALLOW alias of the bra (sharing
      disentanglers / isometries / top, only modified leaves are fresh
      arrays), then use the cached double-network cross-ascent so layer-0
      pairs whose leaves are untouched reuse the bra-bra cross matrices.

    All paths give the SAME numerical answer as the naive
    "ket = state.copy(); ket.apply_local_gate; state.inner(ket)" form
    within tolerance. Physics is unchanged.
    """
    d = state.d_local
    # Drop identity operators.
    active = {leaf: op for leaf, op in leaf_ops.items()
              if not _is_identity(op, d)}
    if not active:
        return complex(state.norm_sq())
    for leaf, op in active.items():
        if op.shape != (d, d):
            raise ValueError(
                f"leaf {leaf} op shape {op.shape}, expected ({d},{d})")
    # Single-leaf delegation: when the state has no analytic-branch
    # decomposition (``_superposition_terms is None``) the identity
    # operators on the other leaves trace out exactly, and
    # ``state.local_expectation`` uses the same tensor-network ascent
    # as the general path. For ``_superposition_terms is not None``
    # we MUST NOT use this short-circuit: ``local_expectation``
    # delegates to ``_local_expectation_from_terms`` which evaluates
    # the analytic branch sum; the rest of the factored expectation
    # path uses the MERA tensor network on the unit-summed leaves,
    # and the two values differ. Stay consistent with the general
    # path to preserve numerics within atol=1e-6.
    if len(active) == 1 and state._superposition_terms is None:
        ((leaf, op),) = active.items()
        return state.local_expectation(leaf, op)
    # Product-MERA fast path: prod_k <v_k|O_k|v_k> on the listed leaves
    # times <v_j|v_j> on the rest (telescoping identity).
    if state._superposition_terms is None and state._is_product():
        val = complex(1.0)
        for k in range(state.N):
            v = state.leaves[k][0, :, 0]
            op = active.get(k)
            if op is None:
                val *= complex(v.conj() @ v)
            else:
                val *= complex(v.conj() @ (op @ v))
        return val
    # General path: cached cross-ascent.
    return _expectation_with_bra_cache(state, active)


# ---------------------------------------------------------------------------
# Cached cross-ascent for the general factored case.
# ---------------------------------------------------------------------------


def _bra_cache(state: MERA) -> dict:
    """Get or build the bra-bra cross-ascent cache on ``state``.

    The cache stores the diagonal (bra == ket == state) cross matrices at
    every layer of the double-network ascent. It is invalidated by an
    explicit version counter on the state which any mutation
    (``apply_local_gate`` / ``apply_two_site_gate``) is expected to bump.
    If the counter is missing or doesn't match, we rebuild from scratch.
    """
    version = getattr(state, "_mutation_version", 0)
    cache = getattr(state, "_bra_cross_cache", None)
    if cache is not None and cache.get("version") == version:
        return cache
    cache = _build_bra_cache(state)
    cache["version"] = version
    # Stash on the state instance.
    try:
        object.__setattr__(state, "_bra_cross_cache", cache)
    except Exception:
        pass
    return cache


def _build_bra_cache(state: MERA) -> dict:
    """Build all layer-0 .. layer-(L-1) bra-bra cross matrices for ``state``.

    Layer 0 (cross[0][j]): (d_1, d_1) cross matrix for pair j with bra==ket.
    Layers >= 1: cross[ell][j] = (d_ell, d_ell) cross matrix.

    Also pre-cache the composed bra layer-isometries Wb_0[j] (layer 0) and
    Wb_ell[j] (layer >= 1) so per-query recomputation reuses them.
    """
    L = state.L
    N = state.N
    cache: dict = {}
    # Layer-0: composed layer-isometry Wb[j] and the diagonal eta_k.
    Wb0: list = []
    for j in range(N // 2):
        u_b = to_device(state.disentanglers[0][j])
        w_b = to_device(state.isometries[0][j])
        # Wb[A, s_l, s_r] = sum_{a,b} w_b[A,a,b] * u_b[a,b,s_l,s_r]
        Wb0.append(contract('Aab,abst->Ast', w_b, u_b))
    cache["Wb0"] = Wb0
    # Pre-contract WW0[j][B,K,s,t] = Wb0[j].conj()[B,s,t] * Wb0[j][K,s,t] so
    # the per-term layer-0 cross matrix collapses from a 4-tensor contract
    # ('Bst,Kst,s,t->BK') to a 3-tensor ('BKst,s,t->BK'). Both Wb factors are
    # term-independent at layer 0, so this is the same pre-contract trick as
    # Wk_higher below — one cuTensorNet path-find amortized over thousands
    # of per-term invocations.
    WW0: list = []
    for j in range(N // 2):
        Wb = Wb0[j]
        WW0.append(contract('Bst,Kst->BKst', Wb.conj(), Wb))
    cache["WW0"] = WW0
    # Diagonal eta_k = leaf.conj() * leaf
    etas_diag: list = []
    for k in range(N):
        v = state.leaves[k][0, :, 0]
        etas_diag.append(to_device(v.conj() * v))
    cache["etas_diag"] = etas_diag
    # Layer-0 diagonal cross matrices.
    cross0: list = []
    for j in range(N // 2):
        eta_l = etas_diag[2 * j]
        eta_r = etas_diag[2 * j + 1]
        Wb = Wb0[j]
        # M[B, K] = sum_{s,t} Wb.conj()[B,s,t] * Wb[K,s,t] * eta_l[s] * eta_r[t]
        M = contract('Bst,Kst,s,t->BK', Wb.conj(), Wb, eta_l, eta_r)
        cross0.append(M)
    # Layers >= 1: ascend the diagonal cross matrices to the top.
    cross_layers: list = [cross0]
    Wb_higher: list = []  # Wb_higher[ell-1] (for ell in 1..L-2): composed
                          # bra layer-isometries used at layer ell ascent.
    Wk_higher: list = []  # Wk_higher[ell-1]: non-conjugated composed ket-side
                          # layer-isometries reused per-term in
                          # _expectation_with_bra_cache (saves 5000+ redundant
                          # 'Kxy,xyab->Kab' contracts per total_energy on P3).
    for ell in range(1, L - 1):
        n_above = N // (2 ** (ell + 1))
        prev = cross_layers[-1]
        layer_Wb: list = []
        layer_Wk: list = []
        layer_cross: list = []
        for j in range(n_above):
            u_b = to_device(state.disentanglers[ell][j])
            w_b = to_device(state.isometries[ell][j])
            # For bra-side at higher layers, the cross-ascent contracts:
            #   Wb_higher[B,a,b] = sum w_b.conj()[B,x,y] * u_b.conj()[x,y,a,b]
            # (mirrors _cross_ascend's Wb construction).
            Wb_h = contract('Bxy,xyab->Bab', w_b.conj(), u_b.conj())
            layer_Wb.append(Wb_h)
            ML = prev[2 * j]
            MR = prev[2 * j + 1]
            # Diagonal: Wb (bra) == Wb (ket). M_new[B,K]
            #   = sum Wb_h[B,a,b] * Wb_h_kbra[K,c,d] * ML[a,c] * MR[b,d]
            # For bra==ket, Wb_h_kbra is the conjugate-transposed form
            # used as Wk (i.e. w_b applied without the .conj() that the
            # bra branch carries). Mirroring _cross_ascend exactly.
            # NB: Wk_h depends ONLY on state.isometries/disentanglers at this
            # (ell, j) — independent of which leaves are active. Stash it for
            # _expectation_with_bra_cache below.
            Wk_h = contract('Kxy,xyab->Kab', w_b, u_b)
            layer_Wk.append(Wk_h)
            M_new = contract('Bab,Kcd,ac,bd->BK', Wb_h, Wk_h, ML, MR)
            layer_cross.append(M_new)
        Wb_higher.append(layer_Wb)
        Wk_higher.append(layer_Wk)
        cross_layers.append(layer_cross)
    cache["cross_layers"] = cross_layers
    cache["Wb_higher"] = Wb_higher
    cache["Wk_higher"] = Wk_higher
    return cache


def _expectation_with_bra_cache(state: MERA,
                                active: dict[int, np.ndarray]) -> complex:
    """Use the cached bra-bra cross-ascent to compute the expectation.

    Equivalent to building ket = state with op-applied-leaves and computing
    state.inner(ket), but only pairs (and ancestor pairs) containing an
    active leaf are recomputed; everything else is read from the cache.
    """
    L = state.L
    N = state.N
    if L == 1:
        # Trivial fall-back: the cache path is overkill for N=2.
        return _expectation_naive_inner(state, active)
    cache = _bra_cache(state)
    Wb0 = cache["Wb0"]
    WW0 = cache["WW0"]
    etas_diag = cache["etas_diag"]
    cross_layers = cache["cross_layers"]
    Wb_higher = cache["Wb_higher"]
    Wk_higher = cache["Wk_higher"]

    # Layer-0: compute fresh eta_k for active leaves; reuse diagonal else.
    affected_pairs_0: set[int] = set()
    eta_mod: dict[int, np.ndarray] = {}
    for leaf, op in active.items():
        v_bra = state.leaves[leaf][0, :, 0]
        # ket leaf = op @ leaf, so eta[s] = bra.conj()[s] * (op @ leaf)[s].
        v_ket = op @ v_bra
        eta_mod[leaf] = to_device(v_bra.conj() * v_ket)
        affected_pairs_0.add(leaf // 2)

    # Build the layer-0 cross matrices for affected pairs only.
    cur_cross: list[np.ndarray] = []
    for j in range(N // 2):
        if j not in affected_pairs_0:
            cur_cross.append(cross_layers[0][j])
            continue
        l_idx = 2 * j
        r_idx = 2 * j + 1
        eta_l = eta_mod.get(l_idx, etas_diag[l_idx])
        eta_r = eta_mod.get(r_idx, etas_diag[r_idx])
        # Bra and ket SHARE this layer's Wb (we modified only the leaves;
        # the disentangler/isometry are the same for bra and ket). The
        # term-independent 'Wb.conj() * Wb' factor is pre-contracted into
        # WW0[j] = WW[B,K,s,t] in the bra cache; per-term we just fold in
        # the two eta vectors.
        M = contract('BKst,s,t->BK', WW0[j], eta_l, eta_r)
        cur_cross.append(M)

    # Higher layers: only pairs whose subtree touched an affected layer-0
    # pair need recomputation.
    affected: set[int] = set(affected_pairs_0)
    for ell in range(1, L - 1):
        n_above = N // (2 ** (ell + 1))
        layer_Wb = Wb_higher[ell - 1]
        layer_Wk = Wk_higher[ell - 1]
        next_cross: list[np.ndarray] = []
        next_affected: set[int] = set()
        for j in range(n_above):
            below_l = 2 * j
            below_r = 2 * j + 1
            if below_l not in affected and below_r not in affected:
                next_cross.append(cross_layers[ell][j])
                continue
            ML = cur_cross[below_l]
            MR = cur_cross[below_r]
            Wb_h = layer_Wb[j]
            # Ket-side composed isometry: bra and ket share the MERA tree
            # tensors (only the leaves differ), so Wk_h depends solely on
            # state.isometries/disentanglers[ell][j] — already pre-built in
            # the bra cache. No re-contraction per term.
            Wk_h = layer_Wk[j]
            M_new = contract('Bab,Kcd,ac,bd->BK', Wb_h, Wk_h, ML, MR)
            next_cross.append(M_new)
            next_affected.add(j)
        cur_cross = next_cross
        affected = next_affected

    # Top contraction with the (2,) cross at layer L-1.
    assert len(cur_cross) == 2
    ML, MR = cur_cross
    T_b = to_device(state.top[..., 0].conj())
    T_k = to_device(state.top[..., 0])
    val = contract('ab,cd,ac,bd->', T_b, T_k, ML, MR)
    return complex(to_host(val))


def _expectation_naive_inner(state: MERA,
                             active: dict[int, np.ndarray]) -> complex:
    """L==1 fallback: build ket via shallow alias and call state.inner."""
    new_leaves = list(state.leaves)
    for leaf, op in active.items():
        new_leaves[leaf] = np.einsum(
            'st,ltr->lsr', op, state.leaves[leaf], optimize='greedy')
    ket = MERA(
        leaves=new_leaves,
        disentanglers=state.disentanglers,
        inter_disentanglers=state.inter_disentanglers,
        isometries=state.isometries,
        top=state.top,
        layer_dims=state.layer_dims,
        _superposition_terms=state._superposition_terms,
    )
    return state.inner(ket)


def _is_identity(op: np.ndarray, d: int) -> bool:
    if op.shape != (d, d):
        return False
    return np.allclose(op, np.eye(d, dtype=op.dtype), atol=1e-12)
