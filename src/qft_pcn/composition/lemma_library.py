"""Lemma library: storage, indexing, registration. Spec §3, §4.

A lemma is a solved sub-problem -- a MERA ground state with residual
energy below eps_register. By Curry-Howard it is a proof object: it
inhabits the proposition its Hamiltonian encodes. The library caches it
so a future QPCN run clamps it rather than re-deriving it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta

FINGERPRINT_DIM = 32


@dataclass(frozen=True)
class MeraTensorBundle:
    """A serialization-friendly snapshot of a MERA's tensors. NOT a live
    MERA; LemmaLibrary.materialize rebuilds a MERA from it.

    Mirrors the storage layout of ``qft.mera.MERA``: leaves are the raw
    per-site (1, d_local, 1) tensors, and disentanglers / inter_disentanglers
    / isometries are flat lists ordered (layer, pair). Together with
    ``layer_dims`` and ``top`` this preserves the full physical state for
    a faithful round-trip via ``mera_from_bundle``.
    """
    n_leaves: int
    leaf_dim: int
    n_layers: int
    leaf_vectors: list[np.ndarray]
    disentanglers: list[np.ndarray]
    isometries: list[np.ndarray]
    inter_disentanglers: list[np.ndarray] = field(default_factory=list)
    top: np.ndarray | None = None
    layer_dims: tuple[int, ...] = ()


@dataclass(frozen=True)
class DerivationMetadata:
    """Provenance of a lemma (spec §3.2)."""
    hamiltonian_id: str
    residual_energy: float
    energy_gap: float
    trotter_steps: int
    assumptions: tuple[str, ...]
    lemma_deps: tuple[str, ...]
    conditional: bool
    source_run_id: str


from src.qft_pcn.qft.mera import MERA


def _flatten_layered(layered: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Flatten a list-of-lists of tensors to a flat list (layer-major)."""
    return [np.asarray(t).copy() for layer in layered for t in layer]


def _layered_shape(state: MERA) -> tuple[list[int], list[int]]:
    """Per-layer pair counts for (disentanglers/isometries, inter_disentanglers).

    Mirrors MERA.__post_init__: at layer ell with N leaves and L layers,
    n_l = N // 2**ell, expected_pairs = n_l // 2,
    expected_inter = max(0, expected_pairs - 1).
    """
    N = state.N
    L = state.L
    intra_counts = [(N // (2 ** ell)) // 2 for ell in range(L)]
    inter_counts = [max(0, c - 1) for c in intra_counts]
    return intra_counts, inter_counts


def _unflatten_layered(
    flat: list[np.ndarray], counts: list[int],
) -> list[list[np.ndarray]]:
    """Inverse of _flatten_layered given per-layer counts."""
    out: list[list[np.ndarray]] = []
    k = 0
    for c in counts:
        out.append([np.asarray(flat[k + j]).copy() for j in range(c)])
        k += c
    if k != len(flat):
        raise ValueError(
            f"_unflatten_layered: consumed {k} tensors, got {len(flat)}")
    return out


def bundle_from_mera(state: MERA) -> MeraTensorBundle:
    """Snapshot a live MERA into a serializable bundle (spec §3.2)."""
    return MeraTensorBundle(
        n_leaves=state.N,
        leaf_dim=state.d_local,
        n_layers=state.L,
        leaf_vectors=[np.asarray(v).copy() for v in state.leaves],
        disentanglers=_flatten_layered(state.disentanglers),
        isometries=_flatten_layered(state.isometries),
        inter_disentanglers=_flatten_layered(state.inter_disentanglers),
        top=np.asarray(state.top).copy(),
        layer_dims=tuple(state.layer_dims),
    )


def mera_from_bundle(bundle: MeraTensorBundle) -> MERA:
    """Rebuild a live MERA from a bundle (inverse of bundle_from_mera).

    Reconstructs the exact tensor content, not just the product-state
    skeleton, so superposition states built via from_term_superposition
    survive the round-trip too.
    """
    if bundle.top is None:
        raise ValueError("bundle.top is None; cannot rebuild MERA")
    # Compute per-layer counts from N and L (matches MERA.__post_init__).
    N = bundle.n_leaves
    L = bundle.n_layers
    intra_counts = [(N // (2 ** ell)) // 2 for ell in range(L)]
    inter_counts = [max(0, c - 1) for c in intra_counts]
    leaves = [np.asarray(v).copy() for v in bundle.leaf_vectors]
    disentanglers = _unflatten_layered(bundle.disentanglers, intra_counts)
    isometries = _unflatten_layered(bundle.isometries, intra_counts)
    inter_disentanglers = _unflatten_layered(
        bundle.inter_disentanglers, inter_counts)
    return MERA(
        leaves=leaves,
        disentanglers=disentanglers,
        inter_disentanglers=inter_disentanglers,
        isometries=isometries,
        top=np.asarray(bundle.top).copy(),
        layer_dims=list(bundle.layer_dims),
    )


def save_bundle_npz(bundle: MeraTensorBundle, path) -> None:
    """Serialize a bundle to .npz (spec §4.1)."""
    if bundle.top is None:
        raise ValueError("bundle.top is None; nothing to serialize")
    arrs: dict[str, np.ndarray] = {
        "n_leaves": np.array(bundle.n_leaves),
        "leaf_dim": np.array(bundle.leaf_dim),
        "n_layers": np.array(bundle.n_layers),
        "n_disent": np.array(len(bundle.disentanglers)),
        "n_iso": np.array(len(bundle.isometries)),
        "n_inter": np.array(len(bundle.inter_disentanglers)),
        "layer_dims": np.array(bundle.layer_dims, dtype=np.int64),
        "top": bundle.top,
    }
    for i, v in enumerate(bundle.leaf_vectors):
        arrs[f"leaf_{i}"] = v
    for i, d in enumerate(bundle.disentanglers):
        arrs[f"disent_{i}"] = d
    for i, w in enumerate(bundle.isometries):
        arrs[f"iso_{i}"] = w
    for i, u in enumerate(bundle.inter_disentanglers):
        arrs[f"inter_{i}"] = u
    np.savez_compressed(path, **arrs)


def load_bundle_npz(path) -> MeraTensorBundle:
    """Inverse of save_bundle_npz."""
    z = np.load(path, allow_pickle=False)
    n_leaves = int(z["n_leaves"])
    n_disent = int(z["n_disent"])
    n_iso = int(z["n_iso"])
    n_inter = int(z["n_inter"])
    return MeraTensorBundle(
        n_leaves=n_leaves,
        leaf_dim=int(z["leaf_dim"]),
        n_layers=int(z["n_layers"]),
        leaf_vectors=[np.asarray(z[f"leaf_{i}"]) for i in range(n_leaves)],
        disentanglers=[np.asarray(z[f"disent_{i}"]) for i in range(n_disent)],
        isometries=[np.asarray(z[f"iso_{i}"]) for i in range(n_iso)],
        inter_disentanglers=[
            np.asarray(z[f"inter_{i}"]) for i in range(n_inter)
        ],
        top=np.asarray(z["top"]),
        layer_dims=tuple(int(x) for x in z["layer_dims"]),
    )


@dataclass(frozen=True)
class Lemma:
    """A cached proof object (spec §3.2)."""
    lemma_id: str
    proposition_type: str
    mera_tensors: MeraTensorBundle
    encoding_meta: MeraEncodingMeta
    derivation: DerivationMetadata
    fingerprint: np.ndarray


from typing import Callable
from src.qft_pcn.composition.errors import CompressionError


def compress_bundle(bundle: MeraTensorBundle,
                    energy_fn: Callable[[MeraTensorBundle], float],
                    eps_compress: float = 1e-9) -> MeraTensorBundle:
    """SVD-truncate the bundle's isometry bonds to the smallest bond
    dimension that keeps energy_fn within eps_compress (spec §4.1).

    energy_fn(bundle) is <Psi|H_L|Psi>; the caller (register_lemma) binds
    it to the real M2 Hamiltonian. Truncation never grows storage: if no
    singular value can be dropped, the bundle is returned unchanged.

    The compression operates only on isometries. Other bundle fields
    (leaf_vectors, disentanglers, inter_disentanglers, top, layer_dims)
    are carried through unchanged.
    """
    base = energy_fn(bundle)
    isos = [np.asarray(w).copy() for w in bundle.isometries]
    for k, w in enumerate(isos):
        mat = w.reshape(w.shape[0], -1)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        for cut in range(len(s) - 1, 0, -1):
            trial = u[:, :cut] @ np.diag(s[:cut]) @ vh[:cut, :]
            cand = list(isos)
            cand[k] = trial.reshape(w.shape)
            trial_bundle = MeraTensorBundle(
                n_leaves=bundle.n_leaves,
                leaf_dim=bundle.leaf_dim,
                n_layers=bundle.n_layers,
                leaf_vectors=bundle.leaf_vectors,
                disentanglers=bundle.disentanglers,
                isometries=cand,
                inter_disentanglers=bundle.inter_disentanglers,
                top=bundle.top,
                layer_dims=bundle.layer_dims,
            )
            if abs(energy_fn(trial_bundle) - base) < eps_compress:
                isos[k] = trial.reshape(w.shape)
            else:
                break
    out = MeraTensorBundle(
        n_leaves=bundle.n_leaves,
        leaf_dim=bundle.leaf_dim,
        n_layers=bundle.n_layers,
        leaf_vectors=bundle.leaf_vectors,
        disentanglers=bundle.disentanglers,
        isometries=isos,
        inter_disentanglers=bundle.inter_disentanglers,
        top=bundle.top,
        layer_dims=bundle.layer_dims,
    )
    if abs(energy_fn(out) - base) >= eps_compress:
        raise CompressionError(
            f"compression drifted energy by >= {eps_compress}")
    return out


def structural_fingerprint(state: MERA) -> np.ndarray:
    """Sorted eigenvalue spectrum of the canonical-bond reduced density
    matrix, padded/truncated to FINGERPRINT_DIM (spec §4.3).

    Implementation note (architectural):
    The spec calls for "the reduced density matrix at the canonical (top)
    bond." For the logic encoder's concrete (hole-free) MERAs, however,
    the top tensor is mathematically forced to ``|0,0> * scalar``: the
    product-MERA constructor (``MERA.from_product``) uses
    ``_orthonormal_isometry`` to ascend each pair, which by construction
    rotates the pair amplitude onto basis vector ``e_0`` at every layer.
    Consequently, every concrete program produces a top tensor whose
    nonzero entry sits in the same single slot, and the top-bond RDM
    spectrum is structurally constant (rank-1, eigenvalue 1) across all
    concrete programs -- it cannot distinguish them.

    The canonical bond that *does* carry program-distinguishing
    information for product MERAs is the **leaf bond**: the multiset of
    per-site leaf state vectors is exactly the encoder's payload (spec
    §5.6). Per §1.1 the leaf assignment is the structural carrier when
    no holes induce entanglement; alpha-equivalence preserves it because
    bid leaves use depth-based BID indices, not surface names. We
    therefore compute the fingerprint as the spectrum of the leaf-bond
    Gram matrix ``G[i,j] = <v_i | v_j>`` over all n_leaves leaf vectors,
    which is alpha-invariant and program-distinguishing.

    For non-product MERAs (hole-bearing or term-superposition states),
    the leaf-Gram spectrum still captures the structural feature: the
    superposition's branch overlaps appear in G's eigenvalues.
    """
    leaves = state.leaves
    # leaf shape: (1, d_local, 1). Stack into V of shape (n_leaves, d_local).
    V = np.stack([np.asarray(s)[0, :, 0] for s in leaves], axis=0)
    # Gram matrix in the leaf basis (Hermitian PSD).
    G = np.einsum("ij,kj->ik", V.conj(), V, optimize="greedy")
    eig = np.linalg.eigvalsh(G)
    eig = np.sort(np.real(eig))[::-1]
    fp = np.zeros(FINGERPRINT_DIM)
    take = min(FINGERPRINT_DIM, len(eig))
    fp[:take] = eig[:take]
    return fp


def fingerprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 (trace-distance-style) distance between fingerprints (spec §4.3)."""
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))
