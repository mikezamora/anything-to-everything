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
