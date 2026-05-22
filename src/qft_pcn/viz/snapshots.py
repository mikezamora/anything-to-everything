"""Non-invasive snapshot extractors.

Each `snapshot_*` function takes a *live* substrate object and returns a plain
dict of JSON-ready values (lists / floats / nested dicts). The functions read
only public attributes and never call mutating methods, so recording a run
leaves the simulation byte-for-byte identical to an unrecorded run.

Every extractor is defensive: a missing or optional attribute yields `None`
rather than raising, so the same extractor works across substrate variants.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# Grids larger than this (per axis) are downsampled before serialization so a
# Frame stays small enough to stream cheaply.
_MAX_GRID = 32


def _safe(fn: Callable[[], Any]) -> Any:
    """Call `fn`, returning None on any AttributeError/KeyError/TypeError."""
    try:
        return fn()
    except (AttributeError, KeyError, TypeError, IndexError, ValueError):
        return None


def _downsample(arr: np.ndarray, max_side: int = _MAX_GRID) -> np.ndarray:
    """Stride-subsample the last two axes of `arr` down to <= max_side."""
    a = np.asarray(arr)
    if a.ndim < 2:
        return a
    sx = max(1, int(np.ceil(a.shape[-2] / max_side)))
    sy = max(1, int(np.ceil(a.shape[-1] / max_side)))
    if sx == 1 and sy == 1:
        return a
    return a[..., ::sx, ::sy]


def _grid(arr: Any) -> Any:
    """Downsample + tolist a grid-like array, defensively."""
    if arr is None:
        return None
    a = np.asarray(arr)
    return _downsample(a).tolist()


# ---- network / manifold ------------------------------------------------------

def snapshot_network(net: Any) -> dict:
    """Snapshot a `QFTPCNNetwork`: metric, curvature, per-layer fields, F."""
    manifold = _safe(lambda: net.manifold)

    metric_h = None
    ricci = None
    if manifold is not None:
        metric_h = {
            "h_xx": _grid(_safe(lambda: manifold.h_xx)),
            "h_xy": _grid(_safe(lambda: manifold.h_xy)),
            "h_yy": _grid(_safe(lambda: manifold.h_yy)),
        }
        ricci = _grid(_safe(lambda: manifold.ricci_scalar()))

    fields = []
    for layer in _safe(lambda: net.layers) or []:
        fields.append({
            "phi": _grid(_safe(lambda l=layer: l.phi.values)),
            "E": _grid(_safe(lambda l=layer: l.error.values)),
            "Pi": _grid(_safe(lambda l=layer: l.precision.pi)),
            "channels": _safe(lambda l=layer: l.phi.channels),
        })

    # free_energy needs an observation; expose the cheap aggregate instead.
    free_energy = _safe(
        lambda: float(np.abs(net.manifold.ricci_scalar()).mean())
    )

    return {
        "metric_h": metric_h,
        "ricci": ricci,
        "fields": fields,
        "free_energy": free_energy,
        "step": _safe(lambda: int(net._step)),
    }


# ---- quantum predictive coder ------------------------------------------------

def snapshot_qpcn(q: Any) -> dict:
    """Snapshot a `QPCN`: energy, errors, params, bond dims, entropies."""
    state = _safe(lambda: q.state)

    bond_dims = _safe(lambda: list(state.bond_dimensions()))
    entropies = None
    if bond_dims is not None:
        n_cuts = _safe(lambda: state.N)
        if n_cuts:
            entropies = [
                _safe(lambda c=c: float(state.entanglement_entropy(c)))
                for c in range(1, n_cuts)
            ]

    occupations = None
    if state is not None:
        occupations = _safe(
            lambda: [float(np.real(np.linalg.norm(t))) for t in state.tensors]
        )

    return {
        "energy": _safe(lambda: float(np.real(q._last_energy))),
        "pred_errors": _safe(lambda: dict(q._last_errors)),
        "params": _safe(lambda: dict(q.cfg.learnable_params)),
        "bond_dims": bond_dims,
        "entropies": entropies,
        "occupations": occupations,
        "step": _safe(lambda: int(q._step)),
    }


# ---- matrix product state ----------------------------------------------------

def snapshot_mps(mps: Any) -> dict:
    """Snapshot an `MPS`: bond dims and per-bond entanglement entropy."""
    bond_dims = _safe(lambda: list(mps.bond_dimensions()))

    entropies = None
    n = _safe(lambda: int(mps.N))
    if n:
        entropies = [
            _safe(lambda c=c: float(mps.entanglement_entropy(c)))
            for c in range(1, n)
        ]

    return {
        "bond_dims": bond_dims,
        "entropies": entropies,
        "n_sites": n,
        "d_local": _safe(lambda: int(mps.d)),
    }


# ---- multifield network ------------------------------------------------------

def snapshot_multifield(mf: Any) -> dict:
    """Snapshot a `MultiFieldNetwork`: per-field fields + couplings."""
    fields = {}
    for name, layer in (_safe(lambda: mf.fields) or {}).items():
        fields[name] = {
            "phi": _grid(_safe(lambda l=layer: l.phi.values)),
            "E": _grid(_safe(lambda l=layer: l.error.values)),
            "Pi": _grid(_safe(lambda l=layer: l.precision.pi)),
        }

    couplings = _safe(
        lambda: {f"{a}|{b}": float(v) for (a, b), v in mf.couplings.items()}
    )

    return {
        "fields": fields,
        "couplings": couplings,
        "step": _safe(lambda: int(mf._step)),
    }


# ---- hamiltonian -------------------------------------------------------------

def snapshot_hamiltonian(H: Any) -> dict:
    """Snapshot a `Hamiltonian`: site count, local dim, curvature."""
    return {
        "n_sites": _safe(lambda: int(H.N)),
        "d_local": _safe(lambda: int(H.d_local)),
        "species_dims": _safe(lambda: list(H.species_dims)),
        "species": _safe(lambda: [s.name for s in H.species]),
        "curvature": _safe(lambda: np.asarray(H.curvature).tolist()),
    }


# ---- MERA --------------------------------------------------------------------

def snapshot_mera(m: Any) -> dict:
    """Snapshot a `MERA`: leaf count, per-layer bond dims, cut entropies."""
    leaves = _safe(lambda: m.leaves)
    n_leaves = len(leaves) if leaves is not None else None

    entropies = None
    if n_leaves:
        entropies = [
            _safe(lambda c=c: float(m.entanglement_entropy(c)))
            for c in range(1, n_leaves)
        ]

    return {
        "n_leaves": n_leaves,
        "layer_dims": _safe(lambda: list(m.layer_dims)),
        "bond_dims": _safe(lambda: list(m.bond_dimensions())),
        "entropies": entropies,
    }


# ---- variational quantum circuit ---------------------------------------------

def snapshot_vqc(vqc: Any) -> dict:
    """Snapshot a variational quantum circuit / quantum generative map."""
    params = _safe(lambda: np.asarray(vqc.params).tolist())
    if params is None:
        params = _safe(lambda: np.asarray(vqc.theta).tolist())

    return {
        "params": params,
        "n_qubits": _safe(lambda: int(vqc.n_qubits)),
        "n_layers": _safe(lambda: int(vqc.n_layers)),
        "kernel": _safe(lambda: np.asarray(vqc.kernel).tolist()),
    }


# ---- logic encoder -----------------------------------------------------------

def snapshot_logic(enc: Any) -> dict:
    """Snapshot a logic encoder / encoding result. Minimal + defensive."""
    return {
        "n_sites": _safe(lambda: int(enc.n_sites)),
        "d_local": _safe(lambda: int(enc.d_local)),
        "bond_dims": _safe(lambda: list(enc.state.bond_dimensions())),
        "term_count": _safe(lambda: len(enc.terms)),
    }
