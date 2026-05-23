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
    """Call `fn`, returning None for a missing optional attribute.

    Only `(AttributeError, KeyError, TypeError)` are swallowed — those signal
    an optional attribute simply not present on this substrate variant.
    `ValueError`/`IndexError` indicate a real coding error in a snapshot
    (e.g. an out-of-range entropy cut) and are deliberately allowed to surface.
    """
    try:
        return fn()
    except (AttributeError, KeyError, TypeError):
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

    # A true free energy needs an observation; expose the cheap curvature
    # aggregate instead, under an honest key name.
    mean_abs_ricci = _safe(
        lambda: float(np.abs(net.manifold.ricci_scalar()).mean())
    )

    return {
        "metric_h": metric_h,
        "ricci": ricci,
        "fields": fields,
        "mean_abs_ricci": mean_abs_ricci,
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
            # entanglement_entropy uses 0-indexed bonds: valid 0..N-2.
            entropies = [
                _safe(lambda c=c: float(state.entanglement_entropy(c)))
                for c in range(n_cuts - 1)
            ]

    # `tensor_norms` is the Frobenius norm of each MPS site tensor —
    # a property of the canonical-form / gauge, NOT a particle observable.
    # Kept as an honest canonical-form sanity diagnostic.
    tensor_norms = None
    if state is not None:
        tensor_norms = _safe(
            lambda: [float(np.real(np.linalg.norm(t))) for t in state.tensors]
        )

    # `occupations_n` is the real particle occupation ⟨n_k⟩ per species
    # per site, computed via state.local_expectation(site, H.n(species)).
    # Shape: { species_name: [⟨n_0⟩, ⟨n_1⟩, ...] }. None if state/H absent.
    occupations_n: dict[str, list[float]] | None = None
    H_obj = _safe(lambda: q.H)
    if state is not None and H_obj is not None:
        try:
            n_sites = int(state.N)
            species_names = [s.name for s in H_obj.species]
            occupations_n = {}
            for s in species_names:
                op = H_obj.n(s)
                occupations_n[s] = [
                    float(np.real(state.local_expectation(k, op)))
                    for k in range(n_sites)
                ]
        except (AttributeError, KeyError, ValueError, TypeError):
            occupations_n = None

    # learnable_params is a list[str] of parameter names; map each to its
    # current value on the Hamiltonian. A missing param yields None rather
    # than aborting the whole dict.
    params = None
    names = _safe(lambda: list(q.cfg.learnable_params))
    if names is not None:
        params = {
            name: _safe(lambda n=name: float(q.H.get_param(n)))
            for name in names
        }

    return {
        "energy": _safe(lambda: float(np.real(q._last_energy))),
        "pred_errors": _safe(lambda: dict(q._last_errors)),
        "params": params,
        "bond_dims": bond_dims,
        "entropies": entropies,
        "occupations_n": occupations_n,
        "tensor_norms": tensor_norms,
        "step": _safe(lambda: int(q._step)),
    }


# ---- matrix product state ----------------------------------------------------

def snapshot_mps(mps: Any) -> dict:
    """Snapshot an `MPS`: bond dims and per-bond entanglement entropy."""
    bond_dims = _safe(lambda: list(mps.bond_dimensions()))

    entropies = None
    n = _safe(lambda: int(mps.N))
    if n:
        # entanglement_entropy uses 0-indexed bonds: valid 0..N-2.
        entropies = [
            _safe(lambda c=c: float(mps.entanglement_entropy(c)))
            for c in range(n - 1)
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

    # Cheap scalar series for line-plot consumers (e.g. the Manim scene):
    # the mean absolute coupling strength. 0.0 when there are no couplings.
    if couplings:
        mean_abs_coupling = float(
            np.mean([abs(v) for v in couplings.values()])
        )
    else:
        mean_abs_coupling = 0.0

    return {
        "fields": fields,
        "couplings": couplings,
        "mean_abs_coupling": mean_abs_coupling,
        "step": _safe(lambda: int(mf._step)),
    }


# ---- hamiltonian -------------------------------------------------------------

def snapshot_hamiltonian(H: Any) -> dict:
    """Snapshot a `Hamiltonian`: real generative-model coefficients per §3.3.4.

    Surfaces the per-species one-site coefficients (`bare_mass`, `kinetic`,
    `quartic`, `source`), the cross-species `density_couplings` g_{ab} and
    `yukawa_couplings` λ_{ab} dicts, the scalar `curvature_xi`, and the
    1D per-site `curvature` field R(x_k) (NOT a 2D matrix; the prior name
    misled the panel into treating it as a coupling map).
    """
    species_names = _safe(lambda: [s.name for s in H.species]) or []

    per_species: dict[str, dict[str, float]] | None = None
    cfg = _safe(lambda: H.cfg)
    if cfg is not None and species_names:
        per_species = {}
        for s in cfg.species:
            per_species[s.name] = {
                "bare_mass": float(s.bare_mass),
                "kinetic": float(s.kinetic),
                "quartic": float(s.quartic),
                "source": float(s.source),
            }

    def _pair_dict(d: Any) -> dict[str, float] | None:
        if d is None:
            return None
        out: dict[str, float] = {}
        for key, val in dict(d).items():
            if isinstance(key, tuple) and len(key) == 2:
                a, b = key
                out[f"{a}|{b}"] = float(val)
            else:
                out[str(key)] = float(val)
        return out

    density_couplings = _safe(lambda: _pair_dict(cfg.density_couplings))
    yukawa_couplings = _safe(lambda: _pair_dict(cfg.yukawa_couplings))
    curvature_xi = _safe(lambda: float(cfg.curvature_xi))

    return {
        "n_sites": _safe(lambda: int(H.N)),
        "d_local": _safe(lambda: int(H.d_local)),
        "species_dims": _safe(lambda: list(H.species_dims)),
        "species": species_names,
        "per_species": per_species,
        "density_couplings": density_couplings,
        "yukawa_couplings": yukawa_couplings,
        "curvature_xi": curvature_xi,
        # 1D per-site R(x_k), shape (N,). Honest representation: a strip,
        # not a 2D coupling matrix.
        "curvature": _safe(lambda: np.asarray(H.curvature).tolist()),
    }


# ---- MERA --------------------------------------------------------------------

def snapshot_mera(m: Any) -> dict:
    """Snapshot a `MERA`: leaf count, per-layer bond dims, cut entropies.

    `iso_residuals` is a per-layer mean of `||W W† − I||_F` over each
    layer's 2->1 isometries (W has shape `(d_out, d_in_left, d_in_right)`,
    reshaped to `(d_out, d_in_left * d_in_right)` so the isometry condition
    is `W W† = I_{d_out}`). It's a cheap drift indicator for variational
    MERA training.
    """
    leaves = _safe(lambda: m.leaves)
    n_leaves = len(leaves) if leaves is not None else None

    entropies = None
    if n_leaves:
        # entanglement_entropy uses 0-indexed cuts: valid 0..n_leaves-2.
        entropies = [
            _safe(lambda c=c: float(m.entanglement_entropy(c)))
            for c in range(n_leaves - 1)
        ]

    iso_residuals = None
    isos = _safe(lambda: m.isometries)
    if isos is not None:
        try:
            layer_means = []
            for layer_isos in isos:
                errs = []
                for W in layer_isos:
                    W2 = np.asarray(W)
                    d_out = W2.shape[0]
                    M = W2.reshape(d_out, -1)
                    gram = M @ M.conj().T
                    errs.append(float(np.linalg.norm(
                        gram - np.eye(d_out), ord='fro')))
                layer_means.append(float(np.mean(errs)) if errs else 0.0)
            iso_residuals = layer_means
        except (ValueError, IndexError, AttributeError):
            iso_residuals = None

    return {
        "n_leaves": n_leaves,
        "layer_dims": _safe(lambda: list(m.layer_dims)),
        "bond_dims": _safe(lambda: list(m.bond_dimensions())),
        "entropies": entropies,
        "iso_residuals": iso_residuals,
    }


# ---- variational quantum circuit ---------------------------------------------

def snapshot_vqc(vqc: Any) -> dict:
    """Snapshot a `QuantumGenerativeMap` or `QuantumConvMap`.

    `QuantumGenerativeMap` stores trainable angles on `theta` (shape
    n_layers x n_qubits x 2). `QuantumConvMap` wraps one in `.qmap` and adds
    a classical `.bias`; we read through to the inner map for the angles.
    """
    # QuantumConvMap delegates to an inner QuantumGenerativeMap (`.qmap`).
    inner = _safe(lambda: vqc.qmap)
    src = inner if inner is not None else vqc

    return {
        "theta": _safe(lambda: np.asarray(src.theta).tolist()),
        "n_qubits": _safe(lambda: int(src.n_qubits)),
        "n_layers": _safe(lambda: int(src.n_layers)),
        "input_scale": _safe(lambda: float(src.input_scale)),
        "bias": _safe(lambda: np.asarray(vqc.bias).tolist()),
    }


# ---- logic encoder -----------------------------------------------------------

def snapshot_logic(enc: Any, state: Any = None) -> dict:
    """Snapshot a logic Hamiltonian (e.g. `EvalHamiltonian`).

    These classes expose the site count as `N` and the rule-term list as
    `terms`; there is no `n_sites`/`d_local`/`state` attribute.

    Each `EvalTerm` in `enc.terms` is a frozen dataclass with `rule_id`
    (str), `site` (int) and `arity` (int) fields; we emit those verbatim as
    JSON-ready dicts so a panel can lay out the real AST/term structure.

    When `state` is supplied, the per-term `residuals` list (one float per
    term, parallel to `terms`) and the scalar `total_energy` are emitted so
    the panel can show relaxation progress.
    """
    def _terms() -> list:
        out = []
        for t in enc.terms:
            out.append({
                "rule_id": str(t.rule_id),
                "site": int(t.site),
                "arity": int(t.arity),
            })
        return out

    residuals: list[float] | None = None
    total_energy: float | None = None
    # Per-bond entanglement entropy on the logic-encoded MPS state. This is
    # the *real* binder-entanglement signal per §1.1 / §8 / §10.1: a binder's
    # use→declaration path is realized as bond entropy along that path. The
    # previous panel rendered evenly-spaced arcs whose opacity was a function
    # of three global λ scalars; those arcs had no relationship to any
    # binder pair (see deviation D-4). Bond entropies, by contrast, ARE the
    # observable consequence of binding-as-entanglement.
    bond_entropies: list[float | None] | None = None
    if state is not None:
        try:
            res = enc.residuals(state)
            residuals = [float(res[(t.rule_id, t.site)]) for t in enc.terms]
            total_energy = float(sum(residuals))
        except (AttributeError, KeyError, ValueError, TypeError):
            residuals = None
            total_energy = None
        n_state = _safe(lambda: int(state.N))
        if n_state:
            bond_entropies = []
            for c in range(n_state - 1):
                bond_entropies.append(
                    _safe(lambda c=c: float(state.entanglement_entropy(c)))
                )

    return {
        "n_sites": _safe(lambda: int(enc.N)),
        "term_count": _safe(lambda: len(enc.terms)),
        "terms": _safe(_terms) or [],
        "lambda_beta": _safe(lambda: float(enc.lambda_beta)),
        "lambda_arith": _safe(lambda: float(enc.lambda_arith)),
        "lambda_if": _safe(lambda: float(enc.lambda_if)),
        "residuals": residuals,
        "total_energy": total_energy,
        "bond_entropies": bond_entropies,
    }
