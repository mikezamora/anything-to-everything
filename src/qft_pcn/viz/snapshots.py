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


# ---- MERA imag-time relaxation (§10.10 induction-theorem demo) ---------------


def _pretty_extended_ty(ty: Any) -> str:
    """Renderer for the extended type vocabulary (TNat / TList / TEq /
    TProp) that ``logic.ast._pretty_ty`` does not yet handle."""
    from ..logic import ast as _ast
    if isinstance(ty, _ast.TInt):
        return "Int"
    if isinstance(ty, _ast.TBool):
        return "Bool"
    if isinstance(ty, _ast.TNat):
        return "Nat"
    if isinstance(ty, _ast.TArrow):
        return f"{_pretty_extended_ty(ty.src)} -> {_pretty_extended_ty(ty.dst)}"
    if isinstance(ty, _ast.TList):
        return f"List ({_pretty_extended_ty(ty.elem)})"
    if isinstance(ty, _ast.TEq):
        return "Eq"
    if isinstance(ty, _ast.TProp):
        return "Prop"
    return f"<{type(ty).__name__}>"


def _pretty_extended_ast(node: Any) -> str:
    """Renderer for the extended-calculus AST including the new nodes
    (Forall / Eq / Nat / Zero / Succ / NatLit / Cons / Nil) that the
    surface parser accepts but `logic.ast.pretty` does not yet handle.

    Defensive: any unrecognised sub-node renders as ``<ClassName>`` rather
    than raising, so a decode artefact never crashes the snapshot pipe.
    """
    from ..logic import ast as _ast
    p = _pretty_extended_ast
    if isinstance(node, _ast.Var):
        return node.name
    if isinstance(node, _ast.IntLit):
        return str(node.val)
    if isinstance(node, _ast.BoolLit):
        return "true" if node.val else "false"
    if isinstance(node, _ast.Zero):
        return "Zero"
    if isinstance(node, _ast.Succ):
        return f"Succ ({p(node.arg)})"
    if isinstance(node, _ast.NatLit):
        return f"NatLit {node.val}"
    if isinstance(node, _ast.Nil):
        return "Nil"
    if isinstance(node, _ast.Cons):
        return f"Cons ({p(node.head)}) ({p(node.tail)})"
    if isinstance(node, _ast.Eq):
        return f"Eq ({p(node.lhs)}) ({p(node.rhs)})"
    if isinstance(node, _ast.Forall):
        return (f"forall {node.param}:{_pretty_extended_ty(node.param_ty)}. "
                f"{p(node.body)}")
    if isinstance(node, _ast.Lam):
        return (f"\\{node.param}:{_pretty_extended_ty(node.param_ty)}. "
                f"{p(node.body)}")
    if isinstance(node, _ast.App):
        return f"({p(node.fn)}) ({p(node.arg)})"
    if isinstance(node, _ast.If):
        return f"if {p(node.cond)} then {p(node.then_b)} else {p(node.else_b)}"
    if isinstance(node, _ast.Bin):
        return f"({p(node.lhs)}) {node.op} ({p(node.rhs)})"
    if isinstance(node, _ast.Fix):
        return (f"fix {node.param}:{_pretty_extended_ty(node.param_ty)}. "
                f"{p(node.body)}")
    return f"<{type(node).__name__}>"


def snapshot_mera_relax(H: Any, state: Any, meta: Any) -> dict:
    """Snapshot a MERA imag-time relaxation step (§10.10 induction-theorem).

    Emits per-term residuals (keyed by ``rule_id`` and ``site`` where
    ``site = term.node``), the scalar ``total_energy``, the per-layer bond
    dimensions, the leaf count, the indices of the Forall-protected leaves
    (kept frozen by ``mera_trotter_step`` to enforce ∀-quantification —
    the load-bearing §1.1 / §10.10 invariant), and an AST text round-trip
    decoded from the live MERA leaves.

    Defensive: missing/optional attributes yield ``None`` rather than
    raising.
    """
    terms_list = _safe(lambda: list(H.terms)) or []

    residuals: list[dict] | None = None
    total_energy: float | None = _safe(lambda: float(H.total_energy(state)))
    try:
        res = H.residuals(state)
        residuals = []
        for t in terms_list:
            key = (t.rule_id, t.node)
            residuals.append({
                "rule_id": str(t.rule_id),
                "site": int(t.node),
                "value": float(res[key]),
            })
    except (AttributeError, KeyError, ValueError, TypeError):
        residuals = None

    n_leaves = _safe(lambda: int(state.N))
    layer_bond_dims = _safe(lambda: list(state.layer_dims))

    forall_protected_leaves: list[int] = []
    try:
        s = meta.forall_protected_leaves or set()
        forall_protected_leaves = sorted(int(i) for i in s)
    except (AttributeError, TypeError):
        forall_protected_leaves = []

    ast_text: str | None = None
    try:
        from ..logic.mera_decoder import decode_mera
        decoded = decode_mera(state, meta)
        ast_text = _pretty_extended_ast(decoded.ast)
    except Exception:
        # decode_mera can raise DecodeError after a relaxation step has
        # transiently driven a leaf out of its node's expected basis; the
        # snapshot must NOT crash the simulation loop. Surface as None.
        ast_text = None

    return {
        "total_energy": total_energy,
        "residuals": residuals,
        "n_leaves": n_leaves,
        "layer_bond_dims": layer_bond_dims,
        "forall_protected_leaves": forall_protected_leaves,
        "ast_text": ast_text,
        "step": _safe(lambda: int(state._step)),
    }


# ---- bridge RunResult --------------------------------------------------------


def snapshot_run_result(result: Any) -> dict:
    """Snapshot a bridge `RunResult`: MPS + Hamiltonian miniatures + scalars.

    Surfaces the M2 composition-layer additions (`meta`, `ground_state`,
    `solved_ast`, `hamiltonian`, `trotter_steps`) for live inspection.
    Per-field accessors are defensive — a `None` substrate field yields
    `None` in the snapshot rather than raising.
    """
    mps_snap: dict | None = None
    try:
        gs = getattr(result, "ground_state", None)
        if gs is not None:
            mps_snap = snapshot_mps(gs)
    except (AttributeError, KeyError, ValueError, TypeError):
        mps_snap = None

    ham_snap: dict | None = None
    try:
        H = getattr(result, "hamiltonian", None)
        if H is not None:
            ham_snap = snapshot_hamiltonian(H)
    except (AttributeError, KeyError, ValueError, TypeError):
        ham_snap = None

    solved_ast_text: str | None = None
    try:
        ast = getattr(result, "solved_ast", None)
        if ast is not None:
            solved_ast_text = _pretty_extended_ast(ast)
    except (AttributeError, KeyError, ValueError, TypeError):
        solved_ast_text = None

    meta_n_leaves: int | None = None
    try:
        meta = getattr(result, "meta", None)
        if meta is not None:
            meta_n_leaves = int(getattr(meta, "n_nodes", 0)) or None
    except (AttributeError, TypeError, ValueError):
        meta_n_leaves = None

    return {
        "mps": mps_snap,
        "hamiltonian": ham_snap,
        "trotter_steps": _safe(lambda: int(result.trotter_steps)),
        "energy": _safe(lambda: float(result.energy)),
        "converged": _safe(lambda: bool(result.converged)),
        "solved_ast_text": solved_ast_text,
        "meta_n_leaves": meta_n_leaves,
    }


# ---- PCN-side: hierarchical fields stack -----------------------------------

def snapshot_pcn_fields(net: Any) -> dict:
    """Snapshot the full PCN layer stack (Phi/E/Pi per layer)."""
    layers_out = []
    for layer in _safe(lambda: net.layers) or []:
        layers_out.append({
            "phi": _grid(_safe(lambda l=layer: l.phi.values)),
            "E":   _grid(_safe(lambda l=layer: l.error.values)),
            "Pi":  _grid(_safe(lambda l=layer: l.precision.pi)),
            "channels": _safe(lambda l=layer: l.phi.channels),
        })
    return {
        "layers": layers_out,
        "step": _safe(lambda: int(net._step)),
    }


# ---- PCN-side: free-energy / learning dynamics -----------------------------

def snapshot_pcn_dynamics(net: Any) -> dict:
    """Snapshot PCN free-energy aggregates.

    Reads the substrate's `QFTPCNLayer.free_energy(phi_below, kappa_R)`.
    For the bottom layer phi_below is a zero observation array (matching
    the run-time pattern); for higher layers it is the layer-below's phi.
    """
    layers = _safe(lambda: net.layers) or []
    manifold = _safe(lambda: net.manifold)
    kappa_R = _safe(lambda: float(net.cfg.kappa_R))

    per_layer_F: list[float | None] = []
    per_layer_kl: list[float | None] = []
    per_layer_e_norm: list[float | None] = []
    per_layer_pi_mean: list[float | None] = []

    if layers and manifold is not None and kappa_R is not None:
        zero_obs = None
        try:
            c0 = layers[0].phi.channels
            zero_obs = np.zeros((c0, manifold.nx, manifold.ny))
        except (AttributeError, IndexError):
            zero_obs = None
        below = zero_obs
        for layer in layers:
            f = _safe(lambda l=layer, b=below:
                      float(l.free_energy(b, kappa_R))) \
                if below is not None else None
            per_layer_F.append(f)
            kl = _safe(lambda l=layer, b=below:
                       float(l.kl_divergence(b))) \
                if below is not None else None
            per_layer_kl.append(kl)
            per_layer_e_norm.append(_safe(
                lambda l=layer: float(np.linalg.norm(l.error.values))))
            per_layer_pi_mean.append(_safe(
                lambda l=layer: float(np.mean(l.precision.pi))))
            below = _safe(lambda l=layer: l.phi.values)

    total_F = None
    if per_layer_F and all(v is not None for v in per_layer_F):
        total_F = float(sum(per_layer_F))

    return {
        "total_free_energy": total_F,
        "per_layer_free_energy": per_layer_F,
        "per_layer_kl": per_layer_kl,
        "per_layer_e_norm": per_layer_e_norm,
        "per_layer_pi_mean": per_layer_pi_mean,
        "n_layers": len(layers),
        "step": _safe(lambda: int(net._step)),
    }


# ---- PCN-side: QFT <-> PCN coupling bridge ---------------------------------

def snapshot_pcn_coupling(net: Any, qpcn: Any = None) -> dict:
    """Snapshot the PCN <-> QFT coupling state.

    PCN -> QFT: bottom-layer error field's stress-energy magnitude.
    QFT -> PCN: mean curvature (the geometry the metric is currently in).
    Optional qpcn argument lets the panel surface QFT operator expectations
    feeding back into PCN observations; if absent the QFT->PCN arrow is
    rendered with curvature only.
    """
    manifold = _safe(lambda: net.manifold)
    layers = _safe(lambda: net.layers) or []
    kappa_R = _safe(lambda: float(net.cfg.kappa_R))

    mean_abs_stress = None
    if manifold is not None and layers:
        e0 = _safe(lambda: layers[0].error.values)
        if e0 is not None:
            ses = _safe(lambda: manifold.stress_energy(e0))
            if ses is not None:
                t_xx, t_xy, t_yy = ses
                mean_abs_stress = float(np.mean(
                    np.abs(t_xx) + np.abs(t_xy) + np.abs(t_yy)) / 3.0)

    mean_abs_ricci = _safe(
        lambda: float(np.abs(net.manifold.ricci_scalar()).mean())
    )

    qpcn_observable_energy = None
    if qpcn is not None:
        qpcn_observable_energy = _safe(
            lambda: float(np.real(qpcn._last_energy)))

    return {
        "kappa_R": kappa_R,
        "mean_abs_stress_energy": mean_abs_stress,
        "mean_abs_ricci": mean_abs_ricci,
        "qpcn_observable_energy": qpcn_observable_energy,
        "step": _safe(lambda: int(net._step)),
    }
