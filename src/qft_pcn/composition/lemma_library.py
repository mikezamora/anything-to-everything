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

# FINGERPRINT_DIM: §4.3 truncation parameter for the leaf-bond Gram
# spectrum. ``structural_fingerprint`` computes the eigenvalues of the
# leaf-Gram matrix (``G[i,j] = <v_i|v_j>`` over leaf vectors), sorts
# them in descending magnitude, and pads/truncates to this length. The
# value 32 is calibrated against M1 leaf counts: the M1 encoder
# (``logic.mera_encoder.encode_mera``) produces MERAs with leaf counts
# bounded above by the AST node count, which for the M1 acceptance
# corpus rarely exceeds 16. A 32-dim ceiling gives headroom for M2/M3
# multi-statement programs (leaf count scales with statement count via
# the §5 leaf-bond fusion) while keeping the fingerprint vector small
# enough that the §12.18 library covariance ``C`` (FINGERPRINT_DIM x
# FINGERPRINT_DIM) diagonalizes in microseconds. If a future encoder
# generates programs with >32 leaves, increasing this constant is the
# correct knob -- the §4.3 distance metric and the §10.8 index are
# both length-agnostic.
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
    """A cached proof object (spec §3.2).

    ``tier`` is a string literal in
    ``{"core", "dynamic", "primitive", "consolidated"}``:

    * ``"core"`` — foundational lemma, immune to pruning per spec §3.3.
    * ``"dynamic"`` — wake-phase solved problem; the default.
    * ``"primitive"`` — sleep-phase :class:`CanonicalPrimitive` persisted by
      :meth:`LemmaLibraryAdapter._save_primitive`; tensor-only abstraction
      (spec §5.4). Filtered out of consolidation walks.
    * ``"consolidated"`` — replacement ``L'`` produced by
      :meth:`LemmaLibraryAdapter.replace` (D35). Inherits the parent
      lemma's ``proposition_type`` and delegates bulk substructure to a
      promoted primitive.

    The field replaces the prior sidecar ``LemmaLibraryAdapter._tiers``
    map (D19 EXTENSIONS resolution): the tier round-trips through
    save/load on the underlying ``.npz`` so §3.3 core-immunity is
    enforceable from the persisted record itself.
    """
    lemma_id: str
    proposition_type: str
    mera_tensors: MeraTensorBundle
    encoding_meta: MeraEncodingMeta
    derivation: DerivationMetadata
    fingerprint: np.ndarray
    tier: str = "dynamic"


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


import json
import hashlib
import typing
from dataclasses import asdict, fields
from src.qft_pcn.composition.errors import LemmaHashCollision, LemmaNotFound


_META_INT_KEY_DICTS = (
    "site_to_ast_path", "binder_leaves", "use_to_binder",
    "nested_type_index", "children_of_node",
)


def _jsonable(v):
    """Recursively coerce tuples / sets to lists for JSON; pass through scalars.

    Sets serialize as sorted lists (deterministic round-trip); tuples become
    lists; dicts have str-keyed values recursed. The set branch must precede
    any iterable handling because Python's `set` is iterable but unordered --
    sorting keeps the on-disk form stable across processes. We call
    ``sorted(v)`` directly: every current ``MeraEncodingMeta`` set field is
    ``set[int]`` (orderable). A future non-orderable set element type should
    raise ``TypeError`` here loudly -- a silent ``key=repr`` coercion would
    hide a real schema-vs-loader mismatch (the ``_meta_from_json`` set-field
    loader assumes ``int`` elements).
    """
    if isinstance(v, set):
        return [_jsonable(x) for x in sorted(v)]
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(val) for k, val in v.items()}
    return v


# Autodetected from MeraEncodingMeta type hints; restored as
# ``set(int(x) for x in ...)`` in ``_meta_from_json``. Single source of truth:
# adding a ``set[...]``-typed field to MeraEncodingMeta automatically extends
# the round-trip without touching a registry. (Caveat: the loader coerces
# elements via ``int(x)`` -- if a future field uses ``set[tuple[...]]`` or
# similar, the loader needs an update; the introspection test pins this
# assumption so the drift surfaces as a failing contract test.)
_META_SET_FIELDS: tuple[str, ...] = tuple(
    name for name, hint in typing.get_type_hints(MeraEncodingMeta).items()
    if typing.get_origin(hint) is set
)


def _ty_to_json(ty) -> dict:
    """Recursive Ty -> JSON-safe dict. Tagged union shape.

    Covers the canonical Ty subclasses the encoder may park in
    ``nested_type_index``. Unknown subclasses raise loudly so that a
    future Ty kind cannot silently round-trip as ``None`` (the prior
    best-effort ``json.dumps`` filter silently dropped every Ty instance,
    so cross-session ``forall xs:List Bool`` loaded back as
    ``forall xs:List Nat`` -- a silent data-loss bug).
    """
    from src.qft_pcn.logic.ast import (
        TNat, TBool, TInt, TProp, TList, TArrow,
    )
    if isinstance(ty, TNat):
        return {"kind": "TNat"}
    if isinstance(ty, TBool):
        return {"kind": "TBool"}
    if isinstance(ty, TInt):
        return {"kind": "TInt"}
    if isinstance(ty, TProp):
        return {"kind": "TProp"}
    if isinstance(ty, TList):
        return {"kind": "TList", "elem": _ty_to_json(ty.elem)}
    if isinstance(ty, TArrow):
        return {"kind": "TArrow",
                "src": _ty_to_json(ty.src),
                "dst": _ty_to_json(ty.dst)}
    raise TypeError(f"unknown Ty subclass: {type(ty).__name__}")


def _ty_from_json(d: dict):
    """Inverse of ``_ty_to_json``. Unknown ``kind`` raises ``TypeError``."""
    from src.qft_pcn.logic.ast import (
        TNat, TBool, TInt, TProp, TList, TArrow,
    )
    kind = d["kind"]
    if kind == "TNat":
        return TNat()
    if kind == "TBool":
        return TBool()
    if kind == "TInt":
        return TInt()
    if kind == "TProp":
        return TProp()
    if kind == "TList":
        return TList(elem=_ty_from_json(d["elem"]))
    if kind == "TArrow":
        return TArrow(src=_ty_from_json(d["src"]),
                      dst=_ty_from_json(d["dst"]))
    raise TypeError(f"unknown Ty kind tag: {kind!r}")


def _meta_to_json(meta: MeraEncodingMeta) -> str:
    """Serialize MeraEncodingMeta to JSON. `layout` is dropped (non-JSON,
    reconstructible by consumers from species_of_leaf + node_of_leaf via
    M1 helpers). `nested_type_index` values are ``Ty`` dataclasses; they
    are routed through ``_ty_to_json`` so the full type structure
    survives save/load (prior best-effort ``json.dumps`` filter silently
    dropped every Ty instance -- see _ty_to_json docstring)."""
    d: dict = {}
    for f in fields(meta):
        if f.name == "layout":
            continue
        v = getattr(meta, f.name)
        if f.name == "nested_type_index":
            d[f.name] = {str(k): _ty_to_json(val) for k, val in v.items()}
        elif isinstance(v, dict):
            d[f.name] = {str(k): _jsonable(val) for k, val in v.items()}
        elif isinstance(v, (list, tuple, set)):
            d[f.name] = _jsonable(v)
        else:
            d[f.name] = v
    return json.dumps(d)


def _meta_from_json(s: str) -> MeraEncodingMeta:
    d = json.loads(s)
    # Coerce int-keyed dicts back to int keys.
    for key in _META_INT_KEY_DICTS:
        if key in d and isinstance(d[key], dict):
            coerced: dict = {}
            for k, val in d[key].items():
                ik = int(k)
                if key == "site_to_ast_path":
                    coerced[ik] = tuple(val)
                elif key == "nested_type_index":
                    # Values are tagged-union dicts emitted by _ty_to_json;
                    # restore the original Ty dataclass.
                    coerced[ik] = _ty_from_json(val)
                else:
                    coerced[ik] = val
            d[key] = coerced
    # hole_regions / witness_node_ranges are lists-of-tuples in spirit;
    # JSON gives lists-of-lists. Coerce inner lists back to tuples where
    # the original carried tuples.
    if "witness_node_ranges" in d:
        d["witness_node_ranges"] = [
            tuple(x) if isinstance(x, list) else x
            for x in d["witness_node_ranges"]
        ]
    # Restore set-typed fields (Gap D): JSON only carries lists, so the
    # serializer wrote sorted lists -- coerce them back to sets of ints
    # so callers that depend on ``in``-test / set-algebra semantics
    # (e.g. evolution drivers consulting forall_protected_leaves) see the
    # exact type the encoder produced.
    for key in _META_SET_FIELDS:
        if key in d and isinstance(d[key], list):
            d[key] = set(int(x) for x in d[key])
    # layout is reconstructible by consumers from species/node info; set None.
    d["layout"] = None
    return MeraEncodingMeta(**d)


def _deriv_from_dict(d: dict) -> DerivationMetadata:
    return DerivationMetadata(
        hamiltonian_id=d["hamiltonian_id"],
        residual_energy=float(d["residual_energy"]),
        energy_gap=float(d["energy_gap"]),
        trotter_steps=int(d["trotter_steps"]),
        assumptions=tuple(d["assumptions"]),
        lemma_deps=tuple(d["lemma_deps"]),
        conditional=bool(d["conditional"]),
        source_run_id=d["source_run_id"],
    )


def _n_leaves_L(lemma: Lemma) -> int:
    """Logical-leaf count used by cost-tier indexing (spec §4.2).

    Prefers an explicit ``:N`` suffix in the lemma_id if present (so
    callers can persist multiple sketch widths against the same AST);
    otherwise falls back to ``5 * n_nodes`` from the encoding meta.
    """
    tail = lemma.lemma_id.rsplit(":", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return 5 * lemma.encoding_meta.n_nodes


class LemmaLibrary:
    """File-backed, append-only store of lemmas with three-tier indexing
    (spec §4): by proposition type, by structural fingerprint, by
    derivation cost.

    The store persists each lemma as a single `.npz` per lemma_id; the
    bundle tensors (leaves, intra/inter disentanglers, isometries, top,
    layer_dims) plus JSON-encoded sidecars for encoding_meta and
    derivation are bundled together. A `manifest.json` at the library
    root maps lemma_id -> light index entry for the three-tier lookups
    (type, cost, fingerprint).
    """

    def __init__(self, root, eps_compress: float = 1e-9):
        self.root = Path(root)
        self.eps_compress = eps_compress
        (self.root / "lemmas").mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._manifest: dict = {}
        if self._manifest_path.exists():
            self._manifest = json.loads(self._manifest_path.read_text())

    def _flush_manifest(self) -> None:
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def _path(self, lemma_id: str) -> Path:
        h = hashlib.sha1(lemma_id.encode()).hexdigest()
        return self.root / "lemmas" / f"{h}.npz"

    def save(self, lemma: Lemma) -> None:
        existing = self._manifest.get(lemma.lemma_id)
        if existing is not None:
            if existing["proposition_type"] != lemma.proposition_type:
                raise LemmaHashCollision(
                    f"{lemma.lemma_id} already maps to a different lemma")
            return  # append-only: identical re-save is a no-op
        path = self._path(lemma.lemma_id)
        b = lemma.mera_tensors
        if b.top is None:
            raise ValueError(
                f"lemma {lemma.lemma_id} bundle.top is None; cannot persist")
        arrs: dict[str, np.ndarray] = {
            "n_leaves": np.array(b.n_leaves),
            "leaf_dim": np.array(b.leaf_dim),
            "n_layers": np.array(b.n_layers),
            "n_disent": np.array(len(b.disentanglers)),
            "n_iso": np.array(len(b.isometries)),
            "n_inter": np.array(len(b.inter_disentanglers)),
            "layer_dims": np.array(b.layer_dims, dtype=np.int64),
            "top": np.asarray(b.top),
            "fingerprint": np.asarray(lemma.fingerprint),
            "meta_json": np.array(_meta_to_json(lemma.encoding_meta)),
            "deriv_json": np.array(json.dumps(asdict(lemma.derivation))),
            "proposition_type": np.array(lemma.proposition_type),
            "tier": np.array(lemma.tier),
        }
        for i, v in enumerate(b.leaf_vectors):
            arrs[f"leaf_{i}"] = np.asarray(v)
        for i, d in enumerate(b.disentanglers):
            arrs[f"disent_{i}"] = np.asarray(d)
        for i, w in enumerate(b.isometries):
            arrs[f"iso_{i}"] = np.asarray(w)
        for i, u in enumerate(b.inter_disentanglers):
            arrs[f"inter_{i}"] = np.asarray(u)
        np.savez_compressed(path, **arrs)
        self._manifest[lemma.lemma_id] = {
            "proposition_type": lemma.proposition_type,
            "trotter_steps": lemma.derivation.trotter_steps,
            "n_leaves_L": _n_leaves_L(lemma),
            # D2 (DEVIATIONS.md §8): persist source goal_id so the
            # orchestrator can do a pre-dispatch cache lookup keyed by
            # goal_id — siblings re-proving the same sub-goal hit the
            # cache instead of re-running the QPCN.
            "source_run_id": lemma.derivation.source_run_id,
        }
        self._flush_manifest()

    def load(self, lemma_id: str) -> Lemma:
        if lemma_id not in self._manifest:
            raise LemmaNotFound(lemma_id)
        z = np.load(self._path(lemma_id), allow_pickle=False)
        n = int(z["n_leaves"])
        n_disent = int(z["n_disent"])
        n_iso = int(z["n_iso"])
        n_inter = int(z["n_inter"])
        bundle = MeraTensorBundle(
            n_leaves=n,
            leaf_dim=int(z["leaf_dim"]),
            n_layers=int(z["n_layers"]),
            leaf_vectors=[np.asarray(z[f"leaf_{i}"]) for i in range(n)],
            disentanglers=[np.asarray(z[f"disent_{i}"]) for i in range(n_disent)],
            isometries=[np.asarray(z[f"iso_{i}"]) for i in range(n_iso)],
            inter_disentanglers=[
                np.asarray(z[f"inter_{i}"]) for i in range(n_inter)
            ],
            top=np.asarray(z["top"]),
            layer_dims=tuple(int(x) for x in z["layer_dims"]),
        )
        # ``tier`` was added with the D19 EXTENSIONS resolution. Old .npz
        # files (pre-D19) lack the key; fall back to ``"dynamic"`` so
        # legacy stores load without re-keying. ``z.files`` is the
        # NpzFile's archive name list.
        tier = str(z["tier"]) if "tier" in z.files else "dynamic"
        return Lemma(
            lemma_id=lemma_id,
            proposition_type=str(z["proposition_type"]),
            mera_tensors=bundle,
            encoding_meta=_meta_from_json(str(z["meta_json"])),
            derivation=_deriv_from_dict(json.loads(str(z["deriv_json"]))),
            fingerprint=np.asarray(z["fingerprint"]),
            tier=tier,
        )

    def materialize(self, lemma_id: str) -> MERA:
        return mera_from_bundle(self.load(lemma_id).mera_tensors)

    def all_ids(self) -> list[str]:
        return list(self._manifest.keys())

    def find_by_type(self, proposition_type: str) -> list[Lemma]:
        return [self.load(lid) for lid, m in self._manifest.items()
                if m["proposition_type"] == proposition_type]

    def find_by_goal_id(self, goal_id: str) -> "Lemma | None":
        """Reverse index: lookup a cached lemma by its source goal_id.

        Spec §8: a sub-goal already proven (i.e. a lemma was registered
        whose ``derivation.source_run_id`` is this ``goal_id``) must
        short-circuit the orchestrator -- siblings re-asking the same
        sub-goal hit the cache and skip the QPCN dispatch.

        Returns the most-recently-registered match (manifest is
        append-only; insertion order is preserved by dict semantics in
        CPython 3.7+). Returns ``None`` if no lemma was derived under
        the queried ``goal_id``.

        ANTI-SHORTCUT: the reverse index is keyed on
        ``derivation.source_run_id`` -- the same field that namespaces
        the content hash in ``_content_id``. This guarantees that the
        cache-hit returns the lemma the original child registered, NOT
        a structurally-identical lemma from a different goal_id (which
        would cross-contaminate L §10.11 hierarchical proofs).
        """
        if not goal_id:
            return None
        match = None
        for lid, m in self._manifest.items():
            if m.get("source_run_id") == goal_id:
                match = lid
        if match is None:
            return None
        return self.load(match)

    def cheapest_for_type(self, proposition_type: str):
        # D28: skip primitive lemmas (proposition_type starting with
        # "primitive:"). Primitives are tensor-only abstractions per spec
        # §5.4: they carry a stub MeraEncodingMeta(n_nodes=0, ...) because
        # they have no AST, which means _n_leaves_L returns 0 for them and
        # they would otherwise rank ahead of every concrete lemma of the
        # same type. Concrete-AST candidate selection must consider only
        # concrete lemmas. Primitive recovery for canonical-AST candidates
        # is a future enhancement — see EXTENSIONS.md.
        cands = [(m["n_leaves_L"], m["trotter_steps"], lid)
                 for lid, m in self._manifest.items()
                 if m["proposition_type"] == proposition_type
                 and not m["proposition_type"].startswith("primitive:")]
        if not cands:
            return None
        cands.sort()
        return self.load(cands[0][2])

    def find_similar(self, query_fingerprint, max_distance: float = 0.1):
        out = []
        for lid in self._manifest:
            lem = self.load(lid)
            d = fingerprint_distance(query_fingerprint, lem.fingerprint)
            if d <= max_distance:
                out.append((lem, d))
        out.sort(key=lambda t: t[1])
        return out

    def _drop(self, lemma_id: str) -> None:
        """Remove a lemma from the manifest + delete its on-disk .npz.

        Used by :meth:`re_evaluate_provisional` to evict a provisional
        lemma whose re-checked residual now exceeds CONJECTURE_CEILING
        (spec §8 / §6.3). Outside of provisional eviction the library is
        append-only — callers should not invoke this for housekeeping.
        """
        if lemma_id not in self._manifest:
            raise LemmaNotFound(lemma_id)
        path = self._path(lemma_id)
        if path.exists():
            path.unlink()
        del self._manifest[lemma_id]
        self._flush_manifest()

    def _rewrite_with_derivation(
        self, lemma_id: str, new_derivation: DerivationMetadata
    ) -> None:
        """Rewrite a stored lemma's derivation in place (used to flip
        ``conditional=True`` -> ``False`` when re-evaluation promotes a
        provisional lemma to ground-state status).

        The mera tensors / encoding meta / fingerprint are unchanged; the
        lemma_id (content hash) therefore stays valid.
        """
        lem = self.load(lemma_id)
        new_lem = Lemma(
            lemma_id=lem.lemma_id,
            proposition_type=lem.proposition_type,
            mera_tensors=lem.mera_tensors,
            encoding_meta=lem.encoding_meta,
            derivation=new_derivation,
            fingerprint=lem.fingerprint,
            tier=lem.tier,
        )
        # Bypass save's "append-only no-op" guard by dropping then re-saving.
        path = self._path(lemma_id)
        if path.exists():
            path.unlink()
        del self._manifest[lemma_id]
        self.save(new_lem)

    def re_evaluate_provisional(
        self,
        energy_fn: Callable[["Lemma"], float | None],
        *,
        residual_gate: float = 1e-6,
        ceiling: float = 1e-3,
    ) -> dict:
        """Walk provisional (``derivation.conditional=True``) lemmas, recompute
        the residual against the supplied ``energy_fn``, and either promote
        (residual <= ``residual_gate`` -> rewrite with ``conditional=False``)
        or drop (residual > ``ceiling`` -> evict from store). Lemmas in the
        middle band stay provisional with their residual updated.

        Implements spec §8 / §6.3 lazy re-evaluation: a provisional lemma
        was integrated as a conjecture; when sibling boundaries change (or
        an upstream solve provides a sharper Hamiltonian), the orchestrator
        invokes this hook so the cached entry is re-checked against fresh
        physics before being relied on by downstream goals.

        ``energy_fn(lemma)`` may return ``None`` to skip a lemma whose
        Hamiltonian cannot be resolved by the current caller (e.g. the
        lemma was registered under a hamiltonian_id the resolver does not
        recognize). Such lemmas are left unchanged.

        Returns a summary dict ``{"promoted": [...], "dropped": [...],
        "kept_provisional": [...], "skipped": [...]}`` of lemma_ids.

        ANTI-SHORTCUT: this is NOT a no-op. The default residual_gate /
        ceiling mirror :mod:`result_integrator` so a caller that supplies
        no thresholds gets the same gates that admitted the lemma in the
        first place.
        """
        summary: dict[str, list[str]] = {
            "promoted": [], "dropped": [], "kept_provisional": [],
            "skipped": [],
        }
        # Snapshot ids first: _drop / _rewrite_with_derivation mutate the
        # manifest while we iterate.
        for lid in list(self._manifest.keys()):
            lem = self.load(lid)
            if not lem.derivation.conditional:
                continue
            try:
                new_res = energy_fn(lem)
            except Exception:  # noqa: BLE001 -- resolver failure is per-lemma
                summary["skipped"].append(lid)
                continue
            if new_res is None:
                summary["skipped"].append(lid)
                continue
            new_res = float(new_res)
            if new_res > ceiling:
                self._drop(lid)
                summary["dropped"].append(lid)
            elif new_res <= residual_gate:
                promoted = DerivationMetadata(
                    hamiltonian_id=lem.derivation.hamiltonian_id,
                    residual_energy=new_res,
                    energy_gap=lem.derivation.energy_gap,
                    trotter_steps=lem.derivation.trotter_steps,
                    assumptions=lem.derivation.assumptions,
                    lemma_deps=lem.derivation.lemma_deps,
                    conditional=False,
                    source_run_id=lem.derivation.source_run_id,
                )
                self._rewrite_with_derivation(lid, promoted)
                summary["promoted"].append(lid)
            else:
                # Mid-band: still provisional, but record the refreshed residual.
                updated = DerivationMetadata(
                    hamiltonian_id=lem.derivation.hamiltonian_id,
                    residual_energy=new_res,
                    energy_gap=lem.derivation.energy_gap,
                    trotter_steps=lem.derivation.trotter_steps,
                    assumptions=lem.derivation.assumptions,
                    lemma_deps=lem.derivation.lemma_deps,
                    conditional=True,
                    source_run_id=lem.derivation.source_run_id,
                )
                self._rewrite_with_derivation(lid, updated)
                summary["kept_provisional"].append(lid)
        return summary


# ---- validated registration (spec §4.5) -----------------------------------


from src.qft_pcn.logic.mera_decoder import decode_mera


@dataclass(frozen=True)
class RegistrationResult:
    """Outcome of register_lemma. Total: register_lemma always returns
    one of these and never throws for a bad candidate."""
    accepted: bool
    lemma_id: str | None
    reason: str


def _validate_decoded(decoded_ast, hamiltonian) -> tuple[bool, str]:
    """Classically type-check the decoded AST (spec §4.5 step 2).

    Returns ``(ok, detail)``. The synthesis stack today exposes no
    standalone AST type-checker (only the problem-level `validate_problem`
    in `logic/synthesis/_validate.py`, which validates SynthesisProblem
    shape, not a decoded AST). Until one lands, this is a soft gate: the
    residual gate carries the weight, and tests monkeypatch this function
    to drive the rejection branch. See plan §I.7 — the
    "no-checker-available" fallback.

    ANTI-SHORTCUT (§1.6 operator-algebraic /
    memory:anti-shortcut-directive): the temptation here is to add a
    Python-side AST typecheck pass and call register_lemma "validated".
    Don't. The §1.6 contract is that lemma admission is gated by the
    *residual energy* of the converged state under the Hamiltonian, not
    by a classical type tree walk. The residual gate
    (``eps_register``, §4.5 step 1) IS the principled check; this
    function is intentionally a tautology until a tensor-network-side
    typechecker lands. Record any planned classical checker as an
    EXTENSIONS.md entry first; do not inline a syntactic AST walk here.
    """
    return True, "no-checker-available"


def _proposition_type(decoded_ast) -> str:
    """Canonical, alpha-normalized type-signature string (spec §4.2)."""
    from src.qft_pcn.logic.ast import canonical_type_string
    return canonical_type_string(decoded_ast)


def _content_id(bundle: MeraTensorBundle, proposition_type: str,
                source_run_id: str | None = None) -> str:
    """Compute content-addressed ID for a lemma bundle.

    ``source_run_id=None`` preserves the original pure content-addressing
    contract: structurally-identical lemmas dedup to one entry.
    ``source_run_id=<goal_id>`` namespaces the hash so the same proposition
    proven under different goal_ids registers as distinct entries
    (required by L §10.11 hierarchical decomposition).
    """
    h = hashlib.sha1()
    h.update(proposition_type.encode())
    # Namespace the content hash by the derivation's source_run_id so two
    # sub-proofs of the same proposition under DIFFERENT goal_ids (e.g.
    # the L §10.11 hierarchical decomposition where L1 and L2 each
    # produce a sub-QPCN proving the same theorem) register as DISTINCT
    # lemma entries. With no source_run_id, the hash falls back to pure
    # content-addressing (the prior single-goal contract).
    if source_run_id:
        h.update(b"\x00source_run_id=")
        h.update(source_run_id.encode())
    for v in bundle.leaf_vectors:
        h.update(np.ascontiguousarray(v).tobytes())
    for d in bundle.disentanglers:
        h.update(np.ascontiguousarray(d).tobytes())
    for w in bundle.isometries:
        h.update(np.ascontiguousarray(w).tobytes())
    for u in bundle.inter_disentanglers:
        h.update(np.ascontiguousarray(u).tobytes())
    if bundle.top is not None:
        h.update(np.ascontiguousarray(bundle.top).tobytes())
    if bundle.layer_dims:
        h.update(np.array(bundle.layer_dims, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def register_lemma(library: LemmaLibrary, state, meta, hamiltonian,
                   derivation: DerivationMetadata,
                   eps_register: float = 1e-8,
                   tier: str = "dynamic",
                   expected_type=None) -> RegistrationResult:
    """Validated registration (spec §4.5). Total: always returns a
    RegistrationResult, never throws for a bad candidate. Failures are
    appended to ``<library.root>/near_misses.log`` and surface via the
    returned ``reason``.

    ``expected_type``: optional :class:`logic.ast.Ty` to gate admission
    on the tensor-network typechecker (EXTENSIONS.md E11 resolution).
    When supplied, the bundle's bond structure is checked against the
    expected type via
    :func:`composition.tn_typechecker.tn_typecheck_bundle` *in addition
    to* the residual gate. The TN typechecker reads only leaf-vector
    one-hot indices + ``meta.forall_protected_leaves`` — no AST walk
    (§1.6 anti-shortcut). Default ``None`` preserves prior behavior:
    the soft ``_validate_decoded`` tautology gate runs alone."""
    near_log = library.root / "near_misses.log"

    # 1. residual gate
    if derivation.residual_energy >= eps_register:
        with near_log.open("a") as fh:
            fh.write(f"residual_too_high {derivation.residual_energy}\n")
        return RegistrationResult(False, None, "residual_too_high")

    # 2. validation pass (resolved through the module namespace so that
    # tests can monkeypatch ``L._validate_decoded`` and have register_lemma
    # see the patched function). Wrap decode_mera in try/except so the
    # function stays total per spec §4.5 — a corrupt or un-parseable
    # state is reported via the result, not propagated as a raise.
    import sys as _sys
    _mod = _sys.modules[__name__]
    try:
        decoded = decode_mera(state, meta)
    except Exception as exc:
        with near_log.open("a") as fh:
            fh.write(f"validation_failed decode_error:{exc}\n")
        return RegistrationResult(
            False, None, f"validation_failed:decode_error:{exc}")
    ast = getattr(decoded, "ast", decoded)
    ok, detail = _mod._validate_decoded(ast, hamiltonian)
    if not ok:
        with near_log.open("a") as fh:
            fh.write(f"validation_failed {detail}\n")
        return RegistrationResult(False, None, f"validation_failed:{detail}")

    # 3. proposition type, 4. fingerprint
    prop_type = _proposition_type(ast)
    fp = structural_fingerprint(state)

    # 5. compress + persist
    bundle = bundle_from_mera(state)

    # 4b. Tensor-network typecheck (EXTENSIONS.md E11): if the caller
    # supplied an ``expected_type``, gate admission on the substrate-
    # side bond-structure check. The check is bond-only (leaf one-hot
    # tags + forall_protected_leaves), not an AST walk; it complements
    # the residual gate without re-doing classical typechecking.
    if expected_type is not None:
        # local import: tn_typechecker imports Lemma from this module, so module-scope import would cycle
        from src.qft_pcn.composition.tn_typechecker import (
            tn_typecheck_bundle, TypeCheckError,
        )
        tn_verdict = tn_typecheck_bundle(bundle, meta, expected_type)
        if isinstance(tn_verdict, TypeCheckError):
            with near_log.open("a") as fh:
                fh.write(
                    f"tn_typecheck_failed {tn_verdict.kind}:"
                    f"{tn_verdict.detail}\n")
            return RegistrationResult(
                False, None,
                f"tn_typecheck_failed:{tn_verdict.kind}")
    if hamiltonian is not None:
        # MeraTypingHamiltonian / MeraEvalHamiltonian expose `.total_energy`;
        # fall back to `.energy` if a future Hamiltonian API renames it.
        def energy_fn(b: MeraTensorBundle) -> float:
            m = mera_from_bundle(b)
            if hasattr(hamiltonian, "total_energy"):
                return float(hamiltonian.total_energy(m))
            return float(hamiltonian.energy(m))
        bundle = compress_bundle(bundle, energy_fn, library.eps_compress)
    lemma_id = _content_id(
        bundle, prop_type,
        source_run_id=getattr(derivation, "source_run_id", None),
    )
    lemma = Lemma(lemma_id=lemma_id, proposition_type=prop_type,
                  mera_tensors=bundle, encoding_meta=meta,
                  derivation=derivation, fingerprint=fp, tier=tier)
    library.save(lemma)
    return RegistrationResult(True, lemma_id, "ok")
