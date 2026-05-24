"""Tensor-network-level typechecker for cached lemmas.

Resolves EXTENSIONS.md entry "Tensor-network typechecker for lemma
decode validation" (E11). The §1.6 anti-shortcut directive forbids
inlining a Python AST type-tree walk in :func:`lemma_library._validate_decoded`:
lemma admission must be gated by the *bond structure of the encoded
substrate*, not by classical name lookup on the decoded AST.

The verdict is computed by reading the encoded MERA's leaf vectors and
the encoding metadata's bookkeeping (binder leaves, use_to_binder,
forall_protected_leaves) — i.e. the same bond-bookkeeping that §1.1
identifies as the carrier of variable binding ("binding = bond
entanglement"). The decoded AST never enters the check.

Surface contract:

* :func:`tn_typecheck(lemma, expected)` returns :class:`TypeCheckOk` on
  success or :class:`TypeCheckError` on failure. Total — never raises
  for shape mismatches; callers branch on the dataclass tag.
* :func:`tn_typecheck_bundle(bundle, meta, expected)` is the substrate
  primitive (no Lemma wrapper) used by tests and by
  ``register_lemma``'s validation pass.

Supported expected types:

* :class:`TInt`, :class:`TBool`, :class:`TNat`, :class:`TList`,
  :class:`TProp` — flat atomic checks against the root node's type-leaf
  one-hot tag.
* :class:`TArrow` — checks the root is a ``Lam`` (kind leaf == KIND_LAM)
  and the type-leaf tag is the corresponding flat ARR_* tag (for
  Int/Bool combinations) or TYPE_ARR_NESTED with the full Ty matching
  in :attr:`MeraEncodingMeta.nested_type_index`.
* :class:`TPi` — checks the root is a ``Forall`` (kind leaf ==
  KIND_FORALL), the value-leaf tag matches the param type tag, AND
  that ``meta.forall_protected_leaves`` is non-empty so the bond
  structure carries the dependency. This is the load-bearing check
  for "dependent product as fiber bundle over the substrate index":
  a non-dependent ``TArrow`` with no Forall binder is *rejected*
  against a ``TPi`` query.

What is explicitly NOT done (§1.6 anti-shortcut):

* No call to ``decode_mera`` or any AST-level typechecker — the only
  inputs that touch the AST surface are the user-supplied
  ``expected: Ty``, which is computed once in :func:`ty_to_tag` and
  stays as flat integer tags / nested-Ty references thereafter.
* No name-based lookup. The verdict reads leaf vectors by their
  ``(node_index, species)`` coordinates (the same coordinates the
  encoder's ``layout.leaf_of`` writes); cross-references go through
  the bond-bookkeeping (``binder_leaves``, ``use_to_binder``,
  ``forall_protected_leaves``), not through a name table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.qft_pcn.logic.ast import (
    Ty, TInt, TBool, TArrow, TNat, TList, TProp, TEq, TPi,
)
from src.qft_pcn.logic.encoding import (
    KIND_LAM, KIND_PAD,
    TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
)
from src.qft_pcn.logic.mera_encoding import (
    LEAVES_PER_NODE, SPECIES_LEAF_OFFSET,
    KIND_FORALL, KIND_FIX,
    TYPE_NAT, TYPE_LIST, TYPE_PROP, TYPE_EQ,
)
from src.qft_pcn.logic._types import ty_to_tag
from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta
from src.qft_pcn.composition.lemma_library import (
    Lemma, MeraTensorBundle, mera_from_bundle,
)


# ---- result types ---------------------------------------------------------


@dataclass(frozen=True)
class TypeCheckOk:
    """The encoded substrate is compatible with the expected type.

    ``bond_signature`` is a short string summary of the bond features
    the typechecker consulted (kind tag at root, type-leaf tag, and
    whether ``forall_protected_leaves`` was non-empty) so a passing
    verdict carries an audit trail of what was inspected.
    """
    bond_signature: str


@dataclass(frozen=True)
class TypeCheckError:
    """Substrate-level type mismatch.

    ``kind`` is one of the constants in :data:`ERROR_KINDS` (machine-
    readable); ``detail`` is a human-readable message naming the
    leaves / tags involved. The bond-structure inspection that produced
    the verdict is named in ``detail`` so consumers can debug the
    encoding-vs-spec mismatch without rerunning the typechecker.
    """
    kind: str
    detail: str


ERROR_KINDS = (
    "empty_meta",
    "leaf_not_onehot",
    "type_tag_mismatch",
    "kind_tag_mismatch",
    "missing_nested_type_entry",
    "nested_type_mismatch",
    "pi_missing_forall_entanglement",
    "value_tag_mismatch",
    "unsupported_expected_type",
)


# ---- leaf-vector readback -------------------------------------------------


def _leaf_index(node_index: int, species: str) -> int:
    """Absolute leaf index of ``species`` for ``node_index`` (node-major)."""
    return LEAVES_PER_NODE * node_index + SPECIES_LEAF_OFFSET[species]


def _read_leaf_onehot(
    leaf_vectors: list[np.ndarray], leaf_idx: int,
    tol: float = 1e-6,
) -> Optional[int]:
    """Return the basis index a leaf points to, if it is (approximately)
    a one-hot vector; otherwise ``None``.

    The encoder writes concrete leaves as exactly one-hot 16-dim vectors
    (``_one_hot`` in ``logic/_mera_leaves.py``). Compression /
    SVD-truncation preserves rank-1 leaves to within numerical tolerance.
    A non-one-hot leaf signals a superposition state (TypeHole /
    structural hole) — those are out of scope for the deterministic
    typechecker and surface as ``leaf_not_onehot``.
    """
    if leaf_idx < 0 or leaf_idx >= len(leaf_vectors):
        return None
    v = np.asarray(leaf_vectors[leaf_idx])
    # leaf shape is (1, d_local, 1); reduce to (d_local,).
    if v.ndim == 3:
        v = v[0, :, 0]
    elif v.ndim == 2:
        v = v.reshape(-1)
    # One-hot detection: exactly one entry near 1, the rest near 0.
    mags = np.abs(v)
    if mags.size == 0:
        return None
    idx = int(np.argmax(mags))
    peak = float(mags[idx])
    rest = float(mags.sum() - peak)
    if peak < 1.0 - tol or rest > tol:
        return None
    return idx


# ---- expected-type -> substrate-feature lowering --------------------------


_ARROW_FLAT_TAGS = {TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB}
_ATOMIC_TAGS = {
    TYPE_INT, TYPE_BOOL, TYPE_NAT, TYPE_LIST, TYPE_PROP, TYPE_EQ,
}


def _expected_root_kind(expected: Ty) -> Optional[int]:
    """The substrate kind tag that the *root* leaf must carry.

    * TArrow / TPi -> a binder kind (Lam / Forall). For TArrow we expect
      KIND_LAM (term-level arrow). For TPi we expect KIND_FORALL —
      a dependent product is realised as a universally-quantified
      proposition over its src.
    * Atomic types impose no kind constraint at the root (a `42 : Int`
      lemma has kind=KIND_INT, a `true : Bool` has KIND_BOOL, etc.);
      returning ``None`` means "do not constrain root kind".
    """
    if isinstance(expected, TArrow):
        return KIND_LAM
    if isinstance(expected, TPi):
        return KIND_FORALL
    return None


# ---- substrate-level typechecker ------------------------------------------


def _check_atomic(
    type_tag: int, expected: Ty,
) -> Optional[TypeCheckError]:
    """Atomic-type tag check. Returns None on success."""
    expected_tag, _ = ty_to_tag(expected)
    if type_tag == expected_tag:
        return None
    return TypeCheckError(
        kind="type_tag_mismatch",
        detail=(
            f"root type-leaf carries tag {type_tag}; "
            f"expected {expected_tag} for {type(expected).__name__}"
        ),
    )


def _check_arrow(
    type_tag: int, expected: TArrow,
    meta: MeraEncodingMeta,
) -> Optional[TypeCheckError]:
    """Arrow check: bond product factorization on the type leaf.

    A flat arrow (Int/Bool -> Int/Bool) writes a single ARR_II/IB/BI/BB
    tag on the root type leaf. A higher-order arrow writes
    TYPE_ARR_NESTED and stores the full Ty in
    ``meta.nested_type_index[0]``. Both forms are substrate-side
    factorizations — the type leaf is the canonical bond carrying the
    function-type product.
    """
    expected_tag, expected_nested = ty_to_tag(expected)
    if expected_tag != TYPE_ARR_NESTED:
        if type_tag != expected_tag:
            return TypeCheckError(
                kind="type_tag_mismatch",
                detail=(
                    f"root type-leaf carries tag {type_tag}; "
                    f"expected flat arrow tag {expected_tag} "
                    f"for {expected!r}"
                ),
            )
        return None
    # Nested arrow: type-leaf must be TYPE_ARR_NESTED and the meta side
    # table must carry the matching full Ty at the root node.
    if type_tag != TYPE_ARR_NESTED:
        return TypeCheckError(
            kind="type_tag_mismatch",
            detail=(
                f"root type-leaf carries tag {type_tag}; expected "
                f"TYPE_ARR_NESTED ({TYPE_ARR_NESTED}) for nested arrow "
                f"{expected!r}"
            ),
        )
    full = meta.nested_type_index.get(0)
    if full is None:
        return TypeCheckError(
            kind="missing_nested_type_entry",
            detail=(
                "root carries TYPE_ARR_NESTED but meta.nested_type_index"
                " has no entry for node 0"
            ),
        )
    if full != expected_nested:
        return TypeCheckError(
            kind="nested_type_mismatch",
            detail=(
                f"nested_type_index[0] = {full!r}; "
                f"expected {expected_nested!r}"
            ),
        )
    return None


def _check_pi(
    leaf_vectors: list[np.ndarray], meta: MeraEncodingMeta,
    expected: TPi,
) -> Optional[TypeCheckError]:
    """Dependent product: Forall binder + entangled fiber bundle.

    Substrate signature for ``Π(x:src). dst``:

    1. Root kind leaf must be KIND_FORALL (already checked by the
       caller via ``_expected_root_kind``).
    2. Root value leaf must carry the flat tag for ``src`` — the
       encoder writes the binder's param_ty into the value leaf
       (``_mera_leaves.node_leaf_vectors`` for KIND_FORALL).
    3. ``meta.forall_protected_leaves`` must be non-empty: the
       Forall's own bid leaf plus every bound-Var's five species
       leaves are flagged as fiber-bundle indices over the substrate
       (``_collect_forall_protected_leaves`` in
       ``logic/mera_encoder.py``). An empty set means there is no
       dependency on the bound variable — the lemma is a vacuous
       quantification, NOT a genuine Π type.
    """
    src_tag, _ = ty_to_tag(expected.src)
    root_value_leaf = _leaf_index(0, "value")
    actual_value = _read_leaf_onehot(leaf_vectors, root_value_leaf)
    if actual_value is None:
        return TypeCheckError(
            kind="leaf_not_onehot",
            detail=(
                f"root value leaf (idx {root_value_leaf}) is not a "
                f"one-hot vector; cannot read param-type tag for "
                f"TPi check"
            ),
        )
    if actual_value != src_tag:
        return TypeCheckError(
            kind="value_tag_mismatch",
            detail=(
                f"root value leaf carries tag {actual_value}; expected "
                f"{src_tag} for Π param type {expected.src!r}"
            ),
        )
    if not meta.forall_protected_leaves:
        return TypeCheckError(
            kind="pi_missing_forall_entanglement",
            detail=(
                "expected TPi (dependent product) but "
                "meta.forall_protected_leaves is empty — the encoded "
                "substrate carries no fiber-bundle entanglement over "
                "the bound variable"
            ),
        )
    return None


def tn_typecheck_bundle(
    bundle: MeraTensorBundle, meta: MeraEncodingMeta, expected: Ty,
) -> TypeCheckOk | TypeCheckError:
    """Substrate-level typecheck against ``expected`` (no AST walk).

    The lemma's encoded MERA is inspected leaf-by-leaf:

    * Root **kind** leaf -> kind tag (KIND_LAM / KIND_FORALL / ...).
    * Root **type** leaf -> type tag (TYPE_INT, TYPE_ARR_II, ...).
    * Root **value** leaf -> parameter-type tag for Forall/Fix binders.
    * ``meta.forall_protected_leaves`` -> non-empty iff a genuine
      dependency on the bound variable is encoded in the bond
      structure.

    ANTI-SHORTCUT: this function MUST NOT call ``decode_mera`` or any
    AST type-tree walker. The §1.6 contract is that the check operates
    on the bond bookkeeping, not on a reconstructed Python AST.
    """
    if meta.n_nodes < 1 or not bundle.leaf_vectors:
        return TypeCheckError(
            kind="empty_meta",
            detail="bundle has no leaves / meta.n_nodes < 1",
        )
    leaf_vectors = bundle.leaf_vectors

    # Root-kind constraint (binders only).
    expected_kind = _expected_root_kind(expected)
    root_kind_idx = _read_leaf_onehot(leaf_vectors, _leaf_index(0, "kind"))
    if root_kind_idx is None:
        return TypeCheckError(
            kind="leaf_not_onehot",
            detail="root kind leaf is not a one-hot vector",
        )
    if root_kind_idx == KIND_PAD:
        return TypeCheckError(
            kind="kind_tag_mismatch",
            detail="root node is KIND_PAD (encoded empty AST)",
        )
    if expected_kind is not None and root_kind_idx != expected_kind:
        return TypeCheckError(
            kind="kind_tag_mismatch",
            detail=(
                f"root kind leaf carries kind {root_kind_idx}; "
                f"expected {expected_kind} for {type(expected).__name__}"
            ),
        )

    # Root type-leaf readback.
    root_type_idx = _read_leaf_onehot(leaf_vectors, _leaf_index(0, "type"))
    if root_type_idx is None:
        return TypeCheckError(
            kind="leaf_not_onehot",
            detail="root type leaf is not a one-hot vector",
        )

    # Dispatch on expected.
    if isinstance(expected, (TInt, TBool, TNat, TList, TProp, TEq)):
        err = _check_atomic(root_type_idx, expected)
        if err is not None:
            return err
        sig = (
            f"atomic(kind={root_kind_idx},type={root_type_idx})"
        )
        return TypeCheckOk(bond_signature=sig)

    if isinstance(expected, TArrow):
        err = _check_arrow(root_type_idx, expected, meta)
        if err is not None:
            return err
        sig = (
            f"arrow(kind={root_kind_idx},type={root_type_idx},"
            f"nested={'yes' if root_type_idx == TYPE_ARR_NESTED else 'no'})"
        )
        return TypeCheckOk(bond_signature=sig)

    if isinstance(expected, TPi):
        # Root is KIND_FORALL (already enforced via expected_kind).
        # Type leaf for a Forall body should be TYPE_PROP (the
        # proposition sort) -- the body of "forall x:T. P" inhabits
        # TProp. The encoder writes TYPE_PROP for the Forall site.
        if root_type_idx != TYPE_PROP:
            return TypeCheckError(
                kind="type_tag_mismatch",
                detail=(
                    f"root type leaf carries tag {root_type_idx}; "
                    f"expected TYPE_PROP ({TYPE_PROP}) for the body "
                    f"of a Π / Forall"
                ),
            )
        err = _check_pi(leaf_vectors, meta, expected)
        if err is not None:
            return err
        n_protected = len(meta.forall_protected_leaves)
        sig = (
            f"pi(kind={root_kind_idx},type={root_type_idx},"
            f"protected_leaves={n_protected})"
        )
        return TypeCheckOk(bond_signature=sig)

    return TypeCheckError(
        kind="unsupported_expected_type",
        detail=f"expected type {type(expected).__name__} not supported",
    )


def tn_typecheck(
    lemma: Lemma, expected: Ty,
) -> TypeCheckOk | TypeCheckError:
    """Bond-structure typecheck of a cached :class:`Lemma`.

    Thin wrapper over :func:`tn_typecheck_bundle` that pulls the
    bundle + meta off the lemma. Provided so the lemma-library
    validation pass and downstream consumers do not have to unwrap
    the dataclass.
    """
    return tn_typecheck_bundle(
        lemma.mera_tensors, lemma.encoding_meta, expected)


# ---- Π_type projector measurement ----------------------------------------


def pi_type_projector_expectation(
    lemma: Lemma, expected: Ty,
) -> float:
    """The {0, 1}-valued substrate verdict cast as a projector
    measurement (EXTENSIONS.md E11 "<Psi|Π_type|Psi> ≈ 1" surface).

    Returns ``1.0`` when :func:`tn_typecheck` accepts and ``0.0``
    otherwise. The numeric form is what
    :func:`lemma_library._validate_decoded` callers expect when they
    want a residual-style verdict alongside ``eps_register``. Round-
    tripping through the MERA materialization is not necessary for the
    verdict — the bundle's leaf vectors already carry the one-hot
    indices — but the function rebuilds the MERA on the side to keep
    the API symmetric with the residual gate (both consume "the
    encoded state").
    """
    # The MERA rebuild is intentionally a side-effect-only sanity step:
    # if the bundle is malformed the rebuild raises, which we surface
    # as a 0.0 verdict instead of letting it propagate (callers expect
    # a total numeric projector measurement).
    try:
        _ = mera_from_bundle(lemma.mera_tensors)
    except Exception:  # noqa: BLE001
        return 0.0
    verdict = tn_typecheck(lemma, expected)
    return 1.0 if isinstance(verdict, TypeCheckOk) else 0.0
