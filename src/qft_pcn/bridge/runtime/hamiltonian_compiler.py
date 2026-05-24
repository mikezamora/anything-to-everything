"""§2.4 constraint-kind compilers (W3 of the DSL programming plan).

Each compiler returns a Hermitian, positive-semidefinite operator on the
appropriate local Hilbert space. Energy = 0 iff the constraint is
satisfied (§13.2).

Conventions:
- Local Hilbert space for a single site of `kind_cutoff` labels = a
  (cutoff x cutoff) computational basis indexed by label position in the
  spec's vocab.
- Multi-site operators are built via numpy tensor products. The encoder
  cap D_LOCAL=65536 applies upstream (validated in dsl/schema.py).
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


def compile_vocabulary(
    *,
    primitives: Sequence[str],
    vocab: Sequence[str],
    n_sites: int,
    kind_cutoff: int,
    weight: float,
) -> np.ndarray:
    """Sum of per-site projectors onto disallowed labels.

    P_site = diag( 0 if label in primitives else 1 )
    H = weight * Σ_site (I ⊗ ... ⊗ P_site ⊗ ... ⊗ I)

    PSD by construction (each P_site is a diagonal projector, sum of PSDs is PSD).
    """
    if kind_cutoff != len(vocab):
        raise ValueError(
            f"kind_cutoff={kind_cutoff} but vocab has {len(vocab)} labels"
        )
    allowed = {vocab.index(p) for p in primitives}
    diag1 = np.array([0.0 if i in allowed else 1.0 for i in range(kind_cutoff)])
    proj_site = np.diag(diag1)
    I = np.eye(kind_cutoff)
    dim = kind_cutoff ** n_sites
    H = np.zeros((dim, dim), dtype=float)
    for k in range(n_sites):
        ops = [I] * n_sites
        ops[k] = proj_site
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H = H + weight * term
    return H


def compile_well_typed_subtree(
    *,
    root: int,
    n_sites: int,
    kind_cutoff: int,
    type_cutoff: int,
    vocab: Sequence[str],
    types: Sequence[str],
    weight: float,
) -> np.ndarray:
    """First-cut well-typed-subtree projector (§10.2 simplified).

    Penalises sites whose `node_kind` is in {Var, App, Lambda} but whose
    `type` is "unknown". Full T-Var/T-App/T-Abs elaboration is W3.T2b.

    PSD by construction (each per-site term is a rank-1 projector ⊗ I).

    Note: `root` is currently unused in the first cut; the full §10.2
    elaboration (W3.T2b) will scope per-site projectors to the subtree
    rooted at `root`. See EXTENSIONS.md for the documented gap.
    """
    # If 'unknown' isn't a declared type or no typed kinds are in vocab,
    # the constraint is vacuous → return zero op (still Hermitian and PSD).
    if "unknown" not in types:
        unknown_idx = -1
    else:
        unknown_idx = types.index("unknown")
    typed_kinds = {vocab.index(k) for k in ("Var", "App", "Lambda") if k in vocab}
    d_site = kind_cutoff * type_cutoff
    dim = d_site ** n_sites
    H = np.zeros((dim, dim), dtype=float)
    if unknown_idx < 0 or not typed_kinds:
        return H
    proj_kind = np.zeros((kind_cutoff, kind_cutoff))
    for k in typed_kinds:
        proj_kind[k, k] = 1.0
    proj_type = np.zeros((type_cutoff, type_cutoff))
    proj_type[unknown_idx, unknown_idx] = 1.0
    p_site = np.kron(proj_kind, proj_type)
    I = np.eye(d_site)
    for s in range(n_sites):
        ops = [I] * n_sites
        ops[s] = p_site
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H = H + weight * term
    return H
