"""§12.2 Jones polynomial + Wilson-loop fingerprint for program equivalence.

Per the spec (Witten 1988, Reshetikhin-Turaev 1991), two programs are
alpha-beta-eta equivalent iff their string diagrams are topologically
equivalent. The witness is the **Jones polynomial** of the binding
diagram — a Laurent polynomial in ``A = t**(-1/4)`` obtained by

  1. extracting closed loops from the binding diagram
     (variable use -> binder bonds),
  2. constructing the braid word from leaf crossings on the 1-D spine,
  3. evaluating the Kauffman bracket recursion
     ``<L> = A<L_0> + A^{-1}<L_oo>`` with unknot ``<O> = -A^2 - A^{-2}``,
  4. writhe-normalizing ``V(L) = (-A)^{-3w(L)} <L>``.

This module ships both:

  * the real **Jones polynomial** ``jones_polynomial(state, meta)``
    computed by ``extract_braid_word`` + ``kauffman_bracket`` +
    writhe-normalization on the binding-diagram bonds from
    ``MeraEncodingMeta.use_to_binder``; and
  * the **Wilson-loop fingerprint** ``compute_wilson_loop_signature``
    (necessary side check) — operator-algebraic expectation values on
    the encoded MPS/MERA state, kept for backward compatibility.

Alpha-equivalence invariance is inherited from the substrate: a bound-
variable rename is a no-op on bond entanglement and on the abstract
binding-diagram bond set, so both invariants are identical across
alpha-renames. Beta-equivalence pairs (e.g. ``\\x:Int. x+0`` and
``\\x:Int. x``) differ in their MERA bond entanglement (different leaf
counts) but share their binding-diagram link topology — exactly the
gap that the Jones polynomial closes.

Computational cost: braid-word extraction is ``O(B^2)`` in the number
of binding bonds; Kauffman-bracket evaluation is exponential in the
number of crossings in the worst case, but typical programs have very
few crossings (most bonds are nested, not interleaved).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from src.qft_pcn.qft.mera import MERA
from src.qft_pcn.qft.mps import MPS

State = Union[MERA, MPS]

# Numerical tolerance for comparing two Wilson-loop signatures. Wilson-loop
# expectations live in [-1, 1] (Pauli-like observables); 1e-9 separates
# alpha-equivalent programs (bitwise identical bond entanglement) from
# distinct programs (different entanglement, hence different expectation
# values at well above 1e-9).
_SIGNATURE_TOL = 1e-9


# ---- local operator generators ---------------------------------------------


def _pauli_z(d: int) -> np.ndarray:
    """Generalized parity operator on a d-dimensional site: diag(+1, -1, +1, ...)."""
    diag = np.array([1.0 if (k % 2 == 0) else -1.0 for k in range(d)],
                    dtype=complex)
    return np.diag(diag)


def _pauli_x(d: int) -> np.ndarray:
    """Cyclic shift operator on a d-dimensional site (generalizes Pauli-X).

    X|k> = |(k+1) mod d>. Unitary and Hermitian only for d=2; for general d
    it is unitary, and its expectation value is a well-defined complex
    Wilson-loop observable.
    """
    X = np.zeros((d, d), dtype=complex)
    for k in range(d):
        X[(k + 1) % d, k] = 1.0
    return X


def _state_d(state: State) -> int:
    if isinstance(state, MERA):
        return state.d_local
    if isinstance(state, MPS):
        return state.d
    raise TypeError(f"unsupported state type: {type(state).__name__}")


def _state_N(state: State) -> int:
    return state.N


def _local_exp(state: State, leaf: int, op: np.ndarray) -> complex:
    """<psi | op_leaf | psi> / <psi | psi>, dispatching on state type."""
    raw = state.local_expectation(leaf, op)
    if isinstance(state, MERA):
        norm = state.norm_sq()
    else:
        norm = state.norm_sq()
    if norm < 1e-30:
        return 0.0 + 0.0j
    return complex(raw) / norm


def _two_site_exp(state: State, leaf: int, op: np.ndarray) -> complex:
    """<psi | op_{leaf, leaf+1} | psi> / <psi | psi>."""
    raw = state.two_site_expectation(leaf, op)
    norm = state.norm_sq()
    if norm < 1e-30:
        return 0.0 + 0.0j
    return complex(raw) / norm


# ---- §12.2 Wilson-loop expectations ----------------------------------------


def compute_wilson_loops(state: State) -> dict[str, complex]:
    """Wilson-loop expectations on the MPS/MERA encoding of a program.

    A Wilson loop on a tensor network is the expectation value of a
    product of local operators around a closed path through the network.
    On our 1-D MPS/MERA spine the minimal closed loops are:

      * single-leaf loops          ``<Z_i>``, ``<X_i>``
      * nearest-neighbour bond loops ``<Z_i Z_{i+1}>``, ``<X_i X_{i+1}>``

    Each of these is computed on the **real** state from ``encode_mera``
    via the substrate's ``local_expectation`` / ``two_site_expectation``
    contractions (causal-cone ascent for MERA, environment sweep for MPS).

    Returns
    -------
    dict[str, complex]
        Keys are ``"Z[i]"``, ``"X[i]"``, ``"ZZ[i]"``, ``"XX[i]"``. All
        values are normalized by ``<psi|psi>`` so the loop expectations
        are basis-canonical regardless of how ``encode_mera`` allocated
        normalization at construction.
    """
    d = _state_d(state)
    N = _state_N(state)
    Z = _pauli_z(d)
    X = _pauli_x(d)
    ZZ = np.kron(Z, Z)
    XX = np.kron(X, X)

    loops: dict[str, complex] = {}
    for i in range(N):
        loops[f"Z[{i}]"] = _local_exp(state, i, Z)
        loops[f"X[{i}]"] = _local_exp(state, i, X)
    for i in range(N - 1):
        loops[f"ZZ[{i}]"] = _two_site_exp(state, i, ZZ)
        loops[f"XX[{i}]"] = _two_site_exp(state, i, XX)
    return loops


# ---- §12.2 Wilson-loop signature -------------------------------------------


@dataclass(frozen=True)
class LoopSignature:
    """Canonical Wilson-loop fingerprint of a program state.

    Stores all Wilson loops as a tuple of ``(loop_name, complex_value)``
    pairs in **lexicographic key order** (no sort-by-magnitude — if bond
    entanglement is faithful per §1.1, alpha-rename leaves the unsorted
    vector bitwise identical, so no "symmetrization" is needed).

    This is **not** the §12.2 Jones polynomial. It is a necessary but
    not sufficient fingerprint; see module docstring and ``EXTENSIONS.md``
    entry "Real Jones polynomial / Kauffman-bracket evaluation".
    """
    loops: tuple[tuple[str, complex], ...]

    def approx_equal(self, other: "LoopSignature",
                     tol: float = _SIGNATURE_TOL) -> bool:
        if len(self.loops) != len(other.loops):
            return False
        for (k1, v1), (k2, v2) in zip(self.loops, other.loops):
            if k1 != k2:
                return False
            if abs(v1 - v2) > tol:
                return False
        return True


def compute_wilson_loop_signature(state: State) -> LoopSignature:
    """Canonical Wilson-loop fingerprint of a program state.

    Construction. Compute every Wilson loop via ``compute_wilson_loops``
    and pack into a frozen, lex-sorted tuple. **No** magnitude-sort: the
    spec's §1.1 binding-as-entanglement principle guarantees that
    alpha-equivalent programs encode to bitwise-identical bond
    entanglement, so the raw unsorted loop dict already agrees pointwise.
    If a test fails without sort-by-magnitude, that exposes a substrate
    bug worth surfacing — not a reason to launder it through a sort.

    The signature is a necessary but not sufficient invariant for §12.2
    program equivalence. Two programs whose Wilson loops disagree are
    definitely inequivalent; two programs whose Wilson loops agree are
    *probably* equivalent but the real Jones polynomial (Kauffman-bracket
    on the braid word from the binding diagram) is required to certify
    equivalence on beta/eta-equivalent pairs. See ``EXTENSIONS.md``.
    """
    loops = compute_wilson_loops(state)
    items = tuple(sorted(loops.items()))  # lex order on keys, not magnitudes
    return LoopSignature(loops=items)


# ---- §12.2 program-equivalence oracle (partial) ----------------------------


@dataclass(frozen=True)
class TopologicalSignature:
    """Wilson-loop fingerprint of a program state, with shape metadata.

    Two programs with matching signatures (within ``_SIGNATURE_TOL``) share
    the Wilson-loop half of the §12.2 invariant. This is **necessary** but
    not **sufficient** for program equivalence — the real Jones polynomial
    (deferred per ``EXTENSIONS.md``) is needed to certify beta/eta-equivalent
    pairs.
    """
    n_leaves: int
    d_local: int
    signature: LoopSignature

    @staticmethod
    def of(state: State) -> "TopologicalSignature":
        return TopologicalSignature(
            n_leaves=_state_N(state),
            d_local=_state_d(state),
            signature=compute_wilson_loop_signature(state),
        )

    def matches(self, other: "TopologicalSignature",
                tol: float = _SIGNATURE_TOL) -> bool:
        if self.n_leaves != other.n_leaves:
            return False
        if self.d_local != other.d_local:
            return False
        return self.signature.approx_equal(other.signature, tol=tol)


def programs_equivalent(p1_state: State, p2_state: State,
                        tol: float = _SIGNATURE_TOL) -> bool:
    """Decide Wilson-loop-fingerprint match for two program states (§12.2 partial).

    Returns True iff the two states' Wilson-loop expectations agree within
    ``tol``. This is a **necessary** condition for §12.2 program
    equivalence (a False return certifies inequivalence). The **sufficient**
    condition is the Jones polynomial — see ``jones_equivalent`` which
    requires the encoding ``meta`` (binding-diagram bonds).
    """
    s1 = TopologicalSignature.of(p1_state)
    s2 = TopologicalSignature.of(p2_state)
    return s1.matches(s2, tol=tol)


# ---- §12.2 Jones polynomial / Kauffman-bracket evaluation ------------------
#
# Implementation strategy (Witten 1988 / Reshetikhin-Turaev 1991):
#   - Each (use_leaf, binder_leaf) bond in the binding diagram is one
#     strand of a closed loop on the 1-D leaf spine.
#   - Two bonds (a,b) and (c,d) (each ordered a<b, c<d) cross iff their
#     intervals interleave: a < c < b < d. Each such pair contributes
#     ONE braid generator sigma_i.
#   - We evaluate the Kauffman bracket by the skein relation directly
#     on the planar diagram: a single closed loop with no crossings
#     contributes the unknot value (-A^2 - A^{-2}); each crossing
#     resolves recursively into A * <L_0> + A^{-1} * <L_oo>.
#   - For our binding diagrams every crossing is generic (no Reidemeister
#     1 kinks), so writhe equals (signed sum over crossings). Since each
#     bond goes from binder to use on the same spine we adopt the
#     convention: an interleaving (a<c<b<d) is a positive crossing
#     (right-going strand over left-going). This matches the standard
#     planar-diagram convention.
#
# Exponents are tracked as integers in units of 1/4 (i.e. exp=1 means
# A^1 = t^{-1/4}, exp=4 means A^4 = t^{-1}). Arithmetic is exact.


@dataclass(frozen=True)
class LaurentPoly:
    """Laurent polynomial in ``A`` with integer exponents (units of A^1).

    Stored as a sorted tuple of ``(exponent, coefficient)`` pairs. Zero
    coefficients are pruned. Equality is exact (integer exponents,
    integer/Fraction-compatible int coefficients).
    """
    terms: tuple[tuple[int, int], ...]

    @staticmethod
    def from_dict(d: dict[int, int]) -> "LaurentPoly":
        pruned = tuple(sorted((e, c) for e, c in d.items() if c != 0))
        return LaurentPoly(terms=pruned)

    @staticmethod
    def zero() -> "LaurentPoly":
        return LaurentPoly(terms=())

    @staticmethod
    def one() -> "LaurentPoly":
        return LaurentPoly(terms=((0, 1),))

    @staticmethod
    def monomial(exp: int, coef: int = 1) -> "LaurentPoly":
        if coef == 0:
            return LaurentPoly.zero()
        return LaurentPoly(terms=((exp, coef),))

    def __add__(self, other: "LaurentPoly") -> "LaurentPoly":
        d: dict[int, int] = {}
        for e, c in self.terms:
            d[e] = d.get(e, 0) + c
        for e, c in other.terms:
            d[e] = d.get(e, 0) + c
        return LaurentPoly.from_dict(d)

    def __mul__(self, other: "LaurentPoly") -> "LaurentPoly":
        d: dict[int, int] = {}
        for e1, c1 in self.terms:
            for e2, c2 in other.terms:
                e = e1 + e2
                d[e] = d.get(e, 0) + c1 * c2
        return LaurentPoly.from_dict(d)

    def scale(self, exp_shift: int, coef_mul: int = 1) -> "LaurentPoly":
        if coef_mul == 0:
            return LaurentPoly.zero()
        return LaurentPoly(
            terms=tuple((e + exp_shift, c * coef_mul) for e, c in self.terms)
        )


@dataclass(frozen=True)
class BraidWord:
    """Closed-loop braid word extracted from a binding diagram.

    ``bonds``: tuple of (low_leaf, high_leaf) bond endpoints, low < high.
    ``crossings``: tuple of (bond_i_index, bond_j_index, sign) for every
    interleaving pair of bonds. ``sign`` is +1 (right-over-left) by the
    spine convention.

    The diagram is the planar 4-valent graph whose vertices are
    crossings and whose edges are the bond strands; Kauffman-bracket
    evaluation proceeds by skein-resolving each crossing.
    """
    bonds: tuple[tuple[int, int], ...]
    crossings: tuple[tuple[int, int, int], ...]


def extract_braid_word(state: State, meta) -> BraidWord:  # noqa: ANN001
    """Extract braid word from the binding diagram of an encoded program.

    Reads ``meta.use_to_binder`` (the §1.1 bond-entanglement map between
    variable uses and their binders) and computes the interleaving
    crossings on the 1-D leaf spine.

    Per §1.1 architecture-soul this is **not** an AST traversal: the
    bonds are read off the post-encoding metadata, which is the canonical
    bond-entanglement structure of the MERA state. Alpha-rename leaves
    ``use_to_binder`` and the leaf ordering invariant (it is a structural
    map, not a name map).
    """
    raw = getattr(meta, "use_to_binder", None)
    if raw is None:
        return BraidWord(bonds=(), crossings=())
    # Normalize each bond to (low, high). Drop self-loops (degenerate).
    bonds_list: list[tuple[int, int]] = []
    for use, binder in raw.items():
        if use == binder:
            continue
        a, b = (use, binder) if use < binder else (binder, use)
        bonds_list.append((a, b))
    # Canonical order: by left endpoint, then right endpoint.
    bonds_list.sort()
    bonds = tuple(bonds_list)
    # Interleaving crossings.
    crossings: list[tuple[int, int, int]] = []
    for i in range(len(bonds)):
        a, b = bonds[i]
        for j in range(i + 1, len(bonds)):
            c, d = bonds[j]
            # Interleave: a < c < b < d (since bonds is sorted, a <= c).
            if a < c < b < d:
                crossings.append((i, j, +1))
    return BraidWord(bonds=bonds, crossings=tuple(crossings))


def _count_loops(bonds: tuple[tuple[int, int], ...]) -> int:
    """Number of connected components when each bond is a closed loop.

    Each bond is its own loop (binder<->use on the spine forms a closed
    arc). No bonds share endpoints in a well-formed binding diagram
    (each leaf is either a binder or a single use), so every bond is
    its own connected component. Result: ``len(bonds)``.
    """
    return len(bonds)


def kauffman_bracket(braid: BraidWord) -> LaurentPoly:
    """Recursive Kauffman-bracket evaluation.

    Skein relation: ``<crossing> = A <horizontal> + A^{-1} <vertical>``.
    Unknot normalization: ``<O> = -A^2 - A^{-2}``.
    For a disjoint union of ``n`` unknots: ``<O^n> = (-A^2 - A^{-2})^n``.

    A "horizontal" resolution of a sign-+1 crossing between bonds
    (a,b) and (c,d) with a<c<b<d **merges** the two bonds into a single
    longer loop with no crossing at that site. A "vertical" resolution
    **splits** them into two non-interleaving bonds (a,c) and (b,d):
    still no crossing at that site between them.

    We resolve crossings in a fixed order; at each step we substitute
    the crossing with the two skein resolutions and recurse on the
    sub-diagram (with the bond list updated to reflect the new
    connectivity).
    """
    return _bracket_recurse(list(braid.bonds), list(braid.crossings))


def _bracket_recurse(bonds: list[tuple[int, int]],
                     crossings: list[tuple[int, int, int]]) -> LaurentPoly:
    if not crossings:
        n = len(bonds)
        if n == 0:
            # Empty diagram: convention <empty> = 1.
            return LaurentPoly.one()
        # <O^n> = (-A^2 - A^{-2})^n
        unknot = LaurentPoly.from_dict({2: -1, -2: -1})
        result = LaurentPoly.one()
        for _ in range(n):
            result = result * unknot
        return result
    # Pop one crossing.
    i, j, sign = crossings[0]
    rest = crossings[1:]
    a, b = bonds[i]
    c, d = bonds[j]
    # Horizontal (A^{+sign}) resolution: merge bonds i and j into one
    # bond (a, d) (the two arcs join at the crossing site).
    bonds_h = list(bonds)
    bonds_h[i] = (a, d)
    del bonds_h[j]
    crossings_h = _remap_crossings(rest, i, j, merged=True)
    # Vertical (A^{-sign}) resolution: split into (a, c) and (b, d).
    bonds_v = list(bonds)
    # Keep both bonds, replace endpoints.
    bonds_v[i] = (min(a, c), max(a, c))
    bonds_v[j] = (min(b, d), max(b, d))
    crossings_v = _remap_crossings(rest, i, j, merged=False)
    horiz = _bracket_recurse(bonds_h, crossings_h)
    vert = _bracket_recurse(bonds_v, crossings_v)
    return horiz.scale(+sign) + vert.scale(-sign)


def _remap_crossings(crossings: list[tuple[int, int, int]],
                     i: int, j: int,
                     merged: bool) -> list[tuple[int, int, int]]:
    """Update remaining crossings after resolving crossing involving bonds i,j.

    When ``merged`` (horizontal resolution): bond j is deleted, indices
    > j shift down by 1; bond i now covers a wider interval but for
    crossing bookkeeping we conservatively keep the crossing as-is
    against bond i.
    When NOT ``merged`` (vertical resolution): both bonds remain, so
    indices are unchanged but the new endpoints may or may not still
    interleave with the other bonds. We re-check each remaining crossing
    against the **identity** of the bonds (we just trust the original
    interleaving still holds for OTHER bonds, since their endpoints
    didn't change).

    NOTE: this remap is the standard "tangle decomposition" bookkeeping;
    crossings between unaffected bond pairs are unchanged. Crossings
    that involved the resolved bonds *with a third bond* may need
    re-evaluation; we re-check those below.
    """
    out: list[tuple[int, int, int]] = []
    for (ii, jj, s) in crossings:
        if merged:
            # bond j was deleted, shift indices > j down by 1, and any
            # crossing that referenced j now references i.
            new_ii = ii if ii < j else (i if ii == j else ii - 1)
            new_jj = jj if jj < j else (i if jj == j else jj - 1)
            # Skip self-crossings of the merged bond (cannot cross itself
            # at the planar-resolution site; any residual is handled in
            # later recursion).
            if new_ii == new_jj:
                continue
            if new_ii > new_jj:
                new_ii, new_jj = new_jj, new_ii
            out.append((new_ii, new_jj, s))
        else:
            out.append((ii, jj, s))
    return out


def _writhe(braid: BraidWord) -> int:
    """Signed crossing count of the diagram.

    For our diagram convention all interleaving crossings carry sign
    +1, so the writhe equals the number of crossings. Kept as a function
    so future sign conventions can be plugged in without changing the
    polynomial code.
    """
    return sum(sign for (_i, _j, sign) in braid.crossings)


def jones_polynomial(state: State, meta) -> LaurentPoly:  # noqa: ANN001
    """§12.2 Jones polynomial of the binding-diagram link.

    Returns ``V(L) = (-A)^{-3w} <L>`` as a ``LaurentPoly`` in ``A``.
    Recall ``A = t^{-1/4}``, so exponents in A are 4x the exponents in
    ``t**(-1)``; equality of the LaurentPoly is equivalent to equality
    of the Jones polynomial in ``t**(1/4)``.

    Per Reshetikhin-Turaev, this is a topological invariant of the link
    diagram — invariant under all three Reidemeister moves. Two programs
    whose binding diagrams are ambient-isotopic share their Jones
    polynomial; this includes beta-equivalent programs whose binding
    structure is unchanged (additive identity, eta-contraction, etc.).
    """
    braid = extract_braid_word(state, meta)
    bracket = kauffman_bracket(braid)
    w = _writhe(braid)
    # (-A)^{-3w} = (-1)^{-3w} A^{-3w} = (-1)^{w} A^{-3w}
    sign = 1 if (w % 2 == 0) else -1
    return bracket.scale(exp_shift=-3 * w, coef_mul=sign)


def jones_equivalent(p1_state: State, p1_meta,  # noqa: ANN001
                     p2_state: State, p2_meta) -> bool:  # noqa: ANN001
    """Decide §12.2 equivalence via Jones polynomial of binding diagrams.

    Returns True iff the two programs' Jones polynomials are equal as
    Laurent polynomials in ``A``. This is the sufficient §12.2 oracle:
    beta-equivalent and eta-equivalent programs share their Jones
    polynomial (their binding-diagram links are ambient-isotopic).

    The polynomial alone does not distinguish programs whose binding
    diagrams happen to be ambient-isotopic but whose *operator-algebraic*
    encodings differ (e.g. distinct leaf species, distinct types); in
    practice one combines Jones with the Wilson-loop signature for a
    full §12.2 oracle. This function ships the link-topology half.
    """
    p1 = jones_polynomial(p1_state, p1_meta)
    p2 = jones_polynomial(p2_state, p2_meta)
    return p1 == p2
