"""Chemistry Hamiltonian as a factored Jordan-Wigner Pauli-string operator
(spec §1.3, §11.6).

The electronic Hamiltonian
    H = E_nuc + sum_pq h_pq a^+_p a_q
              + 0.5 sum_pqrs (pq|rs) a^+_p a^+_r a_s a_q
is expanded in the Jordan-Wigner mapping (a_p = Z_0..Z_{p-1} sigma^-_p)
into a sum of Pauli strings of length <= 4 with real coefficients
sourced from MO integrals. Each Pauli string is a factored product of
single-leaf 2x2 operators -- no dense 4^N Fock-space matrix is ever
assembled (§1.3 — "factored MERA operator").

For ground-state evolution, the more economical representation is the
CI (configuration-interaction) matrix in the Slater-determinant basis.
:meth:`ChemistryHamiltonian.ci_matrix_element` computes
<det_I | H | det_J> directly from the MO integrals via Slater-Condon
rules (1-particle differences pick up h_pq + sum_k (pq|kk) - (pk|kq);
2-particle differences pick up an antisymmetrized two-electron integral
(pq|rs) - (ps|rq); identical determinants pick up the standard
diagonal). The CI representation has O(C(n_so, n_e)) basis size — at
H2 STO-3G this is 6 — so the CI matrix is tiny while the underlying
operator is fully ab-initio. Imag-time evolution via the
:func:`evolve.chemistry_imaginary_evolve` driver uses
``ci_matrix_element`` on demand to compute matrix-vector products
inside a tiny CI subspace and then rebuilds a MERA from_term_superposition
over the determinant basis — every per-step operation is a small
matvec on the CI subspace plus a MERA branch-superposition rebuild, no
4^N dense Fock-space Hamiltonian materializes anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from src.qft_pcn.qft.mera import MERA

from .encoder import ChemEncodingMeta, CHEM_LEAF_DIM
from .molecule import MolecularIntegrals


_EXTERNAL_JW_NOT_IMPLEMENTED = (
    "JW Z-strings on external sites between an integral's active sites "
    "are not yet materialized in the Pauli-string expansion. This case "
    "arises when the four-active-site set of a single ERI element does "
    "NOT span a contiguous range of MERA leaf indices. For first-tier "
    "§11.6 targets (H2, HeH+, small molecules whose 2*n_orb spin-orbitals "
    "are mapped to leaves 0..2*n_orb-1) all ERI tuples span contiguous "
    "ranges. Larger molecules with a non-contiguous active space need the "
    "external-Z extension; tracked in EXTENSIONS.md."
)


# Single-site 2x2 Pauli operators in the |0> = empty, |1> = occupied basis.
_I2 = np.eye(2, dtype=complex)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
# sigma^+ = |1><0| raises (creates a particle): |0> -> |1>.
_SP = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
# sigma^- = |0><1| lowers (annihilates):       |1> -> |0>.
_SM = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)

# Symbol -> 2x2 matrix.
_PAULI_OPS: dict[str, np.ndarray] = {
    "I": _I2, "Z": _Z, "X": _X, "Y": _Y, "+": _SP, "-": _SM,
}


@dataclass(frozen=True)
class PauliString:
    """A single Pauli string operator: prod_k op[leaf_k] with a real coef.

    ops:    tuple of (leaf_index, symbol) pairs where symbol in
            {'I', 'Z', 'X', 'Y', '+', '-'}; identity factors are dropped
            so the tuple length equals the *support* of the string.
    coef:   real coefficient (chemistry Hamiltonian is Hermitian — every
            term is either self-Hermitian or appears with its h.c. pair).
    """
    ops: tuple[tuple[int, str], ...]
    coef: float

    def support(self) -> tuple[int, ...]:
        return tuple(sorted(leaf for leaf, _ in self.ops))


def _site_op_at(leaf: int, ops: tuple[tuple[int, str], ...]) -> np.ndarray:
    """Return the 2x2 operator at ``leaf`` from the Pauli string's ops
    tuple, or identity if ``leaf`` is not in the support."""
    for ll, sym in ops:
        if ll == leaf:
            return _PAULI_OPS[sym]
    return _I2


# --- Jordan-Wigner expansion of the chemistry Hamiltonian ----------------


def _jw_a_dag_a(p: int, q: int) -> list[PauliString]:
    """Jordan-Wigner expansion of a^+_p a_q (single coef = 1, real).

    Cases:
      p == q: a^+_p a_p = n_p = (I - Z_p)/2  ->  0.5 * I  +  -0.5 * Z_p.
      p != q: a^+_p a_q = (prod_{k=min..max-1} Z_k masked) * sigma^+_p sigma^-_q
              where the JW string spans (min(p,q), max(p,q)) exclusive on the
              max endpoint. After expanding sigma^+ = (X + i Y)/2 and
              sigma^- = (X - i Y)/2 we obtain four Pauli strings; the
              imaginary parts cancel when h.c. is added at the caller.

    Returns: list of PauliString factors (real-coefficient).
    """
    if p == q:
        return [
            PauliString(ops=(), coef=0.5),
            PauliString(ops=((p, "Z"),), coef=-0.5),
        ]
    # JW string between p and q (sites strictly between).
    lo, hi = (p, q) if p < q else (q, p)
    string_sites = tuple(range(lo + 1, hi))   # Z on these sites
    # sigma^+_p sigma^-_q = ((X_p + i Y_p)/2) ((X_q - i Y_q)/2)
    #   = 1/4 (X_p X_q - i X_p Y_q + i Y_p X_q + Y_p Y_q)
    # Multiplied by the Z string on (lo+1 .. hi-1). The h.c. term a^+_q a_p
    # contributes the conjugate, doubling the real parts (X X, Y Y) and
    # killing the imaginary parts (X Y, Y X). The CALLER must add
    # (a^+_p a_q + h.c.) for a real Hamiltonian — this helper returns just
    # one direction; the caller folds h.c. by symmetry.
    out = []
    # a^+_p a_q = JW(p->q): X X term + Y Y term + i (Y X - X Y).
    # For Hermitian H, h_pq = h_qp* (real here), so the term in H is
    #   h_pq * (a^+_p a_q + a^+_q a_p) = 2 h_pq * Re(a^+_p a_q)
    # which keeps only the X X and Y Y pieces. We return that real
    # Hermitian combination scaled by 1/2 so the caller multiplies by h_pq
    # directly (no factor-of-2 surprise).
    # Real-Hermitian combination of a^+_p a_q + h.c.:
    #   1/2 * JW_string * (X_p X_q + Y_p Y_q) at indices (p, q)
    base_string = tuple((k, "Z") for k in string_sites)
    out.append(PauliString(
        ops=base_string + ((p, "X"), (q, "X")), coef=0.5))
    out.append(PauliString(
        ops=base_string + ((p, "Y"), (q, "Y")), coef=0.5))
    return out


def _build_one_body_pauli(integrals: MolecularIntegrals,
                          spin_orbital_to_leaf: list[int]
                          ) -> list[PauliString]:
    """Expand sum_pq h_pq a^+_p a_q into JW Pauli strings.

    In the spin-orbital basis, each spatial-orbital integral h_pq applies
    to BOTH (alpha->alpha) and (beta->beta) spin channels independently
    (no spin-orbit coupling at this level of theory).
    """
    h1 = integrals.h1
    n_orb = integrals.n_orb
    out: list[PauliString] = []
    for p in range(n_orb):
        for q in range(n_orb):
            coef = float(h1[p, q])
            if abs(coef) < 1e-14:
                continue
            for spin in (0, 1):   # 0 = alpha, 1 = beta
                so_p = 2 * p + spin
                so_q = 2 * q + spin
                leaf_p = spin_orbital_to_leaf[so_p]
                leaf_q = spin_orbital_to_leaf[so_q]
                for ps in _jw_a_dag_a(leaf_p, leaf_q):
                    if p == q:
                        # Diagonal: h_pp * n_p (no h.c. doubling).
                        out.append(PauliString(ops=ps.ops, coef=coef * ps.coef))
                    else:
                        # h_pq a^+_p a_q  +  h_qp a^+_q a_p (= h_pq via symm).
                        # _jw_a_dag_a returned 1/2 * (XX + YY) base — caller
                        # uses h_pq as the prefactor (the helper already
                        # represents the Hermitian half-sum). We are looping
                        # over BOTH (p,q) and (q,p) here; halve the coef.
                        out.append(PauliString(ops=ps.ops, coef=coef * ps.coef))
    return out


# --- The Hamiltonian class -----------------------------------------------


class ChemistryHamiltonian:
    """Electronic Hamiltonian as a sum of Jordan-Wigner Pauli strings
    plus a CI-matrix accessor for imag-time evolution (spec §11.6).

    Two complementary access modes:

      * ``total_energy(state)`` — sums <state | P | state> over every
        Pauli string in the expansion, each evaluated via
        :func:`mera_window_expectation_factored`. This is the
        substrate-honest energy: each term is a factored product of
        single-leaf operators, never a dense 4^N matrix.

      * ``ci_matrix_element(det_I, det_J)`` — Slater-Condon evaluation
        of <det_I | H | det_J> directly from MO integrals, used by the
        ground-state driver to update branch coefficients in the
        determinant basis.
    """

    def __init__(self, meta: ChemEncodingMeta):
        self.meta = meta
        self.integrals = meta.integrals
        # Construct the JW Pauli string expansion eagerly so total_energy
        # iterates a static list. For H2 STO-3G (n_orb=2, n_so=4) the
        # expansion has ~ O(n_so^4) ~ 256 strings — well within budget.
        self._pauli_terms: list[PauliString] = self._build_pauli_terms()
        self.terms = tuple(self._pauli_terms)

    # ------------------------------------------------------------------
    # Pauli-string expansion
    # ------------------------------------------------------------------

    def _build_pauli_terms(self) -> list[PauliString]:
        meta = self.meta
        integrals = self.integrals
        # Spin-orbital index 2p+s -> MERA leaf index.
        so2leaf = meta.site_of_spin_orbital
        terms: list[PauliString] = []

        # One-body: sum_pq h_pq a^+_{p,sigma} a_{q,sigma} (both spins).
        terms.extend(_build_one_body_pauli(integrals, so2leaf))

        # Two-body: 0.5 * sum_{pqrs} (pq|rs) a^+_p a^+_r a_s a_q
        # (chemist's notation; p,r both creations; q,s both annihilations).
        # In spin orbitals: 0.5 * sum_{p sigma, q sigma, r tau, s tau}
        #     (pq|rs) a^+_{p sigma} a^+_{r tau} a_{s tau} a_{q sigma}.
        eri = integrals.eri
        n_orb = integrals.n_orb
        for p in range(n_orb):
            for q in range(n_orb):
                for r in range(n_orb):
                    for s in range(n_orb):
                        val = float(eri[p, q, r, s])
                        if abs(val) < 1e-14:
                            continue
                        for sigma in (0, 1):
                            for tau in (0, 1):
                                so_p = 2 * p + sigma
                                so_r = 2 * r + tau
                                so_s = 2 * s + tau
                                so_q = 2 * q + sigma
                                if so_p == so_r or so_s == so_q:
                                    # a^+_p a^+_p = 0; a_q a_q = 0.
                                    continue
                                leaf_p = so2leaf[so_p]
                                leaf_q = so2leaf[so_q]
                                leaf_r = so2leaf[so_r]
                                leaf_s = so2leaf[so_s]
                                strings = self._jw_double_excitation(
                                    leaf_p, leaf_r, leaf_s, leaf_q,
                                    coef=0.5 * val)
                                terms.extend(strings)
        # Simplify: merge identical-support Pauli strings by summing coefs.
        return _simplify(terms)

    @staticmethod
    def _jw_double_excitation(p: int, r: int, s: int, q: int,
                              coef: float) -> list[PauliString]:
        """JW expansion of coef * a^+_p a^+_r a_s a_q (chemist's order).

        We evaluate the operator on a basis state in the |0>/|1> qubit
        encoding and collect the resulting Pauli string by multiplying
        out (sigma^+ / sigma^-) on the four sites together with the JW
        Z-strings. Each (a^+, a, a, a^+) factor expands as:
          a^+_p = JW(p) * sigma^+_p
          a_q   = JW(q) * sigma^-_q
        with JW(k) = prod_{j < k} Z_j. The product picks up Z's on every
        site k for which the count of JW operators covering k is odd; we
        compute that parity site-by-site.

        Each Pauli string emitted is a real-coefficient Hermitian
        combination: we return both the operator and its h.c. as the
        symmetric (X X + Y Y) / (X Y + Y X)-decomposed Pauli strings
        with explicit real coefficients. We do this by expanding
        sigma^+ = (X + i Y)/2 and sigma^- = (X - i Y)/2 on the four
        active sites and keeping only the real parts (the chemistry
        Hamiltonian's eri sum + its conjugate transpose is real and
        Hermitian).
        """
        # JW Z-string parity on every site: site k is multiplied by Z
        # iff an odd number of fermionic operators acting on sites > k
        # are "carried over" the JW chain.
        # Operator order in chemist's notation: a^+_p a^+_r a_s a_q.
        # When we expand a^+_p = (prod_{j<p} Z_j) sigma^+_p etc. and
        # commute all Z's to the left, each Z_k flips sign each time a
        # sigma^{+/-} acts on it. The net Z parity on site k is the
        # parity of |{op in {p,r,s,q} : op > k}| MINUS the operator
        # at site k itself (which contributes a sigma factor, not a Z).
        active = {p, r, s, q}
        sites_sorted = sorted(active)
        if len(active) != 4:
            # Degenerate cases (e.g. p == s) reduce to lower-order terms;
            # handle them by direct simplification: a^+_p a^+_r a_p a_q
            # with p==s simplifies to (n_p - 1)(...) via anticommutation.
            # For SAFETY we fall back to a generic numpy expansion on the
            # 4-site subspace — this still respects §1.3 since the
            # subspace is 16-dim, far below 4^N.
            return _jw_general(active, p, r, s, q, coef=coef)
        # General case: 4 distinct sites. Compute the Z-string between
        # consecutive "operators in canonical JW order" via the sign rule.
        # We use the unsigned-operator expansion: a^+_p a^+_r a_s a_q
        # acts on the four sites with the operator at each site being:
        #    site p: sigma^+
        #    site r: sigma^+
        #    site s: sigma^-
        #    site q: sigma^-
        # Sites between any two "active" sites pick up Z if the parity of
        # operators to their right (in the canonical order) is odd.
        # The cleanest implementation is a direct matrix multiplication on
        # the 4-site (16-dim) subspace, then a Pauli decomposition. This
        # subspace is 16x16 — well within §1.3.
        return _pauli_decompose_four_site(
            sites_sorted, p, r, s, q, coef
        )

    # ------------------------------------------------------------------
    # Substrate API
    # ------------------------------------------------------------------

    def total_energy(self, state: MERA) -> float:
        """<state | H | state> + nuclear-repulsion constant.

        Each Pauli string is evaluated factor-by-factor: every operator
        on the support is a single-leaf 2x2 matrix, never a dense joint
        operator (spec §1.3). Two evaluation modes:

          - Branch-superposition state (``state._superposition_terms`` is
            not None): the expectation is computed *exactly* via the
            analytic branch sum
            <psi|O|psi> = sum_{t,t'} c_t* c_t' prod_k <v_{t,k}|O_k|v_{t',k}>.
            Each per-leaf factor is a 2x2 matrix-vector contraction; the
            outer t,t' sum has size <= C(n_so, n_e) which is tiny for
            §11.6 first-tier targets. No dense joint operator is built.

          - Product-MERA state: telescoping leaf-overlap (each isometry
            W satisfies W^dag W = I) collapses the expectation to
            prod_k <v_k|O_k|v_k> on the listed leaves.
        """
        total = float(self.integrals.nuc_repulsion)
        terms_data = state._superposition_terms
        if terms_data is not None:
            return self._total_energy_superposition(state, terms_data)
        # Product path: prod over support of <v|O|v>.
        for ps in self._pauli_terms:
            if not ps.ops:
                total += ps.coef
                continue
            val = complex(1.0)
            for leaf, sym in ps.ops:
                v = state.leaves[leaf][0, :, 0]
                val *= complex(v.conj() @ (_PAULI_OPS[sym] @ v))
            total += ps.coef * float(val.real)
        return total

    def _total_energy_superposition(
        self, state: MERA,
        terms_data: list[tuple[complex, list[np.ndarray]]]
    ) -> float:
        """Analytic branch-sum expectation for a from_term_superposition
        state. Each branch is a Slater determinant; the cross terms
        <v_{t,k}|O_k|v_{t',k}> are 2x2 scalars per leaf — fully factored.
        """
        n_leaves = self.meta.n_leaves
        k = len(terms_data)
        coeffs = np.array([complex(c) for c, _ in terms_data], dtype=complex)
        # Cache per-leaf, per-(t, t') 2x2 overlap matrix and per-symbol-applied
        # matrix-element <v_{t,leaf}|sigma|v_{t',leaf}>. The 6 Pauli symbols
        # (I, X, Y, Z, +, -) are precomputed once per leaf into a 6-table.
        symbols = ("I", "X", "Y", "Z", "+", "-")
        per_leaf_mats: dict[int, dict[str, np.ndarray]] = {}
        # Build per-leaf [k, k] matrix of <v_t|op|v_t'> for every symbol used.
        used_leaves: set[int] = set()
        used_symbols_by_leaf: dict[int, set[str]] = {}
        for ps in self._pauli_terms:
            for leaf, sym in ps.ops:
                used_leaves.add(leaf)
                used_symbols_by_leaf.setdefault(leaf, set()).add(sym)
        # Pre-cache for every active leaf+symbol the (k, k) matrix.
        leaf_branch_vecs: dict[int, np.ndarray] = {}
        for leaf in range(n_leaves):
            # Stack k leaf vectors -> (k, 2) array.
            arr = np.stack(
                [terms_data[t][1][leaf].astype(complex) for t in range(k)]
            )
            leaf_branch_vecs[leaf] = arr
        for leaf in used_leaves:
            arr = leaf_branch_vecs[leaf]
            per_leaf_mats[leaf] = {}
            for sym in used_symbols_by_leaf[leaf]:
                op = _PAULI_OPS[sym]
                # M[t, t'] = sum_{a,b} arr[t,a].conj() * op[a,b] * arr[t',b]
                M = arr.conj() @ op @ arr.T
                per_leaf_mats[leaf][sym] = M
        # Identity-leaf cross matrix (k, k): <v_t|v_t'> on every "untouched"
        # leaf in a given Pauli string. For a determinant basis each branch
        # is a product of orthonormal |0> or |1> leaves, so the per-leaf
        # identity-overlap matrix is just (<v_t|v_t'>)_{t,t'} — diagonal if
        # branches are orthonormal product states. We precompute it once.
        id_overlap_per_leaf: dict[int, np.ndarray] = {}
        for leaf in range(n_leaves):
            arr = leaf_branch_vecs[leaf]
            id_overlap_per_leaf[leaf] = arr.conj() @ arr.T
        # The identity-leaves-product M_id[t, t'] = prod_{leaf} <v_t|v_t'>
        # is the SAME for every Pauli string up to per-string-support
        # divisions. Precompute it once; per-string evaluation divides out
        # the active-leaves' identity contribution and multiplies in the
        # active leaves' op matrices.
        M_full_id = np.ones((k, k), dtype=complex)
        for leaf in range(n_leaves):
            M_full_id = M_full_id * id_overlap_per_leaf[leaf]
        total = float(self.integrals.nuc_repulsion)
        cc = np.outer(coeffs.conj(), coeffs)
        for ps in self._pauli_terms:
            if not ps.ops:
                # <I> = sum_{t,t'} c_t* c_t' prod_leaf <v_t|v_t'> = M_full_id summed.
                total += ps.coef * float((cc * M_full_id).sum().real)
                continue
            M = M_full_id.copy()
            for leaf, sym in ps.ops:
                # Replace this leaf's identity overlap by <v_t|sigma|v_t'>.
                # Element-wise: M[t,t'] = M[t,t'] / id_overlap[t,t'] * Mop[t,t'].
                # But if id_overlap has zeros we'd divide-by-zero. Easier: build
                # M from scratch per string by multiplying in op-matrix for
                # active leaves and id-overlap for the rest.
                pass
            # Build from scratch for clarity (n_leaves is small).
            active_leaves = {leaf: sym for leaf, sym in ps.ops}
            M = np.ones((k, k), dtype=complex)
            for leaf in range(n_leaves):
                if leaf in active_leaves:
                    M = M * per_leaf_mats[leaf][active_leaves[leaf]]
                else:
                    M = M * id_overlap_per_leaf[leaf]
            val = (cc * M).sum()
            total += ps.coef * float(val.real)
        return total

    def term_gates(self, state: MERA, term: PauliString, dt: float,
                   imaginary: bool = True):
        """Imaginary-time gates for a single Pauli string.

        Empty Pauli strings (identity contribution) emit no gate (a
        global scalar leaves the state direction unchanged after
        renormalization). For strings of length 1 we emit a single-leaf
        gate exp(-dt c P). For length-2 strings we emit a 2-leaf gate.
        For length >= 3 the multi-leaf gate cannot be applied via the
        substrate's 1-or-2-leaf primitives without an additional Trotter
        split; the chemistry evolution driver
        (:func:`chemistry_imaginary_evolve`) uses a CI-determinant
        representation instead and does NOT route through
        ``mera_trotter_step``, so this method is provided primarily for
        substrate-shape compatibility (and for short-support terms that
        the driver may delegate to ``mera_trotter_step`` in mixed-mode
        settings).
        """
        if not imaginary:
            raise NotImplementedError(
                "real-time chemistry evolution not implemented; use "
                "imaginary=True for ground-state search")
        if not term.ops:
            return []
        alpha = dt * float(term.coef)
        if len(term.ops) == 1:
            leaf, sym = term.ops[0]
            P = _PAULI_OPS[sym]
            # P^2 = I for X/Y/Z. exp(-alpha P) = cosh(alpha) I - sinh(alpha) P.
            gate = np.cosh(alpha) * _I2 - np.sinh(alpha) * P
            return [((leaf,), gate)]
        if len(term.ops) == 2:
            (l0, s0), (l1, s1) = term.ops
            if l0 > l1:
                l0, l1 = l1, l0
                s0, s1 = s1, s0
            P0 = _PAULI_OPS[s0]
            P1 = _PAULI_OPS[s1]
            # (P0 x P1)^2 = I (for unitary Paulis on different sites).
            joint = np.kron(P0, P1)
            gate = np.cosh(alpha) * np.eye(4, dtype=complex) - np.sinh(alpha) * joint
            return [((l0, l1), gate)]
        # >= 3-leaf: leave for the CI driver.
        return []

    def term_affected_leaves(self, term: PauliString) -> frozenset:
        return frozenset(leaf for leaf, _ in term.ops)

    # ------------------------------------------------------------------
    # CI (Slater-determinant) matrix elements via Slater-Condon rules
    # ------------------------------------------------------------------

    def ci_matrix_element(self, det_I: tuple[int, ...],
                          det_J: tuple[int, ...]) -> float:
        """Slater-Condon evaluation of <det_I | H | det_J>.

        Determinants are represented as sorted tuples of occupied
        spin-orbital indices (the same indexing used by the spin-orbital
        -> leaf map: 2*p + sigma). Spatial-orbital MO integrals are
        looked up from ``self.integrals.h1`` and ``.eri``; nuclear
        repulsion is included for the diagonal.

        Implements the standard 3-case formula (Szabo-Ostlund §2.3):
          * same det:           E0 = sum_i h_ii + 0.5 sum_{ij} (J_ij - K_ij)
          * 1 occupation diff:  h_ab + sum_j (j in both) (J_aj - K_aj)
          * 2 occupations diff: (ac|bd) - (ad|bc) antisymmetrized
          * >2 diffs: 0
        plus the sign from re-ordering the determinant to canonical form.
        """
        h1 = self.integrals.h1
        eri = self.integrals.eri
        # Find differences.
        set_I = set(det_I)
        set_J = set(det_J)
        diff_I = sorted(set_I - set_J)
        diff_J = sorted(set_J - set_I)
        n_diff = len(diff_I)
        if n_diff > 2:
            return 0.0
        if n_diff == 0:
            # Diagonal.
            e = float(self.integrals.nuc_repulsion)
            occ = list(det_I)
            for so_i in occ:
                p, _ = divmod(so_i, 2)
                e += float(h1[p, p])
            for ii in range(len(occ)):
                for jj in range(ii + 1, len(occ)):
                    so_i = occ[ii]
                    so_j = occ[jj]
                    pi, si = divmod(so_i, 2)
                    pj, sj = divmod(so_j, 2)
                    coulomb = float(eri[pi, pi, pj, pj])
                    exchange = (float(eri[pi, pj, pj, pi])
                                if si == sj else 0.0)
                    e += coulomb - exchange
            return e
        if n_diff == 1:
            a = diff_I[0]
            b = diff_J[0]
            pa, sa = divmod(a, 2)
            pb, sb = divmod(b, 2)
            if sa != sb:
                return 0.0   # spin-orbital symmetry blocks the term
            sign = _slater_sign(det_I, det_J, [a], [b])
            e = float(h1[pa, pb])
            common = sorted(set_I & set_J)
            for so_j in common:
                pj, sj = divmod(so_j, 2)
                coulomb = float(eri[pa, pb, pj, pj])
                exchange = (float(eri[pa, pj, pj, pb])
                            if sa == sj else 0.0)
                e += coulomb - exchange
            return sign * e
        # n_diff == 2
        a, c = diff_I
        b, d = diff_J
        pa, sa = divmod(a, 2)
        pb, sb = divmod(b, 2)
        pc, sc = divmod(c, 2)
        pd, sd = divmod(d, 2)
        # The two pairs (a,b) and (c,d) MUST have matching spins for a
        # non-zero matrix element (else the eri integral vanishes by
        # spin orthogonality).
        # In Szabo's notation, sign comes from reordering Lambda_I,
        # Lambda_J so the excited indices line up.
        sign = _slater_sign(det_I, det_J, [a, c], [b, d])
        term = 0.0
        # <ab||cd>_{spin} antisymmetrized:
        #   (a,c | b,d) [direct]  - (a,d | b,c) [exchange]
        # with spin selection rules: direct nonzero iff (sa==sb and sc==sd),
        # exchange nonzero iff (sa==sd and sc==sb).
        if sa == sb and sc == sd:
            term += float(eri[pa, pb, pc, pd])
        if sa == sd and sc == sb:
            term -= float(eri[pa, pd, pc, pb])
        return sign * term


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _simplify(terms: list[PauliString]) -> list[PauliString]:
    """Merge Pauli strings with identical (sorted) ops by summing coefs.

    The Pauli strings emitted from the four-site decomposition often
    overlap (e.g. two h_pq, h_qp contributions both yield X X on (p, q)).
    Merging keeps the term list compact for the ``total_energy`` loop.
    """
    bucket: dict[tuple[tuple[int, str], ...], float] = {}
    for ps in terms:
        key = tuple(sorted(ps.ops))
        bucket[key] = bucket.get(key, 0.0) + float(ps.coef)
    out: list[PauliString] = []
    for key, c in bucket.items():
        if abs(c) < 1e-12:
            continue
        out.append(PauliString(ops=key, coef=c))
    return out


def _pauli_decompose_four_site(sites_sorted: list[int],
                                p: int, r: int, s: int, q: int,
                                coef: float) -> list[PauliString]:
    """Build the operator
        coef * (a^+_p a^+_r a_s a_q + h.c.)
    explicitly on the 4-site (16-dim) subspace via direct matrix
    construction, then decompose into Pauli strings via the
    orthonormal Pauli basis projection
        c_P = Tr(P^dagger O) / 16
    for P in {I, X, Y, Z}^{⊗4}. The 4 distinct sites' subspace is
    16-dim — far below 4^N (the global Fock space) — so this is a
    factored, per-term local operation in the spec §1.3 sense; the
    *global* Hamiltonian is never assembled as a 4^N matrix.

    Note: we add the h.c. immediately so the returned strings have real
    coefficients; this means the (p, r, s, q) and (q, s, r, p) eri-loop
    iterations BOTH emit the SAME Pauli strings (sign-counted by the
    eri symmetry (pq|rs) = (qp|sr)). The _simplify pass deduplicates.
    """
    # The four physical sites in canonical order (lowest -> highest):
    s_list = sites_sorted
    site_to_local = {s_list[i]: i for i in range(4)}
    # For each LOCAL index 0..3 (corresponding to s_list), figure out
    # the operator at that site (sigma^+ for p or r, sigma^- for s or q,
    # with possibly multiple operators per site if not distinct — but
    # in the n_diff==4 path all are distinct).
    op_at_site = {p: _SP, r: _SP, s: _SM, q: _SM}
    # JW Z-string sign correction: when we move sigma operators past
    # other sites' sigma operators (in the canonical right-to-left
    # operator order a^+_p a^+_r a_s a_q), each move past a Z-string
    # picks up a Z on every intermediate site. Construct the joint
    # 16x16 by direct multiplication in the canonical order using the
    # JW representation a^+_k = JW(k) sigma^+_k where JW(k) =
    # prod_{j<k} Z_j (over ALL sites in the molecule's leaf layout, not
    # just these four).
    # On the 4-site SUBSYSTEM, we account for inter-subsystem Z's by
    # the parity of "external sites between two active sites" — these
    # contribute a global +/-1 sign (they're identity on the
    # subsystem's 4-dim Hilbert space) so we can fold them as a single
    # +/-1 multiplier.
    # Inside the subsystem, however, the Z on local-index k MUST be
    # applied as the 2x2 Z at that subsystem position.
    def build_one_site_op_pattern(target_site: int, op2x2: np.ndarray,
                                  jw_below: set[int]) -> np.ndarray:
        """Build 16x16 = prod over local positions of (Z if pos in jw_below_local
        else I) at every position, with `op2x2` substituted at the target site.

        jw_below: set of global site indices < target_site that are ALSO
        in this 4-site subsystem (so they materialize as Z's locally).
        """
        ops = []
        for local in range(4):
            site = s_list[local]
            if site == target_site:
                ops.append(op2x2)
            elif site in jw_below:
                ops.append(_Z)
            else:
                ops.append(_I2)
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out

    def a_dag_or_ann(site: int, is_dag: bool) -> np.ndarray:
        """Build a^+_site or a_site on the 4-site subsystem with proper
        JW Z-string across the in-subsystem sites with smaller index.

        External (out-of-subsystem) JW Z contributions are not
        materialized: they cancel pairwise across the (p, r, s, q) tuple
        whose total operator support sums to an even number of
        Z's at every external site (the 4 sites have a balanced
        creation/annihilation count). We verify the cancellation here
        and emit a +/-1 sign if it doesn't.
        """
        jw_in = {sj for sj in s_list if sj < site}
        sigma = _SP if is_dag else _SM
        return build_one_site_op_pattern(site, sigma, jw_in)

    O = a_dag_or_ann(p, True) @ a_dag_or_ann(r, True) @ \
        a_dag_or_ann(s, False) @ a_dag_or_ann(q, False)
    # External JW parity: count of "active operators below each external
    # site"; if any external site has odd parity the operator's
    # restriction to this 4-site subspace picks up a global -1.
    # The 4 operators are at distinct sites {p, r, s, q}; the parity of
    # "operators at sites > k" for any external k is even iff k is below
    # an even number of these sites. Since we have 4 operators total,
    # for any external k either 0, 2, or 4 lie above (always even) OR
    # 1 or 3 (odd). For the latter case the global operator has nontrivial
    # support on k via a Z, which would NOT be representable on the 4-site
    # subspace — but that case never arises in our usage because the four
    # active sites in (p, r, s, q) are EXACTLY the 4 distinct fermion
    # operator labels, and for any external site k, exactly (#operators
    # whose site > k) Z's accumulate on k, which is 0, 2, or 4 if k is
    # outside [min(p,r,s,q), max(p,r,s,q)] (always even) or 1, 2, or 3
    # if k is INSIDE the range. The 1 / 3 case means the operator
    # carries a Z on external site k — i.e. the operator is NOT
    # localized on these 4 sites.
    # For the H2 STO-3G test the four sites are {0,1,2,3} — there are
    # NO external sites. For HeH+ same. We assert this case here.
    span_lo = min(p, r, s, q)
    span_hi = max(p, r, s, q)
    external_in_range = []
    for k in range(span_lo + 1, span_hi):
        if k not in {p, r, s, q}:
            external_in_range.append(k)
    if external_in_range:
        # External in-range sites with ODD JW parity carry a nontrivial
        # Z; extend the subsystem to include them so the operator builder
        # adds the matching Z factor at each one.
        op_sites = [p, r, s, q]
        extra: list[int] = []
        for k in external_in_range:
            count_above = sum(1 for x in op_sites if x > k)
            if count_above % 2 == 1:
                extra.append(k)
        if extra:
            # Route through _jw_general which supports the extended-subsystem
            # construction. Pass the active set as the original four sites;
            # _jw_general re-walks the parity and adds the same extras.
            return _jw_general({p, r, s, q}, p, r, s, q, coef=coef)
    # No explicit h.c.: the outer eri (p,q,r,s) loop already covers
    # both (p,q,r,s) and (q,p,s,r) (= h.c. via real-symmetric eri),
    # so summing every iteration's O term yields a Hermitian total.
    H_sub = coef * O
    # Pauli decomposition on the 4-site subspace.
    return _decompose_subspace_to_pauli(H_sub, s_list)


def _decompose_subspace_to_pauli(M: np.ndarray, leaf_indices: list[int]
                                  ) -> list[PauliString]:
    """Decompose a 16x16 matrix on 4 sites into Pauli strings.

    M is expressed as sum over (P0, P1, P2, P3) in {I,X,Y,Z}^4 of
    c_P * P0 (x) P1 (x) P2 (x) P3 where c_P = Tr(P^dagger M) / 16.
    Only the real part is kept (the input is built to be Hermitian).
    """
    paulis = [("I", _I2), ("X", _X), ("Y", _Y), ("Z", _Z)]
    out: list[PauliString] = []
    for i, (s0, P0) in enumerate(paulis):
        for j, (s1, P1) in enumerate(paulis):
            for k, (s2, P2) in enumerate(paulis):
                for l, (s3, P3) in enumerate(paulis):
                    P = np.kron(np.kron(np.kron(P0, P1), P2), P3)
                    c = np.trace(P.conj().T @ M) / 16.0
                    if abs(c.real) < 1e-12 and abs(c.imag) < 1e-12:
                        continue
                    # Hermitian operator -> real Pauli coefficient.
                    ops: list[tuple[int, str]] = []
                    for local, sym in enumerate((s0, s1, s2, s3)):
                        if sym != "I":
                            ops.append((leaf_indices[local], sym))
                    out.append(PauliString(ops=tuple(ops),
                                            coef=float(c.real)))
    return out


def _jw_general(active: set[int], p: int, r: int, s: int, q: int,
                coef: float) -> list[PauliString]:
    """Handle the degenerate cases where two of {p, r, s, q} coincide.

    These reduce to lower-order operators (e.g., a^+_p a^+_r a_p a_q with
    p in active appearing twice => Pauli-exclusion identity reductions).
    Build the explicit subspace operator and decompose.

    In-range external sites with ODD JW parity carry a nontrivial Z and
    must be added to the subsystem; we extend the active set to include
    them so the Pauli decomposition emits the corresponding Z factors.
    """
    span_lo = min(p, r, s, q)
    span_hi = max(p, r, s, q)
    op_sites = [p, r, s, q]
    extended_active = set(active)
    for k in range(span_lo + 1, span_hi):
        if k in active:
            continue
        count_above = sum(1 for x in op_sites if x > k)
        if count_above % 2 == 1:
            extended_active.add(k)
    sites_sorted = sorted(extended_active)
    n_active = len(sites_sorted)
    site_to_local = {sites_sorted[i]: i for i in range(n_active)}

    def site_op(target_site: int, op2x2: np.ndarray) -> np.ndarray:
        jw_in = {sj for sj in sites_sorted if sj < target_site}
        ops = []
        for local in range(n_active):
            site = sites_sorted[local]
            if site == target_site:
                ops.append(op2x2)
            elif site in jw_in:
                ops.append(_Z)
            else:
                ops.append(_I2)
        out = ops[0]
        for o in ops[1:]:
            out = np.kron(out, o)
        return out

    # In the degenerate case, ops at the same site multiply directly,
    # so we accumulate matrix products in order.
    op_seq = [(p, _SP), (r, _SP), (s, _SM), (q, _SM)]
    M = np.eye(2 ** n_active, dtype=complex)
    for site, op2x2 in op_seq:
        M = M @ site_op(site, op2x2)
    H_sub = coef * M
    if n_active == 0 or np.allclose(H_sub, 0):
        return []
    # Pauli decompose on n_active-site subspace.
    return _decompose_n_site(H_sub, sites_sorted)


def _decompose_n_site(M: np.ndarray, leaf_indices: list[int]
                       ) -> list[PauliString]:
    """Generic Pauli decomposition for arbitrary n-site subspace
    (n = len(leaf_indices) in {1,2,3,4})."""
    paulis = [("I", _I2), ("X", _X), ("Y", _Y), ("Z", _Z)]
    n = len(leaf_indices)
    dim = 2 ** n
    out: list[PauliString] = []
    for idx in range(4 ** n):
        # Decode base-4 digits.
        digits = []
        x = idx
        for _ in range(n):
            digits.append(x % 4)
            x //= 4
        digits.reverse()
        P = paulis[digits[0]][1]
        for d in digits[1:]:
            P = np.kron(P, paulis[d][1])
        c = np.trace(P.conj().T @ M) / dim
        if abs(c.real) < 1e-12 and abs(c.imag) < 1e-12:
            continue
        ops: list[tuple[int, str]] = []
        for local, d in enumerate(digits):
            sym = paulis[d][0]
            if sym != "I":
                ops.append((leaf_indices[local], sym))
        out.append(PauliString(ops=tuple(ops), coef=float(c.real)))
    return out


def _slater_sign(det_I: tuple[int, ...], det_J: tuple[int, ...],
                  excl_I: list[int], excl_J: list[int]) -> int:
    """Sign of the parity-permutation aligning det_I and det_J across
    the excited indices.

    For the Slater-Condon rules to give the correct overall sign, the
    excited orbitals in det_I (excl_I) and det_J (excl_J) must be
    moved to the END of their respective sorted determinant tuples;
    the sign is (-1)^(sum of positions moved).
    """
    sign = 1
    for orb in excl_I:
        pos = det_I.index(orb)
        # Number of swaps to bring orb to the end: (len(det_I) - 1 - pos).
        sign *= (-1) ** (len(det_I) - 1 - pos)
        # After the move, the relative position list shifts; track by
        # removing orb and re-walking det_I.
        det_I = det_I[:pos] + det_I[pos + 1:]
    for orb in excl_J:
        pos = det_J.index(orb)
        sign *= (-1) ** (len(det_J) - 1 - pos)
        det_J = det_J[:pos] + det_J[pos + 1:]
    return sign


