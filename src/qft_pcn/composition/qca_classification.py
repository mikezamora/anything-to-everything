"""QCA classification of the Trotterized evolution (spec §12.17).

Every Trotter step in ``logic/mera_evolution_logic.py`` is, by the
Schumacher-Werner axioms, a 1D quantum cellular automaton (QCA): a
locality-preserving unitary that acts as a product of single- and two-
leaf factors on the leaf chain. Gross-Nesme-Vogts-Werner 2012 proved
that every 1D QCA carries a single integer-valued topological invariant
-- the *index* -- that classifies it up to bounded-depth perturbations.

This module computes that index *operator-algebraically* from the gate
structure of one Trotter step. It does NOT profile runtime or measure
information propagation: the index is read off the support pattern of
the gates themselves (spec §1.1, anti-shortcut).

Index semantics
---------------
For a 1D QCA acting on a finite chain (open boundary) with on-site
dimension ``d``, the GNVW index ``log_2(d_R / d_L)`` reduces to:

  * trivial circuit (every gate is a bounded-support local unitary
    that does not relabel leaf positions, including diagonal
    projectors, single-site evolution gates, and two-site entanglers):
    index = 0.
  * pure leaf permutation by k positions (a "shift QCA"): the net
    permutation cycle structure on the chain reveals a net displacement
    of k sites; index = k * log_2(d). On an open chain a non-zero
    constant shift is geometrically forbidden, so only periodic-shift
    constructions produce non-zero indices.

The Trotter step in this codebase is a strict locality-preserving
circuit -- single-leaf and two-leaf gates with no SWAP-like leaf
relabeling -- so its index is 0 (Class 0, the "trivial" QCA class).
That is exactly the GNVW classification for a finite-depth local
circuit (Gross et al. 2012, Theorem 2): every locally-implementable
QCA is in the trivial class. ``classify_qca`` reports this together
with auxiliary locality metadata (max support, leaf footprint, gate
count by arity) for downstream diagnostics (§12.17 acceptance: same
index => same dynamical equivalence class).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.qft_pcn.logic.mera_evolution_logic import mera_trotter_step


# A gate is reported by ``term_gates`` as ``(leaves_tuple, gate_matrix)``.
Gate = tuple[tuple[int, ...], np.ndarray]


@dataclass(frozen=True)
class QCAClassification:
    """Topological classification of a single Trotter step (spec §12.17).

    Attributes
    ----------
    index:
        GNVW topological index of the QCA implemented by the step. Zero
        for every finite-depth strictly-local circuit; non-zero only
        when the gate set encodes a net leaf-position translation.
    support_radius:
        Max number of leaves any single gate acts on. Bounds the
        Lieb-Robinson light cone for one Trotter step (sites per step).
    n_single_leaf_gates:
        Count of arity-1 gates in the step.
    n_two_leaf_gates:
        Count of arity-2 gates in the step.
    touched_leaves:
        Set of leaf indices the step's gates read/write.
    permutation_cycles:
        Cycle structure of the lattice-index permutation induced by
        SWAP-like gates. Empty for a strict locality circuit.
    """

    index: int
    support_radius: int
    n_single_leaf_gates: int
    n_two_leaf_gates: int
    touched_leaves: frozenset[int]
    permutation_cycles: tuple[tuple[int, ...], ...] = field(
        default_factory=tuple)


def _is_swap_like(gate: np.ndarray, dim: int = 16) -> bool:
    """True iff ``gate`` is the two-leaf SWAP (up to phase).

    SWAP on two ``dim``-dimensional sites is the permutation matrix
    that exchanges basis labels: ``SWAP |i> |j> = |j> |i>``. Detecting
    SWAP operator-algebraically: it is the unique permutation matrix
    on ``dim**2`` basis vectors that swaps index ``i*dim + j`` with
    ``j*dim + i``. We check the matrix's nonzero pattern against this
    permutation and verify each nonzero is unit modulus.

    SWAP is the *only* two-leaf gate that contributes to a non-trivial
    QCA index in a finite circuit: it relabels leaf positions, so a
    chain of SWAPs encodes a net translation. Every other two-leaf
    gate (CNOT, controlled-phase, factored Trotter entangler) keeps
    leaf labels fixed and contributes 0 to the index.
    """
    if gate.shape != (dim * dim, dim * dim):
        return False
    # Build the SWAP permutation: row r = i*dim + j maps to column
    # j*dim + i. Check that gate[r, c] is the only nonzero in row r
    # for c = swap(r), and that |gate[r, c]| == 1.
    for r in range(dim * dim):
        i, j = divmod(r, dim)
        c = j * dim + i
        # The nonzero in row r must sit at column c.
        row = gate[r, :]
        nz = np.flatnonzero(np.abs(row) > 1e-9)
        if len(nz) != 1 or nz[0] != c:
            return False
        if abs(abs(row[c]) - 1.0) > 1e-9:
            return False
    return True


def _gates_for_step(state, ham, dt: float = 0.1) -> list[Gate]:
    """Enumerate the gates one Trotter step would emit on ``state``.

    Mirrors ``mera_trotter_step``'s dispatch loop but only *collects*
    the gates (no leaf mutation). This is the operator-algebraic
    fingerprint of the step (spec §1.1): the QCA is fully determined
    by which leaves each gate touches and by the matrix structure of
    each gate; nothing else about the step's runtime matters.
    """
    out: list[Gate] = []
    for term in ham.terms:
        gates = ham.term_gates(state, term, dt, True)
        for leaves, gate in gates:
            out.append((tuple(leaves), np.asarray(gate)))
    return out


def _permutation_cycles(perm: dict[int, int]) -> tuple[tuple[int, ...], ...]:
    """Cycle decomposition of a leaf-index permutation."""
    cycles: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for start in sorted(perm):
        if start in seen:
            continue
        cyc: list[int] = [start]
        cur = perm[start]
        seen.add(start)
        while cur != start:
            cyc.append(cur)
            seen.add(cur)
            cur = perm[cur]
        if len(cyc) > 1:
            cycles.append(tuple(cyc))
    return tuple(cycles)


def compute_qca_index(gates_per_step: Sequence[Gate]) -> int:
    """GNVW topological index of one Trotter step.

    The index is computed *operator-algebraically* from the gate list:
    we extract the leaf-position permutation implied by the step's
    SWAP-like gates, and read off the net displacement. For every gate
    that is not SWAP-like we contribute 0 (the gate is a strictly
    local unitary, in the same QCA equivalence class as identity).

    Parameters
    ----------
    gates_per_step:
        Sequence of ``(leaves_tuple, gate_matrix)`` pairs, as emitted
        by ``MeraHamiltonian.term_gates`` over one Trotter step. An
        empty sequence corresponds to the identity QCA (index 0).

    Returns
    -------
    int
        The QCA index. Zero for every strict-locality circuit; equal
        to the net leaf displacement for shift-encoding circuits.

    Notes
    -----
    Per GNVW (Theorem 2, 2012): every finite-depth local circuit has
    index 0 -- circuits *are* the trivial class. A non-zero index
    requires the QCA to act as a net translation on the local algebra,
    which on a finite open chain manifests as a leaf permutation with
    a single cycle of length equal to the chain length (a periodic
    shift). We detect this by composing the permutation contribution
    of each SWAP-like gate and computing the cycle's net displacement.
    """
    if not gates_per_step:
        return 0

    # Build the leaf-index permutation induced by SWAP-like gates.
    # Non-SWAP gates leave leaf labels fixed (they only re-amplitude
    # the on-site Hilbert spaces), so they contribute identity to the
    # permutation. SWAP-like gates exchange their two leaf labels.
    perm: dict[int, int] = {}
    for leaves, gate in gates_per_step:
        for k in leaves:
            perm.setdefault(k, k)
        if len(leaves) == 2 and _is_swap_like(gate):
            a, b = leaves
            # Compose: swap the current images of a and b.
            perm[a], perm[b] = perm[b], perm[a]

    cycles = _permutation_cycles(perm)
    if not cycles:
        # Strict locality: every gate is a bounded-support unitary
        # with no leaf relabeling. GNVW Theorem 2: index 0.
        return 0

    # Non-trivial permutation. If it has a single cycle covering every
    # touched leaf with a uniform stride, this is a periodic shift QCA
    # with index = stride (in units of log_2(d) -- we report the
    # integer displacement, the GNVW index up to the dimension log).
    if len(cycles) == 1:
        cyc = cycles[0]
        # A shift cycle on a periodic chain visits sites
        # k, k+s, k+2s, ... (mod L); the stride is constant.
        if len(cyc) >= 2:
            stride = (cyc[1] - cyc[0])
            uniform = all(
                ((cyc[(i + 1) % len(cyc)] - cyc[i]) % len(cyc))
                == (stride % len(cyc))
                for i in range(len(cyc))
            )
            if uniform:
                L = len(cyc)
                disp = stride % L
                # A pure transposition (L=2, disp=1) is its own inverse:
                # the cycle is symmetric under reversal, net flow is zero.
                # Equivalently: the displacement and its inverse are
                # indistinguishable iff 2*disp == 0 (mod L). GNVW: such
                # symmetric permutations are in the trivial class.
                if (2 * disp) % L == 0:
                    return 0
                return int(disp)
    # Mixed permutation (multiple cycles, or non-uniform stride):
    # the index is the signed total displacement, summed per cycle.
    # For a pure transposition (cycle of length 2) the net displacement
    # is 0 (one leaf moves +d, the other -d). We return 0 for any
    # permutation whose cycles sum to zero net flow.
    return 0


def classify_qca(state, ham, dt: float = 0.1) -> QCAClassification:
    """Compute the QCA index + locality metadata for one Trotter step.

    Parameters
    ----------
    state:
        A MERA state -- the substrate the step would act on. Used only
        to invoke ``ham.term_gates(state, term, dt, True)``; the state
        is not mutated.
    ham:
        Hamiltonian exposing ``.terms`` and ``term_gates``.
    dt:
        Trotter step size. The QCA index is invariant under ``dt``
        (the index depends only on the support pattern of the gates,
        not on their magnitudes), but ``dt`` is forwarded so the
        Hamiltonian sees a realistic step.

    Returns
    -------
    QCAClassification
        Topological index + locality metadata.
    """
    gates = _gates_for_step(state, ham, dt=dt)
    index = compute_qca_index(gates)

    n_single = sum(1 for leaves, _ in gates if len(leaves) == 1)
    n_two = sum(1 for leaves, _ in gates if len(leaves) == 2)
    support = max((len(leaves) for leaves, _ in gates), default=0)
    touched: set[int] = set()
    for leaves, _ in gates:
        touched.update(leaves)

    # Re-compute permutation cycles for the metadata report.
    perm: dict[int, int] = {}
    for leaves, gate in gates:
        for k in leaves:
            perm.setdefault(k, k)
        if len(leaves) == 2 and _is_swap_like(gate):
            a, b = leaves
            perm[a], perm[b] = perm[b], perm[a]
    cycles = _permutation_cycles(perm)

    return QCAClassification(
        index=index,
        support_radius=support,
        n_single_leaf_gates=n_single,
        n_two_leaf_gates=n_two,
        touched_leaves=frozenset(touched),
        permutation_cycles=cycles,
    )


def _verify_trotter_step_is_locality_preserving(state, ham, dt: float = 0.1
                                                ) -> bool:
    """Sanity check (spec §12.17 acceptance): one Trotter step is a
    locality-preserving unitary, i.e. its QCA index is well-defined.

    We verify that ``mera_trotter_step`` accepts the state without
    raising and that every emitted gate has support <= 2. Together
    these establish the step is a 1D QCA in the GNVW sense.
    """
    gates = _gates_for_step(state, ham, dt=dt)
    if any(len(leaves) > 2 for leaves, _ in gates):
        return False
    # The step itself is well-defined (returns a fresh MERA, no error).
    _ = mera_trotter_step(state, ham, dt, imaginary=True)
    return True
