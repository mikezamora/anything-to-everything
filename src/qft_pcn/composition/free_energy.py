"""Substrate-level free-energy and ΔF estimator (spec §13.6 / §7).

This module implements the principled variational free-energy proxy that
backs the orchestrator's lemma-selection / frontier-ordering schedule
(spec §5.3 expected-ΔF). The free energy is computed directly from
tensor-network contractions on the MERA substrate state -- it is NOT a
structural heuristic over node fan-out or depth.

Spec §13.6 (variational principle):

    F_var[ρ] = tr(ρH) + T · tr(ρ log ρ)   ≥   F_true = −T log Z

For a pure MERA state ``|ψ⟩`` we read F_var as

    F(ψ, H, T) = ⟨ψ|H|ψ⟩  +  T · S(ρ_cut)

where ``S(ρ_cut)`` is the von Neumann entanglement entropy across the
canonical mid-network bond cut (the same §12.16 measurement used by
:func:`goal_graph._bond_entanglement_of` and the orchestrator's existing
bond-entanglement signal). Both terms are operator-algebraic readings of
the substrate: the energy comes from the Hamiltonian's
``total_energy(state)`` (a sum of MERA local / two-site expectations);
the entropy comes from ``state.entanglement_entropy(cut)`` (a Schmidt-
spectrum SVD on the materialized state). No classical lookup, no cache
keyed by lemma name — the value is recomputed from the live substrate
on every call (anti-shortcut directive §1.1).

ΔF is the diff of F across a hypothetical lemma application:

    ΔF = F(after_state, H, T) − F(before_state, H, T)

A negative ΔF marks a state transformation that compresses the
substrate's free energy — the favoured frontier candidate per spec §5.3.
A ΔF ≈ 0 marks a no-op transformation (the lemma did not move the
substrate).

The caller supplies both substrate states; this module does NOT clone
the goal_graph or virtually dispatch a lemma (the heavier
snapshot-and-compare path noted in EXTENSIONS.md). The split is
intentional: substrate ΔF is the cheap operator-algebraic primitive,
the schedule-level snapshot is the orchestrator's responsibility.
"""
from __future__ import annotations

from typing import Any, Protocol


class _HasTotalEnergy(Protocol):
    def total_energy(self, state: Any) -> float: ...


class _HasMpsEnergyShape(Protocol):
    # The qft.evolution.energy(state, H) shape: H exposes local_op + bond_op
    N: int
    def local_op(self, site: int) -> Any: ...
    def bond_op(self, site: int) -> Any: ...


class FreeEnergyError(ValueError):
    """Raised when the substrate state and Hamiltonian disagree on shape
    or the requested cut is out of range for the state.
    """


def _energy_expectation(state: Any, hamiltonian: Any) -> float:
    """⟨ψ|H|ψ⟩ on the live substrate.

    Two surfaces are accepted:

    * ``hamiltonian.total_energy(state)`` -- the MERA-shaped Hamiltonian
      contract used by :class:`MeraEvalHamiltonian` and the §15.4
      chemistry Hamiltonians. This is the preferred path because it is
      the SAME operator-algebraic reading that drives imaginary-time
      descent on the substrate.
    * ``qft.evolution.energy(state, H)`` -- the MPS-shaped Hamiltonian
      contract (``H.local_op(k)`` + ``H.bond_op(k)``). Used as a fallback
      so the primitive is callable from MPS-only substrate contexts.

    Neither path materialises a global dense state vector beyond what
    the underlying expectation surfaces already require; both read the
    energy as a sum of local and two-site expectations.
    """
    if hasattr(hamiltonian, "total_energy"):
        return float(hamiltonian.total_energy(state))
    if hasattr(hamiltonian, "local_op") and hasattr(hamiltonian, "bond_op"):
        from ..qft.evolution import energy as _mps_energy
        return float(_mps_energy(state, hamiltonian))
    raise FreeEnergyError(
        f"Hamiltonian of type {type(hamiltonian).__name__!r} exposes "
        "neither total_energy(state) nor local_op/bond_op; cannot read "
        "⟨H⟩ from the substrate"
    )


def _canonical_cut(state: Any) -> int | None:
    """Mid-network cut for the entropy reading, or None if the state is
    too small / does not expose the §5.8 entropy surface.

    Matches the cut convention in :func:`goal_graph._bond_entanglement_of`
    so substrate ΔF stays compatible with the existing bond-entanglement
    schedule signal: ``cut = N // 2 − 1``, clamped to ``[0, N − 2]``.
    """
    N = getattr(state, "N", None)
    if not isinstance(N, int) or N < 2:
        return None
    if not hasattr(state, "entanglement_entropy"):
        return None
    return max(0, min(N - 2, N // 2 - 1))


def _entropy_term(state: Any, cut: int | None) -> float:
    """T-independent von Neumann entropy contribution to F_var.

    Returns ``0.0`` when no cut is available (1-site state, non-MERA
    backend without the entropy surface). The fallback is silent on
    *absence* of the surface but loud on a substrate fault inside the
    surface -- matching the §1.1 architecture-soul invariant that the
    Schmidt-spectrum reading is a measurement, never a fabrication.
    """
    if cut is None:
        return 0.0
    fn = getattr(state, "entanglement_entropy", None)
    if fn is None:
        return 0.0
    try:
        return float(fn(cut))
    except (IndexError, ValueError):
        # Documented contract failures (cut out of range / ill-shaped
        # state) degrade to zero in the §12.16 path-fitness signal --
        # same convention as _bond_entanglement_of. Genuine substrate
        # faults (NumericalInstability, NotImplementedError) propagate.
        return 0.0


def free_energy(
    state: Any,
    hamiltonian: Any,
    *,
    temperature: float = 1.0,
    cut: int | None = None,
) -> float:
    """Variational free energy F = ⟨H⟩ + T · S(ρ_cut) on the substrate.

    Parameters
    ----------
    state:
        A MERA / MPS substrate state. Must expose the energy contract
        consumed by :func:`_energy_expectation` and (optionally) the
        §5.8 ``entanglement_entropy(cut)`` surface. Internal-node
        synthetic results without a ground_state have no substrate
        reading and MUST be filtered out by the caller -- this routine
        deliberately raises rather than fabricating a zero in that case.
    hamiltonian:
        The Hamiltonian operator. Must expose either
        ``total_energy(state)`` (MERA-shaped) or ``local_op``/``bond_op``
        (MPS-shaped). The lemma-scoring caller passes the composed
        Hamiltonian under which the substrate state was measured (the
        ``ChildResult.hamiltonian`` field is the source of truth).
    temperature:
        ``T`` in the §13.6 expression. Defaults to ``1.0`` (the natural
        scale: a single bit of entropy weighs the same as a single unit
        of energy). Strictly non-negative; the variational principle is
        defined for ``T ≥ 0`` and the §13.6 inequality is tight at
        ``ρ = e^{−H/T}/Z``.
    cut:
        Optional explicit bond cut. Defaults to the canonical mid-
        network cut (``N // 2 − 1``, clamped), matching the §12.16
        path-fitness convention used by ``_bond_entanglement_of``.

    Returns
    -------
    float
        ``⟨H⟩ + T · S(ρ_cut)``. Real-valued by construction (``H`` is
        Hermitian, ``S`` is non-negative).

    Raises
    ------
    FreeEnergyError
        If ``state`` is missing (None), if the Hamiltonian exposes no
        energy surface, or if ``temperature`` is negative. The §1.1
        anti-shortcut directive: a missing substrate is NOT silently
        zeroed -- the caller must filter.
    """
    if state is None:
        raise FreeEnergyError(
            "free_energy requires a real substrate state; got None. "
            "Synthetic _JointResult (internal-node joins) carry no "
            "ground_state and must be filtered by the caller."
        )
    if not isinstance(temperature, (int, float)) or temperature < 0.0:
        raise FreeEnergyError(
            f"temperature must be a non-negative real; got {temperature!r}"
        )
    if hamiltonian is None:
        raise FreeEnergyError(
            "free_energy requires a Hamiltonian; got None. The §13.6 "
            "variational principle is defined relative to an operator H."
        )
    e = _energy_expectation(state, hamiltonian)
    chosen_cut = cut if cut is not None else _canonical_cut(state)
    s = _entropy_term(state, chosen_cut)
    return float(e + float(temperature) * s)


def delta_free_energy(
    before_state: Any,
    after_state: Any,
    hamiltonian: Any,
    *,
    temperature: float = 1.0,
    cut: int | None = None,
) -> float:
    """ΔF = F(after) − F(before) computed on the live substrate.

    The principled signed lemma-scoring signal: a negative ΔF marks a
    state transformation that compresses the substrate free energy
    (spec §5.3 ΔF-favoured frontier candidate); ΔF ≈ 0 marks a no-op
    transformation (the lemma did not move the substrate).

    Both ``before_state`` and ``after_state`` are read from the live
    substrate on every call -- this routine does NOT cache by lemma id
    (anti-shortcut directive §1.1). The Hamiltonian is shared (the
    spec §13.6 comparison is between two states under the SAME
    operator); a separate per-state Hamiltonian would be a category
    error, not a refinement.

    Parameters
    ----------
    before_state, after_state:
        Substrate states (MERA / MPS) bracketing the hypothetical
        transformation. Both MUST be real states (anti-shortcut: no
        sentinel "absence" is silently treated as zero free energy).
    hamiltonian:
        The shared operator. See :func:`free_energy`.
    temperature, cut:
        Forwarded to both :func:`free_energy` calls.

    Returns
    -------
    float
        Signed free-energy delta. Negative ⇒ the transformation is
        favoured (lower F); positive ⇒ disfavoured; ≈ 0 ⇒ no-op.
    """
    f_before = free_energy(
        before_state, hamiltonian,
        temperature=temperature, cut=cut,
    )
    f_after = free_energy(
        after_state, hamiltonian,
        temperature=temperature, cut=cut,
    )
    return float(f_after - f_before)
