"""Holographic compilation: MERA-layer-sequence record (§12.10, partial).

Spec reference: ``QFT_PCN_ARCHITECTURE.md`` §12.10. The §12.10 vision is
that a MERA-structured QPCN is literally a compilation pipeline: each
MERA layer is an RG (renormalization group) coarse-graining pass — a
compiler pass that lifts leaf-level operations into higher-level
effective tensors. High-level constructs live near the root; low-level
operations live at the leaves; layer ℓ is the ℓ-th lowering /
optimization pass.

Honest scope (D15): this module currently records the layer sequence
of an already-encoded MERA and delegates equivalence to shared-state
identity + Wilson-loop signature comparison. Full §12.10 compilation
passes — RG-flow optimization, layer truncation, equivalence-preserving
local rewrites (compress, fuse, eliminate) — are substrate-wide future
work tracked in ``EXTENSIONS.md``. Per §1.1 (binding = bond
entanglement, never classical lookup), every record carries REAL MERA
tensors at its layer index; there is no AST-pass surrogate.

API
---
- :class:`CompilationLayer` — one MERA-layer record, with its real
  tensors and a snapshot of the full encoded state at that layer.
- :func:`record_mera_layer_sequence` — split ``encode_mera``'s output
  into a list of ``CompilationLayer`` records, one per MERA layer
  ℓ ∈ [0, L). This is a record-only pass: no optimization or rewrite
  is applied.
- :func:`verify_layer_state_identical` — tautological identity check
  on the shared encoded state via §12.2
  ``compute_wilson_loop_signature``. Honestly named: because all
  layers from a single ``record_mera_layer_sequence`` call reference
  the same state, this returns ``True`` trivially for intra-call
  layer pairs; cross-call pairs reduce to a Wilson-signature compare.
  This is NOT a §12.10 semantics-preserving-rewrite equivalence
  oracle (deferred to ``EXTENSIONS.md``).
- :func:`detect_compilation_convergence` — uses §12.8
  ``detect_dpt_in_wake_sleep_log`` to detect a compilation phase
  transition (i.e., a snapshot sequence has NOT converged when a DPT
  event fires between consecutive snapshots).

Cost: ``record_mera_layer_sequence`` is O(L) tensor handle copies plus
one ``encode_mera`` call; ``verify_layer_state_identical`` is one
Wilson-loop signature pair (§12.2 cost: O(N·d^4·L));
``detect_compilation_convergence`` runs L-1 ``MERA.inner`` calls (§12.8
cost: double-network ascent per pair).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.qft_pcn.composition.dynamical_pt import (
    DEFAULT_DPT_ECHO_THRESHOLD,
    DEFAULT_DPT_RELATIVE_DROP,
    DPTEvent,
    WakeSleepSnapshot,
    detect_dpt_in_wake_sleep_log,
)
from src.qft_pcn.composition.topological_invariants import (
    LoopSignature,
    compute_wilson_loop_signature,
)
from src.qft_pcn.logic.ast import Node
from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta, encode_mera
from src.qft_pcn.qft.mera import MERA


# ---------------------------------------------------------------------------
# CompilationLayer record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompilationLayer:
    """One MERA-layer record.

    ``layer_index`` is the MERA layer ℓ ∈ [0, L); ``isometries``,
    ``disentanglers``, ``inter_disentanglers`` are the REAL per-layer
    tensor lists from the source ``MERA`` (shared references — these are
    read-only views into the encoded state, never mutated).

    ``state`` is the full encoded MERA standing in for the program's
    semantic content at this layer (used by §12.2 Wilson-loop signature
    verification and §12.8 DPT convergence detection). The same MERA is
    shared across all layers of a single recording; the layer index
    distinguishes which RG pass the record refers to.

    ``meta`` is the ``MeraEncodingMeta`` from the ``encode_mera`` call
    that produced ``state``; retained so callers can correlate a layer
    back to leaf/node bookkeeping (binder leaves, type tags, ...).
    """
    layer_index: int
    isometries: tuple[np.ndarray, ...]
    disentanglers: tuple[np.ndarray, ...]
    inter_disentanglers: tuple[np.ndarray, ...]
    state: MERA
    meta: MeraEncodingMeta

    def wilson_signature(self) -> LoopSignature:
        """Wilson-loop signature of this layer's encoded state (§12.2)."""
        return compute_wilson_loop_signature(self.state)


# ---------------------------------------------------------------------------
# MERA-layer-sequence recording (no optimization pass)
# ---------------------------------------------------------------------------


def record_mera_layer_sequence(
    ast: Node,
    *,
    n_nodes_max: int = 32,
    chi_layer: int = 16,
) -> list[CompilationLayer]:
    """Record the MERA-layer sequence of an AST's encoded state.

    Calls ``encode_mera`` once (the real §6 MERA encoder, not an AST
    walker) and emits one :class:`CompilationLayer` per MERA layer
    ℓ ∈ [0, L). Each record carries:

    * The per-layer disentangler / inter-disentangler / isometry tensors
      from the encoded state (the actual MERA operations that perform
      the coarse-graining at depth ℓ — §12.10 "isometries that
      introduce abstractions ... disentanglers that simplify
      entanglement structure").
    * A reference to the same encoded ``MERA`` state, so any layer's
      :meth:`CompilationLayer.wilson_signature` reads the §12.2
      invariant of the whole encoded program.

    Honest scope (D15): this is a RECORD-ONLY operation. No
    equivalence-preserving rewrite, no layer truncation, no
    RG-flow optimization pass is applied. The returned list mirrors
    the layer indexing of the already-encoded MERA. Real §12.10
    compilation passes are tracked in ``EXTENSIONS.md``.

    The full RG-flow semantics (§12.10): layer 0 is the lowest-level
    pass (leaves → first coarse-graining), layer L-1 is the highest
    level (just below the top tensor). Walking the returned list in
    order = walking the layer sequence from low-level to high-level.

    Returns the layer sequence in MERA layer order. The list is always
    non-empty for any AST that encodes to N ≥ 2 leaves (L ≥ 1).
    """
    state, meta = encode_mera(
        ast, n_nodes_max=n_nodes_max, chi_layer=chi_layer)
    layers: list[CompilationLayer] = []
    for ell in range(state.L):
        layers.append(CompilationLayer(
            layer_index=ell,
            isometries=tuple(state.isometries[ell]),
            disentanglers=tuple(state.disentanglers[ell]),
            inter_disentanglers=tuple(state.inter_disentanglers[ell]),
            state=state,
            meta=meta,
        ))
    return layers


# ---------------------------------------------------------------------------
# Shared-state identity check (tautological for intra-call layer pairs)
# ---------------------------------------------------------------------------


def verify_layer_state_identical(
    layer_a: CompilationLayer,
    layer_b: CompilationLayer,
    *,
    tol: float | None = None,
) -> bool:
    """Check whether two records reference Wilson-signature-equal states.

    Honest scope (D15): because :func:`record_mera_layer_sequence`
    makes all layers from one call share the SAME encoded ``state``,
    this function is tautologically ``True`` for any two layers from
    the same recording — there is no per-layer rewrite that could
    change the underlying state. For two layers from independent
    ``record_mera_layer_sequence`` calls on the same AST, the result is
    ``True`` because ``encode_mera`` is deterministic at the AST level
    (per §1.1, bond entanglement is a function of the AST). For
    layers from compilations of different programs, the result is
    ``False`` unless the programs are accidentally Wilson-loop-
    equivalent.

    This is NOT the §12.10 "optimized-vs-original program equivalence"
    oracle described in the spec — that requires real equivalence-
    preserving rewrite passes to exist, which are tracked as deferred
    work in ``EXTENSIONS.md``. The function name reflects the honest
    semantic: a shared-state identity test backed by §12.2 Wilson-loop
    signature comparison (a necessary, not sufficient, program
    invariant per ``topological_invariants``'s module docstring).

    ``tol`` is forwarded to :meth:`LoopSignature.approx_equal`; default
    is the topological_invariants module's ``_SIGNATURE_TOL``.
    """
    sig_a = layer_a.wilson_signature()
    sig_b = layer_b.wilson_signature()
    if tol is None:
        return sig_a.approx_equal(sig_b)
    return sig_a.approx_equal(sig_b, tol=tol)


# ---------------------------------------------------------------------------
# §12.8-backed convergence detection
# ---------------------------------------------------------------------------


def detect_compilation_convergence(
    snapshots: Sequence[CompilationLayer | MERA],
    *,
    echo_threshold: float = DEFAULT_DPT_ECHO_THRESHOLD,
    relative_drop: float = DEFAULT_DPT_RELATIVE_DROP,
) -> bool:
    """Decide whether a compilation snapshot sequence has converged.

    Per §12.10's risk table: "Optimization 'infinite regress' — keep
    applying passes without termination | Use the §12.8
    dynamical-phase-transition detector to identify when optimization
    has converged."

    Construction: wrap each snapshot in a :class:`WakeSleepSnapshot`
    (using the snapshot's MERA state and its position in the sequence
    as a synthetic "cycle index"), then run
    :func:`detect_dpt_in_wake_sleep_log` over the wrapped list. The
    DPT detector flags non-analyticities in the Loschmidt echo between
    consecutive snapshots:

    * **No DPT events** = compilation has converged (consecutive
      snapshots are smoothly related; no regime change).
    * **One or more DPT events** = compilation has NOT converged
      (some pass triggered a structural reorganization).

    This function returns ``True`` when there are zero DPT events, i.e.
    when the compilation pass sequence has reached a stable point.
    Fewer than two snapshots are vacuously convergent (no pass-induced
    transition could be detected).

    Accepts either :class:`CompilationLayer` records (a same-program
    layer walk) or raw :class:`MERA` states (an arbitrary pass-sequence
    snapshot list); both are routed through :meth:`MERA.inner` per
    §12.8. Note: when real §12.10 compilation passes land (per
    ``EXTENSIONS.md``), this detector becomes the termination gate; in
    the current record-only configuration, a snapshot sequence is
    typically constructed by the caller from independent encodings.
    """
    snaps = list(snapshots)
    if len(snaps) < 2:
        return True
    wake_sleep: list[WakeSleepSnapshot] = []
    for i, snap in enumerate(snaps):
        if isinstance(snap, CompilationLayer):
            state = snap.state
        elif isinstance(snap, MERA):
            state = snap
        else:
            raise TypeError(
                "detect_compilation_convergence requires "
                "CompilationLayer or MERA snapshots (§1.1: real bond "
                f"states, not surrogate descriptors); got {type(snap).__name__}")
        wake_sleep.append(WakeSleepSnapshot(cycle_index=i, state=state))
    events = detect_dpt_in_wake_sleep_log(
        wake_sleep,
        echo_threshold=echo_threshold,
        relative_drop=relative_drop,
    )
    return len(events) == 0


__all__ = [
    "CompilationLayer",
    "record_mera_layer_sequence",
    "detect_compilation_convergence",
    "verify_layer_state_identical",
]
