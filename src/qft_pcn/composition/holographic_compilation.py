"""Holographic compilation: MERA-as-compilation (§12.10).

Spec reference: ``QFT_PCN_ARCHITECTURE.md`` §12.10. A MERA-structured QPCN is
literally a compilation pipeline. Each MERA layer is an RG (renormalization
group) coarse-graining pass — a compiler pass that lifts leaf-level
operations into higher-level effective tensors. High-level constructs live
near the root; low-level operations live at the leaves; layer ℓ is the
ℓ-th lowering/optimization pass.

Per §1.1 (binding = bond entanglement, never classical lookup), each
``CompilationLayer`` carries the REAL MERA tensors at its layer index — the
intra-pair disentanglers, inter-pair disentanglers, and 2-to-1 isometries
from ``encode_mera``'s output. There is no AST-pass surrogate, no parser
trick: an equivalence check between two layers reduces to a Wilson-loop
signature comparison (§12.2) on the encoded states, and convergence of a
compilation snapshot sequence reduces to a Loschmidt-echo DPT detection
(§12.8) on the encoded states.

API
---
- :class:`CompilationLayer` — one MERA-layer pass, with its real tensors and
  a snapshot of the full encoded state at that layer.
- :func:`compile_to_mera_layers` — split ``encode_mera``'s output into a
  list of ``CompilationLayer`` records, one per MERA layer ℓ ∈ [0, L).
- :func:`verify_layer_equivalence` — uses §12.2 ``compute_wilson_loop_signature``
  to decide whether two layers compute the same function.
- :func:`detect_compilation_convergence` — uses §12.8
  ``detect_dpt_in_wake_sleep_log`` to detect a compilation phase
  transition (i.e., the pass sequence has NOT converged when a DPT
  event fires between consecutive snapshots).

Cost: ``compile_to_mera_layers`` is O(L) tensor handle copies plus one
``encode_mera`` call; ``verify_layer_equivalence`` is one Wilson-loop
signature pair (§12.2 cost: O(N·d^4·L)); ``detect_compilation_convergence``
runs L-1 ``MERA.inner`` calls (§12.8 cost: double-network ascent per pair).
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
    """One MERA-layer compilation pass.

    ``layer_index`` is the MERA layer ℓ ∈ [0, L); ``isometries``,
    ``disentanglers``, ``inter_disentanglers`` are the REAL per-layer
    tensor lists from the source ``MERA`` (shared references — these are
    read-only views into the encoded state, never mutated).

    ``state`` is the full encoded MERA standing in for the program's
    semantic content at this layer (used by §12.2 Wilson-loop signature
    verification and §12.8 DPT convergence detection). The same MERA is
    shared across all layers of a single compilation; the layer index
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
# Compilation: encode_mera -> per-layer passes
# ---------------------------------------------------------------------------


def compile_to_mera_layers(
    ast: Node,
    *,
    n_nodes_max: int = 32,
    chi_layer: int = 16,
) -> list[CompilationLayer]:
    """Compile an AST into a sequence of MERA-layer compilation passes.

    Calls ``encode_mera`` once (the real §6 MERA encoder, not an AST
    walker) and emits one :class:`CompilationLayer` per MERA layer
    ℓ ∈ [0, L). Each layer carries:

    * The per-layer disentangler / inter-disentangler / isometry tensors
      from the encoded state (the actual MERA operations that perform
      the coarse-graining at depth ℓ — §12.10 "isometries that
      introduce abstractions ... disentanglers that simplify
      entanglement structure").
    * A reference to the same encoded ``MERA`` state, so any layer's
      :meth:`CompilationLayer.wilson_signature` reads the §12.2
      invariant of the whole encoded program.

    The full RG-flow semantics (§12.10): layer 0 is the lowest-level
    pass (leaves → first coarse-graining), layer L-1 is the highest
    level (just below the top tensor). Walking the returned list in
    order = walking the compilation pipeline from low-level to
    high-level.

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
# §12.2-backed equivalence verification
# ---------------------------------------------------------------------------


def verify_layer_equivalence(
    layer_a: CompilationLayer,
    layer_b: CompilationLayer,
    *,
    tol: float | None = None,
) -> bool:
    """Decide whether two compilation layers compute the same function.

    Per §12.10 acceptance: "Verify by ... equivalence checking (§12.2
    topological invariants) that the optimized program has identical
    observable behavior."

    Construction: compute :func:`compute_wilson_loop_signature` on each
    layer's encoded state and compare via
    :meth:`LoopSignature.approx_equal`. This is the §12.2 Wilson-loop
    half of the program-equivalence oracle (a necessary but not
    sufficient invariant per the module docstring of
    ``topological_invariants``; sufficient discrimination would require
    the full Jones polynomial, deferred per ``EXTENSIONS.md``).

    Two layers from the SAME compiled state always agree trivially
    (their states are reference-shared). Two layers from independent
    ``compile_to_mera_layers`` calls on the same AST agree because
    ``encode_mera`` is deterministic at this layer (per §1.1, the bond
    entanglement is a function of the AST). Two layers from compilations
    of different programs disagree unless the programs are accidentally
    Wilson-loop-equivalent.

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
    §12.8.
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
    "compile_to_mera_layers",
    "detect_compilation_convergence",
    "verify_layer_equivalence",
]
