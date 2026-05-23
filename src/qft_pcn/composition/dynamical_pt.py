"""Dynamical phase transitions for self-detecting curriculum (§12.8).

Spec reference: ``QFT_PCN_ARCHITECTURE.md`` §12.8 (Heyl-Polkovnikov-Kehrein
2013 / Heyl 2018 review). When a quantum system is driven across a phase
boundary, the Loschmidt echo

    L(t) = |<Psi_0 | Psi(t)>|^2

exhibits non-analyticities at *critical times* — dynamical phase
transitions (DPTs). The QPCN wake-sleep cycle (§10.9) is exactly such a
quench sequence: each cycle adds new primitives, which moves the library
"ground state" |Psi_t> to a new configuration. A sudden drop in the
return amplitude between successive cycles signals that the architecture
has reorganized into a structurally different regime — i.e., gained a
qualitatively new capability.

This module is operator-algebraic per §1.1: the overlap is computed on
REAL bond contractions via :meth:`MERA.inner` (the double-network
cross-tensor algorithm — entanglement-bond pattern, never AST diffing).
The DPT detector consumes real wake-sleep snapshots produced by
``wake_sleep_cycle``.

API
---
- :func:`compute_loschmidt_echo` — return-amplitude <Psi_0|Psi_t> as a
  complex number (zeros / near-zeros signal DPTs).
- :func:`detect_dpt_in_wake_sleep_log` — scan a snapshot sequence and
  surface DPT events at consecutive-cycle return-amplitude dips.
- :class:`WakeSleepSnapshot` — minimal snapshot record (cycle index +
  representative state).
- :class:`DPTEvent` — structured DPT detection (cycle pair, echo
  magnitudes, gap).

Cost: one :meth:`MERA.inner` call per consecutive pair, dominated by the
double-network ascent (negligible vs the wake-sleep cycle itself per
§12.8 "computational cost").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.qft_pcn.qft.mera import MERA


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# A Loschmidt echo |<Psi_0|Psi_t>|^2 below this threshold is treated as a
# dynamical-phase-transition signal. The reference physics literature
# (Heyl 2018 review) characterises DPTs as actual zero crossings of the
# return amplitude in the thermodynamic limit; at finite system size the
# echo dips toward — but does not exactly reach — zero. A threshold of
# 1e-2 corresponds to a ~10x amplitude suppression vs the typical
# slow-drift baseline (~0.5–1.0) between similar-regime library states,
# which empirically separates regime reorganizations from incremental
# updates. Tune per corpus.
DEFAULT_DPT_ECHO_THRESHOLD = 1e-2

# A consecutive-cycle |L| drop of at least this factor (current/previous)
# is flagged even if the absolute echo magnitude is above the zero
# threshold above — captures the *non-analyticity* aspect of a DPT (a
# sudden derivative change) rather than just absolute smallness.
DEFAULT_DPT_RELATIVE_DROP = 0.25


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WakeSleepSnapshot:
    """One wake-sleep cycle's representative state.

    ``cycle_index`` is the §10.9 cycle counter. ``state`` is the MERA
    standing in for the library's joint "ground state" after the cycle
    (in practice the most recent registered solution's state, or a
    designated library-summary state).
    """
    cycle_index: int
    state: MERA


@dataclass(frozen=True)
class DPTEvent:
    """A detected dynamical phase transition between two successive cycles.

    ``cycle_before`` / ``cycle_after`` index the snapshot pair; ``echo``
    is |<Psi_before|Psi_after>|^2 (the Loschmidt echo); ``amplitude`` is
    the raw complex overlap <Psi_before|Psi_after> (sign / phase
    information is preserved so the caller can distinguish a zero
    crossing from a unitarily evolved phase).

    ``trigger`` is one of:
      * ``"absolute"`` — echo magnitude fell below the zero threshold.
      * ``"relative"`` — echo dropped by at least the relative-drop
        factor versus the previous consecutive overlap (non-analytic
        derivative signal).
      * ``"both"`` — both criteria fired.
    """
    cycle_before: int
    cycle_after: int
    amplitude: complex
    echo: float
    trigger: str


# ---------------------------------------------------------------------------
# Loschmidt echo (real bond contraction)
# ---------------------------------------------------------------------------


def compute_loschmidt_echo(state_before: MERA,
                           state_after: MERA) -> complex:
    """Return-amplitude <Psi_0|Psi_t> for two MERA states.

    The overlap is computed on REAL bond contractions via
    :meth:`MERA.inner` (the double-network cross-tensor algorithm). NOT
    an AST diff, NOT a heuristic — this is the entanglement-bond
    structure of §1.1: every shared physical index is summed, every
    isometry is contracted in the causal cone, the result is the exact
    Hilbert-space inner product.

    The Loschmidt *echo* is |<Psi_0|Psi_t>|^2; this function returns the
    raw complex amplitude so callers can examine the phase (zero
    crossings of the amplitude, not just dips in the echo, are the
    signature of a DPT per Heyl 2018).

    States must agree on (N, L, d_local). Neither argument is mutated.
    """
    if not isinstance(state_before, MERA) or not isinstance(state_after, MERA):
        raise TypeError(
            "compute_loschmidt_echo requires two MERA states (§1.1: real "
            "bond contractions, not surrogate descriptors)")
    return complex(state_before.inner(state_after))


# ---------------------------------------------------------------------------
# DPT detector
# ---------------------------------------------------------------------------


def detect_dpt_in_wake_sleep_log(
    snapshots: Sequence[WakeSleepSnapshot],
    *,
    echo_threshold: float = DEFAULT_DPT_ECHO_THRESHOLD,
    relative_drop: float = DEFAULT_DPT_RELATIVE_DROP,
) -> list[DPTEvent]:
    """Scan consecutive wake-sleep snapshots for DPT events.

    For each consecutive pair (snapshots[i], snapshots[i+1]):
      1. Compute the complex overlap and the Loschmidt echo
         L = |<Psi_i|Psi_{i+1}>|^2.
      2. Flag an *absolute* DPT if L falls below ``echo_threshold``
         (Heyl 2018: zero crossings in the thermodynamic limit; at
         finite N a near-zero echo is the operative signal).
      3. Flag a *relative* DPT if L is at most ``relative_drop`` times
         the previous consecutive echo (a non-analyticity in the rate
         — the derivative-style criterion suggested by §12.8's risk
         table for "smooth L(t) over a moving window").

    Returns events in cycle order. An empty list means no transitions
    detected (per §12.8 "at apparent stagnation ... perturb the library
    or change the problem distribution"). With fewer than two snapshots
    no detection is possible and the result is empty.
    """
    if echo_threshold <= 0.0:
        raise ValueError(f"echo_threshold must be > 0 (got {echo_threshold})")
    if not 0.0 < relative_drop < 1.0:
        raise ValueError(
            f"relative_drop must be in (0, 1) (got {relative_drop})")
    snaps = list(snapshots)
    if len(snaps) < 2:
        return []
    events: list[DPTEvent] = []
    prev_echo: float | None = None
    for i in range(len(snaps) - 1):
        a, b = snaps[i], snaps[i + 1]
        amp = compute_loschmidt_echo(a.state, b.state)
        echo = float(abs(amp) ** 2)
        abs_hit = echo < echo_threshold
        rel_hit = (prev_echo is not None
                   and prev_echo > 0.0
                   and echo <= relative_drop * prev_echo)
        trigger: str | None = None
        if abs_hit and rel_hit:
            trigger = "both"
        elif abs_hit:
            trigger = "absolute"
        elif rel_hit:
            trigger = "relative"
        if trigger is not None:
            events.append(DPTEvent(
                cycle_before=a.cycle_index,
                cycle_after=b.cycle_index,
                amplitude=amp,
                echo=echo,
                trigger=trigger,
            ))
        prev_echo = echo
    return events


__all__ = [
    "DEFAULT_DPT_ECHO_THRESHOLD",
    "DEFAULT_DPT_RELATIVE_DROP",
    "DPTEvent",
    "WakeSleepSnapshot",
    "compute_loschmidt_echo",
    "detect_dpt_in_wake_sleep_log",
]
