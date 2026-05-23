"""Acceptance tests for §12.8 dynamical phase transitions / Loschmidt echo.

These tests exercise the real M2 substrate:
  * MERA states are built from ``MERA.from_product`` over real 16-dim
    leaves (the same factory used by ``encode_mera`` and the rest of
    the §12 composition tests).
  * The Loschmidt echo is computed by :func:`compute_loschmidt_echo`,
    which calls :meth:`MERA.inner` — the double-network bond
    contraction (§1.1 entanglement-bond pattern; NOT an AST walk).
  * Wake-sleep snapshots are real ``WakeSleepSnapshot`` records over
    real MERA states.

No mocks, no stubs, no skips.
"""
from __future__ import annotations

import numpy as np

from src.qft_pcn.composition.dynamical_pt import (
    DEFAULT_DPT_ECHO_THRESHOLD,
    DPTEvent,
    WakeSleepSnapshot,
    compute_loschmidt_echo,
    detect_dpt_in_wake_sleep_log,
)
from src.qft_pcn.composition.tests.conftest import (
    basis_leaf,
    make_product_mera,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _basis_mera(leaf_indices: list[int]):
    """A normalized product MERA whose leaves are computational-basis
    vectors at the given indices."""
    return make_product_mera([basis_leaf(i) for i in leaf_indices])


# ---------------------------------------------------------------------------
# Loschmidt echo primitive
# ---------------------------------------------------------------------------


def test_identical_states_have_unit_echo():
    """<Psi|Psi> = 1 exactly for a normalized state.

    The amplitude is computed by real bond contraction (MERA.inner),
    not by an identity short-circuit on the object reference: we pass
    two DIFFERENT MERA objects whose leaves happen to encode the same
    product state. The unit overlap must come out of the double-network
    contraction itself (§1.1).
    """
    state_a = _basis_mera([1, 2, 3, 0])
    state_b = _basis_mera([1, 2, 3, 0])
    # Sanity: distinct objects so we are not short-circuiting on `is`.
    assert state_a is not state_b
    amp = compute_loschmidt_echo(state_a, state_b)
    assert isinstance(amp, complex)
    assert abs(amp - 1.0) < 1e-10
    # And the echo magnitude itself is 1.
    assert abs(abs(amp) ** 2 - 1.0) < 1e-10


def test_orthogonal_states_have_zero_echo():
    """Two product MERAs that differ on at least one leaf by an
    orthogonal basis vector have exactly zero overlap.

    For product states, ``<Psi_a|Psi_b> = prod_k <a_k|b_k>``. Choosing
    a_k != b_k at one site forces that factor — and the whole inner
    product — to zero. The MERA.inner double-network contraction
    reproduces this exactly (verified by the MERA inner-product tests).
    """
    state_a = _basis_mera([0, 1, 2, 3])
    state_b = _basis_mera([0, 1, 2, 4])  # last leaf differs orthogonally
    amp = compute_loschmidt_echo(state_a, state_b)
    assert abs(amp) < 1e-12
    # And the Loschmidt echo magnitude is also zero.
    assert abs(amp) ** 2 < 1e-24


# ---------------------------------------------------------------------------
# DPT detector on real wake-sleep snapshot sequences
# ---------------------------------------------------------------------------


def test_dpt_detector_finds_known_zero():
    """A synthetic but REAL wake-sleep snapshot sequence with an
    engineered zero crossing.

    Cycles 0, 1, 2: identical product state (library is consolidating —
    no transition). Cycle 3: a leaf flips to an orthogonal basis vector
    (library has reorganized — DPT). Cycle 4: stays at the new state
    (no further transition).

    The detector must surface exactly ONE event, located at the
    (cycle 2 -> cycle 3) boundary, with echo well below the absolute
    threshold (a true zero crossing in finite-N).
    """
    consolidating = _basis_mera([1, 2, 3, 0])
    reorganized = _basis_mera([1, 2, 3, 5])  # last leaf flipped, orthogonal
    snapshots = [
        WakeSleepSnapshot(cycle_index=0, state=_basis_mera([1, 2, 3, 0])),
        WakeSleepSnapshot(cycle_index=1, state=_basis_mera([1, 2, 3, 0])),
        WakeSleepSnapshot(cycle_index=2, state=consolidating),
        WakeSleepSnapshot(cycle_index=3, state=reorganized),
        WakeSleepSnapshot(cycle_index=4, state=_basis_mera([1, 2, 3, 5])),
    ]
    events = detect_dpt_in_wake_sleep_log(snapshots)
    assert len(events) == 1, f"expected 1 DPT event, got {len(events)}: {events}"
    ev = events[0]
    assert isinstance(ev, DPTEvent)
    assert ev.cycle_before == 2
    assert ev.cycle_after == 3
    assert ev.echo < DEFAULT_DPT_ECHO_THRESHOLD
    # Absolute trigger (echo near zero) — both criteria fire here since the
    # previous consecutive echo was ~1, so the relative drop also hits.
    assert ev.trigger in ("absolute", "both")
    # Sanity: detector is idempotent under identity input prefix (the
    # 0->1 and 1->2 boundaries are above threshold and not in events).
    cycles_seen = {(e.cycle_before, e.cycle_after) for e in events}
    assert (0, 1) not in cycles_seen
    assert (1, 2) not in cycles_seen
    assert (3, 4) not in cycles_seen


def test_dpt_detector_returns_empty_for_short_log():
    """Fewer than two snapshots cannot produce a DPT signal (per §12.8
    "one overlap calculation per cycle" — two cycles minimum)."""
    assert detect_dpt_in_wake_sleep_log([]) == []
    one = [WakeSleepSnapshot(cycle_index=0,
                             state=_basis_mera([0, 0, 0, 0]))]
    assert detect_dpt_in_wake_sleep_log(one) == []
