"""E30 — Pin the expressivity wall as a monotone (breadth, χ) transition.

Per spec §13.5 the MERA substrate is area-law bounded. For hole-bearing
programs the §5.3 / §1.1 binding-as-entanglement principle forces a
rank-k superposition between the hole bid leaf and the k candidate binder
bid leaves. The substrate can represent that state iff χ ≥ k; below the
wall the encoder MUST refuse (never silently degrade to a classical
lookup — §1.1).

This test runs the E30 sweep on a small CI-sized corpus and asserts:

  (a) At large χ, acceptance is high — substrate IS expressive enough.
  (b) At small χ (below the wall), acceptance is low — substrate IS NOT.
  (c) The transition is monotone in χ at every (breadth, depth) cell.
  (d) The wall locus matches the principled prediction χ* = breadth.

It also writes the artifact under ``experiments/results/expressivity_wall/``
so the test doubles as the canonical sweep run for the E30 EXTENSIONS.md
entry.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.expressivity_wall.sweep import (
    DEFAULT_BREADTHS,
    DEFAULT_DEPTHS,
    DEFAULT_CHIS,
    is_monotone_transition,
    run_sweep,
    wall_loci_by_breadth,
    write_artifacts,
)


# CI-sized cube. Each cell runs in well under a second except a handful
# at large (breadth, depth) which still finish in a few seconds; the full
# cube completes in a few tens of seconds.
CI_BREADTHS = (2, 3, 4, 5, 6)
CI_DEPTHS = (0, 1)
CI_CHIS = (2, 3, 4, 6, 8)
THRESHOLD = 1e-2


@pytest.fixture(scope="module")
def sweep_rows():
    rows = run_sweep(
        CI_BREADTHS, CI_DEPTHS, CI_CHIS,
        dt=0.1, steps=20, threshold=THRESHOLD,
    )
    # Persist artifact so the EXTENSIONS.md entry can point at a fresh
    # CSV/JSON pair every time the test runs.
    out_dir = (Path(__file__).resolve().parents[4]
               / "experiments" / "results" / "expressivity_wall")
    write_artifacts(
        rows, out_dir,
        config={
            "breadths": list(CI_BREADTHS),
            "depths": list(CI_DEPTHS),
            "chis": list(CI_CHIS),
            "dt": 0.1, "steps": 20, "threshold": THRESHOLD,
        },
    )
    return rows


def test_large_chi_accepts_full_corpus(sweep_rows):
    """(a) — at the largest χ in the sweep, every (breadth, depth) cell
    must converge below threshold. If this fails the substrate cannot
    represent even simple hole-binding states at the top of our χ range."""
    big = max(CI_CHIS)
    failing = [
        (r.breadth, r.depth, r.residual, r.error_kind)
        for r in sweep_rows
        if r.chi == big and not r.accepted
    ]
    assert not failing, (
        f"§13.5 sanity: at χ={big} every cell should converge; "
        f"got failures {failing}"
    )


def test_small_chi_refused_for_wide_holes(sweep_rows):
    """(b) — at χ=2 every breadth ≥ 3 MUST be refused by the substrate.
    A passing cell here would mean the encoder silently classicalised
    the hole binding — a §1.1 violation."""
    small = min(CI_CHIS)
    leaked = [
        (r.breadth, r.depth, r.residual, r.error_kind)
        for r in sweep_rows
        if r.chi == small and r.breadth >= 3 and r.accepted
    ]
    assert not leaked, (
        f"§1.1 violation: at χ={small} breadth≥3 cells accepted "
        f"(should be refused as the substrate cannot encode the "
        f"rank-k binding superposition): {leaked}"
    )
    # And at least one cell at small χ must actually refuse — otherwise
    # the sweep didn't reach the wall.
    refused = [r for r in sweep_rows if r.chi == small and r.refused]
    assert refused, (
        f"sweep failed to locate the wall: no refusals at χ={small}; "
        f"expand the breadth range or shrink the χ range"
    )


def test_transition_is_monotone(sweep_rows):
    """(c) — once χ crosses the wall the substrate must stay accepting.
    A non-monotone residual curve (accept → refuse → accept across χ)
    would be a numerical bug."""
    violations = []
    for b in CI_BREADTHS:
        for d in CI_DEPTHS:
            if not is_monotone_transition(sweep_rows, b, d):
                violations.append((b, d))
    assert not violations, (
        f"§13.5 wall must be monotone in χ; non-monotone cells: "
        f"{violations}"
    )


def test_wall_locus_matches_principled_prediction(sweep_rows):
    """(d) — the §1.1 principle predicts the wall at χ* = #candidate
    binders: every sweep χ < breadth must refuse, and the first sweep
    χ ≥ breadth must accept. If the encoder ever accepts at χ < breadth
    the binding-as-entanglement invariant has been broken (rank
    deficiency in the superposition)."""
    chis_sorted = sorted(CI_CHIS)
    mismatches = []
    for d in CI_DEPTHS:
        loci = wall_loci_by_breadth(sweep_rows, d)
        for breadth, observed_min in loci.items():
            assert observed_min is not None, (
                f"sweep failed to find an accepting χ for "
                f"(breadth={breadth}, depth={d}); extend χ range"
            )
            # Smallest χ in the sweep that satisfies χ ≥ breadth.
            predicted_min = next(
                (c for c in chis_sorted if c >= breadth), None
            )
            if predicted_min is None:
                pytest.fail(
                    f"sweep χ range {chis_sorted} cannot reach "
                    f"breadth={breadth}; widen the χ range"
                )
            if observed_min != predicted_min:
                mismatches.append(
                    (breadth, d, observed_min, predicted_min)
                )
    assert not mismatches, (
        f"§1.1 prediction (min accepting χ = smallest sweep χ ≥ "
        f"breadth) violated; (breadth, depth, observed, predicted) "
        f"= {mismatches}"
    )


def test_artifact_written(sweep_rows):
    """(e) — confirm the E30 artifact lands under experiments/results."""
    out_dir = (Path(__file__).resolve().parents[4]
               / "experiments" / "results" / "expressivity_wall")
    assert (out_dir / "sweep.csv").exists(), (
        f"missing E30 CSV artifact under {out_dir}"
    )
    assert (out_dir / "sweep.json").exists(), (
        f"missing E30 JSON artifact under {out_dir}"
    )
