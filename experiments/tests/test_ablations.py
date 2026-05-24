"""Tests for the ablation harness."""
from __future__ import annotations

from experiments.ablations import (
    ABLATION_CONFIGS, run_ablation_matrix,
)
from experiments.baselines import QPCNBaseline
from experiments.benchmarks import load_minif2f


def test_ablation_configs_cover_spec_table():
    labels = {cfg.label for cfg in ABLATION_CONFIGS}
    # spec §14.4 names A1..A8 + a baseline column
    for name in ("BASELINE", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"):
        assert name in labels, f"missing ablation row {name}"


def test_ablation_matrix_produces_one_row_per_config():
    problems = load_minif2f(limit=1)
    rows = run_ablation_matrix(
        problems,
        benchmark_label="minif2f",
        solver_factory=lambda: QPCNBaseline(),
    )
    assert len(rows) == len(ABLATION_CONFIGS)
    labels = [r.config_label for r in rows]
    assert labels.count("BASELINE") == 1
    # Wired configs report real diagnostics; not-yet-wired surface the flag.
    for r in rows:
        for a in r.attempts:
            cfg = next(c for c in ABLATION_CONFIGS if c.label == r.config_label)
            if not cfg.wired:
                assert a.diagnostics.get("not_yet_wired") is True
                assert a.error == "not_yet_wired"
            else:
                # BASELINE / A4 / A7 share the BASELINE attempt outcome.
                # ``not_yet_wired`` must NOT be present.
                assert a.diagnostics.get("not_yet_wired") is not True
