"""End-to-end smoke for the top-level runner.

The runner is the spec §14 "load corpus, run QPCN, collect metrics,
apply stats, write report" pipeline. This test exercises it on a
small mixed-benchmark slice with baselines DISABLED (so we don't
require AlphaProof/Synquid/ReProver binaries) but ablations enabled.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from experiments.runner import run, write_report


@pytest.mark.timeout(600)
def test_runner_e2e_smoke(tmp_path: Path):
    results, stats = run(
        benchmarks=["myth", "qm9"],
        limit_per_benchmark=2,
        include_baselines=False,
        include_ablations=True,
        seed=42,
    )
    assert results, "runner produced no rows"
    # qpcn baseline row present per benchmark.
    bench_solvers = {(r.benchmark, r.solver, r.config_label) for r in results}
    assert ("myth", "qpcn", "BASELINE") in bench_solvers
    assert ("qm9", "qpcn", "BASELINE") in bench_solvers

    # statistical report has per-run metric blocks for every row.
    for r in results:
        key = f"{r.benchmark}/{r.solver}/{r.config_label}"
        assert key in stats.per_run, f"missing metrics block for {key}"

    out_path = tmp_path / "smoke_report.md"
    write_report(results, stats, out_path,
                 git_sha="testsha", n_problems_total=4)
    body = out_path.read_text(encoding="utf-8")
    assert "QPCN evaluation initial run" in body
    assert "Per-(benchmark, solver, config) metrics" in body
