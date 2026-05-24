"""Top-level evaluation runner (spec §14).

Pipeline:
  1. Load each benchmark corpus (small-subset mode by default).
  2. Run the QPCN solver + each available baseline against the corpus.
  3. Apply the statistical protocol.
  4. Write a markdown report to ``experiments/reports/initial_run_<SHA>.md``.

Usage::

    uv run python -m experiments.runner            # small-subset run
    uv run python -m experiments.runner --full     # honour full corpora
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from .ablations import ABLATION_CONFIGS, run_ablation_matrix
from .baselines import (
    AlphaProofBaseline, BaselineUnavailable, QPCNBaseline,
    ReProverBaseline, SynquidBaseline,
)
from .benchmarks import (
    load_dreamcoder, load_hazel, load_humaneval, load_minif2f, load_myth,
    load_qm9,
)
from .schema import BenchmarkResult, ProblemSpec, ProofAttempt
from .stats import run_statistical_protocol


_BENCHMARK_LOADERS = {
    "minif2f": load_minif2f,
    "humaneval_typed": lambda limit: load_humaneval(limit=limit, only_typed_subset=True),
    "humaneval_full": lambda limit: load_humaneval(limit=limit),
    "myth": load_myth,
    "dreamcoder": load_dreamcoder,
    "hazel": load_hazel,
    "qm9": load_qm9,
}


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _run_solver_on(
    solver, problems: Sequence[ProblemSpec], *,
    solver_name: str, benchmark_label: str,
) -> BenchmarkResult:
    """Invoke ``solver.solve`` on each problem; collect attempts.

    A ``BaselineUnavailable`` raised by an adapter is converted into a
    single structured no-attempt row (so the report STILL surfaces the
    baseline's absence; we don't silently drop it).
    """
    attempts: list[ProofAttempt] = []
    for p in problems:
        try:
            attempts.append(solver.solve(p))
        except BaselineUnavailable as exc:
            # One row per problem so cross-baseline aggregates line up.
            attempts.append(ProofAttempt(
                solver=solver_name,
                problem_id=p.problem_id,
                solved=False,
                well_typed=False,
                residual_energy=None,
                candidates=(),
                wall_time_s=0.0,
                error=str(exc),
                diagnostics={"unavailable": True,
                             "extensions_anchor": exc.extensions_anchor},
            ))
    return BenchmarkResult(
        solver=solver_name,
        benchmark=benchmark_label,
        config_label="BASELINE",
        attempts=tuple(attempts),
    )


def run(
    *,
    benchmarks: Sequence[str],
    limit_per_benchmark: Optional[int],
    include_baselines: bool,
    include_ablations: bool,
    seed: int,
) -> tuple[list[BenchmarkResult], Any]:
    """Execute the full evaluation pipeline and return (results, stats)."""
    all_results: list[BenchmarkResult] = []
    qpcn = QPCNBaseline(seed=seed)

    for bench in benchmarks:
        if bench not in _BENCHMARK_LOADERS:
            print(f"[runner] skip unknown benchmark {bench!r}", file=sys.stderr)
            continue
        loader = _BENCHMARK_LOADERS[bench]
        problems = loader(limit_per_benchmark)
        if not problems:
            print(f"[runner] benchmark {bench}: empty corpus", file=sys.stderr)
            continue
        print(f"[runner] benchmark {bench}: {len(problems)} problems")
        t0 = time.time()
        qpcn_res = _run_solver_on(
            qpcn, problems, solver_name="qpcn", benchmark_label=bench,
        )
        all_results.append(qpcn_res)
        print(f"[runner]   qpcn   wall={time.time()-t0:.1f}s "
              f"solved={sum(1 for a in qpcn_res.attempts if a.solved)}/{len(problems)}")

        if include_baselines:
            for solver_cls, solver_name in (
                (AlphaProofBaseline, "alphaproof"),
                (ReProverBaseline, "reprover"),
                (SynquidBaseline, "synquid"),
            ):
                b_res = _run_solver_on(
                    solver_cls(), problems,
                    solver_name=solver_name, benchmark_label=bench,
                )
                all_results.append(b_res)
                n_solved = sum(1 for a in b_res.attempts if a.solved)
                n_unavail = sum(
                    1 for a in b_res.attempts
                    if a.diagnostics.get("unavailable")
                )
                print(f"[runner]   {solver_name:9s} solved={n_solved}/{len(problems)} "
                      f"unavailable={n_unavail}")

        if include_ablations:
            abl_rows = run_ablation_matrix(
                problems,
                benchmark_label=bench,
                solver_factory=lambda s=seed: QPCNBaseline(seed=s),
            )
            all_results.extend(abl_rows)

    print(f"[runner] applying statistical protocol over {len(all_results)} rows")
    stats = run_statistical_protocol(all_results)
    return all_results, stats


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return repr(obj)


def write_report(
    results: Sequence[BenchmarkResult],
    stats,
    out_path: Path,
    *,
    git_sha: str,
    n_problems_total: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# QPCN evaluation initial run -- HEAD {git_sha}")
    lines.append("")
    lines.append("Spec anchor: §14 (evaluation methodology) + §14.1 "
                 "(HumanEval typed subset).")
    lines.append("")
    lines.append(f"- Total problems across benchmarks: {n_problems_total}")
    lines.append(f"- Total result rows: {len(results)}")
    lines.append("")
    lines.append("## Per-(benchmark, solver, config) metrics")
    lines.append("")
    lines.append("| benchmark | solver | config | n | solved | pass@1 | type@1 | residual_mean | CI95 (pass@1) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        key = f"{r.benchmark}/{r.solver}/{r.config_label}"
        m = stats.per_run.get(key)
        n_solved = sum(1 for a in r.attempts if a.solved)
        if m is None:
            lines.append(
                f"| {r.benchmark} | {r.solver} | {r.config_label} | "
                f"{len(r.attempts)} | {n_solved} | - | - | - | - |"
            )
            continue
        ci = m.pass_at_1_ci
        ci_str = (
            f"[{ci.lower:.3f}, {ci.upper:.3f}]"
            if ci.n_observations > 0 else "-"
        )
        lines.append(
            f"| {r.benchmark} | {r.solver} | {r.config_label} | "
            f"{len(r.attempts)} | {n_solved} | "
            f"{m.pass_at_1:.3f} | {m.type_at_1:.3f} | "
            f"{m.residual_energy_mean:.3f} | {ci_str} |"
        )
    lines.append("")
    lines.append("## Pairwise comparisons (Cohen's d / Cliff's delta)")
    lines.append("")
    if stats.comparisons:
        lines.append("| metric | solver_a | solver_b | mean_a | mean_b | Cohen d | Cliff delta |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in stats.comparisons:
            lines.append(
                f"| {c.metric} | {c.solver_a} | {c.solver_b} | "
                f"{c.mean_a:.3f} | {c.mean_b:.3f} | "
                f"{c.cohens_d:+.3f} | {c.cliffs_delta:+.3f} |"
            )
    else:
        lines.append("_No pairwise comparisons computed (baselines unavailable)._")
    lines.append("")
    lines.append("## Holm-Bonferroni corrected p-values")
    lines.append("")
    if stats.corrected_p:
        lines.append("| label | raw_p | adjusted_p | significant @ 0.05 |")
        lines.append("|---|---|---|---|")
        for cp in stats.corrected_p:
            lines.append(
                f"| {cp.label} | {cp.raw_p:.4f} | {cp.adjusted_p:.4f} | {cp.significant} |"
            )
    else:
        lines.append("_No paired tests run (no overlapping solver pairs)._")
    lines.append("")
    lines.append("## Per-problem attempts (raw)")
    lines.append("")
    lines.append("```json")
    raw_payload = {
        "results": [_to_jsonable(r) for r in results[:200]],
    }
    lines.append(json.dumps(raw_payload, indent=2, default=repr)[:30000])
    lines.append("```")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- ``unavailable=true`` rows mark baselines whose upstream "
                 "binary is not present on this host. See ``EXTENSIONS.md`` "
                 "for the per-baseline deferral.")
    lines.append("- ``not_yet_wired=true`` ablation rows mark configurations "
                 "whose substrate-flip is not yet implemented (A1, A2, A3, "
                 "A5, A6, A8). The BASELINE / A4 / A7 rows are real.")
    lines.append("- ``out_of_substrate`` errors on QPCN proof rows are "
                 "honest no-attempts on miniF2F problems that lie outside "
                 "the K-8 Nat-arithmetic fragment.")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[runner] report written to {out_path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["minif2f", "humaneval_typed", "myth", "dreamcoder",
                 "hazel", "qm9"],
        help="Benchmarks to run.",
    )
    parser.add_argument("--limit", type=int, default=4,
                        help="Per-benchmark cap (small-subset mode).")
    parser.add_argument("--full", action="store_true",
                        help="Disable the per-benchmark cap.")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip baseline adapters (avoid BaselineUnavailable noise).")
    parser.add_argument("--no-ablations", action="store_true",
                        help="Skip ablation matrix.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Report output path; default reports/initial_run_<SHA>.md")
    args = parser.parse_args(argv)

    limit = None if args.full else args.limit
    sha = _git_sha()
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = (
            Path(__file__).resolve().parent / "reports"
            / f"initial_run_{sha}.md"
        )

    results, stats = run(
        benchmarks=args.benchmarks,
        limit_per_benchmark=limit,
        include_baselines=not args.no_baselines,
        include_ablations=not args.no_ablations,
        seed=args.seed,
    )
    n_total = sum(len(r.attempts) for r in results if r.solver == "qpcn"
                  and r.config_label == "BASELINE")
    write_report(results, stats, out_path,
                 git_sha=sha, n_problems_total=n_total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
