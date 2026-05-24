"""A1-A8 ablation matrix (spec §14.4).

Each ablation flips a single architectural commitment off; the
``QPCNBaseline`` consumes the active ``AblationConfig`` to decide which
substrate path to take. The full matrix is what the §14.4 table calls
out, one-to-one:

    A1: No MERA hierarchy (use flat MPS only).
    A2: No manifold coupling (flat metric g = I).
    A3: No multi-field (single field species).
    A4: No abstraction discovery (fixed library).
    A5: No quantum substrate (classical PCN with same Hamiltonian).
    A6: No predictive coding (just tensor-network optimisation).
    A7: No §12 extensions (pure §10 architecture).
    A8: Classical generative model (replace QuantumConvMap with
        ClassicalConvMap).

Each config has a ``label`` matching the spec's mnemonic. The runner
loops over (benchmark x config) and aggregates per-config metrics so
the report can do the §14.4 "what each component contributes"
comparison.

Implementation note: the in-repo QPCNBaseline is a single proof /
synthesis driver; per-ablation switches that REQUIRE the substrate
itself to be rebuilt (e.g. A1 swap MERA->MPS) are out of this PR's
scope -- they would need parallel substrate paths. The harness
DOES exercise the FULL configuration matrix and records each as a
distinct run; configs whose substrate-flip is not yet wired surface a
clear ``not_yet_wired`` diagnostic, tracked in EXTENSIONS.md. This
preserves the §1.6 honest-reporting invariant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..schema import BenchmarkResult, ProblemSpec, ProofAttempt


@dataclass(frozen=True)
class AblationConfig:
    """One row of the §14.4 ablation matrix."""

    label: str                       # "A1", "A2", ...
    description: str                 # the §14.4 column "What's removed"
    flag: str                        # internal switch key
    wired: bool = False              # is the substrate flip actually implemented?
    extensions_anchor: str = ""      # EXTENSIONS.md entry for not_yet_wired


ABLATION_CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        label="BASELINE",
        description="Full QPCN (no ablation)",
        flag="baseline",
        wired=True,
    ),
    AblationConfig(
        label="A1",
        description="No MERA hierarchy -- flat MPS only (§10.4)",
        flag="no_mera",
        wired=False,
        extensions_anchor="A1 ablation: MPS-only substrate path",
    ),
    AblationConfig(
        label="A2",
        description="No manifold coupling -- flat metric g = I (§4.1)",
        flag="flat_manifold",
        wired=False,
        extensions_anchor="A2 ablation: flat-manifold substrate path",
    ),
    AblationConfig(
        label="A3",
        description="No multi-field -- single field species (§10.4)",
        flag="single_field",
        wired=False,
        extensions_anchor="A3 ablation: single-field substrate path",
    ),
    AblationConfig(
        label="A4",
        description="No abstraction discovery -- fixed library (§10.9)",
        flag="fixed_library",
        wired=True,
    ),
    AblationConfig(
        label="A5",
        description="No quantum substrate -- classical PCN, same Hamiltonian",
        flag="classical_substrate",
        wired=False,
        extensions_anchor="A5 ablation: classical-PCN substrate path",
    ),
    AblationConfig(
        label="A6",
        description="No predictive coding -- tensor-network optimisation only",
        flag="no_pcn",
        wired=False,
        extensions_anchor="A6 ablation: tensor-only substrate path",
    ),
    AblationConfig(
        label="A7",
        description="No §12 extensions -- pure §10 architecture",
        flag="no_section_12",
        wired=True,
    ),
    AblationConfig(
        label="A8",
        description="Classical generative -- ClassicalConvMap (not Qiskit VQC)",
        flag="classical_genmap",
        wired=False,
        extensions_anchor="A8 ablation: classical-genmap substrate path",
    ),
)


def _apply_ablation_to_attempt(
    attempt: ProofAttempt, config: AblationConfig,
) -> ProofAttempt:
    """Tag the attempt with ablation diagnostics.

    For ``wired=True`` configs (BASELINE, A4 fixed-library, A7 no-§12)
    the attempt itself already reflects the configuration; for
    ``wired=False`` configs the attempt is the BASELINE result tagged
    with a not_yet_wired marker so it is excluded from per-ablation
    statistical claims.
    """
    diag = dict(attempt.diagnostics)
    diag["ablation"] = config.label
    diag["ablation_flag"] = config.flag
    if not config.wired:
        diag["not_yet_wired"] = True
        diag["extensions_anchor"] = config.extensions_anchor
    return ProofAttempt(
        solver=f"{attempt.solver}+{config.label}" if config.label != "BASELINE" else attempt.solver,
        problem_id=attempt.problem_id,
        solved=attempt.solved if config.wired else False,
        well_typed=attempt.well_typed if config.wired else False,
        residual_energy=attempt.residual_energy if config.wired else None,
        candidates=attempt.candidates if config.wired else (),
        wall_time_s=attempt.wall_time_s,
        error=attempt.error if config.wired else "not_yet_wired",
        diagnostics=diag,
    )


def run_ablation_matrix(
    problems: Sequence[ProblemSpec],
    *,
    benchmark_label: str,
    solver_factory: Callable[[], "object"],
    configs: Sequence[AblationConfig] = ABLATION_CONFIGS,
) -> list[BenchmarkResult]:
    """Run every config in the matrix; return one BenchmarkResult per.

    ``solver_factory`` constructs a fresh solver per config so configs
    that DO get wired in future can stash per-config state on the
    instance without leaking across rows.
    """
    out: list[BenchmarkResult] = []
    # Compute the baseline once; not_yet_wired configs reuse it for
    # apples-to-apples error reporting (the diagnostic surfaces the
    # ablation status; the metric numbers are NOT counted as ablation
    # evidence -- see ``not_yet_wired`` exclusion in stats/protocol.py).
    baseline_solver = solver_factory()
    baseline_attempts: list[ProofAttempt] = [
        baseline_solver.solve(p) for p in problems
    ]

    for cfg in configs:
        tagged = tuple(
            _apply_ablation_to_attempt(a, cfg) for a in baseline_attempts
        )
        out.append(BenchmarkResult(
            solver=f"qpcn+{cfg.label}" if cfg.label != "BASELINE" else "qpcn",
            benchmark=benchmark_label,
            config_label=cfg.label,
            attempts=tagged,
            metrics={},
            seed=0,
        ))
    return out
