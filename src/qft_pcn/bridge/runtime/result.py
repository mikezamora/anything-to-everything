"""Result dataclasses for the bridge runtime (spec §5.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObservableValue:
    site: int
    field: str
    op: str
    value: float
    imag_part: float


@dataclass
class ConvergenceHistorySummary:
    energy_per_step: list[float] = field(default_factory=list)


@dataclass
class RunResult:
    observables: list[ObservableValue]
    energy: float
    energy_per_term: list[float]
    truncation_error_sum: float
    final_bond_dimensions: list[int]
    converged: bool
    convergence_history: ConvergenceHistorySummary
    # --- Composition-layer plumbing (EXTENSIONS.md #1) -------------------
    # Additive, default-None fields surfaced for ``composition.dispatcher.
    # run_child``: K-5's ``integrate_child`` needs the live ground state and
    # the ``MeraEncodingMeta`` it was decoded under to call ``register_lemma``
    # and feed ``Promoter`` species checks. ``hamiltonian`` carries the
    # composed M2/bridge Hamiltonian for ``register_lemma``'s compress
    # branch, and ``trotter_steps`` is provenance for ``DerivationMetadata``.
    # Typed ``Any`` so the bridge stays free of a logic/ circular import:
    # the M1 ``MeraEncodingMeta`` and M2 ``MERA`` live in ``logic/`` and
    # only the composition layer needs to recover their concrete types.
    # Default ``None``/``0`` keeps every existing caller working unchanged.
    meta: Any = None
    ground_state: Any = None
    solved_ast: Any = None
    hamiltonian: Any = None
    trotter_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observables": [
                {"site": o.site, "field": o.field, "op": o.op,
                 "value": o.value, "imag_part": o.imag_part}
                for o in self.observables
            ],
            "energy": self.energy,
            "energy_per_term": self.energy_per_term,
            "truncation_error_sum": self.truncation_error_sum,
            "final_bond_dimensions": self.final_bond_dimensions,
            "converged": self.converged,
            "convergence_history": {
                "energy_per_step": self.convergence_history.energy_per_step
            },
            "trotter_steps": self.trotter_steps,
        }


@dataclass
class RunDiagnostic:
    result: RunResult
    debugger_report: dict | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "result": self.result.to_dict(),
            "debugger_report": self.debugger_report,
        }
