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
