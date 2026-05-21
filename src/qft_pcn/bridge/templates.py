"""Builder dataclasses for constructing DSL specs from typed templates.

This module knows nothing of LLM SDKs; it builds pure JSON-compatible dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FieldSpec:
    name: str
    cutoff: int


@dataclass
class ConstraintSpec:
    kind: str                       # "local" or "two_site"
    term: str
    weight: float = 1.0
    site: int | None = None
    sites: tuple[int, int] | None = None


@dataclass
class ObservableSpec:
    site: int
    field: str
    op: str


@dataclass
class SearchSpec:
    method: str = "imag_time"
    steps: int = 50
    chi_max: int = 32
    dt: float = 0.05


@dataclass
class Problem:
    fields: list[FieldSpec]
    sites: int
    constraints: list[ConstraintSpec] = field(default_factory=list)
    boundary: dict[int, dict[str, Any]] = field(default_factory=dict)
    observables: list[ObservableSpec] = field(default_factory=list)
    search: SearchSpec = field(default_factory=SearchSpec)

    def to_dsl(self) -> dict[str, Any]:
        cons: list[dict[str, Any]] = []
        for c in self.constraints:
            entry: dict[str, Any] = {"kind": c.kind, "term": c.term,
                                     "weight": c.weight}
            if c.kind == "local":
                entry["site"] = c.site
            else:
                entry["sites"] = list(c.sites or ())
            cons.append(entry)
        return {
            "fields": [{"name": f.name, "cutoff": f.cutoff}
                       for f in self.fields],
            "sites": self.sites,
            "constraints": cons,
            "boundary": {str(k): dict(v) for k, v in self.boundary.items()},
            "observables": [{"site": o.site, "field": o.field, "op": o.op}
                            for o in self.observables],
            "search": {"method": self.search.method, "steps": self.search.steps,
                       "chi_max": self.search.chi_max, "dt": self.search.dt},
        }


def stlc_synthesis(*, sites: int = 32,
                   holes: list[dict[str, Any]] | None = None,
                   constraints: list[ConstraintSpec] | None = None,
                   observables: list[ObservableSpec] | None = None
                   ) -> Problem:
    """Canonical builder for §10.7 STLC synthesis problems."""
    fields = [
        FieldSpec("kind", 8), FieldSpec("type", 8),
        FieldSpec("bid", 8),  FieldSpec("value", 16),
    ]
    return Problem(
        fields=fields,
        sites=sites,
        constraints=list(constraints or []),
        observables=list(observables or [
            ObservableSpec(site=0, field="kind", op="n")
        ]),
    )
