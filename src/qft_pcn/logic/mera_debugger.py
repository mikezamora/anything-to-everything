"""MERA constraint debugger (spec §3).

D's generic constraint reporter retargeted to MERA states. Consumes named
per-term operators (M2's .terms) satisfying NamedMeraTerm; computes
per-term residual energy; emits a JSON-serializable DiagnosticReport.
Generic over any MERA Hamiltonian — no per-rule branching, no
isinstance(H, ...) dispatch (principle 8).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from src.qft_pcn.qft.mera import MERA
from .mera_encoder import MeraEncodingMeta


# --- Protocol -------------------------------------------------------------


@runtime_checkable
class NamedMeraTerm(Protocol):
    """One named, individually measurable MERA Hamiltonian term."""

    name: str
    rule_class: str
    node: int
    leaves: tuple

    def expectation(self, state: MERA) -> float: ...


# --- Explanation registry -------------------------------------------------

ExplanationContext = dict[str, Any]
ContextExtractor = Callable[
    ["NamedMeraTerm", MERA, MeraEncodingMeta], ExplanationContext
]

_EXPLANATION_REGISTRY: dict[str, tuple[str, ContextExtractor | None]] = {}


def register_explanation(
    rule_class: str,
    template: str,
    context_extractor: ContextExtractor | None = None,
) -> None:
    """Register a human-readable template for a rule class.

    Idempotent for the same (rule_class, template, extractor) tuple.
    Raises ValueError on conflicting re-registration.
    """
    existing = _EXPLANATION_REGISTRY.get(rule_class)
    if existing is not None:
        existing_tpl, existing_ext = existing
        if existing_tpl == template and existing_ext is context_extractor:
            return
        raise ValueError(
            f"rule_class {rule_class!r} already registered with a "
            f"different template/extractor"
        )
    _EXPLANATION_REGISTRY[rule_class] = (template, context_extractor)


def get_explanation(
    rule_class: str,
) -> tuple[str, ContextExtractor | None] | None:
    return _EXPLANATION_REGISTRY.get(rule_class)


def clear_explanations() -> None:
    _EXPLANATION_REGISTRY.clear()


# --- Report dataclasses ---------------------------------------------------


@dataclass(frozen=True)
class TermEvaluationError:
    """Recorded when a term's expectation() raised."""

    term_name: str
    rule_class: str
    node: int
    exception_type: str
    exception_message: str

    def to_dict(self) -> dict:
        return {
            "term_name": self.term_name,
            "rule_class": self.rule_class,
            "node": self.node,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TermEvaluationError":
        return cls(
            term_name=d["term_name"],
            rule_class=d["rule_class"],
            node=int(d["node"]),
            exception_type=d["exception_type"],
            exception_message=d["exception_message"],
        )


@dataclass(frozen=True)
class RuleViolation:
    """One MERA Hamiltonian term whose |<H_term>| >= threshold."""

    name: str
    rule_class: str
    node: int
    leaves: list
    energy_contribution: float
    ast_path: list | None
    lookup_failed: bool
    explanation: str
    context: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rule_class": self.rule_class,
            "node": self.node,
            "leaves": list(self.leaves),
            "energy_contribution": float(self.energy_contribution),
            "ast_path": (list(self.ast_path)
                         if self.ast_path is not None else None),
            "lookup_failed": bool(self.lookup_failed),
            "explanation": self.explanation,
            "context": dict(self.context),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RuleViolation":
        return cls(
            name=d["name"],
            rule_class=d["rule_class"],
            node=int(d["node"]),
            leaves=list(d["leaves"]),
            energy_contribution=float(d["energy_contribution"]),
            ast_path=(list(d["ast_path"])
                      if d["ast_path"] is not None else None),
            lookup_failed=bool(d["lookup_failed"]),
            explanation=d["explanation"],
            context=dict(d["context"]),
        )


@dataclass(frozen=True)
class DiagnosticReport:
    """Top-level report from diagnose()."""

    total_energy: float
    threshold: float
    n_terms_evaluated: int
    n_violations: int
    rule_violations: list
    by_rule_class: dict
    errors: list

    def to_dict(self) -> dict:
        return {
            "total_energy": float(self.total_energy),
            "threshold": float(self.threshold),
            "n_terms_evaluated": int(self.n_terms_evaluated),
            "n_violations": int(self.n_violations),
            "rule_violations": [v.to_dict() for v in self.rule_violations],
            "by_rule_class": {k: float(v)
                              for k, v in self.by_rule_class.items()},
            "errors": [e.to_dict() for e in self.errors],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosticReport":
        return cls(
            total_energy=float(d["total_energy"]),
            threshold=float(d["threshold"]),
            n_terms_evaluated=int(d["n_terms_evaluated"]),
            n_violations=int(d["n_violations"]),
            rule_violations=[RuleViolation.from_dict(v)
                             for v in d["rule_violations"]],
            by_rule_class={k: float(v)
                           for k, v in d["by_rule_class"].items()},
            errors=[TermEvaluationError.from_dict(e)
                    for e in d["errors"]],
        )


# --- diagnose -------------------------------------------------------------

_VALID_SORTS = ("descending", "ascending", "node")
_PROTO_ATTRS = ("name", "rule_class", "node", "leaves", "expectation")


def _render_explanation(
    term: Any,
    energy: float,
    ast_path: list | None,
    state: MERA,
    meta: MeraEncodingMeta,
    errors: list,
) -> tuple[str, dict]:
    """Render explanation string for a violation. See spec §3."""
    entry = _EXPLANATION_REGISTRY.get(term.rule_class)
    fallback = (
        f"{term.rule_class} violated at node {term.node} "
        f"(energy {energy:.3g})"
    )
    default_ctx: dict = {
        "node": int(term.node),
        "ast_path": ast_path,
        "rule_class": term.rule_class,
        "name": term.name,
        "energy": f"{energy:.3g}",
    }
    if entry is None:
        return fallback, {}

    template, extractor = entry
    context: dict = dict(default_ctx)
    if extractor is not None:
        try:
            extra = extractor(term, state, meta)
        except Exception as exc:  # noqa: BLE001
            errors.append(TermEvaluationError(
                term_name=term.name,
                rule_class=term.rule_class,
                node=int(term.node),
                exception_type=type(exc).__name__,
                exception_message=f"explanation extractor: {exc}",
            ))
            return fallback, {}
        if not isinstance(extra, dict):
            errors.append(TermEvaluationError(
                term_name=term.name,
                rule_class=term.rule_class,
                node=int(term.node),
                exception_type="TypeError",
                exception_message=(
                    f"explanation extractor returned "
                    f"{type(extra).__name__}, expected dict"
                ),
            ))
            return fallback, {}
        context.update(extra)

    try:
        rendered = template.format(**context)
    except (KeyError, IndexError) as exc:
        errors.append(TermEvaluationError(
            term_name=term.name,
            rule_class=term.rule_class,
            node=int(term.node),
            exception_type=type(exc).__name__,
            exception_message=f"template missing slot: {exc}",
        ))
        return fallback, {}

    extracted = {k: v for k, v in context.items() if k not in default_ctx}
    return rendered, extracted


def diagnose(
    state: MERA,
    meta: MeraEncodingMeta,
    hamiltonian_terms: list,
    *,
    threshold: float = 1e-6,
    sort: str = "descending",
) -> DiagnosticReport:
    """Structured per-term residual-energy report on a MERA state.

    See spec §3.2 for the full behavioural contract.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if sort not in _VALID_SORTS:
        raise ValueError(
            f"sort must be one of {_VALID_SORTS}, got {sort!r}"
        )
    for i, term in enumerate(hamiltonian_terms):
        if not isinstance(term, NamedMeraTerm):
            missing = next(
                (a for a in _PROTO_ATTRS if not hasattr(term, a)), "?"
            )
            raise TypeError(
                f"term at index {i} does not satisfy NamedMeraTerm "
                f"(missing attribute {missing!r})"
            )

    total = 0.0
    by_rule: dict[str, float] = {}
    errors: list[TermEvaluationError] = []
    measured: list[tuple] = []  # (term, energy)
    for term in hamiltonian_terms:
        try:
            e = float(term.expectation(state))
        except Exception as exc:  # noqa: BLE001 - captured by contract
            errors.append(TermEvaluationError(
                term_name=term.name,
                rule_class=term.rule_class,
                node=int(term.node),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            ))
            continue
        total += e
        by_rule[term.rule_class] = by_rule.get(term.rule_class, 0.0) + e
        measured.append((term, e))

    site_to_path = getattr(meta, "site_to_ast_path", {}) or {}

    violations: list[RuleViolation] = []
    for term, e in measured:
        if abs(e) < threshold:
            continue
        raw_path = site_to_path.get(int(term.node))
        lookup_failed = raw_path is None
        ast_path = None if lookup_failed else list(raw_path)
        text, ctx = _render_explanation(
            term, e, ast_path, state, meta, errors
        )
        violations.append(RuleViolation(
            name=term.name,
            rule_class=term.rule_class,
            node=int(term.node),
            leaves=list(term.leaves),
            energy_contribution=float(e),
            ast_path=ast_path,
            lookup_failed=lookup_failed,
            explanation=text,
            context=ctx,
        ))

    if sort == "descending":
        violations.sort(
            key=lambda v: (-v.energy_contribution, v.node, v.name)
        )
    elif sort == "ascending":
        violations.sort(
            key=lambda v: (v.energy_contribution, v.node, v.name)
        )
    else:  # "node"
        violations.sort(key=lambda v: (v.node, v.name))

    return DiagnosticReport(
        total_energy=float(total),
        threshold=float(threshold),
        n_terms_evaluated=len(hamiltonian_terms),
        n_violations=len(violations),
        rule_violations=violations,
        by_rule_class=by_rule,
        errors=errors,
    )


# --- Pretty-printer -------------------------------------------------------


def format_report(report: DiagnosticReport) -> str:
    """Human-readable pretty-print for CLI / test failures."""
    lines: list[str] = []
    lines.append("Diagnostic Report")
    lines.append("-----------------")
    lines.append(f"Total energy: {report.total_energy:.6g}")
    lines.append(f"Threshold: {report.threshold:.3g}")
    lines.append(
        f"Terms evaluated: {report.n_terms_evaluated}  "
        f"Violations: {report.n_violations}"
    )

    if report.by_rule_class:
        lines.append("")
        lines.append("By rule class:")
        for rc in sorted(report.by_rule_class):
            lines.append(f"  {rc}: {report.by_rule_class[rc]:.6g}")

    lines.append("")
    if not report.rule_violations:
        lines.append("No violations above threshold.")
    else:
        lines.append("Violations:")
        for v in report.rule_violations:
            path_str = (
                "ast_path=" + repr(v.ast_path) if v.ast_path is not None
                else "ast_path=<unmapped node>"
            )
            lines.append(
                f"  [{v.rule_class}] node {v.node} "
                f"energy={v.energy_contribution:.6g} {path_str}"
            )
            lines.append(f"    {v.explanation}")
            if v.context:
                lines.append(f"    context={v.context}")

    if report.errors:
        lines.append("")
        lines.append("Errors (terms that failed to evaluate):")
        for e in report.errors:
            lines.append(
                f"  [{e.rule_class}] {e.term_name} at node {e.node}: "
                f"{e.exception_type}: {e.exception_message}"
            )

    return "\n".join(lines)
