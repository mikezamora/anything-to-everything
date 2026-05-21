"""Constraint debugger for the QPCN logic layer (sub-project D).

This module turns per-term residual energies of a Hamiltonian sum
H = sum_k H_k into a structured, JSON-serializable diagnostic report.

It is generic: any Hamiltonian whose terms satisfy the
NamedHamiltonianTerm protocol can be diagnosed. Sub-projects B (typing
Hamiltonian) and C (evaluation Hamiltonian) are the first consumers
(via thin adapters that wrap their TypingTerm/EvalTerm dataclasses in
this protocol — D itself does not know about B/C internals). The
debugger never re-implements rules; it sorts per-term energies and
renders templates from the explanation registry.

See docs/superpowers/specs/2026-05-21-constraint-debugger-design.md
for the contract.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from src.qft_pcn.qft.mps import MPS


# ---- The Protocol --------------------------------------------------------


@runtime_checkable
class NamedHamiltonianTerm(Protocol):
    """One named, individually measurable Hamiltonian term.

    Sub-projects B and C produce iterables of these (directly, or via
    an adapter that wraps their internal TypingTerm/EvalTerm). The
    debugger consumes them generically — no per-rule branching inside
    the debugger.

    Attributes:
        name: Unique, stable identifier for this term instance.
              Convention: "<rule_class>@site_<k>" or
              "<rule_class>@sites_<k>_<l>".
        rule_class: The reusable rule identifier (no site).
                    Example: "T-App-Arrow".
        site: Primary AST site. The "anchor" the report points at.
        sites: Full tuple of sites touched by this term.

    Methods:
        expectation(state): <state | H_term | state> as a real Python
            float. Implementations MUST take Re(.) explicitly and check
            the imaginary residual is < 1e-10 (Hermitian by contract).
    """

    name: str
    rule_class: str
    site: int
    sites: tuple[int, ...]

    def expectation(self, state: MPS) -> float: ...


# ---- Dataclasses ---------------------------------------------------------


@dataclass(frozen=True)
class RuleViolation:
    """One Hamiltonian term whose |<H_term>| >= threshold.

    All fields are JSON-friendly: site/sites are ints/lists of ints,
    energy_contribution is a Python float, ast_path is list[int] or None,
    context is a dict of JSON-primitive values.
    """

    name: str
    rule_class: str
    site: int
    sites: list[int]
    energy_contribution: float
    ast_path: list[int] | None
    lookup_failed: bool
    explanation: str
    context: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "rule_class": self.rule_class,
            "site": self.site,
            "sites": list(self.sites),
            "energy_contribution": float(self.energy_contribution),
            "ast_path": (list(self.ast_path) if self.ast_path is not None
                         else None),
            "lookup_failed": bool(self.lookup_failed),
            "explanation": self.explanation,
            "context": dict(self.context),
        }


@dataclass(frozen=True)
class TermEvaluationError:
    """Recorded when a term's expectation() raised."""

    term_name: str
    rule_class: str
    site: int
    exception_type: str
    exception_message: str

    def to_dict(self) -> dict:
        return {
            "term_name": self.term_name,
            "rule_class": self.rule_class,
            "site": self.site,
            "exception_type": self.exception_type,
            "exception_message": self.exception_message,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    """Top-level report from diagnose().

    See docs/superpowers/specs/2026-05-21-constraint-debugger-design.md
    §5.1 for the field semantics.
    """

    total_energy: float
    threshold: float
    n_terms_evaluated: int
    n_violations: int
    rule_violations: list[RuleViolation]
    by_rule_class: dict[str, float]
    errors: list[TermEvaluationError]

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
        """Convenience: json.dumps(self.to_dict(), indent=indent)."""
        return _json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosticReport":
        """Inverse of to_dict. Round-trips equal dataclass instances."""
        return cls(
            total_energy=float(d["total_energy"]),
            threshold=float(d["threshold"]),
            n_terms_evaluated=int(d["n_terms_evaluated"]),
            n_violations=int(d["n_violations"]),
            rule_violations=[
                RuleViolation(
                    name=item["name"],
                    rule_class=item["rule_class"],
                    site=int(item["site"]),
                    sites=list(item["sites"]),
                    energy_contribution=float(item["energy_contribution"]),
                    ast_path=(list(item["ast_path"])
                              if item["ast_path"] is not None else None),
                    lookup_failed=bool(item["lookup_failed"]),
                    explanation=item["explanation"],
                    context=dict(item["context"]),
                )
                for item in d["rule_violations"]
            ],
            by_rule_class={k: float(v)
                           for k, v in d["by_rule_class"].items()},
            errors=[
                TermEvaluationError(
                    term_name=item["term_name"],
                    rule_class=item["rule_class"],
                    site=int(item["site"]),
                    exception_type=item["exception_type"],
                    exception_message=item["exception_message"],
                )
                for item in d["errors"]
            ],
        )


# ---- Explanation registry ------------------------------------------------


ExplanationContext = dict[str, Any]

# Callable shape: (term, state, meta) -> ExplanationContext
ContextExtractor = Callable[
    ["NamedHamiltonianTerm", "MPS", Any],   # meta typed as Any to avoid
    ExplanationContext,                      # import cycles
]

# Module-level registry. rule_class -> (template, extractor_or_None).
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
            return  # idempotent
        raise ValueError(
            f"rule_class {rule_class!r} already registered with a different "
            f"template/extractor"
        )
    _EXPLANATION_REGISTRY[rule_class] = (template, context_extractor)


def get_explanation(
    rule_class: str,
) -> tuple[str, ContextExtractor | None] | None:
    """Return (template, extractor) for rule_class, or None if missing."""
    return _EXPLANATION_REGISTRY.get(rule_class)


def clear_explanations() -> None:
    """Test affordance: reset the registry."""
    _EXPLANATION_REGISTRY.clear()


# ---- diagnose() core -----------------------------------------------------


_VALID_SORT_MODES = frozenset({"descending", "ascending", "site"})


def _assert_is_named_term(t: object, index: int) -> None:
    """Validate a term satisfies NamedHamiltonianTerm structurally.

    Raises TypeError with the offending index and the first missing
    attribute (clearer than the bare 'isinstance failed' message).
    """
    for attr in ("name", "rule_class", "site", "sites", "expectation"):
        if not hasattr(t, attr):
            raise TypeError(
                f"term at index {index} does not satisfy "
                f"NamedHamiltonianTerm (missing attribute {attr!r})"
            )
    if not isinstance(t, NamedHamiltonianTerm):
        raise TypeError(
            f"term at index {index} does not satisfy "
            f"NamedHamiltonianTerm (structural check failed)"
        )


def diagnose(
    state: MPS,
    meta: Any,
    hamiltonian_terms: list,
    threshold: float = 1e-6,
    sort: str = "descending",
) -> DiagnosticReport:
    """Compute a structured diagnostic report from per-term residual energies.

    See spec §4 for the full contract. Briefly:
      * Reads <H_term> from each named term.
      * Filters by |<H_term>| >= threshold into rule_violations.
      * Aggregates per-rule-class totals across ALL terms.
      * Captures term-evaluation exceptions into report.errors rather
        than propagating them.
      * Pure / read-only on its inputs.
    """
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if sort not in _VALID_SORT_MODES:
        raise ValueError(
            f"sort must be one of 'descending', 'ascending', 'site', "
            f"got {sort!r}"
        )
    for i, t in enumerate(hamiltonian_terms):
        _assert_is_named_term(t, i)

    successes: list[tuple[Any, float]] = []   # (term, energy)
    errors: list[TermEvaluationError] = []
    by_rule_class: dict[str, float] = {}

    for term in hamiltonian_terms:
        try:
            e = float(term.expectation(state))
        except Exception as exc:
            errors.append(TermEvaluationError(
                term_name=term.name,
                rule_class=term.rule_class,
                site=int(term.site),
                exception_type=type(exc).__name__,
                exception_message=str(exc),
            ))
            continue
        successes.append((term, e))
        by_rule_class[term.rule_class] = (
            by_rule_class.get(term.rule_class, 0.0) + e
        )

    total_energy = sum(e for (_, e) in successes)

    rule_violations: list[RuleViolation] = _build_violations(
        successes=successes, meta=meta, state=state,
        threshold=threshold, sort=sort, errors=errors,
    )

    return DiagnosticReport(
        total_energy=float(total_energy),
        threshold=float(threshold),
        n_terms_evaluated=len(hamiltonian_terms),
        n_violations=len(rule_violations),
        rule_violations=rule_violations,
        by_rule_class=by_rule_class,
        errors=errors,
    )


def _build_violations(
    *,
    successes: list[tuple[Any, float]],
    meta: Any,
    state: MPS,
    threshold: float,
    sort: str,
    errors: list[TermEvaluationError],
) -> list[RuleViolation]:
    """Filter and order successful per-term evaluations into violations.

    Filters by |<H_term>| >= threshold, looks up each site's AST path
    via meta.site_to_ast_path, attaches an explanation via the registry,
    and returns the list ordered per `sort`.

    Extractor failures (callable in the registry raises) are captured
    into `errors` and the affected violation falls back to a default
    explanation.
    """
    site_to_path = getattr(meta, "site_to_ast_path", {}) or {}

    violations: list[RuleViolation] = []
    for term, energy in successes:
        if abs(energy) < threshold:
            continue
        raw_path = site_to_path.get(int(term.site))
        if raw_path is None:
            ast_path: list[int] | None = None
            lookup_failed = True
        else:
            ast_path = list(raw_path)
            lookup_failed = False

        explanation, context = _render_explanation(
            term=term, state=state, meta=meta,
            energy=energy, ast_path=ast_path,
            errors=errors,
        )

        violations.append(RuleViolation(
            name=term.name,
            rule_class=term.rule_class,
            site=int(term.site),
            sites=list(term.sites),
            energy_contribution=float(energy),
            ast_path=ast_path,
            lookup_failed=lookup_failed,
            explanation=explanation,
            context=context,
        ))

    if sort == "descending":
        violations.sort(key=lambda v: (-v.energy_contribution, v.site, v.name))
    elif sort == "ascending":
        violations.sort(key=lambda v: (v.energy_contribution, v.site, v.name))
    elif sort == "site":
        violations.sort(key=lambda v: (v.site, v.name))
    else:
        raise AssertionError(f"unreachable sort mode {sort!r}")

    return violations


def _render_explanation(
    *,
    term: Any,
    state: MPS,
    meta: Any,
    energy: float,
    ast_path: list[int] | None,
    errors: list[TermEvaluationError],
) -> tuple[str, dict]:
    """Build (explanation_string, context_dict) for a violation.

    Looks up the rule_class in the explanation registry:
    - If absent: returns the fallback string and empty context.
    - If present (template, None): substitutes default slots only.
    - If present (template, extractor): calls extractor for extra slots;
      if extractor raises, captures the error and uses fallback.
    """
    entry = _EXPLANATION_REGISTRY.get(term.rule_class)
    fallback = (
        f"{term.rule_class} violated at site {term.site} "
        f"(energy {energy:.3g})"
    )
    default_ctx: dict = {
        "site": int(term.site),
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
        except Exception as exc:
            errors.append(TermEvaluationError(
                term_name=term.name,
                rule_class=term.rule_class,
                site=int(term.site),
                exception_type=type(exc).__name__,
                exception_message=f"explanation extractor: {exc}",
            ))
            return fallback, {}
        if not isinstance(extra, dict):
            errors.append(TermEvaluationError(
                term_name=term.name,
                rule_class=term.rule_class,
                site=int(term.site),
                exception_type="TypeError",
                exception_message=(
                    f"explanation extractor returned {type(extra).__name__}, "
                    f"expected dict"
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
            site=int(term.site),
            exception_type=type(exc).__name__,
            exception_message=f"template missing slot: {exc}",
        ))
        return fallback, {}

    # Report only the EXTRA context (what the extractor produced), not
    # the default slots — the default slots are already redundant with
    # the RuleViolation's own fields.
    extracted = {k: v for k, v in context.items() if k not in default_ctx}
    return rendered, extracted


# ---- Pretty-printer ------------------------------------------------------


def format_report(report: DiagnosticReport) -> str:
    """Human-readable rendering of a DiagnosticReport.

    The JSON form (report.to_json()) is the wire format for downstream
    consumers; this function is a developer convenience for printing in
    tests, CLIs, and log messages. Sub-project G is responsible for
    natural-language verbalization for end users.
    """
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
                else "ast_path=<unmapped site>"
            )
            lines.append(
                f"  [{v.rule_class}] site {v.site} "
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
                f"  [{e.rule_class}] {e.term_name} at site {e.site}: "
                f"{e.exception_type}: {e.exception_message}"
            )

    return "\n".join(lines)


# ---- Seed STLC templates -------------------------------------------------
#
# These templates exist so sub-project D can be tested in isolation. They
# use the rule_class names from sub-project B's TypingHamiltonian
# (T-Var, T-Abs, T-App-Arrow, T-Lit-Int, T-Lit-Bool, T-Bin-Arith,
# T-Bin-Cmp, T-Obligation), plus a few generic STLC class names that the
# mock tests use (T-App, T-If, T-Bin, T-IntLit, T-BoolLit). When B/E
# adapt B's terms to the protocol, they can call clear_explanations()
# and re-register with richer extractors.


_STLC_SEED_TEMPLATES: dict[str, str] = {
    # Mock-test-facing class names (compatible with the §7 mock suite).
    "T-Var": (
        "T-Var violated at site {site} (AST path {ast_path}): "
        "variable use type does not match its binder."
    ),
    "T-Abs": (
        "T-Abs violated at site {site} (AST path {ast_path}): "
        "lambda body type does not match declared return type."
    ),
    "T-App": (
        "T-App violated at site {site} (AST path {ast_path}): "
        "function applied to argument of wrong type "
        "(expected {expected_arg_type}, got {actual_arg_type})."
    ),
    "T-If": (
        "T-If violated at site {site} (AST path {ast_path}): "
        "condition is not Bool, or branches have different types."
    ),
    "T-Bin": (
        "T-Bin violated at site {site} (AST path {ast_path}): "
        "operands do not match expected types for operator."
    ),
    "T-IntLit": (
        "T-IntLit violated at site {site}: literal node has non-Int type tag."
    ),
    "T-BoolLit": (
        "T-BoolLit violated at site {site}: literal node has non-Bool "
        "type tag."
    ),
    # Sub-project B's actual rule_id strings (so a future adapter that
    # wraps B's terms verbatim renders sensibly with no extra work).
    "T-Lit-Int": (
        "T-Lit-Int violated at site {site}: integer literal has non-Int "
        "type tag."
    ),
    "T-Lit-Bool": (
        "T-Lit-Bool violated at site {site}: boolean literal has non-Bool "
        "type tag."
    ),
    "T-Bin-Arith": (
        "T-Bin-Arith violated at site {site} (AST path {ast_path}): "
        "arithmetic binop has non-Int operand or result."
    ),
    "T-Bin-Cmp": (
        "T-Bin-Cmp violated at site {site} (AST path {ast_path}): "
        "comparison binop has non-Bool result."
    ),
    "T-App-Arrow": (
        "T-App-Arrow violated at site {site} (AST path {ast_path}): "
        "application's function type does not yield the application's "
        "result type."
    ),
    "T-Obligation": (
        "T-Obligation violated at site {site} (AST path {ast_path}): "
        "node's posted type-obligation does not match its actual type."
    ),
}


def _seed_t_app_extractor(term, state, meta) -> ExplanationContext:
    """Stub extractor for the T-App seed template.

    Reports the expected/actual argument types as "?" placeholders. The
    real extractor lives in B/E and reads the type register off the
    state. This stub only exists so the template renders without a
    KeyError before that ships.
    """
    return {"expected_arg_type": "?", "actual_arg_type": "?"}


def register_stlc_seed_templates() -> None:
    """Register the bundled seed templates for STLC typing rules.

    Idempotent. Call this once after `clear_explanations()` (or at
    program start) when running D in isolation.
    """
    for rule_class, template in _STLC_SEED_TEMPLATES.items():
        extractor = _seed_t_app_extractor if rule_class == "T-App" else None
        register_explanation(rule_class, template, extractor)
