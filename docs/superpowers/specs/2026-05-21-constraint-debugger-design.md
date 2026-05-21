# Spec: Constraint Debugger for the QPCN Logic Layer

**Document type**: Implementation specification (sub-project D of the §10 roadmap).
**Date**: 2026-05-21.
**Author**: Brainstorming session, this branch.
**Status**: Approved design, ready for implementation plan.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` (root). This spec realizes §10.6 and operationalizes §8.2's "constraint debugging" claim.
**Acceptance owner**: human review of the structured-report test suite passing.

---

## 0. How to read this spec

This document is the contract for one sub-project. It exists because the §10 roadmap was decomposed into seven sub-projects (A–G); this is **sub-project D: the constraint debugger**. The other sub-projects are:

- **A** — AST ↔ MPS encoder/decoder (`docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md`). Already specified; produces the lattice and `EncodingMeta` we consume.
- **B** — typing-rule Hamiltonian compiler. Being specified in parallel; produces `H_typing`, a sum of named per-term Hermitian operators on the encoded lattice.
- **C** — evaluation Hamiltonian. Being specified in parallel; produces `H_eval` with the same per-term-named structure.
- **E** — STLC synthesis demo.
- **F** — MERA upgrade.
- **G** — LLM bridge / DSL runtime.

D **does not implement any Hamiltonian.** D is a *reporter*. It consumes a list of named per-term Hamiltonian operators (exposed by B, C, or any future constraint set), computes `⟨H_term⟩` per term on a given MPS state, and emits a structured, JSON-serializable diagnostic report. Every section below is part of the contract; if something is missing here that you need to decide while implementing, **stop and ask** (see §1).

---

## 1. Driving principles (non-negotiable)

The architecture document's §8.2 claims that **per-term residual energies give us a constraint-debugger for arbitrary logic** — not type-systems, not lambda calculus, *arbitrary* logic expressed as a sum of local Hermitian operators. This sub-project is where that claim is made operational. The principles below are the ones that, if violated, silently destroy the generality of the mechanism.

### 1.1 Per-term energies are first-class

Every Hamiltonian we build (B, C, and any future user-defined set) emits its constraint terms as **independently named, independently measurable operators**. The Hamiltonian object as a whole is `H = Σ_k H_k` where each `H_k` carries:

- a **stable name** (e.g. `"T-App@site_5"`),
- a **rule class** (e.g. `"T-App"`), reusable across many term instances,
- a **primary site** (the AST position the term diagnoses),
- and an `expectation(state)` method that computes `⟨state | H_k | state⟩`.

The debugger consumes this list. It does *not* re-derive the operators from the AST, and it does *not* peek inside the Hamiltonian object to break out subterms. The naming convention is the contract; B and C must satisfy it, and D consumes it via a `Protocol` (§3).

The "easy shortcut" — having D re-implement the typing rules or evaluation rules to know which AST positions are at fault — is **rejected**. It duplicates B/C logic, drifts out of sync, and makes the debugger STLC-specific. The architecture's claim is *generic* structured error reporting; D must be generic too.

### 1.2 The debugger does not re-implement the Hamiltonian

D reads `⟨H_term⟩` from the existing operators provided to it. It does not parse rule names to reconstruct algebra. It does not contract MPS tensors itself (beyond what `MPS.local_expectation` / `bond_op` already does — see Hamiltonian.bond_op in `src/qft_pcn/qft/hamiltonian.py`). It sorts by energy, filters by threshold, and emits structure. **D is a reporter, not an implementor.**

The "easy shortcut" — recomputing the rule's local operator inside the debugger because "it's small" — is rejected. The moment we have two definitions of `T-App`, they will drift.

### 1.3 Reports are JSON-serializable

The downstream consumer (G — LLM bridge) needs structured data to verbalize. `DiagnosticReport` and every nested dataclass have a `to_dict()` method whose output is `json.dumps`-able with no custom encoder. Floats are floats, ints are ints, strings are strings, lists/dicts of the same. No numpy scalars, no enum objects, no tuples in keys.

The "easy shortcut" — emitting the report as a Python `repr` string or a custom `__str__` — is rejected. G must operate on structured data.

### 1.4 Locality is preserved — every violation cites an AST path

Each rule violation has a `site: int` and an `ast_path: tuple[int, ...]`. The `ast_path` is recovered via `EncodingMeta.site_to_ast_path` (provided by sub-project A). The report points at the *AST node*, not just a site index in the encoded lattice, because the consumer reasons about programs, not about lattices. If the path lookup fails (site out of range, PAD site, etc.) the debugger emits `ast_path=null` and a `lookup_failed: true` flag — it does **not** raise. See §9 for the error model.

### 1.5 The debugger works on partial states

During imaginary-time evolution (sub-project E will exercise this), the state has not necessarily relaxed to the ground state. D reports whatever the *current* `⟨H_term⟩` values are. There is no notion of "the answer is wrong because evolution didn't finish" — the debugger reports residuals at any point in evolution and lets the caller decide whether the energy is small enough to count as solved. Threshold filtering (§4) is configurable; the default is `1e-6`.

The "easy shortcut" — requiring the state to be a definite (collapsed) configuration — is rejected. The whole point of the QPCN is that states are superpositions during search.

### 1.6 No special-case logic per Hamiltonian

D is generic over any `H` that exposes named per-term operators satisfying the `NamedHamiltonianTerm` protocol (§3). There is **no** per-Hamiltonian dispatch table inside the debugger. The same `diagnose()` function works on `H_typing`, `H_eval`, `H_typing + H_eval`, and any user-defined sum. Human-readable rule names come from B's and C's term-naming convention; D learns nothing STLC-specific.

The "easy shortcut" — adding an `isinstance(H, TypingHamiltonian)` branch to extract typing-specific structure — is rejected. The debugger gets more powerful as B, C, and future sub-projects add named rules; it must not grow special cases.

### 1.7 Explanations are templates, not free text

D maintains a small **registry** mapping `rule_class` strings to template strings with substitution slots (`{site}`, `{ast_path}`, `{type_expected}`, `{type_actual}`, …). Templates are filled with values extracted from a per-rule `context_extractor` callable that B/C may register; if no extractor is registered for a rule class, the explanation is the rule class name plus the site index. The mechanism is generic; the *content* is rule-specific and provided by the rule's authors (B, C, future).

The "easy shortcut" — hardcoding STLC error messages in the debugger — is rejected. It is the same special-case sin as §1.6 in different clothing.

---

## 2. Scope

### 2.1 In scope (this sub-project)

- The `NamedHamiltonianTerm` protocol (§3) — the contract B and C must satisfy.
- A `diagnose()` function (§4) that takes a state, encoding metadata, and a list of named terms; returns a `DiagnosticReport`.
- `DiagnosticReport`, `RuleViolation`, and `ExplanationContext` dataclasses (§5), all with `to_dict()` methods.
- An explanation **registry** (§6): `register_explanation(rule_class, template, context_extractor)`. Includes a small bundled set of templates for STLC typing rules (the ones B will publish) so the acceptance tests in §7 work end-to-end.
- Acceptance tests (§7): ill-typed STLC programs producing structured reports with the expected `rule_class`, `site`, and `ast_path`.
- A pretty-printer for the report (`format_report(report) -> str`) for CLI use. JSON is the wire format; the pretty-printer is for humans reading test failures.

### 2.2 Out of scope (deferred / belongs elsewhere)

- The actual typing rules. Those are B's job (and B will register its templates by calling `register_explanation` at module import time).
- The actual evaluation rules. C's job, same registration mechanism.
- Mapping reports to natural-language messages. That is G's job; D produces structured JSON, G verbalizes.
- Suggesting *fixes*. D identifies *which* constraints are stuck; it does not propose how to unstick them. Fix-suggestion may be future work; it is not part of the §8.2 claim.
- Per-bond observables not associated with a named term. The protocol is term-keyed.
- Visualization (term-energy histograms, AST overlays). Could be a future tool that consumes `DiagnosticReport`; it is not part of D.

### 2.3 Will not do, even if asked later

- Inline a typing-rule check inside the debugger. The debugger is generic; STLC is just one client.
- Mutate the input state or the Hamiltonian. `diagnose()` is read-only.
- Emit non-JSON-friendly types in the report (numpy scalars, enums, tuples as keys).
- Hide low-energy terms below threshold by default; threshold is a parameter and `0.0` is a valid value that reports every term including those at zero (for instrumentation / training-curve plotting).

---

## 3. The per-term Hamiltonian protocol

This is the contract that B and C must satisfy. It lives in `src/qft_pcn/logic/debugger.py` (the debugger module owns the protocol because the debugger is the consumer; B and C `import` it).

```python
# src/qft_pcn/logic/debugger.py

from typing import Protocol, runtime_checkable
import numpy as np

from qft_pcn.qft.mps import MPS


@runtime_checkable
class NamedHamiltonianTerm(Protocol):
    """One named, individually measurable Hamiltonian term.

    Sub-projects B and C produce iterables of these. The debugger consumes
    them generically — no per-rule branching inside the debugger.

    Attributes:
        name: Unique, stable identifier for this term instance.
              Convention: "<rule_class>@site_<k>" for one-site terms,
              "<rule_class>@sites_<k>_<l>" for two-site terms.
              Example: "T-App@site_5", "E-Beta@sites_3_7".
        rule_class: The reusable rule identifier (no site).
                    Example: "T-App", "T-Var", "E-Beta", "T-If".
        site: Primary AST site. For multi-site terms this is the *anchor*
              (typically the root of the constraint, e.g. the App site
              for T-App, even though the term touches fn/arg sites).
        sites: Full tuple of sites touched by this term. For one-site
               terms this is (site,). For two-site terms it's (i, j)
               with i < j. Required for tie-breaking when one site
               participates in multiple terms.
    """

    name: str
    rule_class: str
    site: int
    sites: tuple[int, ...]

    def expectation(self, state: MPS) -> float:
        """⟨state | H_term | state⟩ as a real number.

        MUST return a Python float (not numpy.float64). The debugger
        sorts and serializes these directly to JSON.

        For a Hermitian H_term the expectation is real by construction;
        implementations should take the real part explicitly and discard
        imaginary noise below 1e-10. Larger imaginary parts MUST raise
        ValueError (indicates a non-Hermitian term — a bug in B or C).
        """
        ...
```

**Why a Protocol and not an abstract base class.** B and C will likely have their own dataclass hierarchies (`OneSiteTerm`, `TwoSiteTerm`, …) for their own internal use. A Protocol lets them satisfy the contract structurally without inheriting from a debugger-defined class. `@runtime_checkable` lets the debugger assert protocol conformance at the entry point of `diagnose()` for a fail-fast error.

**Why `sites: tuple[int, ...]` in addition to `site: int`.** The primary `site` is the AST-path anchor — what the report points at. But a violation may *involve* other sites (e.g. T-App touches the App, the function, and the argument). Tools that want to highlight every involved site (G, future visualizers) use `sites`; tools that just want a single "where" use `site`. Both are required so consumers don't have to invent fallback logic.

**Hermiticity guarantee.** Every `H_term` is Hermitian (it represents a real energy contribution). The protocol's `expectation` returns a real float; implementations must take `Re ⟨ψ|H|ψ⟩` and check the imaginary residual is < 1e-10. The debugger does **not** re-check this — if B or C ship a non-Hermitian term, that is their bug, not D's.

---

## 4. The reporting API

```python
# src/qft_pcn/logic/debugger.py

def diagnose(
    state: MPS,
    meta: EncodingMeta,
    hamiltonian_terms: list[NamedHamiltonianTerm],
    threshold: float = 1e-6,
    sort: str = "descending",
) -> DiagnosticReport:
    """Compute a structured diagnostic report from per-term residual energies.

    Args:
        state: An MPS encoded by sub-project A (or a partially-evolved
               state from sub-project E). Read-only — not mutated.
        meta: The EncodingMeta returned by sub-project A's encode().
              Used for the site-to-AST-path lookup in violations.
        hamiltonian_terms: List of named per-term operators (typically
              obtained from sub-project B and/or C). Each must satisfy
              the NamedHamiltonianTerm protocol; the debugger asserts
              this at entry with a clear error message naming the
              offending term.
        threshold: Minimum |⟨H_term⟩| for a term to count as a violation.
              Defaults to 1e-6. Set to 0.0 to include every term
              (instrumentation mode). Negative values are rejected
              (ValueError).
        sort: "descending" (default) — violations sorted by energy
              contribution, largest first.
              "ascending" — smallest first (rarely useful).
              "site" — sorted by site index ascending (for stable
              report comparison in tests).
              Any other value: ValueError.

    Returns:
        A DiagnosticReport. See §5 for the shape.

    Raises:
        TypeError: a term in hamiltonian_terms does not satisfy
                   NamedHamiltonianTerm (with the offending index and
                   the missing attribute named).
        ValueError: threshold < 0 or sort not in valid set.

    Does NOT raise on:
        - terms whose expectation() raises (these are captured into the
          report's `errors` field; see §5.2).
        - sites that don't have an ast_path in meta (these violations are
          still reported, with ast_path=None and lookup_failed=True).
        - empty hamiltonian_terms (returns an empty report with zero total).
    """
```

**Behavioral contract:**

1. The function is **read-only** on its arguments. No mutation of `state`, `meta`, or any term.
2. `total_energy` is the sum of every term's `expectation`, regardless of threshold. The threshold gates only `rule_violations` membership.
3. `rule_violations` includes terms with `|⟨H_term⟩| >= threshold` (note: absolute value — Hermitian terms with negative expectation, while uncommon for the conventionally-positive Hamiltonian constraints we build in B/C, are not filtered out by sign).
4. Order of `rule_violations` matches the `sort` argument exactly. For `"descending"` and `"ascending"`, ties are broken by `site` ascending, then by `name` lexicographic — deterministic.
5. If a term's `expectation(state)` raises, the failure is captured into `report.errors` (term name, exception type, exception message); the term is **not** included in `rule_violations` or `total_energy`. Other terms continue to be evaluated. This is required because the debugger must remain useful during development of B/C, when individual terms may be buggy.

---

## 5. The report data shape (JSON-friendly)

All dataclasses live in `src/qft_pcn/logic/debugger.py`. All have a `to_dict()` method returning a plain Python structure (no numpy, no enums, no tuples) that `json.dumps` accepts.

### 5.1 `DiagnosticReport`

```python
@dataclass(frozen=True)
class DiagnosticReport:
    """Top-level report from diagnose()."""

    total_energy: float
    """Sum of every term's expectation, before threshold filtering."""

    threshold: float
    """The threshold the report was computed with."""

    n_terms_evaluated: int
    """Number of terms in the input list."""

    n_violations: int
    """len(rule_violations). For convenience in the JSON consumer."""

    rule_violations: list["RuleViolation"]
    """Terms with |expectation| >= threshold, ordered per `sort` argument."""

    by_rule_class: dict[str, float]
    """rule_class -> summed energy contribution across all sites.
       Aggregated over EVERY term (not filtered by threshold), so a caller
       can see e.g. that T-App as a class contributes 3.7 total even if
       no single site exceeds threshold."""

    errors: list["TermEvaluationError"]
    """Terms whose expectation() raised. Empty in the happy path."""

    def to_dict(self) -> dict: ...

    def to_json(self, *, indent: int | None = 2) -> str:
        """Convenience: json.dumps(self.to_dict(), indent=indent)."""
        ...

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosticReport":
        """Inverse of to_dict. Round-trip is a test requirement (§7.5)."""
        ...
```

### 5.2 `RuleViolation`

```python
@dataclass(frozen=True)
class RuleViolation:
    """One Hamiltonian term whose ⟨H_term⟩ is above threshold."""

    name: str
    """The term's full name, e.g. "T-App@site_5"."""

    rule_class: str
    """The reusable rule identifier, e.g. "T-App"."""

    site: int
    """Primary AST site."""

    sites: list[int]
    """Full list of sites touched (was tuple in the protocol; serialized
       as a JSON array). Always includes `site` somewhere in the list."""

    energy_contribution: float
    """⟨state | H_term | state⟩."""

    ast_path: list[int] | None
    """Path in the original AST, looked up via meta.site_to_ast_path.
       None if the lookup failed (PAD site, out-of-range, etc.)."""

    lookup_failed: bool
    """True iff ast_path is None because the site wasn't in
       meta.site_to_ast_path. Distinguishes legitimate None from
       "we just didn't try"."""

    explanation: str
    """Human-readable explanation, generated from the explanation registry
       (§6). For an unregistered rule_class, this is the fallback string
       f"{rule_class} violated at site {site} (energy {energy:.3g})"."""

    context: dict
    """The substitution dictionary that was passed to the explanation
       template (or an empty dict if no context_extractor was registered).
       JSON-serializable. Useful for the LLM bridge in G — even if the
       template was vacuous, the structured context is preserved."""

    def to_dict(self) -> dict: ...
```

### 5.3 `TermEvaluationError`

```python
@dataclass(frozen=True)
class TermEvaluationError:
    """Recorded when a term's expectation() raised."""

    term_name: str
    rule_class: str
    site: int
    exception_type: str       # e.g. "ValueError"
    exception_message: str    # str(exc)

    def to_dict(self) -> dict: ...
```

### 5.4 Example report JSON (from acceptance test §7.1)

For the ill-typed program `((λx:Int. x + 1)(true))`:

```json
{
  "total_energy": 1.0,
  "threshold": 1e-06,
  "n_terms_evaluated": 27,
  "n_violations": 1,
  "rule_violations": [
    {
      "name": "T-App@site_0",
      "rule_class": "T-App",
      "site": 0,
      "sites": [0, 1, 5],
      "energy_contribution": 1.0,
      "ast_path": [],
      "lookup_failed": false,
      "explanation": "T-App violated at site 0 (AST path []): function applied to argument of wrong type (expected Int, got Bool).",
      "context": {
        "site": 0,
        "ast_path": [],
        "expected_arg_type": "Int",
        "actual_arg_type": "Bool",
        "fn_site": 1,
        "arg_site": 5
      }
    }
  ],
  "by_rule_class": {
    "T-App": 1.0,
    "T-Var": 0.0,
    "T-Abs": 0.0,
    "T-IntLit": 0.0,
    "T-BoolLit": 0.0,
    "T-If": 0.0,
    "T-Bin": 0.0
  },
  "errors": []
}
```

---

## 6. The explanation registry

### 6.1 Mechanism

```python
# src/qft_pcn/logic/debugger.py

ExplanationContext = dict[str, object]

# Function type: given a violation's term + state + meta, return the
# substitution dictionary used to fill the template's slots.
ContextExtractor = Callable[
    ["NamedHamiltonianTerm", MPS, "EncodingMeta"],
    ExplanationContext,
]


def register_explanation(
    rule_class: str,
    template: str,
    context_extractor: ContextExtractor | None = None,
) -> None:
    """Register a human-readable template for a rule class.

    Args:
        rule_class: The rule's class identifier, e.g. "T-App".
        template: A format string with named slots, e.g.
            "T-App violated at site {site}: expected {expected_arg_type}, "
            "got {actual_arg_type}."
            Required slots: at minimum {site}. Optional slots: any names
            the context_extractor produces.
        context_extractor: Optional callable that returns a dict of
            template substitutions. If None, the template is filled
            with only {site}, {ast_path}, {rule_class}, {energy} — the
            "default context" available for every violation.

    The registration is idempotent for the same (rule_class, template,
    extractor) tuple, and raises ValueError on conflicting re-registration
    (same rule_class, different template or extractor). Tests need this
    to not be order-dependent.
    """


def get_explanation(rule_class: str) -> tuple[str, ContextExtractor | None] | None:
    """Returns (template, extractor) or None if not registered."""


def clear_explanations() -> None:
    """Test affordance: reset the registry. Not for production use."""
```

The registry is a module-level dict. B and C populate it at import time. The debugger's `diagnose()` reads it.

### 6.2 Default context

Even when no `context_extractor` is registered, the following slots are always available in the template:

| slot | source |
|---|---|
| `{site}` | `term.site` |
| `{ast_path}` | `meta.site_to_ast_path.get(term.site)` (may be `None`) |
| `{rule_class}` | `term.rule_class` |
| `{name}` | `term.name` |
| `{energy}` | `f"{violation.energy_contribution:.3g}"` |

A template that uses only these slots needs no extractor.

### 6.3 Bundled templates for STLC typing rules

Because the acceptance tests in §7 must run end-to-end *before* sub-project B has registered its own templates (the two sub-projects are written in parallel), D bundles a minimal seed set of STLC templates. B will, when it ships, call `clear_explanations()` and re-register its own templates with proper extractors. Until then, the seed set lets D's acceptance tests pass on its own.

The seed templates (in a helper `_register_stlc_seed_templates()` called from `__init__.py` only when the module is run under `pytest` or explicitly enabled — see Task 11 in the plan for the mechanism):

| `rule_class` | template |
|---|---|
| `T-Var` | `"T-Var violated at site {site} (AST path {ast_path}): variable use type does not match its binder."` |
| `T-Abs` | `"T-Abs violated at site {site} (AST path {ast_path}): lambda body type does not match declared return type."` |
| `T-App` | `"T-App violated at site {site} (AST path {ast_path}): function applied to argument of wrong type (expected {expected_arg_type}, got {actual_arg_type})."` |
| `T-If` | `"T-If violated at site {site} (AST path {ast_path}): condition is not Bool, or branches have different types."` |
| `T-Bin` | `"T-Bin violated at site {site} (AST path {ast_path}): operands do not match expected types for operator."` |
| `T-IntLit` | `"T-IntLit violated at site {site}: literal node has non-Int type tag."` |
| `T-BoolLit` | `"T-BoolLit violated at site {site}: literal node has non-Bool type tag."` |

The `T-App` entry references `{expected_arg_type}` and `{actual_arg_type}`. The seed set ships with a single seed extractor for `T-App` that reads the function and argument type tags off the state via single-site projector measurements (it knows about `meta.species` and the `type` register from sub-project A; this is the only spot in D that touches the encoding's *meaning*, and it lives in a clearly-marked seed module that B will overwrite). All other seed templates rely on default-context slots only — they describe the rule but don't try to extract specifics.

This compromise is local: the seed set exists *only* to make D testable in isolation. When B ships, B owns the registry. See §11 open question (closed) and §12 contract for G.

---

## 7. Acceptance tests

Located in `src/qft_pcn/tests/test_logic_debugger.py`. Uses pytest convention from the existing tests. Where possible the tests use a **mock** `NamedHamiltonianTerm` implementation rather than a real Hamiltonian, so D can be tested without B being ready. The end-to-end test (§7.6) is gated on B being available; until then it is skipped with `pytest.skip` carrying an explanatory message.

### 7.1 Structured report shape

```python
def test_diagnose_emits_structured_report_for_ill_typed_program():
    """End-to-end: an ill-typed program produces a report with at least
    one T-App violation citing the application site."""
    # Build an MPS encoding ((λx:Int. x + 1)(true)) — the argument is Bool
    # but T-App expects Int, so T-App@site_0 has ⟨H⟩ = 1.0.
    p = parse("(\\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)

    # Construct mock terms standing in for what B will produce.
    # The mock T-App term returns 1.0 on this state; all other mock terms
    # return 0.0. This isolates D from B's correctness.
    terms = _mock_stlc_terms_for(p, state, meta, violated={"T-App@site_0": 1.0})

    report = diagnose(state, meta, terms, threshold=1e-6)

    assert isinstance(report, DiagnosticReport)
    assert abs(report.total_energy - 1.0) < 1e-10
    assert report.n_violations == 1
    assert len(report.rule_violations) == 1
    v = report.rule_violations[0]
    assert v.rule_class == "T-App"
    assert v.site == 0
    assert v.ast_path == []      # the App is the root
    assert v.lookup_failed is False
    assert "T-App" in v.explanation
```

### 7.2 JSON round-trip

```python
def test_report_is_json_roundtrippable():
    """to_json() output parses back to an equal report."""
    p = parse("(\\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={"T-App@site_0": 1.0})
    report = diagnose(state, meta, terms)

    s = report.to_json()
    d = json.loads(s)
    rt = DiagnosticReport.from_dict(d)
    assert rt == report                       # frozen dataclass equality
```

### 7.3 Threshold filtering and ordering

```python
def test_threshold_filters_low_energy_terms():
    p = parse("\\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(
        p, state, meta,
        violated={
            "T-App@site_0": 0.5,
            "T-Abs@site_0": 1.5,
            "T-Var@site_1": 1e-9,           # below default threshold
        },
    )
    report = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    assert report.n_violations == 2
    assert report.rule_violations[0].rule_class == "T-Abs"   # 1.5 > 0.5
    assert report.rule_violations[1].rule_class == "T-App"
    # by_rule_class sums every term, including sub-threshold ones:
    assert abs(report.by_rule_class["T-Var"] - 1e-9) < 1e-15


def test_threshold_zero_includes_every_term():
    p = parse("\\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={"T-Var@site_1": 1e-12})
    report = diagnose(state, meta, terms, threshold=0.0)
    assert report.n_violations == 1


def test_sort_modes():
    p = parse("\\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(
        p, state, meta,
        violated={"T-App@site_0": 0.3, "T-Abs@site_1": 0.7, "T-Var@site_2": 0.5},
    )
    asc  = diagnose(state, meta, terms, sort="ascending")
    desc = diagnose(state, meta, terms, sort="descending")
    site = diagnose(state, meta, terms, sort="site")
    assert [v.energy_contribution for v in asc.rule_violations] == [0.3, 0.5, 0.7]
    assert [v.energy_contribution for v in desc.rule_violations] == [0.7, 0.5, 0.3]
    assert [v.site for v in site.rule_violations] == [0, 1, 2]
```

### 7.4 Term-evaluation error capture

```python
def test_terms_that_raise_are_captured_into_errors_not_propagated():
    p = parse("\\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    good = _MockTerm("T-Var@site_1", "T-Var", 1, (1,), value=0.5)
    bad  = _MockTermThatRaises("T-App@site_0", "T-App", 0, (0,),
                               exc=ValueError("intentional"))
    report = diagnose(state, meta, [good, bad], threshold=0.0)
    assert report.n_violations == 1                       # only `good`
    assert report.rule_violations[0].rule_class == "T-Var"
    assert len(report.errors) == 1
    assert report.errors[0].term_name == "T-App@site_0"
    assert report.errors[0].exception_type == "ValueError"
    assert "intentional" in report.errors[0].exception_message
```

### 7.5 Protocol enforcement at entry

```python
def test_non_conforming_term_raises_typeerror_with_diagnostic():
    class NotATerm: pass     # missing `name`, `rule_class`, etc.
    state, meta = encode(parse("\\x:Int. x"), N=32, chi_max=16)
    with pytest.raises(TypeError) as exc_info:
        diagnose(state, meta, [NotATerm()])
    msg = str(exc_info.value)
    assert "index 0" in msg
    assert "NamedHamiltonianTerm" in msg
```

### 7.6 End-to-end with the real B Hamiltonian (skip if B not ready)

```python
@pytest.mark.skipif(
    not _has_module("qft_pcn.logic.typing_hamiltonian"),
    reason="sub-project B not yet implemented",
)
def test_end_to_end_ill_typed_program_with_real_typing_hamiltonian():
    """Once B ships, this is the test that proves the whole stack works:
    ill-typed program → real H_typing → diagnose → report with the
    right rule_class and site."""
    from qft_pcn.logic.typing_hamiltonian import build_typing_terms

    p = parse("(\\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)
    terms = build_typing_terms(meta)
    report = diagnose(state, meta, terms, threshold=1e-6)

    rule_classes = {v.rule_class for v in report.rule_violations}
    assert "T-App" in rule_classes
    # The T-App violation must cite the application site:
    app_violations = [v for v in report.rule_violations if v.rule_class == "T-App"]
    assert any(v.site == 0 for v in app_violations)
```

This is the test the acceptance criterion in §10.6 of the architecture doc names ("an ill-typed program produces a structured report identifying the specific typing-rule violation"). Until B is ready, the test is skipped, and the §7.1 mock test stands in for the structural correctness of D itself.

### 7.7 Empty input

```python
def test_empty_terms_list_returns_zero_report():
    state, meta = encode(parse("\\x:Int. x"), N=32, chi_max=16)
    report = diagnose(state, meta, [], threshold=1e-6)
    assert report.total_energy == 0.0
    assert report.n_terms_evaluated == 0
    assert report.n_violations == 0
    assert report.by_rule_class == {}
    assert report.errors == []
```

### 7.8 Partial state (mid-evolution)

```python
def test_diagnose_works_on_partially_evolved_state():
    """The debugger must work on states that haven't fully relaxed."""
    p = parse("(\\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)

    # Build a mock term that drops linearly from 1.0 to 0.3 as a stand-in
    # for partial evolution (we don't depend on evolution.py — that's E's job).
    terms = _mock_stlc_terms_for(p, state, meta, violated={"T-App@site_0": 0.3})
    report = diagnose(state, meta, terms, threshold=1e-6)

    assert abs(report.total_energy - 0.3) < 1e-10
    assert report.n_violations == 1
    # The report is still well-formed; it just shows a smaller residual.
```

### 7.9 Lookup-failure path

```python
def test_ast_path_lookup_failure_does_not_raise():
    """A term anchored at a PAD site (e.g. a global constraint) gets
    ast_path=None, lookup_failed=True — not an exception."""
    p = parse("\\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    # Pick a site that is PAD and therefore not in meta.site_to_ast_path:
    pad_site = max(meta.site_to_ast_path) + 1
    bogus = _MockTerm("Global@site_X", "Global", pad_site, (pad_site,), value=0.7)
    report = diagnose(state, meta, [bogus], threshold=0.0)
    assert report.n_violations == 1
    v = report.rule_violations[0]
    assert v.ast_path is None
    assert v.lookup_failed is True
```

### 7.10 Explanation registry

```python
def test_register_explanation_idempotent():
    clear_explanations()
    register_explanation("T-App", "T-App@{site}: expected={expected_arg_type}")
    register_explanation("T-App", "T-App@{site}: expected={expected_arg_type}")
    # No raise.


def test_register_explanation_conflict_raises():
    clear_explanations()
    register_explanation("T-App", "first template")
    with pytest.raises(ValueError):
        register_explanation("T-App", "different template")


def test_unregistered_rule_class_uses_fallback():
    clear_explanations()
    p = parse("\\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    term = _MockTerm("X-Custom@site_0", "X-Custom", 0, (0,), value=0.5)
    report = diagnose(state, meta, [term], threshold=0.0)
    assert "X-Custom" in report.rule_violations[0].explanation
    assert "0.5" in report.rule_violations[0].explanation        # uses {energy}


def test_template_with_extractor():
    clear_explanations()
    def extractor(term, state, meta):
        return {"foo": "bar"}
    register_explanation("T-X", "rule {foo} at site {site}", extractor)
    p = parse("\\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    term = _MockTerm("T-X@site_0", "T-X", 0, (0,), value=0.5)
    report = diagnose(state, meta, [term], threshold=0.0)
    assert report.rule_violations[0].explanation == "rule bar at site 0"
    assert report.rule_violations[0].context == {"foo": "bar"}
```

### 7.11 Pretty-printer

```python
def test_format_report_is_human_readable():
    p = parse("(\\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={"T-App@site_0": 1.0})
    report = diagnose(state, meta, terms)
    text = format_report(report)
    # No assertion on exact format — just sanity checks.
    assert "Total energy: 1." in text
    assert "T-App" in text
    assert "site 0" in text
```

### 7.12 Mock helpers (`_MockTerm`, `_MockTermThatRaises`, `_mock_stlc_terms_for`)

Defined in the test module. They're not in the production module — they're a test affordance so D can be tested without B/C. The protocol allows any structurally conforming class:

```python
@dataclass
class _MockTerm:
    name: str
    rule_class: str
    site: int
    sites: tuple[int, ...]
    value: float = 0.0
    def expectation(self, state) -> float:
        return float(self.value)
```

`_mock_stlc_terms_for(p, state, meta, violated)` builds a list of mocks: one per rule class × site combination, all returning 0.0 *except* the ones explicitly named in the `violated` dict.

---

## 8. File layout

```
src/qft_pcn/logic/
├── __init__.py                  # public re-exports: diagnose, DiagnosticReport,
│                                #   RuleViolation, NamedHamiltonianTerm,
│                                #   register_explanation, format_report
├── debugger.py                  # protocol + diagnose() + dataclasses + registry
│                                #   + format_report() + seed STLC templates
└── (existing files from sub-project A)
src/qft_pcn/tests/
└── test_logic_debugger.py       # all tests from §7
```

Public exports from `src/qft_pcn/logic/__init__.py` (additive — A's exports stay):

```python
from .debugger import (
    NamedHamiltonianTerm,
    DiagnosticReport,
    RuleViolation,
    TermEvaluationError,
    diagnose,
    format_report,
    register_explanation,
    get_explanation,
    clear_explanations,
)
```

Top-level `src/qft_pcn/__init__.py` re-exports `diagnose` and `DiagnosticReport` so callers can `from qft_pcn import diagnose`.

**No changes to any existing file** except `src/qft_pcn/logic/__init__.py` (additive re-export) and `src/qft_pcn/__init__.py` (additive re-export). In particular, `src/qft_pcn/qft/*` is untouched.

---

## 9. Error model

D defines no new exception types. It uses standard Python exceptions:

| condition | exception | message format |
|---|---|---|
| `threshold < 0` | `ValueError` | `"threshold must be >= 0, got {threshold}"` |
| `sort` not in `{"descending", "ascending", "site"}` | `ValueError` | `"sort must be one of 'descending', 'ascending', 'site', got {sort!r}"` |
| term doesn't satisfy `NamedHamiltonianTerm` | `TypeError` | `"term at index {i} does not satisfy NamedHamiltonianTerm (missing attribute {attr!r})"` |
| `register_explanation` conflicting re-registration | `ValueError` | `"rule_class {rule_class!r} already registered with a different template/extractor"` |

Term-evaluation failures (`term.expectation(state)` raises) are **not** re-raised; they go into `report.errors`. This is required for §1.5 (the debugger must work mid-evolution, when terms may be transiently ill-defined).

Lookup failures (`meta.site_to_ast_path` missing a site) are **not** errors; they produce `ast_path=None, lookup_failed=True`.

---

## 10. Acceptance criteria

The sub-project is complete when:

1. The structured-report test §7.1 passes (mock T-App violation reported).
2. The JSON round-trip test §7.2 passes (`report == from_dict(to_dict(report))`).
3. The threshold and ordering tests §7.3 pass (3 sub-tests).
4. The term-error-capture test §7.4 passes.
5. The protocol-enforcement test §7.5 passes.
6. The end-to-end test §7.6 either passes (if B is ready) or skips cleanly with a non-misleading reason string.
7. The empty-input test §7.7 passes.
8. The partial-state test §7.8 passes.
9. The lookup-failure test §7.9 passes.
10. The explanation-registry tests §7.10 pass (4 sub-tests).
11. The pretty-printer test §7.11 passes (sanity check only).
12. All previously passing tests in `src/qft_pcn/tests/` still pass.
13. `from qft_pcn import diagnose, DiagnosticReport` works.
14. `python -c "from qft_pcn.logic.debugger import NamedHamiltonianTerm; print(NamedHamiltonianTerm.__module__)"` prints `qft_pcn.logic.debugger` (the protocol is importable from the right place; sub-projects B and C will import it from here).

No claim of completion is acceptable without these tests actually running green in a fresh shell. (See `superpowers:verification-before-completion`.)

---

## 11. Open questions

**None.** Every design choice above is locked. If you, while implementing, find a real ambiguity that this document does not resolve — stop and ask the human. Do not paper over it.

(For the record, the questions that *were* asked and *are* resolved:

- *Should D bundle seed STLC templates so it can be tested before B ships?* — Yes (§6.3). The seed set is local to D and is overwritten by B at import time when B ships.
- *Should `expectation` return real or complex?* — Real (§3). Hermitian by contract; implementations take `Re ⟨ψ|H|ψ⟩` and check the imaginary residual.
- *Should the debugger contract use a protocol or an ABC?* — Protocol (§3). B and C should not have to inherit from a debugger class.
- *Should term-evaluation errors propagate or be captured?* — Captured (§4, §1.5). The debugger remains useful during B/C development.
- *Should the threshold compare to `|⟨H⟩|` or `⟨H⟩`?* — Absolute value (§4 behavioral contract). Hermitian-constraint terms are conventionally non-negative, but the debugger does not assume.
- *Should `total_energy` respect the threshold?* — No (§4 behavioral contract). Total is the full sum; threshold gates only the `rule_violations` list.
)

---

## 12. Contract for G (consumes the report JSON)

Sub-project G (LLM bridge / DSL runtime) is the primary consumer of `DiagnosticReport`. The contract:

1. **G receives a JSON object** with the schema described in §5.1–5.4. Every field is required (no field is omitted; absent values are `null` or empty list/dict, never missing keys).
2. **G never mutates the report.** If G wants a transformed view, it constructs a new object.
3. **G uses `rule_class` for routing** (e.g. selecting a verbalization template), not `name` — `name` is per-site and changes every program, `rule_class` is stable.
4. **G uses `ast_path` for highlighting** the location in the rendered source. If `ast_path` is `null` and `lookup_failed` is true, G falls back to citing `site` directly with a phrase like "(at lattice site N, AST location unknown)".
5. **G uses `context` to enrich messages.** The `context` dict carries everything the rule's template knew; G's verbalizer can use the same keys to generate a richer NL message than the bundled template's plain-text fallback. Schema for context keys is **per-`rule_class`** and is owned by whichever sub-project (B, C, future) registered the rule; G must tolerate missing context keys gracefully.
6. **G respects the ordering of `rule_violations`** for display priority. D delivers them sorted by `sort` argument; G should not re-sort unless it has a deliberate reason and documents it.
7. **G is allowed to read `errors`** to surface "the debugger could not evaluate term X" warnings to developers in dev-mode. End users typically don't see this; in production G filters `errors` out of its NL message.
8. **The schema is versioned implicitly via the module.** When D changes the schema, the module version (and the import path) changes. G pins a specific D version. There is no `version` field in the JSON itself for sub-project D (this is fine because both D and G are in the same repo for now; if D is later vendored standalone, a `_schema_version` field is added at the top of `DiagnosticReport`).

---

## 13. Glossary (local)

- **Named term** — A single Hamiltonian operator `H_k` that satisfies the `NamedHamiltonianTerm` protocol (has `name`, `rule_class`, `site`, `sites`, `expectation`).
- **Rule class** — The reusable identifier shared by all term instances of a given rule, e.g. `T-App`. One rule, many sites, one `rule_class`.
- **Site** — One of the `N` positions in the MPS, indexed 0…N-1, as defined by sub-project A.
- **AST path** — A tuple of integers indexing children in the original AST, recovered from `EncodingMeta.site_to_ast_path`. Empty tuple `()` means the root.
- **Residual energy** — The non-zero `⟨H_term⟩` of a term whose constraint is unsatisfied at the current state. Zero (or below threshold) ⇒ the constraint is satisfied at this state.
- **Threshold** — A floating-point number below which `|⟨H_term⟩|` is treated as "satisfied" for the purposes of populating `rule_violations`.
- **Explanation registry** — Module-level dict mapping `rule_class → (template, context_extractor)`. Populated by B, C, future sub-projects; read by D's `diagnose()`.
- **Context extractor** — A callable provided alongside a template that, given the term, the state, and the encoding metadata, returns the dict of substitutions to fill the template's slots.
- **Seed template** — A minimal, low-information template bundled with D so D can be tested in isolation; overwritten by B/C when those modules are imported.
