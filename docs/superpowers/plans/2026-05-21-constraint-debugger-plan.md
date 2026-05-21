# Constraint Debugger Implementation Plan (Sub-project D)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sub-project D from `docs/superpowers/specs/2026-05-21-constraint-debugger-design.md`: a generic, JSON-emitting, per-term-energy-based constraint debugger that consumes any Hamiltonian exposing the `NamedHamiltonianTerm` protocol.

**Architecture:** A single new module `src/qft_pcn/logic/debugger.py` containing the `NamedHamiltonianTerm` Protocol, three dataclasses (`DiagnosticReport`, `RuleViolation`, `TermEvaluationError`), the `diagnose()` entry point, an explanation registry (`register_explanation`, `get_explanation`, `clear_explanations`), a JSON-friendly `to_dict`/`to_json`/`from_dict` round-trip, a pretty-printer (`format_report`), and a seed set of STLC explanation templates. No new exception types; no changes to `src/qft_pcn/qft/*`. Tests live in `src/qft_pcn/tests/test_logic_debugger.py` and use mock terms so they don't depend on sub-project B's Hamiltonian being ready.

**Tech Stack:** Python 3.11, numpy 1.26, pytest. Built on the existing `src/qft_pcn/qft/` (MPS) and `src/qft_pcn/logic/` (encoder, EncodingMeta) machinery from sub-project A.

**Driving principles (from spec §1, non-negotiable):**

1. **Per-term energies are first-class** — every term has a name, rule_class, site(s), and a real-valued `expectation`. D consumes these via a Protocol.
2. **The debugger does not re-implement the Hamiltonian.** D reads `⟨H_term⟩`, it does not contract operators itself, it does not parse rule names to reconstruct algebra.
3. **Reports are JSON-serializable** end-to-end. `to_dict` → `json.dumps` works, `from_dict` round-trips.
4. **Locality is preserved.** Every violation has a `site` and an `ast_path` looked up via `meta.site_to_ast_path`.
5. **Works on partial states.** No assumption of ground-state collapse; reports whatever `⟨H_term⟩` is *now*.
6. **No special-case logic per Hamiltonian.** D is generic; the rule names come from B/C.
7. **Explanations are templates, not free text.** A registry maps `rule_class → (template, optional context_extractor)`.

**Reference:** The spec at `docs/superpowers/specs/2026-05-21-constraint-debugger-design.md` is the authoritative contract. When in doubt, re-read it. If something here disagrees with the spec, the spec wins.

**Test command throughout:** Run pytest via the project venv: `.venv/bin/python -m pytest <path> -v`.

**Sub-project A is already complete.** Imports of `encode`, `parse`, `EncodingMeta`, `MPS` work; you should not need to modify any A or qft/* file.

---

## File Map

Files this plan creates or modifies:

| Path | Role |
|---|---|
| `src/qft_pcn/logic/debugger.py` | The module. Protocol, dataclasses, registry, `diagnose`, `format_report`, seed templates. **Created.** |
| `src/qft_pcn/tests/test_logic_debugger.py` | All §7 tests. **Created.** |
| `src/qft_pcn/logic/__init__.py` | Additive re-exports of debugger names. **Modified.** |
| `src/qft_pcn/__init__.py` | Additive re-export of `diagnose` / `DiagnosticReport` at the top level. **Modified.** |

No other files are touched. In particular `src/qft_pcn/qft/*` is untouched.

---

## Task 1: Establish baseline & inspect EncodingMeta

**Files:** None modified. Read-only inspection.

We need to know the exact shape of `EncodingMeta.site_to_ast_path` (built by A) before we write tests against it.

- [ ] **Step 1: Verify the venv and pytest are present**

Run: `.venv/bin/python -m pytest --version`
Expected: `pytest <version>` (no error).

- [ ] **Step 2: Run the existing logic test suite to confirm green baseline**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -v --ignore=src/qft_pcn/tests/test_quantum.py 2>&1 | tail -10`
Expected: all tests pass (sub-project A complete).

- [ ] **Step 3: Inspect EncodingMeta exports**

Run:
```bash
.venv/bin/python -c "
from qft_pcn.logic.encoding import EncodingMeta
import dataclasses
for f in dataclasses.fields(EncodingMeta):
    print(f.name, f.type)
"
```
Expected: includes a field named `site_to_ast_path` whose value type is `dict[int, tuple[int, ...]]`.

- [ ] **Step 4: Confirm the test-helper `encode` and `parse` are importable**

Run:
```bash
.venv/bin/python -c "
from qft_pcn.logic import encode, parse, EncodingMeta
p = parse(r'\x:Int. x')
state, meta = encode(p, N=32, chi_max=16)
print(type(meta).__name__, sorted(meta.site_to_ast_path.items())[:5])
"
```
Expected: prints `EncodingMeta` and a list of `(site_index, path_tuple)` pairs starting at `(0, ())`.

No commit for this task — pure verification.

---

## Task 2: Create empty debugger module + Protocol

**Files:**
- Create: `src/qft_pcn/logic/debugger.py`
- Create: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §3, the `NamedHamiltonianTerm` Protocol is the first thing B and C will import. It must exist before anything else in this module.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_logic_debugger.py` with this content:

```python
"""Tests for the constraint debugger (sub-project D)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from qft_pcn.qft.mps import MPS
from qft_pcn.logic import encode, parse


def test_named_hamiltonian_term_protocol_importable():
    """The Protocol exists at the expected import path so that B and C
    can import it without circular dependencies."""
    from qft_pcn.logic.debugger import NamedHamiltonianTerm
    assert NamedHamiltonianTerm.__module__ == "qft_pcn.logic.debugger"


def test_named_hamiltonian_term_is_runtime_checkable():
    """Protocol is runtime_checkable so diagnose() can validate inputs."""
    from qft_pcn.logic.debugger import NamedHamiltonianTerm

    @dataclass
    class Conforming:
        name: str
        rule_class: str
        site: int
        sites: tuple[int, ...]
        def expectation(self, state) -> float:
            return 0.0

    c = Conforming("X@0", "X", 0, (0,))
    assert isinstance(c, NamedHamiltonianTerm)


def test_named_hamiltonian_term_rejects_missing_attribute():
    """A class missing `expectation` does not satisfy the Protocol."""
    from qft_pcn.logic.debugger import NamedHamiltonianTerm

    class Missing:
        name = "X@0"
        rule_class = "X"
        site = 0
        sites = (0,)
        # no expectation

    assert not isinstance(Missing(), NamedHamiltonianTerm)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v`
Expected: 3 errors, all `ModuleNotFoundError: No module named 'qft_pcn.logic.debugger'`.

- [ ] **Step 3: Create the debugger module with the Protocol**

Create `src/qft_pcn/logic/debugger.py`:

```python
"""Constraint debugger for the QPCN logic layer (sub-project D).

This module turns per-term residual energies of a Hamiltonian sum
H = Σ_k H_k into a structured, JSON-serializable diagnostic report.

It is generic: any Hamiltonian whose terms satisfy the
NamedHamiltonianTerm protocol can be diagnosed. Sub-projects B
(typing Hamiltonian) and C (evaluation Hamiltonian) are the
first consumers. The debugger never re-implements rules.

See docs/superpowers/specs/2026-05-21-constraint-debugger-design.md
for the contract.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from qft_pcn.qft.mps import MPS


@runtime_checkable
class NamedHamiltonianTerm(Protocol):
    """One named, individually measurable Hamiltonian term.

    Sub-projects B and C produce iterables of these. The debugger
    consumes them generically — no per-rule branching inside the
    debugger.

    Attributes:
        name: Unique, stable identifier for this term instance.
              Convention: "<rule_class>@site_<k>" or
              "<rule_class>@sites_<k>_<l>".
        rule_class: The reusable rule identifier (no site).
                    Example: "T-App".
        site: Primary AST site. The "anchor" the report points at.
        sites: Full tuple of sites touched by this term.

    Methods:
        expectation(state): ⟨state | H_term | state⟩ as a real Python
            float. Implementations MUST take Re(.) explicitly and check
            the imaginary residual is < 1e-10 (Hermitian by contract).
    """

    name: str
    rule_class: str
    site: int
    sites: tuple[int, ...]

    def expectation(self, state: MPS) -> float: ...
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): NamedHamiltonianTerm protocol (sub-project D, scaffolding)

Introduces the runtime-checkable Protocol that sub-projects B (typing
Hamiltonian) and C (evaluation Hamiltonian) will satisfy. The debugger
consumes terms generically through this protocol; it does not re-derive
the operators from the AST.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Dataclasses for the report (`RuleViolation`, `TermEvaluationError`, `DiagnosticReport`)

**Files:**
- Modify: `src/qft_pcn/logic/debugger.py`
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §5. Frozen dataclasses with `to_dict`. JSON serialization is added in Task 4.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_rule_violation_dataclass_shape():
    from qft_pcn.logic.debugger import RuleViolation
    v = RuleViolation(
        name="T-App@site_5",
        rule_class="T-App",
        site=5,
        sites=[5, 6, 8],
        energy_contribution=1.0,
        ast_path=[0, 1],
        lookup_failed=False,
        explanation="T-App violated",
        context={"foo": "bar"},
    )
    assert v.name == "T-App@site_5"
    assert v.rule_class == "T-App"
    assert v.site == 5
    assert v.sites == [5, 6, 8]
    assert v.energy_contribution == 1.0
    assert v.ast_path == [0, 1]
    assert v.lookup_failed is False
    assert v.explanation == "T-App violated"
    assert v.context == {"foo": "bar"}


def test_rule_violation_is_frozen():
    from qft_pcn.logic.debugger import RuleViolation
    import dataclasses
    v = RuleViolation(
        name="X@0", rule_class="X", site=0, sites=[0],
        energy_contribution=0.5, ast_path=None, lookup_failed=True,
        explanation="x", context={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.name = "Y@0"  # type: ignore[misc]


def test_rule_violation_to_dict_is_json_safe():
    from qft_pcn.logic.debugger import RuleViolation
    v = RuleViolation(
        name="T-App@site_5",
        rule_class="T-App",
        site=5,
        sites=[5, 6, 8],
        energy_contribution=1.0,
        ast_path=[0, 1],
        lookup_failed=False,
        explanation="T-App violated",
        context={"a": 1, "b": "x"},
    )
    d = v.to_dict()
    # round-trips through json.dumps without a custom encoder:
    s = json.dumps(d)
    rt = json.loads(s)
    assert rt == d
    # primitive types only:
    assert isinstance(d["sites"], list)
    assert isinstance(d["site"], int)
    assert isinstance(d["energy_contribution"], float)


def test_rule_violation_to_dict_handles_none_ast_path():
    from qft_pcn.logic.debugger import RuleViolation
    v = RuleViolation(
        name="X@0", rule_class="X", site=0, sites=[0],
        energy_contribution=0.0, ast_path=None, lookup_failed=True,
        explanation="x", context={},
    )
    d = v.to_dict()
    assert d["ast_path"] is None
    assert d["lookup_failed"] is True


def test_term_evaluation_error_dataclass_shape():
    from qft_pcn.logic.debugger import TermEvaluationError
    e = TermEvaluationError(
        term_name="T-App@site_0",
        rule_class="T-App",
        site=0,
        exception_type="ValueError",
        exception_message="bad",
    )
    d = e.to_dict()
    assert json.dumps(d)
    assert d["term_name"] == "T-App@site_0"
    assert d["exception_type"] == "ValueError"


def test_diagnostic_report_dataclass_shape():
    from qft_pcn.logic.debugger import DiagnosticReport
    r = DiagnosticReport(
        total_energy=1.0,
        threshold=1e-6,
        n_terms_evaluated=3,
        n_violations=1,
        rule_violations=[],
        by_rule_class={"T-App": 1.0},
        errors=[],
    )
    d = r.to_dict()
    assert json.dumps(d, indent=2)
    assert d["total_energy"] == 1.0
    assert d["by_rule_class"] == {"T-App": 1.0}
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -20`
Expected: 6 new tests fail with `ImportError` / `AttributeError` for `RuleViolation`, `TermEvaluationError`, `DiagnosticReport`.

- [ ] **Step 3: Implement the dataclasses**

Append to `src/qft_pcn/logic/debugger.py`:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RuleViolation:
    """One Hamiltonian term whose |⟨H_term⟩| ≥ threshold.

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
            "by_rule_class": {k: float(v) for k, v in self.by_rule_class.items()},
            "errors": [e.to_dict() for e in self.errors],
        }
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 9 passed (3 from Task 2 + 6 new).

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): frozen dataclasses for the diagnostic report

Adds RuleViolation, TermEvaluationError, and DiagnosticReport. Each
exposes to_dict() that returns a structure of Python primitives — no
numpy scalars, no enums, no tuples — so json.dumps works without a
custom encoder. JSON round-trip is the contract with sub-project G
(LLM bridge), tested for serialization safety here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: JSON round-trip (`to_json`, `from_dict`)

**Files:**
- Modify: `src/qft_pcn/logic/debugger.py`
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §5.1: `DiagnosticReport` round-trips `to_dict ↔ from_dict` and offers a convenience `to_json`.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_diagnostic_report_json_roundtrip():
    from qft_pcn.logic.debugger import (
        DiagnosticReport, RuleViolation, TermEvaluationError,
    )
    v = RuleViolation(
        name="T-App@site_5", rule_class="T-App", site=5,
        sites=[5, 6, 8], energy_contribution=1.0,
        ast_path=[0, 1], lookup_failed=False,
        explanation="T-App violated", context={"foo": "bar"},
    )
    e = TermEvaluationError(
        term_name="bad", rule_class="X", site=0,
        exception_type="ValueError", exception_message="oops",
    )
    r = DiagnosticReport(
        total_energy=1.0, threshold=1e-6,
        n_terms_evaluated=3, n_violations=1,
        rule_violations=[v],
        by_rule_class={"T-App": 1.0, "T-Var": 0.0},
        errors=[e],
    )
    s = r.to_json()
    assert isinstance(s, str)
    rt = DiagnosticReport.from_dict(json.loads(s))
    assert rt == r          # frozen dataclass equality, full structure


def test_diagnostic_report_to_json_default_indent():
    from qft_pcn.logic.debugger import DiagnosticReport
    r = DiagnosticReport(
        total_energy=0.0, threshold=1e-6,
        n_terms_evaluated=0, n_violations=0,
        rule_violations=[], by_rule_class={}, errors=[],
    )
    # indent=2 by default produces a pretty-printed multi-line string:
    s = r.to_json()
    assert "\n" in s
    s_compact = r.to_json(indent=None)
    assert "\n" not in s_compact
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 2 new tests fail with `AttributeError: ... has no attribute 'to_json'` and `'from_dict'`.

- [ ] **Step 3: Implement `to_json` and `from_dict`**

In `src/qft_pcn/logic/debugger.py`, replace the existing `DiagnosticReport` class with this extended version (keep RuleViolation and TermEvaluationError as-is):

```python
import json as _json


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
            "by_rule_class": {k: float(v) for k, v in self.by_rule_class.items()},
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
            by_rule_class={k: float(v) for k, v in d["by_rule_class"].items()},
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
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): JSON round-trip for DiagnosticReport

Adds to_json() / from_dict() to support the contract with sub-project G:
the report is wire-serialized to JSON, parsed back by G, and read for
verbalization. Round-trip preserves frozen-dataclass equality so tests
can assert structural identity.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Explanation registry (`register_explanation`, `get_explanation`, `clear_explanations`)

**Files:**
- Modify: `src/qft_pcn/logic/debugger.py`
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §6.1. A module-level dict; idempotent re-registration; conflict raises.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_register_explanation_basic():
    from qft_pcn.logic.debugger import (
        register_explanation, get_explanation, clear_explanations,
    )
    clear_explanations()
    register_explanation("T-App", "T-App at site {site}")
    tpl, ext = get_explanation("T-App")
    assert tpl == "T-App at site {site}"
    assert ext is None


def test_register_explanation_idempotent_same_template():
    from qft_pcn.logic.debugger import (
        register_explanation, clear_explanations,
    )
    clear_explanations()
    register_explanation("T-App", "tpl")
    register_explanation("T-App", "tpl")  # no error


def test_register_explanation_conflict_raises():
    from qft_pcn.logic.debugger import (
        register_explanation, clear_explanations,
    )
    clear_explanations()
    register_explanation("T-App", "first")
    with pytest.raises(ValueError) as exc:
        register_explanation("T-App", "different")
    assert "T-App" in str(exc.value)


def test_register_explanation_idempotent_same_extractor():
    from qft_pcn.logic.debugger import (
        register_explanation, clear_explanations,
    )
    clear_explanations()
    def ext(term, state, meta):
        return {}
    register_explanation("T-X", "tpl", ext)
    register_explanation("T-X", "tpl", ext)  # same extractor identity, OK


def test_register_explanation_extractor_conflict_raises():
    from qft_pcn.logic.debugger import (
        register_explanation, clear_explanations,
    )
    clear_explanations()
    def ext1(term, state, meta):
        return {}
    def ext2(term, state, meta):
        return {}
    register_explanation("T-X", "tpl", ext1)
    with pytest.raises(ValueError):
        register_explanation("T-X", "tpl", ext2)


def test_get_explanation_missing_returns_none():
    from qft_pcn.logic.debugger import (
        get_explanation, clear_explanations,
    )
    clear_explanations()
    assert get_explanation("UNREGISTERED") is None


def test_clear_explanations_empties_registry():
    from qft_pcn.logic.debugger import (
        register_explanation, get_explanation, clear_explanations,
    )
    register_explanation("Z", "z")
    clear_explanations()
    assert get_explanation("Z") is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 7 new tests fail with `ImportError`.

- [ ] **Step 3: Implement the registry**

Append to `src/qft_pcn/logic/debugger.py`:

```python
from typing import Any, Callable

ExplanationContext = dict[str, Any]

# Callable shape: (term, state, meta) -> ExplanationContext
ContextExtractor = Callable[
    ["NamedHamiltonianTerm", "MPS", Any],  # meta typed as Any to avoid
    ExplanationContext,                     # import cycles
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
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): explanation registry for rule classes

Module-level dict mapping rule_class -> (template, optional extractor).
Sub-projects B and C populate this at import time; the debugger reads
it when building human-readable messages. Idempotent re-registration
of identical tuples is allowed; conflicting re-registration raises.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Mock test fixtures (`_MockTerm`, `_MockTermThatRaises`, `_mock_stlc_terms_for`)

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

We need mock terms in the test module so the diagnose() tests in Tasks 7-9 can run without depending on B/C. Per spec §7.12.

- [ ] **Step 1: Add the mock helpers and a smoke test**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
# ---- Mock fixtures for tests below ----------------------------------------

@dataclass
class _MockTerm:
    """A NamedHamiltonianTerm satisfying the Protocol structurally.

    Returns a constant `value` from expectation(); used to simulate
    Hamiltonian terms in a controlled way without depending on the
    actual B/C implementations.
    """
    name: str
    rule_class: str
    site: int
    sites: tuple[int, ...]
    value: float = 0.0

    def expectation(self, state) -> float:
        return float(self.value)


@dataclass
class _MockTermThatRaises:
    """Mock term whose expectation() raises a configured exception."""
    name: str
    rule_class: str
    site: int
    sites: tuple[int, ...]
    exc: BaseException

    def expectation(self, state) -> float:
        raise self.exc


def _mock_stlc_terms_for(p, state, meta, *, violated: dict[str, float]):
    """Build a list of mock STLC typing terms.

    For each (rule_class, site) we emit one mock returning 0.0, except
    where the term's name appears in `violated`, in which case the mock
    returns the specified energy.

    This mirrors the shape sub-project B's build_typing_terms(meta) will
    eventually produce; nothing real is computed.
    """
    rule_classes = ["T-Var", "T-Abs", "T-App", "T-If", "T-Bin",
                    "T-IntLit", "T-BoolLit"]
    terms = []
    for site in sorted(meta.site_to_ast_path):
        for rc in rule_classes:
            name = f"{rc}@site_{site}"
            terms.append(_MockTerm(
                name=name, rule_class=rc, site=site, sites=(site,),
                value=violated.get(name, 0.0),
            ))
    return terms


def test_mock_term_satisfies_protocol():
    """Smoke test: the mock conforms to the Protocol used by diagnose()."""
    from qft_pcn.logic.debugger import NamedHamiltonianTerm
    m = _MockTerm("T-App@site_0", "T-App", 0, (0,))
    assert isinstance(m, NamedHamiltonianTerm)
    # And it returns a real Python float:
    p = parse(r"\x:Int. x")
    state, _ = encode(p, N=32, chi_max=16)
    e = m.expectation(state)
    assert type(e) is float
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -10`
Expected: 19 passed.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
test(logic/debugger): mock NamedHamiltonianTerm fixtures

Adds _MockTerm, _MockTermThatRaises, and _mock_stlc_terms_for to the
test module so the upcoming diagnose() tests can run before sub-project
B's real H_typing is ready. The mocks satisfy the Protocol structurally
and return controlled per-term energies.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: `diagnose()` core — input validation & total energy

**Files:**
- Modify: `src/qft_pcn/logic/debugger.py`
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §4. This task implements the entry point with input validation, term-protocol check, total energy computation, and the `errors` field. Threshold filtering, ordering, and explanations come in Tasks 8-9.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_diagnose_rejects_negative_threshold():
    from qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    with pytest.raises(ValueError) as exc:
        diagnose(state, meta, [], threshold=-0.1)
    assert "threshold" in str(exc.value)


def test_diagnose_rejects_bad_sort():
    from qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    with pytest.raises(ValueError) as exc:
        diagnose(state, meta, [], sort="banana")
    assert "sort" in str(exc.value)


def test_diagnose_rejects_non_conforming_term():
    from qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)

    class NotATerm:
        pass  # missing every Protocol attribute

    with pytest.raises(TypeError) as exc:
        diagnose(state, meta, [NotATerm()])
    msg = str(exc.value)
    assert "index 0" in msg
    assert "NamedHamiltonianTerm" in msg


def test_diagnose_empty_terms_returns_zero():
    from qft_pcn.logic.debugger import diagnose, DiagnosticReport
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    r = diagnose(state, meta, [], threshold=1e-6)
    assert isinstance(r, DiagnosticReport)
    assert r.total_energy == 0.0
    assert r.threshold == 1e-6
    assert r.n_terms_evaluated == 0
    assert r.n_violations == 0
    assert r.rule_violations == []
    assert r.by_rule_class == {}
    assert r.errors == []


def test_diagnose_total_energy_is_sum_of_all_terms():
    """total_energy is independent of threshold."""
    from qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    terms = [
        _MockTerm("A@0", "A", 0, (0,), value=0.3),
        _MockTerm("B@1", "B", 1, (1,), value=0.7),
        _MockTerm("C@2", "C", 2, (2,), value=1e-9),   # sub-threshold
    ]
    r = diagnose(state, meta, terms, threshold=1.0)
    # threshold = 1.0 means no violations make the list, but total still sums:
    assert abs(r.total_energy - (0.3 + 0.7 + 1e-9)) < 1e-12
    assert r.n_terms_evaluated == 3
    assert r.n_violations == 0


def test_diagnose_captures_term_evaluation_errors():
    """Terms that raise are NOT propagated; they go into report.errors."""
    from qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    good = _MockTerm("T-Var@1", "T-Var", 1, (1,), value=0.5)
    bad = _MockTermThatRaises(
        "T-App@0", "T-App", 0, (0,), exc=ValueError("intentional"),
    )
    r = diagnose(state, meta, [good, bad], threshold=0.0)
    # `bad` is excluded from total_energy and from rule_violations:
    assert abs(r.total_energy - 0.5) < 1e-12
    assert r.n_terms_evaluated == 2     # still counts in n_terms_evaluated
    assert r.n_violations == 1
    assert r.rule_violations[0].rule_class == "T-Var"
    assert len(r.errors) == 1
    e = r.errors[0]
    assert e.term_name == "T-App@0"
    assert e.exception_type == "ValueError"
    assert "intentional" in e.exception_message
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -20`
Expected: 6 new tests fail with `ImportError: cannot import name 'diagnose'`.

- [ ] **Step 3: Implement `diagnose()` (core, no violations yet — Task 8)**

Append to `src/qft_pcn/logic/debugger.py`:

```python
# Constants for sort modes — used for input validation.
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
    # Final check via the runtime_checkable Protocol (catches wrong types
    # if all attributes are present but `expectation` isn't callable, etc.).
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
      * Reads ⟨H_term⟩ from each named term.
      * Filters by |⟨H_term⟩| >= threshold into rule_violations.
      * Aggregates per-rule-class totals across ALL terms.
      * Captures term-evaluation exceptions into report.errors rather
        than propagating them.
      * Pure / read-only on its inputs.
    """
    # ---- Input validation -------------------------------------------------
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if sort not in _VALID_SORT_MODES:
        raise ValueError(
            f"sort must be one of 'descending', 'ascending', 'site', "
            f"got {sort!r}"
        )
    for i, t in enumerate(hamiltonian_terms):
        _assert_is_named_term(t, i)

    # ---- Per-term evaluation ---------------------------------------------
    # We evaluate every term once. Successful evaluations contribute to
    # total_energy and by_rule_class; failures go into errors.
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
        by_rule_class[term.rule_class] = by_rule_class.get(term.rule_class, 0.0) + e

    total_energy = sum(e for (_, e) in successes)

    # ---- Build violations (Task 8 fills this in) -------------------------
    rule_violations: list[RuleViolation] = _build_violations(
        successes=successes, meta=meta, state=state,
        threshold=threshold, sort=sort,
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
) -> list[RuleViolation]:
    """Filter successful per-term evaluations into RuleViolation entries.

    Task 8 will extend this to apply threshold filtering, sort modes,
    AST-path lookup, and explanation generation. For Task 7 it returns
    an empty list so the entry-point plumbing is testable in isolation.
    """
    return []
```

Note the deliberate split: `_build_violations` is a placeholder returning `[]`. Tests in this task only assert plumbing-level behavior (validation, totals, errors). Task 8 implements the body.

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`

Expected: most tests pass, but one will partially fail because `test_diagnose_captures_term_evaluation_errors` expects `n_violations == 1` (the `good` term) and `rule_violations[0].rule_class == "T-Var"`. With the placeholder `_build_violations`, the rule_violations will be empty.

Adjust the test temporarily to defer that assertion to Task 8:

Replace `test_diagnose_captures_term_evaluation_errors` with a Task-7-scope version:

```python
def test_diagnose_captures_term_evaluation_errors():
    """Terms that raise are NOT propagated; they go into report.errors.

    Task 7 scope: verifies errors capture and that total_energy excludes
    the failing term. Task 8 will additionally assert that rule_violations
    correctly lists the surviving terms.
    """
    from qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    good = _MockTerm("T-Var@1", "T-Var", 1, (1,), value=0.5)
    bad = _MockTermThatRaises(
        "T-App@0", "T-App", 0, (0,), exc=ValueError("intentional"),
    )
    r = diagnose(state, meta, [good, bad], threshold=0.0)
    # `bad` is excluded from total_energy:
    assert abs(r.total_energy - 0.5) < 1e-12
    assert r.n_terms_evaluated == 2
    assert len(r.errors) == 1
    e = r.errors[0]
    assert e.term_name == "T-App@0"
    assert e.exception_type == "ValueError"
    assert "intentional" in e.exception_message
```

Run again: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 25 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): diagnose() entry point (input validation, totals, errors)

Implements the validation layer, per-term evaluation loop, total_energy
sum, by_rule_class aggregation, and error capture for terms that raise.
Violation-list construction is a placeholder; threshold filtering and
sorting are implemented in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `_build_violations` — threshold, sort, AST path, explanation

**Files:**
- Modify: `src/qft_pcn/logic/debugger.py`
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §4 (behavioral contract) and §5 (RuleViolation fields). This is the core of D.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_diagnose_threshold_filters_low_energy_terms():
    """|⟨H⟩| < threshold means the term doesn't make rule_violations."""
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 0.5,
        "T-Abs@site_0": 1.5,
        "T-Var@site_1": 1e-9,    # sub-threshold
    })
    r = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    assert r.n_violations == 2
    assert r.rule_violations[0].rule_class == "T-Abs"
    assert r.rule_violations[1].rule_class == "T-App"
    # by_rule_class includes everything, even sub-threshold terms:
    assert abs(r.by_rule_class["T-Var"] - 1e-9) < 1e-15


def test_diagnose_threshold_zero_includes_every_violator():
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-Var@site_1": 1e-12,
    })
    r = diagnose(state, meta, terms, threshold=0.0)
    # threshold = 0.0 means ANY non-zero term is a violation.
    # Mock terms have value=0.0 by default; equality with 0.0 means
    # |x| >= 0.0 is True, so EVERY term is included. We assert at
    # least the explicit violator is in there:
    names = {v.name for v in r.rule_violations}
    assert "T-Var@site_1" in names


def test_diagnose_sort_descending():
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 0.3, "T-Abs@site_1": 0.7, "T-Var@site_2": 0.5,
    })
    r = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    assert [v.energy_contribution for v in r.rule_violations] == [0.7, 0.5, 0.3]


def test_diagnose_sort_ascending():
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 0.3, "T-Abs@site_1": 0.7, "T-Var@site_2": 0.5,
    })
    r = diagnose(state, meta, terms, threshold=1e-6, sort="ascending")
    assert [v.energy_contribution for v in r.rule_violations] == [0.3, 0.5, 0.7]


def test_diagnose_sort_by_site():
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 0.3, "T-Abs@site_1": 0.7, "T-Var@site_2": 0.5,
    })
    r = diagnose(state, meta, terms, threshold=1e-6, sort="site")
    assert [v.site for v in r.rule_violations] == [0, 1, 2]


def test_diagnose_ast_path_is_looked_up():
    """For sites in meta.site_to_ast_path, ast_path is the looked-up tuple."""
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    # Site 0 in this program is the root Lam, ast_path == ().
    term = _MockTerm("T-Abs@site_0", "T-Abs", 0, (0,), value=1.0)
    r = diagnose(state, meta, [term], threshold=1e-6)
    assert r.n_violations == 1
    v = r.rule_violations[0]
    assert v.ast_path == list(meta.site_to_ast_path[0])
    assert v.lookup_failed is False


def test_diagnose_ast_path_lookup_failure_does_not_raise():
    """A term at a site not in meta.site_to_ast_path: ast_path=None,
    lookup_failed=True (NOT an exception)."""
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    pad_site = max(meta.site_to_ast_path) + 5
    bogus = _MockTerm("Global@99", "Global", pad_site, (pad_site,), value=0.7)
    r = diagnose(state, meta, [bogus], threshold=0.0)
    assert r.n_violations >= 1
    matches = [v for v in r.rule_violations if v.name == "Global@99"]
    assert len(matches) == 1
    v = matches[0]
    assert v.ast_path is None
    assert v.lookup_failed is True


def test_diagnose_default_tie_breaking_is_deterministic():
    """When two violations have equal energy, site asc then name lex."""
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = [
        _MockTerm("Z@1", "Z", 1, (1,), value=1.0),
        _MockTerm("A@1", "A", 1, (1,), value=1.0),  # same site, name < Z
        _MockTerm("A@0", "A", 0, (0,), value=1.0),  # smaller site
    ]
    r = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    names = [v.name for v in r.rule_violations]
    assert names == ["A@0", "A@1", "Z@1"]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -25`
Expected: 8 new tests fail (some assert `n_violations >= 1`, but the placeholder returns empty).

- [ ] **Step 3: Implement `_build_violations`**

In `src/qft_pcn/logic/debugger.py`, **replace** the placeholder `_build_violations` with this full implementation:

```python
def _build_violations(
    *,
    successes: list[tuple[Any, float]],
    meta: Any,
    state: MPS,
    threshold: float,
    sort: str,
) -> list[RuleViolation]:
    """Filter and order successful per-term evaluations into violations.

    Filters by |⟨H_term⟩| >= threshold, looks up each site's AST path
    via meta.site_to_ast_path, attaches an explanation via the registry
    (Task 9 wires this fully), and returns the list ordered per `sort`.
    """
    violations: list[RuleViolation] = []
    for term, energy in successes:
        if abs(energy) < threshold:
            continue
        # AST path lookup — None means "not in the map" (PAD site, etc.).
        site_to_path = getattr(meta, "site_to_ast_path", {})
        raw_path = site_to_path.get(int(term.site))
        if raw_path is None:
            ast_path: list[int] | None = None
            lookup_failed = True
        else:
            ast_path = list(raw_path)
            lookup_failed = False

        # Build the explanation. Task 9 wires the registry; for Task 8 we
        # call a helper that returns a sensible fallback if nothing is
        # registered.
        explanation, context = _render_explanation(
            term=term, state=state, meta=meta,
            energy=energy, ast_path=ast_path,
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

    # ---- Ordering ---------------------------------------------------------
    if sort == "descending":
        violations.sort(key=lambda v: (-v.energy_contribution, v.site, v.name))
    elif sort == "ascending":
        violations.sort(key=lambda v: (v.energy_contribution, v.site, v.name))
    elif sort == "site":
        violations.sort(key=lambda v: (v.site, v.name))
    else:
        # Defensive — caller-side validation in diagnose() should prevent this.
        raise AssertionError(f"unreachable sort mode {sort!r}")

    return violations


def _render_explanation(
    *,
    term: Any,
    state: MPS,
    meta: Any,
    energy: float,
    ast_path: list[int] | None,
) -> tuple[str, dict]:
    """Build (explanation_string, context_dict) for a violation.

    Task 8 returns a fallback explanation; Task 9 wires the registry.
    """
    return (
        f"{term.rule_class} violated at site {term.site} "
        f"(energy {energy:.3g})",
        {},
    )
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -25`
Expected: 33 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): violation list — threshold, sort, AST path lookup

Implements _build_violations: filters successful per-term evaluations
by |⟨H_term⟩| >= threshold, attaches the AST path looked up via
meta.site_to_ast_path (or None + lookup_failed=True if the site
isn't mapped), and orders the result per the sort argument. Tie
breaking is deterministic (site ascending, then name lex).

Explanations are fallback strings for now; the explanation registry
is wired in the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Wire the explanation registry into `_render_explanation`

**Files:**
- Modify: `src/qft_pcn/logic/debugger.py`
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §6. Use the registry when a `rule_class` is registered; otherwise the fallback from Task 8. Default-context slots (`{site}`, `{ast_path}`, `{rule_class}`, `{name}`, `{energy}`) are always available.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_unregistered_rule_class_uses_fallback_explanation():
    from qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    t = _MockTerm("X-Custom@site_0", "X-Custom", 0, (0,), value=0.5)
    r = diagnose(state, meta, [t], threshold=0.0)
    v = r.rule_violations[0]
    assert "X-Custom" in v.explanation
    assert "0.5" in v.explanation


def test_registered_template_default_slots():
    """A template using only default slots needs no extractor."""
    from qft_pcn.logic.debugger import (
        diagnose, register_explanation, clear_explanations,
    )
    clear_explanations()
    register_explanation(
        "T-Foo",
        "{rule_class} fired at site {site} (path {ast_path}, e={energy})",
    )
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    t = _MockTerm("T-Foo@site_0", "T-Foo", 0, (0,), value=0.5)
    r = diagnose(state, meta, [t], threshold=0.0)
    v = r.rule_violations[0]
    assert v.explanation.startswith("T-Foo fired at site 0")
    assert "[]" in v.explanation or "()" in v.explanation  # ast_path repr


def test_registered_template_with_extractor():
    from qft_pcn.logic.debugger import (
        diagnose, register_explanation, clear_explanations,
    )
    clear_explanations()
    def ext(term, state, meta):
        return {"foo": "bar", "baz": 42}
    register_explanation(
        "T-X", "rule {foo}/{baz} at site {site}", ext,
    )
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    t = _MockTerm("T-X@site_0", "T-X", 0, (0,), value=0.5)
    r = diagnose(state, meta, [t], threshold=0.0)
    v = r.rule_violations[0]
    assert v.explanation == "rule bar/42 at site 0"
    # context dict carries the substitutions for downstream tools:
    assert v.context == {"foo": "bar", "baz": 42}


def test_extractor_failure_falls_back_gracefully():
    """If a registered extractor raises, the violation gets the fallback
    explanation; the diagnose() call still succeeds. The failure is
    captured as a TermEvaluationError so dev tools can surface it."""
    from qft_pcn.logic.debugger import (
        diagnose, register_explanation, clear_explanations,
    )
    clear_explanations()
    def bad_ext(term, state, meta):
        raise RuntimeError("extractor blew up")
    register_explanation("T-Bad", "rule {missing_slot}", bad_ext)
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    t = _MockTerm("T-Bad@site_0", "T-Bad", 0, (0,), value=0.5)
    r = diagnose(state, meta, [t], threshold=0.0)
    # Violation is still emitted, with fallback explanation:
    assert r.n_violations == 1
    v = r.rule_violations[0]
    assert "T-Bad" in v.explanation
    # And the extractor failure is in errors:
    assert any(e.exception_type == "RuntimeError" for e in r.errors)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -20`
Expected: most of the 4 new tests fail (template substitution not yet wired).

- [ ] **Step 3: Wire the registry**

In `src/qft_pcn/logic/debugger.py`, **replace** `_render_explanation` with this implementation. Also extend the `diagnose()` function to thread an `errors` list reference into `_build_violations` so extractor failures can be appended.

Step 3a: change the signature of `_build_violations` to accept and append-to an `errors` list:

```python
def _build_violations(
    *,
    successes: list[tuple[Any, float]],
    meta: Any,
    state: MPS,
    threshold: float,
    sort: str,
    errors: list[TermEvaluationError],   # NEW: extractor failures appended here
) -> list[RuleViolation]:
    """Filter and order successful per-term evaluations into violations.

    Filters by |⟨H_term⟩| >= threshold, looks up each site's AST path
    via meta.site_to_ast_path, attaches an explanation via the registry,
    and returns the list ordered per `sort`.

    Extractor failures (callable in the registry raises) are captured
    into `errors` and the affected violation falls back to a default
    explanation.
    """
    violations: list[RuleViolation] = []
    for term, energy in successes:
        if abs(energy) < threshold:
            continue
        site_to_path = getattr(meta, "site_to_ast_path", {})
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
```

Step 3b: replace `_render_explanation`:

```python
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
    - If present (template, extractor): calls extractor for the
      substitution dict; if extractor raises, captures the error
      and uses fallback.
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

    # Substitute. Missing slots raise KeyError — we capture and fall back.
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
```

Step 3c: update `diagnose()` to pass the `errors` list:

Find this call inside `diagnose()`:

```python
    rule_violations: list[RuleViolation] = _build_violations(
        successes=successes, meta=meta, state=state,
        threshold=threshold, sort=sort,
    )
```

Replace it with:

```python
    rule_violations: list[RuleViolation] = _build_violations(
        successes=successes, meta=meta, state=state,
        threshold=threshold, sort=sort, errors=errors,
    )
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -25`
Expected: 37 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): wire explanation registry into rule violations

_render_explanation now consults _EXPLANATION_REGISTRY for each
violation. Default slots (site, ast_path, rule_class, name, energy)
are always available; registered extractors can add more. Extractor
failures and template KeyErrors are captured as TermEvaluationErrors
so the diagnose() call never raises on a buggy template — the
violation falls back to a sensible default and the registry-time bug
is surfaced through report.errors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Pretty-printer (`format_report`)

**Files:**
- Modify: `src/qft_pcn/logic/debugger.py`
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §7.11. Human-readable string for CLI / test-failure output.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_format_report_basic():
    from qft_pcn.logic.debugger import (
        diagnose, format_report, clear_explanations,
    )
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 1.0,
    })
    r = diagnose(state, meta, terms, threshold=1e-6)
    text = format_report(r)
    assert isinstance(text, str)
    assert "Total energy" in text
    assert "T-App" in text
    assert "site 0" in text


def test_format_report_empty():
    from qft_pcn.logic.debugger import format_report, DiagnosticReport
    r = DiagnosticReport(
        total_energy=0.0, threshold=1e-6,
        n_terms_evaluated=0, n_violations=0,
        rule_violations=[], by_rule_class={}, errors=[],
    )
    text = format_report(r)
    assert "Total energy: 0" in text
    assert "No violations" in text


def test_format_report_includes_errors_section():
    from qft_pcn.logic.debugger import (
        diagnose, format_report, clear_explanations,
    )
    clear_explanations()
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    bad = _MockTermThatRaises(
        "T-App@0", "T-App", 0, (0,), exc=ValueError("oops"),
    )
    r = diagnose(state, meta, [bad], threshold=0.0)
    text = format_report(r)
    assert "Errors" in text or "errors" in text
    assert "oops" in text
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 3 new tests fail with `ImportError: cannot import name 'format_report'`.

- [ ] **Step 3: Implement `format_report`**

Append to `src/qft_pcn/logic/debugger.py`:

```python
def format_report(report: DiagnosticReport) -> str:
    """Human-readable rendering of a DiagnosticReport.

    The JSON form (report.to_json()) is the wire format for downstream
    consumers; this function is a developer convenience for printing in
    tests, CLIs, and log messages. Sub-project G is responsible for
    natural-language verbalization for end users.
    """
    lines: list[str] = []
    lines.append(f"Diagnostic Report")
    lines.append(f"-----------------")
    lines.append(f"Total energy: {report.total_energy:.6g}")
    lines.append(f"Threshold: {report.threshold:.3g}")
    lines.append(
        f"Terms evaluated: {report.n_terms_evaluated}  "
        f"Violations: {report.n_violations}"
    )

    if report.by_rule_class:
        lines.append("")
        lines.append("By rule class:")
        # Stable ordering: rule_class name ascending.
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
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 40 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): format_report() human-readable pretty-printer

The JSON form is the wire format; this is for developer-facing
inspection in tests, CLIs, and logs. Lays out total energy, by-rule
totals, violations with their explanations and contexts, and any
term-evaluation errors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Seed STLC explanation templates

**Files:**
- Modify: `src/qft_pcn/logic/debugger.py`
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §6.3. A small bundled set so D is testable in isolation. The function is **not** called at import time (sub-project B will own the registry when it ships); callers opt in by importing and calling the function.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_register_stlc_seed_templates_populates_registry():
    from qft_pcn.logic.debugger import (
        register_stlc_seed_templates, get_explanation, clear_explanations,
    )
    clear_explanations()
    register_stlc_seed_templates()
    for rc in ("T-Var", "T-Abs", "T-App", "T-If", "T-Bin",
               "T-IntLit", "T-BoolLit"):
        assert get_explanation(rc) is not None, (
            f"seed template for {rc} not registered"
        )


def test_register_stlc_seed_templates_is_idempotent():
    from qft_pcn.logic.debugger import (
        register_stlc_seed_templates, clear_explanations,
    )
    clear_explanations()
    register_stlc_seed_templates()
    register_stlc_seed_templates()    # second call must not raise


def test_seed_t_app_explanation_renders_default_slots():
    from qft_pcn.logic.debugger import (
        register_stlc_seed_templates, diagnose, clear_explanations,
    )
    clear_explanations()
    register_stlc_seed_templates()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    # Use the T-Abs seed (no extractor), so we exercise pure default-slot
    # substitution without needing the T-App extractor's actual type-tag
    # measurement logic (which we leave to the integration test).
    t = _MockTerm("T-Abs@site_0", "T-Abs", 0, (0,), value=1.0)
    r = diagnose(state, meta, [t], threshold=1e-6)
    v = r.rule_violations[0]
    assert "T-Abs" in v.explanation
    assert "site 0" in v.explanation
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -10`
Expected: 3 new tests fail with `ImportError: cannot import name 'register_stlc_seed_templates'`.

- [ ] **Step 3: Implement seed registration**

Append to `src/qft_pcn/logic/debugger.py`:

```python
# ---- Seed STLC templates ---------------------------------------------------
#
# These templates exist so sub-project D can be tested in isolation, before
# sub-project B ships its real H_typing and registers its own templates.
# When B ships, it calls clear_explanations() and re-registers with richer
# extractors. The seed set uses only default-context slots for most rules;
# T-App ships with a stub extractor that returns the type tags as strings
# read off the encoding metadata's nested_type_index where available.


_STLC_SEED_TEMPLATES: dict[str, str] = {
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
        "T-BoolLit violated at site {site}: literal node has non-Bool type tag."
    ),
}


def _seed_t_app_extractor(term, state, meta) -> ExplanationContext:
    """Stub extractor for the T-App seed template.

    Reports the expected/actual argument types as "?" placeholders. The
    real extractor lives in sub-project B and reads the type register
    off the state. This stub only exists so the template renders without
    a KeyError before B ships.
    """
    return {"expected_arg_type": "?", "actual_arg_type": "?"}


def register_stlc_seed_templates() -> None:
    """Register the bundled seed templates for STLC typing rules.

    Idempotent. Call this once after `clear_explanations()` (or at
    program start) when running D in isolation. When sub-project B
    ships, B owns the registry; this function should not be called
    in that configuration.
    """
    for rule_class, template in _STLC_SEED_TEMPLATES.items():
        extractor = _seed_t_app_extractor if rule_class == "T-App" else None
        register_explanation(rule_class, template, extractor)
```

- [ ] **Step 4: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -15`
Expected: 43 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/debugger.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): seed STLC explanation templates

Bundles a small, opt-in set of templates for the standard STLC typing
rules so D is testable in isolation. Call register_stlc_seed_templates()
to populate the registry. Sub-project B will own the registry when it
ships and re-register with richer extractors; the seed set is purely a
testing scaffold for D.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Acceptance test — structured report for an ill-typed program

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

This is spec §7.1 — the headline acceptance test cited in §10.6 of the architecture doc.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_diagnose_emits_structured_report_for_ill_typed_program():
    """End-to-end with mock H_typing: ((λx:Int. x + 1)(true)) produces
    a report identifying the T-App violation at the application site."""
    from qft_pcn.logic.debugger import (
        diagnose, register_stlc_seed_templates, clear_explanations,
        DiagnosticReport,
    )
    clear_explanations()
    register_stlc_seed_templates()

    p = parse(r"(\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)

    # T-App is violated at the application site (site 0 — App is the root
    # of this pre-order encoding). Other rules report 0 in this mock.
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 1.0,
    })

    report = diagnose(state, meta, terms, threshold=1e-6)

    assert isinstance(report, DiagnosticReport)
    assert abs(report.total_energy - 1.0) < 1e-10
    assert report.n_violations == 1

    v = report.rule_violations[0]
    assert v.rule_class == "T-App"
    assert v.site == 0
    assert v.ast_path == list(meta.site_to_ast_path[0])
    assert v.lookup_failed is False
    assert "T-App" in v.explanation
    # Pretty-printer must not crash on the result:
    from qft_pcn.logic.debugger import format_report
    text = format_report(report)
    assert "T-App" in text
```

- [ ] **Step 2: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py::test_diagnose_emits_structured_report_for_ill_typed_program -v`

Expected: 1 passed. (All the machinery is in place; this is the integration assertion.)

If it fails: re-read the diagnose() and _build_violations implementations — the most likely culprits are a typo in `_mock_stlc_terms_for` (Task 6) or a stale `_render_explanation` not handling the seed extractor.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
test(logic/debugger): headline acceptance test for ill-typed STLC program

((λx:Int. x + 1)(true)) encoded, run through diagnose() with a mock
T-App violation, produces the structured report identified in §10.6
of the architecture doc: rule_class="T-App", site=0, ast_path=[],
explanation references the rule.

The mock stands in for sub-project B's real H_typing; when B ships,
test_end_to_end_ill_typed_program_with_real_typing_hamiltonian (next
task) will exercise the full stack.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: End-to-end test gated on sub-project B (skip-if-not-ready)

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §7.6. The test exists so that when B ships, the proof of the full pipeline is already in place. Until then it skips cleanly.

- [ ] **Step 1: Add the gated test and a JSON-round-trip integration test**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


@pytest.mark.skipif(
    not _has_module("qft_pcn.logic.typing_hamiltonian"),
    reason="sub-project B not yet implemented",
)
def test_end_to_end_ill_typed_program_with_real_typing_hamiltonian():
    """Once B ships, this is the test that proves the whole stack works:
    ill-typed program → real H_typing → diagnose → report with the
    right rule_class and site."""
    from qft_pcn.logic.typing_hamiltonian import build_typing_terms
    from qft_pcn.logic.debugger import diagnose

    p = parse(r"(\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)
    terms = build_typing_terms(meta)
    report = diagnose(state, meta, terms, threshold=1e-6)

    rule_classes = {v.rule_class for v in report.rule_violations}
    assert "T-App" in rule_classes
    app_violations = [v for v in report.rule_violations
                      if v.rule_class == "T-App"]
    assert any(v.site == 0 for v in app_violations)


def test_full_report_json_roundtrip_for_ill_typed_program():
    """Combine §7.1 and §7.2: produce a real-looking report, serialize,
    parse back, and assert equality plus structural correctness."""
    from qft_pcn.logic.debugger import (
        diagnose, register_stlc_seed_templates, clear_explanations,
        DiagnosticReport,
    )
    clear_explanations()
    register_stlc_seed_templates()
    p = parse(r"(\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 1.0,
    })
    r = diagnose(state, meta, terms, threshold=1e-6)
    s = r.to_json()
    rt = DiagnosticReport.from_dict(json.loads(s))
    assert rt == r
    # Well-formed JSON:
    parsed = json.loads(s)
    assert isinstance(parsed["rule_violations"], list)
    assert parsed["rule_violations"][0]["rule_class"] == "T-App"
```

- [ ] **Step 2: Run to verify both behave correctly**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -25`

Expected: the gated test reports `SKIPPED [1]` with reason "sub-project B not yet implemented", and the JSON-roundtrip test passes. Total: 45 passed, 1 skipped.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
test(logic/debugger): gated end-to-end test for sub-project B integration

Adds test_end_to_end_ill_typed_program_with_real_typing_hamiltonian,
skipped while qft_pcn.logic.typing_hamiltonian (B) is absent. Auto-
enables when B ships, proving the full pipeline (encode → real
H_typing → diagnose → structured report) without any further D
changes.

Also adds a JSON-roundtrip integration test combining the §7.1 and
§7.2 scenarios.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Public re-exports

**Files:**
- Modify: `src/qft_pcn/logic/__init__.py` (additive)
- Modify: `src/qft_pcn/__init__.py` (additive)
- Modify: `src/qft_pcn/tests/test_logic_debugger.py`

Per spec §8.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_debugger.py`:

```python
def test_debugger_public_reexports_from_logic_init():
    """logic.__init__ exposes the debugger's public surface."""
    from qft_pcn.logic import (
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
    # The Protocol is the SAME object as the one in debugger.py:
    from qft_pcn.logic.debugger import (
        NamedHamiltonianTerm as NHT_direct,
    )
    assert NamedHamiltonianTerm is NHT_direct


def test_top_level_qft_pcn_reexports_diagnose():
    """qft_pcn top-level re-exports the most commonly-used names."""
    from qft_pcn import diagnose, DiagnosticReport
    from qft_pcn.logic.debugger import (
        diagnose as diagnose_direct,
        DiagnosticReport as DR_direct,
    )
    assert diagnose is diagnose_direct
    assert DiagnosticReport is DR_direct
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -k "reexports" -v`
Expected: 2 tests fail with `ImportError`.

- [ ] **Step 3: Modify `src/qft_pcn/logic/__init__.py`**

Read the current content of `src/qft_pcn/logic/__init__.py`. Then **append** the following block at the end (do not remove A's existing re-exports):

```python
# ---- Sub-project D: constraint debugger ----------------------------------
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
    register_stlc_seed_templates,
)
```

- [ ] **Step 4: Modify `src/qft_pcn/__init__.py`**

Read the current content of `src/qft_pcn/__init__.py`. **Append** at the end:

```python
# Convenience top-level re-exports from sub-project D (debugger):
from qft_pcn.logic.debugger import diagnose, DiagnosticReport  # noqa: E402
```

- [ ] **Step 5: Run to verify the tests pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -20`
Expected: 47 passed, 1 skipped (the B-gated test).

- [ ] **Step 6: Run the full project test suite for regression check**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -v --ignore=src/qft_pcn/tests/test_quantum.py 2>&1 | tail -10`
Expected: all tests pass (sub-project A's tests are still green).

- [ ] **Step 7: Commit**

```bash
git add src/qft_pcn/logic/__init__.py src/qft_pcn/__init__.py src/qft_pcn/tests/test_logic_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/debugger): public re-exports from logic and top-level qft_pcn

Sub-project D's public surface is now available as
  from qft_pcn.logic import diagnose, DiagnosticReport, NamedHamiltonianTerm, ...
  from qft_pcn import diagnose, DiagnosticReport
matching A's pattern. Sub-projects B and C will import
NamedHamiltonianTerm from qft_pcn.logic.debugger to satisfy the
contract.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Final verification & sweep

**Files:** None modified. Verification only.

Per the spec's §10 acceptance criteria. This task exists so the implementer has a single checklist to confirm before declaring D done.

- [ ] **Step 1: Run the full debugger test file**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v 2>&1 | tail -50`

Confirm: 47 passed, 1 skipped (`test_end_to_end_ill_typed_program_with_real_typing_hamiltonian`).

- [ ] **Step 2: Confirm the skip reason is correct**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_debugger.py -v -rs 2>&1 | grep -i skip`
Expected: includes `SKIPPED ... sub-project B not yet implemented`.

- [ ] **Step 3: Run the full project test suite**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -v --ignore=src/qft_pcn/tests/test_quantum.py 2>&1 | tail -10`
Expected: all tests pass.

- [ ] **Step 4: Sanity-check the top-level imports**

Run:
```bash
.venv/bin/python -c "
from qft_pcn import diagnose, DiagnosticReport
from qft_pcn.logic import (
    NamedHamiltonianTerm, RuleViolation, TermEvaluationError,
    format_report, register_explanation, get_explanation,
    clear_explanations, register_stlc_seed_templates,
)
print('OK')
"
```
Expected: `OK`.

- [ ] **Step 5: Confirm the spec's acceptance criteria (§10)**

Walk through §10 of the spec one by one:

1. §7.1 structured-report test — `test_diagnose_emits_structured_report_for_ill_typed_program` — pass.
2. §7.2 JSON round-trip — `test_diagnostic_report_json_roundtrip`, `test_full_report_json_roundtrip_for_ill_typed_program` — pass.
3. §7.3 threshold/ordering — `test_diagnose_threshold_filters_low_energy_terms`, `_threshold_zero_includes_every_violator`, `_sort_descending`, `_sort_ascending`, `_sort_by_site` — pass.
4. §7.4 error capture — `test_diagnose_captures_term_evaluation_errors` — pass.
5. §7.5 protocol enforcement — `test_diagnose_rejects_non_conforming_term` — pass.
6. §7.6 end-to-end with B — `test_end_to_end_ill_typed_program_with_real_typing_hamiltonian` — skipped cleanly.
7. §7.7 empty input — `test_diagnose_empty_terms_returns_zero` — pass.
8. §7.8 partial state — covered implicitly by mock-controlled-value tests (`test_diagnose_threshold_filters_low_energy_terms` with a 0.3 energy is the partial-state scenario; values < ground state max are exactly "partial").
9. §7.9 lookup failure — `test_diagnose_ast_path_lookup_failure_does_not_raise` — pass.
10. §7.10 explanation registry — `test_register_explanation_*`, `test_unregistered_rule_class_uses_fallback_explanation`, `test_registered_template_*` — pass.
11. §7.11 pretty-printer — `test_format_report_*` — pass.
12. Previously passing tests — all of `test_qft.py`, `test_logic_encoder.py`, etc. — still pass (verified Step 3).
13. `from qft_pcn import diagnose, DiagnosticReport` works — verified Step 4.
14. `NamedHamiltonianTerm.__module__ == "qft_pcn.logic.debugger"` — covered by `test_named_hamiltonian_term_protocol_importable`.

- [ ] **Step 6: Confirm clean git status**

Run: `git status --short src/qft_pcn/logic/ src/qft_pcn/tests/ src/qft_pcn/__init__.py 2>&1 | head -20`
Expected: no uncommitted files in those paths (all committed task-by-task).

- [ ] **Step 7: Smoke-test format_report on the headline scenario**

Run:
```bash
.venv/bin/python -c "
from qft_pcn.logic import (
    encode, parse, diagnose, format_report,
    register_stlc_seed_templates, clear_explanations,
)
from dataclasses import dataclass

@dataclass
class MockTerm:
    name: str
    rule_class: str
    site: int
    sites: tuple
    value: float = 0.0
    def expectation(self, state):
        return float(self.value)

clear_explanations()
register_stlc_seed_templates()
p = parse(r'(\x:Int. x + 1)(true)')
state, meta = encode(p, N=32, chi_max=16)
terms = [MockTerm('T-App@site_0', 'T-App', 0, (0,), 1.0)]
r = diagnose(state, meta, terms, threshold=1e-6)
print(format_report(r))
"
```
Expected: a multi-line report ending with the T-App explanation. Eyeball it for readability — no `<bound method>` repr, no numpy scalars in the output.

D is now complete. Sub-projects B and C can build typing/evaluation Hamiltonians against the `NamedHamiltonianTerm` protocol exported from `qft_pcn.logic.debugger`, and sub-project G can consume `DiagnosticReport.to_json()` directly.

---

## Self-Review Summary

This plan covers every section of the spec:

- **§1 principles** — Every task respects them. Principle 6 (no special-case logic per Hamiltonian) is enforced structurally: the only STLC-aware code in D is the opt-in `register_stlc_seed_templates()` helper, which lives in a clearly-named function and is documented as B's eventual responsibility.
- **§2 scope** — In-scope items map to Tasks 2–11 (protocol, dataclasses, registry, diagnose, format_report, seed templates). Out-of-scope items are not implemented (e.g. no NL verbalization — that's G).
- **§3 protocol** — Task 2.
- **§4 reporting API** — Tasks 7–9.
- **§5 report shape** — Tasks 3–4.
- **§6 registry** — Tasks 5, 9, 11.
- **§7 tests** — Tasks 6, 7, 8, 9, 10, 12, 13, with mock fixtures in Task 6.
- **§8 file layout** — Task 14 (re-exports).
- **§9 error model** — Task 7 input validation + Task 9 extractor error capture.
- **§10 acceptance criteria** — Task 15 walks through every one.
- **§12 contract for G** — D delivers JSON-friendly output (Tasks 3, 4, 14). G's responsibilities are not implemented here by design.

No placeholders. Every code block is complete. Function names match across tasks (`diagnose`, `_build_violations`, `_render_explanation`, `register_explanation`, `register_stlc_seed_templates`, `format_report`, etc.). Tests are gated on B's absence via `pytest.skipif` so the suite runs green in isolation today and goes green-with-coverage when B ships.
