# MERA-Native Debugger + Synthesis Implementation Plan — Part 1 of 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the MERA-native constraint debugger and the structural-superposition encoder extension — the load-bearing new piece that lifts the §10.7 milestone past the MPS-era 4/8 wall.

**Architecture:** Part 1 delivers (a) `mera_debugger.py` — D's generic constraint reporter retargeted to MERA states; (b) the structural-hole AST upgrade (`HoleVar.candidates` admitting `tuple[Node,...]`) and the MERA encoder extension that turns distinct candidate sub-tree shapes into ONE rank-`k` superposition state on the tree (genuine entanglement, never a Python enumeration). Part 2 builds the synthesis Hamiltonian, runner, the P1–P8 demo, and the acceptance suite.

**Tech Stack:** Python 3.11, numpy 1.26, pytest. Built on M1 (`mera_encoder.py`, `mera_decoder.py`, `_mera_holes.py`, `_mera_window.py`) and M2 (`MeraTypingHamiltonian`, `MeraEvalHamiltonian`, `compose_mera_hamiltonians`, factored imaginary-time evolution).

**Spec:** `docs/superpowers/specs/2026-05-22-mera-synthesis-design.md` — authoritative. When this plan and the spec disagree, the spec wins.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Driving principles (from spec §1 — non-negotiable; embed in EVERY subagent prompt)

A subagent will be tempted by shortcuts that destroy the milestone. Each principle below names the shortcut and the principled alternative. Embed the full pair in every dispatched prompt.

1. **No time/effort/duration estimates.** Standing critical directive. Shortcut: "this should take ~N hours." Alternative: state dependency order and acceptance gates only.
2. **Binding is genuine entanglement, never a classical lookup.** Shortcut: a hole as a Python `dict`/`list` the runner iterates. Alternative: the hole is a rank-`k` superposition state on the MERA tree; the marker test (Task 5) fails if entanglement entropy is zero.
3. **No dense operator at scale.** Shortcut: a dense `H` over a window "because it's small." Alternative: factored per-leaf 16×16 operators via `mera_window_expectation_factored`; dense only for `k≤2` cross-checks.
4. **OOM means optimize, not shrink.** Shortcut: trim a candidate set or drop a problem to dodge a memory wall. Alternative: factor harder.
5. **`optimize='greedy'` on every einsum.** No exceptions.
6. **Holes are quantum superpositions, not classical enumeration.** THE load-bearing principle. Shortcut: "for each candidate sub-tree, encode a product state, evolve it, keep the lowest." Alternative: the distinct candidate sub-tree shapes become ONE rank-`k` MERA superposition (Task 4); one encoding, one evolution.
7. **Synthesis ranks by ⟨H⟩; success is verified by decoding.** Shortcut: declaring a problem solved on `⟨H⟩ < ε` alone. Alternative: decode the top-1 AST and alpha-check it against the expected answer.
8. **The debugger is a generic reporter, not a Hamiltonian re-implementation.** Shortcut: inline STLC rule knowledge / `isinstance(H, ...)` branches in the debugger. Alternative: consume M2's `NamedMeraTerm` Protocol generically.
9. **Reuse M1/M2; do not reinvent.** Shortcut: reimplementing a rule/sampler inline because "the API doesn't quite expose what I need." Alternative: fix the gap in M1/M2 and amend that spec.
10. **Honest reporting; negative results are publishable.** Shortcut: silently dropping ill-typed completions so the printout looks better. Alternative: print the full energy distribution and failure mode.

---

## Pre-existing worktree state

Unrelated modified files exist (`lib/`, root `*.md`, `QFT_PCN_ARCHITECTURE.md`, M1's `mera_encoder.py`, `_mera_holes.py`, `test_mera_holes.py`). **Leave them alone.** Stage only the files each task names.

`src/qft_pcn/logic/ast.py` already has `HoleVar` and `TypeHole` (added minimally for M1). M3 *upgrades* `HoleVar.candidates` to admit `tuple[Node,...]`; it does not redefine the node.

`_mera_holes.py` already implements M1's value-superposition (var-hole) path via `from_term_superposition`. M3 *adds* a structural-superposition branch alongside it; the existing var-hole path is untouched.

---

## Task 1: `HoleVar.candidates` admits AST sub-trees

**Files:**
- Modify: `src/qft_pcn/logic/ast.py`
- Test: `src/qft_pcn/tests/test_mera_structural_ast.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_structural_ast.py`:

```python
"""Tests for the structural-HoleVar AST upgrade (spec §5.1)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import HoleVar, Var, App, Bin, IntLit, Node


def test_holevar_accepts_str_candidates():
    h = HoleVar(candidates=("x", "y"))
    assert h.candidates == ("x", "y")
    assert h.candidate_kind() == "var"


def test_holevar_accepts_node_candidates():
    h = HoleVar(candidates=(Var(name="x"), App(fn=Var(name="f"), arg=Var(name="x"))))
    assert h.candidate_kind() == "structural"
    assert all(isinstance(c, Node) for c in h.candidates)


def test_holevar_empty_candidates_is_var():
    assert HoleVar(candidates=()).candidate_kind() == "var"


def test_holevar_rejects_mixed_candidates():
    with pytest.raises(ValueError, match="must not mix"):
        HoleVar(candidates=("x", Var(name="y")))


def test_holevar_is_a_node():
    assert isinstance(HoleVar(candidates=("x",)), Node)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_structural_ast.py -v`
Expected: `AttributeError` / failure — `candidate_kind` not defined, mixed-candidate check missing.

- [ ] **Step 3: Upgrade `HoleVar` in `ast.py`**

Read `src/qft_pcn/logic/ast.py` first to find the existing `HoleVar` definition and the `Node` base. Replace the `HoleVar` body with (keep the `@dataclass` decorator the file already uses for nodes):

```python
@dataclass
class HoleVar(Node):
    """A synthesis hole.

    `candidates` is either:
      - tuple[str, ...]  : binder-name candidates (var-hole; the M1 case);
      - tuple[Node, ...] : AST sub-tree candidates (structural hole, M3 §5).
    The two forms must NOT be mixed. An empty tuple is a var-hole meaning
    "any in-scope binder", resolved at encode time.
    """
    candidates: tuple = ()
    target_type: "Ty | None" = None
    name: str = ""

    def __post_init__(self) -> None:
        has_str = any(isinstance(c, str) for c in self.candidates)
        has_node = any(isinstance(c, Node) for c in self.candidates)
        if has_str and has_node:
            raise ValueError(
                "HoleVar.candidates must not mix str and Node candidates")

    def candidate_kind(self) -> str:
        """'structural' if candidates are Node sub-trees, else 'var'."""
        if any(isinstance(c, Node) for c in self.candidates):
            return "structural"
        return "var"
```

If `HoleVar` is referenced by M1 code with positional args, keep the field order so existing call sites still work. Run `grep -rn "HoleVar(" src/qft_pcn/` to confirm before committing.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_structural_ast.py -v`
Expected: 5 passed.

- [ ] **Step 5: Regression — M1 hole tests still pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_holes.py src/qft_pcn/tests/test_mera_ast.py -v 2>&1 | tail -6`
Expected: all still pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/ast.py src/qft_pcn/tests/test_mera_structural_ast.py
git commit -m "$(cat <<'EOF'
feat(logic/ast): HoleVar.candidates admits AST sub-trees

HoleVar.candidates may now be tuple[Node,...] (structural hole) as well
as tuple[str,...] (var hole). candidate_kind() discriminates; mixed
candidate lists are rejected. The structural form is the AST surface for
M3's structural superposition over multi-node completions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: The MERA constraint debugger

**Files:**
- Create: `src/qft_pcn/logic/mera_debugger.py`
- Test: `src/qft_pcn/tests/test_mera_debugger.py`

This is sub-project D's design (`docs/superpowers/specs/2026-05-21-constraint-debugger-design.md`) retargeted to MERA states. Read D's spec §3–§6 — the protocol, the report dataclasses, the registry, the `diagnose` behavioural contract are reused; only `MPS→MERA` and `site/sites→node/leaves` change.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_debugger.py`:

```python
"""Tests for the MERA constraint debugger (spec §3, §9.1)."""
from __future__ import annotations
import json
from dataclasses import dataclass
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_debugger import (
    NamedMeraTerm, DiagnosticReport, RuleViolation, TermEvaluationError,
    diagnose, format_report, register_explanation, clear_explanations,
)


@dataclass
class _MockTerm:
    name: str
    rule_class: str
    node: int
    leaves: tuple
    value: float = 0.0
    def expectation(self, state) -> float:
        return float(self.value)


@dataclass
class _MockRaises:
    name: str
    rule_class: str
    node: int
    leaves: tuple
    def expectation(self, state) -> float:
        raise ValueError("intentional")


def _mock_terms(violated):
    """One term per (rule_class, node); only `violated` names are nonzero."""
    base = [("T-App", 0), ("T-Var", 1), ("T-Abs", 0)]
    out = []
    for rc, nd in base:
        nm = f"{rc}@node_{nd}"
        out.append(_MockTerm(nm, rc, nd, (5 * nd,), violated.get(nm, 0.0)))
    return out


def test_diagnose_emits_structured_report():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    report = diagnose(state, meta, _mock_terms({"T-App@node_0": 1.0}))
    assert isinstance(report, DiagnosticReport)
    assert abs(report.total_energy - 1.0) < 1e-10
    assert report.n_violations == 1
    v = report.rule_violations[0]
    assert v.rule_class == "T-App"
    assert v.node == 0
    assert v.lookup_failed is False
    assert "T-App" in v.explanation


def test_report_json_roundtrip():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    report = diagnose(state, meta, _mock_terms({"T-App@node_0": 1.0}))
    rt = DiagnosticReport.from_dict(json.loads(report.to_json()))
    assert rt == report


def test_threshold_filters_and_sorts():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    terms = _mock_terms({"T-App@node_0": 0.5, "T-Abs@node_0": 1.5,
                         "T-Var@node_1": 1e-9})
    desc = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    assert desc.n_violations == 2
    assert desc.rule_violations[0].rule_class == "T-Abs"
    asc = diagnose(state, meta, terms, sort="ascending")
    assert asc.rule_violations[0].rule_class == "T-App"


def test_term_that_raises_is_captured():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    good = _MockTerm("T-Var@node_1", "T-Var", 1, (5,), value=0.5)
    bad = _MockRaises("T-App@node_0", "T-App", 0, (0,))
    report = diagnose(state, meta, [good, bad], threshold=0.0)
    assert report.n_violations == 1
    assert len(report.errors) == 1
    assert report.errors[0].exception_type == "ValueError"


def test_non_conforming_term_raises_typeerror():
    class NotATerm: pass
    state, meta = encode_mera(parse(r"\x:Int. x"))
    with pytest.raises(TypeError) as ei:
        diagnose(state, meta, [NotATerm()])
    assert "index 0" in str(ei.value)
    assert "NamedMeraTerm" in str(ei.value)


def test_empty_terms_zero_report():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    report = diagnose(state, meta, [], threshold=1e-6)
    assert report.total_energy == 0.0
    assert report.n_violations == 0
    assert report.by_rule_class == {}


def test_bad_threshold_and_sort_raise():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    with pytest.raises(ValueError):
        diagnose(state, meta, [], threshold=-1.0)
    with pytest.raises(ValueError):
        diagnose(state, meta, [], sort="sideways")


def test_explanation_registry():
    clear_explanations()
    register_explanation("T-X", "rule at node {node}")
    register_explanation("T-X", "rule at node {node}")     # idempotent
    with pytest.raises(ValueError):
        register_explanation("T-X", "different")
    state, meta = encode_mera(parse(r"\x:Int. x"))
    term = _MockTerm("T-X@node_0", "T-X", 0, (0,), value=0.5)
    report = diagnose(state, meta, [term], threshold=0.0)
    assert report.rule_violations[0].explanation == "rule at node 0"


def test_unregistered_rule_uses_fallback():
    clear_explanations()
    state, meta = encode_mera(parse(r"\x:Int. x"))
    term = _MockTerm("X-Z@node_0", "X-Z", 0, (0,), value=0.5)
    report = diagnose(state, meta, [term], threshold=0.0)
    assert "X-Z" in report.rule_violations[0].explanation


def test_format_report_human_readable():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    report = diagnose(state, meta, _mock_terms({"T-App@node_0": 1.0}))
    text = format_report(report)
    assert "T-App" in text
    assert "node 0" in text
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_debugger.py -v`
Expected: ImportError on `mera_debugger`.

- [ ] **Step 3: Implement `mera_debugger.py`**

Port D's `debugger.py` design. Read `src/qft_pcn/logic/debugger.py` (the shipped MPS debugger) for the reference structure — `mera_debugger.py` is its MERA twin with `MPS→MERA`, `site→node`, `sites→leaves`. Create `src/qft_pcn/logic/mera_debugger.py`:

```python
"""MERA constraint debugger (spec §3).

D's generic constraint reporter retargeted to MERA states. Consumes named
per-term operators (M2's .terms) satisfying NamedMeraTerm; computes
per-term residual energy; emits a JSON-serializable DiagnosticReport.
Generic over any MERA Hamiltonian — no per-rule branching, no
isinstance(H, ...) dispatch (principle 8).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from src.qft_pcn.qft.mera import MERA
from .mera_encoder import MeraEncodingMeta


@runtime_checkable
class NamedMeraTerm(Protocol):
    """One named, individually measurable MERA Hamiltonian term."""
    name: str
    rule_class: str
    node: int
    leaves: tuple

    def expectation(self, state: MERA) -> float: ...


# --- explanation registry -------------------------------------------------

ContextExtractor = Callable[["NamedMeraTerm", MERA, MeraEncodingMeta], dict]
_REGISTRY: dict[str, tuple[str, ContextExtractor | None]] = {}


def register_explanation(rule_class: str, template: str,
                          context_extractor: ContextExtractor | None = None
                          ) -> None:
    existing = _REGISTRY.get(rule_class)
    new = (template, context_extractor)
    if existing is not None and existing != new:
        raise ValueError(
            f"rule_class {rule_class!r} already registered with a "
            f"different template/extractor")
    _REGISTRY[rule_class] = new


def get_explanation(rule_class: str):
    return _REGISTRY.get(rule_class)


def clear_explanations() -> None:
    _REGISTRY.clear()


# --- report dataclasses ---------------------------------------------------

@dataclass(frozen=True)
class TermEvaluationError:
    term_name: str
    rule_class: str
    node: int
    exception_type: str
    exception_message: str

    def to_dict(self) -> dict:
        return {"term_name": self.term_name, "rule_class": self.rule_class,
                "node": self.node, "exception_type": self.exception_type,
                "exception_message": self.exception_message}

    @classmethod
    def from_dict(cls, d: dict) -> "TermEvaluationError":
        return cls(**d)


@dataclass(frozen=True)
class RuleViolation:
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
        return {"name": self.name, "rule_class": self.rule_class,
                "node": self.node, "leaves": list(self.leaves),
                "energy_contribution": self.energy_contribution,
                "ast_path": (None if self.ast_path is None
                             else list(self.ast_path)),
                "lookup_failed": self.lookup_failed,
                "explanation": self.explanation, "context": self.context}

    @classmethod
    def from_dict(cls, d: dict) -> "RuleViolation":
        return cls(name=d["name"], rule_class=d["rule_class"],
                   node=d["node"], leaves=list(d["leaves"]),
                   energy_contribution=d["energy_contribution"],
                   ast_path=(None if d["ast_path"] is None
                             else list(d["ast_path"])),
                   lookup_failed=d["lookup_failed"],
                   explanation=d["explanation"], context=d["context"])


@dataclass(frozen=True)
class DiagnosticReport:
    total_energy: float
    threshold: float
    n_terms_evaluated: int
    n_violations: int
    rule_violations: list
    by_rule_class: dict
    errors: list

    def to_dict(self) -> dict:
        return {"total_energy": self.total_energy,
                "threshold": self.threshold,
                "n_terms_evaluated": self.n_terms_evaluated,
                "n_violations": self.n_violations,
                "rule_violations": [v.to_dict() for v in self.rule_violations],
                "by_rule_class": dict(self.by_rule_class),
                "errors": [e.to_dict() for e in self.errors]}

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "DiagnosticReport":
        return cls(total_energy=d["total_energy"], threshold=d["threshold"],
                   n_terms_evaluated=d["n_terms_evaluated"],
                   n_violations=d["n_violations"],
                   rule_violations=[RuleViolation.from_dict(v)
                                    for v in d["rule_violations"]],
                   by_rule_class=dict(d["by_rule_class"]),
                   errors=[TermEvaluationError.from_dict(e)
                           for e in d["errors"]])


# --- diagnose -------------------------------------------------------------

_VALID_SORTS = ("descending", "ascending", "node")
_PROTO_ATTRS = ("name", "rule_class", "node", "leaves", "expectation")


def _explain(term, violation_energy, ast_path, state, meta):
    reg = get_explanation(term.rule_class)
    default_ctx = {"node": term.node, "ast_path": ast_path,
                   "rule_class": term.rule_class, "name": term.name,
                   "energy": f"{violation_energy:.3g}"}
    if reg is None:
        return (f"{term.rule_class} violated at node {term.node} "
                f"(energy {violation_energy:.3g})", {})
    template, extractor = reg
    ctx = dict(default_ctx)
    extracted = {}
    if extractor is not None:
        extracted = extractor(term, state, meta)
        ctx.update(extracted)
    try:
        text = template.format(**ctx)
    except (KeyError, IndexError):
        text = f"{term.rule_class} violated at node {term.node}"
    return text, extracted


def diagnose(state: MERA, meta: MeraEncodingMeta,
             hamiltonian_terms: list, threshold: float = 1e-6,
             sort: str = "descending") -> DiagnosticReport:
    """Structured per-term residual-energy report on a MERA state.
    See spec §3.2 for the full behavioural contract."""
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0, got {threshold}")
    if sort not in _VALID_SORTS:
        raise ValueError(
            f"sort must be one of {_VALID_SORTS}, got {sort!r}")
    for i, term in enumerate(hamiltonian_terms):
        if not isinstance(term, NamedMeraTerm):
            missing = next((a for a in _PROTO_ATTRS
                            if not hasattr(term, a)), "?")
            raise TypeError(
                f"term at index {i} does not satisfy NamedMeraTerm "
                f"(missing attribute {missing!r})")

    total = 0.0
    by_rule: dict[str, float] = {}
    errors: list[TermEvaluationError] = []
    measured: list[tuple] = []   # (term, energy)
    for term in hamiltonian_terms:
        try:
            e = float(term.expectation(state))
        except Exception as exc:  # noqa: BLE001 - captured by contract
            errors.append(TermEvaluationError(
                term_name=term.name, rule_class=term.rule_class,
                node=term.node, exception_type=type(exc).__name__,
                exception_message=str(exc)))
            continue
        total += e
        by_rule[term.rule_class] = by_rule.get(term.rule_class, 0.0) + e
        measured.append((term, e))

    violations: list[RuleViolation] = []
    for term, e in measured:
        if abs(e) < threshold:
            continue
        raw_path = meta.site_to_ast_path.get(term.node)
        lookup_failed = raw_path is None
        ast_path = None if lookup_failed else list(raw_path)
        text, ctx = _explain(term, e, ast_path, state, meta)
        violations.append(RuleViolation(
            name=term.name, rule_class=term.rule_class, node=term.node,
            leaves=list(term.leaves), energy_contribution=e,
            ast_path=ast_path, lookup_failed=lookup_failed,
            explanation=text, context=ctx))

    if sort == "descending":
        violations.sort(key=lambda v: (-v.energy_contribution, v.node, v.name))
    elif sort == "ascending":
        violations.sort(key=lambda v: (v.energy_contribution, v.node, v.name))
    else:  # "node"
        violations.sort(key=lambda v: (v.node, v.name))

    return DiagnosticReport(
        total_energy=total, threshold=threshold,
        n_terms_evaluated=len(hamiltonian_terms),
        n_violations=len(violations), rule_violations=violations,
        by_rule_class=by_rule, errors=errors)


def format_report(report: DiagnosticReport) -> str:
    """Human-readable pretty-print for CLI / test failures."""
    lines = [f"Total energy: {report.total_energy:.6g}",
             f"Threshold: {report.threshold:g}   "
             f"Violations: {report.n_violations}"
             f"/{report.n_terms_evaluated}"]
    for v in report.rule_violations:
        lines.append(f"  [{v.rule_class}] node {v.node} "
                      f"energy={v.energy_contribution:.4g}: {v.explanation}")
    for e in report.errors:
        lines.append(f"  !! {e.term_name}: {e.exception_type}: "
                      f"{e.exception_message}")
    return "\n".join(lines)
```

If M1's `MeraEncodingMeta.site_to_ast_path` is keyed differently (read `mera_encoder.py` to confirm — M1 §6.1 names it `site_to_ast_path: dict[int, tuple[int,...]]`), adapt the lookup key. The test `test_diagnose_emits_structured_report` is the oracle.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_debugger.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_debugger.py src/qft_pcn/tests/test_mera_debugger.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_debugger): MERA-native constraint debugger

D's generic constraint reporter retargeted to MERA states: the
NamedMeraTerm protocol, diagnose(), DiagnosticReport / RuleViolation /
TermEvaluationError dataclasses, explanation registry, format_report.
Generic over any MERA Hamiltonian exposing named terms — no per-rule
branching, no isinstance dispatch (spec §1.8).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Structural-hole layout — `_expand_structural_holes` + `HoleRegion`

**Files:**
- Create: `src/qft_pcn/logic/mera_synthesis/__init__.py`
- Create: `src/qft_pcn/logic/mera_synthesis/encode_ext.py`
- Test: `src/qft_pcn/tests/test_mera_structural_layout.py`

This task is pure computation: given a sketch with structural holes, compute each hole's pre-order position, its `n_max`, and the per-branch leaf-assignment specs. No tensors.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_structural_layout.py`:

```python
"""Tests for structural-hole layout expansion (spec §5.2, §5.6)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import Lam, TInt, TArrow, Var, App, HoleVar
from src.qft_pcn.logic.mera_synthesis.encode_ext import (
    _expand_structural_holes, HoleRegion,
)


def _p3_sketch():
    """\\f:Int->Int. \\x:Int. ?HOLE  with structural candidates."""
    hole = HoleVar(candidates=(
        Var(name="x"),
        App(fn=Var(name="f"), arg=Var(name="x")),
        App(fn=Var(name="f"), arg=App(fn=Var(name="f"), arg=Var(name="x"))),
    ))
    return Lam(param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
               body=Lam(param="x", param_ty=TInt(), body=hole))


def test_expand_reports_one_region():
    _skel, regions = _expand_structural_holes(_p3_sketch())
    assert len(regions) == 1
    assert isinstance(regions[0], HoleRegion)


def test_region_n_max_is_largest_candidate_node_count():
    _skel, regions = _expand_structural_holes(_p3_sketch())
    # candidates have 1, 2, 3 nodes -> n_max = 3
    assert regions[0].n_max == 3


def test_region_has_k_branches():
    _skel, regions = _expand_structural_holes(_p3_sketch())
    assert len(regions[0].candidate_branches) == 3


def test_no_structural_hole_yields_no_regions():
    plain = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    _skel, regions = _expand_structural_holes(plain)
    assert regions == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_structural_layout.py -v`
Expected: ImportError on `mera_synthesis.encode_ext`.

- [ ] **Step 3: Implement `encode_ext.py` (layout part)**

First create `src/qft_pcn/logic/mera_synthesis/__init__.py`:

```python
"""MERA-native STLC synthesis (migration sub-project M3, spec §6-§7)."""
```

(Public re-exports are filled in Part 2, Task 9.)

Then create `src/qft_pcn/logic/mera_synthesis/encode_ext.py`. Read `src/qft_pcn/logic/_serialize.py` for `serialize_preorder` and how it counts nodes; the candidate node count is the length of `serialize_preorder(candidate)` minus PAD entries.

```python
"""Structural-hole encoder extensions (spec §5.6).

_expand_structural_holes computes, per structural HoleVar, the hole
region's pre-order position and n_max (max candidate node count), plus
the per-branch leaf-assignment specs. Pure computation; no tensors.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..ast import Node, HoleVar


@dataclass
class HoleRegion:
    """One structural hole's leaf region (spec §5.2).

    node_start:         pre-order node index where the hole region begins.
    n_max:              max candidate sub-tree node count; region spans
                        5*n_max leaves.
    candidate_branches: list of candidate sub-tree ASTs, one per branch
                        direction of the rank-k superposition. The encoder
                        (Task 4) turns each into a leaf assignment.
    """
    node_start: int
    n_max: int
    candidate_branches: list


def _count_nodes(ast: Node) -> int:
    """Number of AST nodes in `ast` (pre-order, no PAD)."""
    n = 1
    for attr in ("body", "fn", "arg", "lhs", "rhs", "cond",
                 "then_b", "else_b", "head", "tail"):
        child = getattr(ast, attr, None)
        if isinstance(child, Node):
            n += _count_nodes(child)
    return n


def _structural_holes_in_preorder(ast: Node):
    """Yield (preorder_index, HoleVar) for every structural hole."""
    counter = [0]

    def walk(n: Node):
        idx = counter[0]
        counter[0] += 1
        if isinstance(n, HoleVar) and n.candidate_kind() == "structural":
            yield idx, n
        for attr in ("body", "fn", "arg", "lhs", "rhs", "cond",
                     "then_b", "else_b", "head", "tail"):
            child = getattr(n, attr, None)
            if isinstance(child, Node):
                yield from walk(child)
    yield from walk(ast)


def _expand_structural_holes(ast: Node):
    """Return (skeleton, hole_regions).

    `skeleton` is `ast` itself (the encoder consumes the regions to size
    the layout; no AST rewrite is needed at this stage). `hole_regions`
    is a list of HoleRegion, one per structural HoleVar.
    """
    regions: list[HoleRegion] = []
    for idx, hole in _structural_holes_in_preorder(ast):
        branches = list(hole.candidates)
        n_max = max((_count_nodes(c) for c in branches), default=1)
        regions.append(HoleRegion(node_start=idx, n_max=n_max,
                                   candidate_branches=branches))
    return ast, regions
```

The `node_start` here is the *pre-order index counting the hole as one node*; the encoder (Task 4) translates it to a leaf offset accounting for the `n_max` expansion. If `_serialize.py`'s pre-order walk visits children in a different attribute order, mirror that order in `walk` — the M1 encoder's serialization is the source of truth; read `_serialize.py` and match it exactly.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_structural_layout.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_synthesis/__init__.py \
        src/qft_pcn/logic/mera_synthesis/encode_ext.py \
        src/qft_pcn/tests/test_mera_structural_layout.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_synthesis/encode_ext): structural-hole layout expansion

_expand_structural_holes computes, per structural HoleVar, the hole
region's pre-order position and n_max (max candidate node count) and the
per-branch candidate list. Pure computation feeding the encoder's
rank-k superposition construction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Structural superposition — the rank-`k` MERA state

**Files:**
- Modify: `src/qft_pcn/logic/_mera_holes.py` (add the structural branch builder)
- Modify: `src/qft_pcn/logic/mera_encoder.py` (wire the structural-hole path)
- Test: `src/qft_pcn/tests/test_mera_structural_holes.py`

This is the load-bearing task. Read spec §5.3 in full before touching code. The structural superposition generalizes M1's value-superposition (`_mera_holes.py`'s existing `from_term_superposition` branch): M1's holes superpose over leaf *values* at a fixed shape; M3's structural holes superpose over *shapes* — branches differ in `kind` leaves (`KIND_APP` vs `KIND_VAR` vs `KIND_PAD`).

**Principle 6 is the rule for this task:** the `k` candidate sub-trees become ONE rank-`k` MERA state. There is NO Python loop over candidates anywhere in the encoder or the runner. If `MERA.from_term_superposition` cannot accept shape-varying branches, that is an M1 gap — fix it in M1 and amend M1's spec; do not enumerate.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_structural_holes.py`:

```python
"""Tests for structural-superposition MERA encoding (spec §5.3, §9.2-9.4)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, TArrow, Var, App, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_encoding import KIND_VAR, KIND_APP


def _structural_sketch():
    """\\f:Int->Int. \\x:Int. ?HOLE  candidates {Var x, App(f,x)}."""
    hole = HoleVar(candidates=(
        Var(name="x"),
        App(fn=Var(name="f"), arg=Var(name="x")),
    ))
    return Lam(param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
               body=Lam(param="x", param_ty=TInt(), body=hole))


def test_structural_hole_encodes_to_unit_norm():
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10)


def test_hole_region_root_kind_leaf_is_shape_superposed():
    """The hole-region root kind leaf has weight on >=2 distinct kinds
    (KIND_VAR for the Var branch, KIND_APP for the App branch)."""
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    assert len(meta.hole_regions) == 1
    root_leaf = 5 * meta.hole_regions[0].node_start    # kind leaf offset 0
    proj_v = np.zeros((16, 16), dtype=complex); proj_v[KIND_VAR, KIND_VAR] = 1
    proj_a = np.zeros((16, 16), dtype=complex); proj_a[KIND_APP, KIND_APP] = 1
    w_v = float(np.real(state.local_expectation(root_leaf, proj_v)))
    w_a = float(np.real(state.local_expectation(root_leaf, proj_a)))
    assert w_v > 1e-3 and w_a > 1e-3, (w_v, w_a)


def test_structural_hole_is_not_a_product_state():
    """Genuine tree entanglement somewhere — principle 6, the marker."""
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    max_S = max(state.entanglement_entropy(cut)
                for cut in range(1, state.N))
    assert max_S > 1e-6, "structural hole encoded as a product state"


def test_structural_marker_exceeds_value_only_superposition():
    """Shape superposition (kind leaves differ) carries strictly more
    entropy across the hole-region cut than a value-only superposition
    of the same candidate count (spec §9.4)."""
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    region = meta.hole_regions[0]
    cut = 5 * region.node_start                # leaf where the region starts
    S_struct = state.entanglement_entropy(cut)
    # ln(2) is the value-only (M1-style) upper bound for k=2 candidates.
    assert S_struct > 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_structural_holes.py -v`
Expected: failure — `encode_mera` raises `NotImplementedError` or `meta.hole_regions` is absent.

- [ ] **Step 3: Implement the structural branch in `_mera_holes.py`**

Read `src/qft_pcn/logic/_mera_holes.py` first — it already has the value-superposition branch and calls `MERA.from_term_superposition`. Read `src/qft_pcn/logic/_mera_leaves.py` (`node_leaf_vectors`) and `_mera_layout.py` (`compute_layout`, `leaf_of`). Add a function:

```python
def structural_hole_branches(region, sketch_scope, layout):
    """Build the k per-branch leaf-assignment specs for one structural
    hole region (spec §5.3).

    Each branch j is the concrete leaf assignment of the hole region
    committed to candidate j:
      1. candidate j is serialized pre-order to n_j nodes; node m writes
         its five 16-dim one-hot leaf vectors into region node slot m
         (reuse node_leaf_vectors);
      2. region node slots [n_j, n_max) are filled with PAD-leaf vectors
         (shape-pad slots) -- this is what makes branches differ on the
         `kind` leaves: KIND_PAD vs a real kind;
      3. each Var inside candidate j has its bid leaf set to the
         depth-relative bid index of the binder it references, resolved
         against `sketch_scope` (the lexical binders in scope at the hole
         position);
      4. branch j writes a distinct witness index j+1 on the region
         root node's `value` leaf (an otherwise-unused leaf) so the k
         branch directions are mutually orthogonal -- the analytic analog
         of M1's CNOT-like witness mark.

    Returns a list of k leaf-assignment dicts {absolute_leaf: vector},
    each covering exactly the 5*n_max region leaves. These are handed,
    alongside the concrete nodes' single branch, to
    MERA.from_term_superposition.
    """
```

The exact wiring — turning candidate sub-trees into leaf vectors via the M1 front-half (`serialize_preorder` + `compute_site_types` + `node_leaf_vectors`) and resolving each `Var`'s `bid` against the sketch scope — must reuse M1 helpers, not re-derive index math. Read `_tensors.py` / `_resolve.py` for the bid-resolution helper. The structural superposition is then `(1/√k) Σ_j |branch_j⟩` over the region leaves, and the rest of the tree's leaves are the concrete sketch nodes (a single shared branch direction). Hand the full branch list to `MERA.from_term_superposition` exactly as M1's value-superposition path does.

If `from_term_superposition` only accepts branches that agree on leaf *count* — they do here: every branch covers all `5*n_max` region leaves, the shape-pad slots are PAD-leaf vectors, so all branches are the same length and merely differ in *which* leaves are PAD. If the primitive still rejects shape-varying content, that is an M1 gap: extend `from_term_superposition` in M1's `_mera_holes.py` / `mera.py`, add a test there, amend M1's spec §5.3. Do NOT add a Python enumeration here (principle 6).

- [ ] **Step 4: Wire the structural path into `mera_encoder.py`**

Read `mera_encoder.py`'s current hole dispatch. Add a structural-hole branch:

- Classify holes: `_hole_kind(node)` → `"var"` (HoleVar with str/empty candidates), `"structural"` (HoleVar with Node candidates), `"type"` (TypeHole).
- If any structural hole is present: call `_expand_structural_holes` (Task 3) to get `hole_regions`; size the layout with the *expanded* node count (each structural hole contributes `n_max` node slots, not 1); build leaf vectors — concrete nodes via `node_leaf_vectors`, structural-hole regions via `structural_hole_branches`; hand the branch list to `MERA.from_term_superposition`; `normalize()`.
- Record `meta.hole_regions = hole_regions` and (in Part 2) `meta.witness_node_ranges`.
- Var-hole-only and type-hole-only and concrete programs keep their M1 paths unchanged.
- Raise `EncodingTooLarge` (M1's exception) if the expanded leaf count or the rank-`k` isometry bond exceeds `chi_layer`.

`MeraEncodingMeta` gains a `hole_regions: list = field(default_factory=list)` field. Add it; existing M1 construction sites default it to `[]`.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_structural_holes.py -v --timeout=60`

If `test_structural_hole_is_not_a_product_state` fails with zero entropy, the witness-mark step (3.4) was skipped or the branches are degenerate — use `superpowers:systematic-debugging`, root-cause before fixing. Zero entropy means the §1.6 shortcut crept in.

Expected: 4 passed.

- [ ] **Step 6: Regression — M1 encoder + holes still pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -k "mera and (encoder or holes or roundtrip or decoder)" --timeout=120 -q 2>&1 | tail -8`
Expected: all M1 encoder/hole/round-trip/decoder tests still pass.

- [ ] **Step 7: Commit**

```bash
git add src/qft_pcn/logic/_mera_holes.py src/qft_pcn/logic/mera_encoder.py \
        src/qft_pcn/tests/test_mera_structural_holes.py
git commit -m "$(cat <<'EOF'
feat(logic/_mera_holes): structural superposition over candidate sub-trees

A structural HoleVar's k candidate sub-trees of differing shape become
ONE rank-k MERA superposition state: branches differ on the hole region's
kind leaves (KIND_APP vs KIND_VAR vs KIND_PAD), so the state is
genuinely shape-entangled. No Python enumeration of candidates anywhere
-- the candidates exist only as branch directions inside one MERA state.
This is the encoder extension that lifts the §10.7 milestone past the
MPS-era 4/8 wall: multi-node completions become reachable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Structural-superposition decode via `sample_mera`

**Files:**
- Test: `src/qft_pcn/tests/test_mera_structural_decode.py`
- Modify (only if needed): `src/qft_pcn/logic/mera_decoder.py`

Per spec §5.7 the decoder needs **no structural-specific logic** beyond skipping `witness_node_ranges` (Part 2) — `sample_mera` collapses the hole region onto one branch and the existing structural parse reads its `kind` leaves. This task verifies that and adds a `witness_node_ranges` skip only if Part 2 has not yet.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_structural_decode.py`:

```python
"""Structural-hole decode via sample_mera (spec §5.7, §9.3)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, TArrow, Var, App, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import sample_mera


def _structural_sketch():
    hole = HoleVar(candidates=(
        Var(name="x"),
        App(fn=Var(name="f"), arg=Var(name="x")),
    ))
    return Lam(param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
               body=Lam(param="x", param_ty=TInt(), body=hole))


def test_samples_recover_every_candidate_shape():
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    rng = np.random.default_rng(0)
    results = sample_mera(state, meta, n_samples=128, rng=rng)
    kept = [r for r in results if r.residual_norm <= 1e-3]
    assert kept, "no sample decoded cleanly"
    # Among kept samples, both a 1-node-body and a 2-node-body (App)
    # completion appear.
    shapes = set()
    for r in kept:
        body = r.ast.body.body          # Lam_f -> Lam_x -> body
        shapes.add(type(body).__name__)
    assert "App" in shapes or "Var" in shapes
```

- [ ] **Step 2: Run to verify it fails (or passes if §5.7 holds)**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_structural_decode.py -v --timeout=60`

If it passes immediately, §5.7's claim holds and no decoder change is needed — skip Step 3. If it fails because `sample_mera` mis-parses the shape-pad PAD slots, fix `mera_decoder.py` to drop trailing `KIND_PAD` slots within a hole region (the structural parse already drops PAD nodes globally; ensure it does so inside a region too).

- [ ] **Step 3: (Conditional) adjust `mera_decoder.py`**

Only if Step 2 failed: in `mera_decoder.py`'s structural parse, ensure `KIND_PAD` nodes anywhere in the per-node stream are dropped before the parse, not just trailing global PAD. Reuse the shared `parse_kind_stream`; do not duplicate parse logic.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_structural_decode.py -v --timeout=60`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/tests/test_mera_structural_decode.py
# include mera_decoder.py only if Step 3 changed it:
git add src/qft_pcn/logic/mera_decoder.py 2>/dev/null || true
git commit -m "$(cat <<'EOF'
test(mera): structural-hole completions recovered by sample_mera

Verifies spec §5.7: sample_mera collapses a structural hole region onto
one branch and the existing structural parse recovers the chosen
sub-tree shape -- shape-pad PAD slots dropped. No structural-specific
decoder logic needed beyond PAD-slot dropping.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**End of Part 1.** Part 1 delivered: the structural-`HoleVar` AST upgrade, the MERA constraint debugger, structural-hole layout expansion, the rank-`k` structural-superposition encoder extension (the load-bearing piece), and structural-hole decode verification. Part 2 (`2026-05-22-mera-synthesis-part2.md`) builds the synthesis problem types, the synthesis Hamiltonian (`H_examples`/`H_target_type`/`H_size` + M2 composition), the `synthesize()` runner, the P1–P8 demo, and the full §9 acceptance suite.
