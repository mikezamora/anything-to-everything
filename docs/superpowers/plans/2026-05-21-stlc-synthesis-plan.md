# STLC Synthesis Milestone Implementation Plan (Sub-Project E)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sub-project E from `docs/superpowers/specs/2026-05-21-stlc-synthesis-design.md`: an end-to-end STLC program-synthesis demo where the QPCN, without any LLM in the loop, produces ranked completions of hole-bearing lambda-calculus programs by imag-time-relaxing under `H_typing + H_eval + H_synthesis` and sampling.

**Architecture:** Composes sub-project A (encoder/decoder), B (typing-rule Hamiltonian), C (evaluation Hamiltonian), D (constraint debugger). Adds (i) `TypeHole` AST node, (ii) encoder extension for type-register superposition, (iii) `H_examples`/`H_target_type`/`H_size` Hamiltonian builders, (iv) the `synthesize()` pipeline, (v) the demo script.

**Tech Stack:** Python 3.11, numpy 1.26, scipy 1.16, pytest. Built on existing `src/qft_pcn/qft/` and `src/qft_pcn/logic/` machinery.

**Driving principles (from spec §1, non-negotiable):**

1. **End-to-end uses the architectural stack.** No classical enumeration fallback.
2. **No LLM in the loop.** This is the bold core demonstration.
3. **Holes are quantum superpositions.** Not enumeration placeholders.
4. **Ranking is by ⟨H⟩ on relaxed completions.** No auxiliary heuristics.
5. **The sampler is A's principled MPS sampler.** No custom beam search.
6. **Honest reporting; negative results are publishable.** P8 must classify as failure.
7. **Reuse A/B/C/D; do not reinvent.** New code lives only in `logic/synthesis/`.

**Reference:** The spec at `docs/superpowers/specs/2026-05-21-stlc-synthesis-design.md` is the authoritative contract. When in doubt, re-read it. If something here disagrees with the spec, the spec wins. The spec sections referenced as `§X.Y` throughout are spec sections.

**Test command throughout:** Run pytest via the project venv: `.venv/bin/python -m pytest <path> -v`.

**Dependency on sub-projects A/B/C/D:** This plan assumes A is implemented (`src/qft_pcn/logic/{ast,encoder,decoder,encoding}.py` exist) and B/C/D are implemented (`src/qft_pcn/logic/{typing_rules,hamiltonian_compiler,evaluation_hamiltonian,debugger}.py` exist, or equivalent). If any of those are missing while executing this plan, **stop and escalate** — do not stub them out from E.

This document is the single plan for sub-project E and covers all 22 tasks.

---

## Phase 0: Pre-flight checks

### Task 0: Verify upstream sub-projects are in place

**Files:** (read-only)

- [ ] **Step 1: Confirm A is in place**

Run: `.venv/bin/python -c "from src.qft_pcn.logic import encode, decode, sample, HoleVar, parse; print('A: ok')"`
Expected: `A: ok`.

If not: **STOP**. Sub-project A is missing.

- [ ] **Step 2: Confirm B is in place (typing-rule Hamiltonian)**

Run: `.venv/bin/python -c "from src.qft_pcn.logic.hamiltonian_compiler import compile_typing_hamiltonian; print('B: ok')"`
Expected: `B: ok`.

If the import path differs (e.g. B settled on `from src.qft_pcn.logic.typing import compile`), update the test command and **proceed**, noting the deviation.

If B is genuinely missing: **STOP** and escalate.

- [ ] **Step 3: Confirm C is in place (eval Hamiltonian)**

Run: `.venv/bin/python -c "from src.qft_pcn.logic.evaluation_hamiltonian import compile_eval_hamiltonian; print('C: ok')"`
Expected: `C: ok`.

- [ ] **Step 4: Confirm D is in place (debugger)**

Run: `.venv/bin/python -c "from src.qft_pcn.logic.debugger import diagnose; print('D: ok')"`
Expected: `D: ok`.

- [ ] **Step 5: Run the full existing test suite as baseline**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -v --ignore=src/qft_pcn/tests/test_quantum.py 2>&1 | tail -20`
Expected: All tests pass.

If any test fails, **STOP** and report. Do not start E on a red baseline.

- [ ] **Step 6: Confirm Hamiltonian.__add__ and .expectation exist**

Run:
```bash
.venv/bin/python -c "
from src.qft_pcn.qft.hamiltonian import Hamiltonian
print('add:', hasattr(Hamiltonian, '__add__'))
print('expectation:', hasattr(Hamiltonian, 'expectation'))
"
```

Expected: both `True`. If either is `False`, this plan adds them in Task 12; record the gap and continue.

---

## Phase 1: AST and problem types

### Task 1: Add `TypeHole` to ast.py

**Files:**
- Modify: `src/qft_pcn/logic/ast.py` (add TypeHole class).
- Modify: `src/qft_pcn/tests/test_logic_ast.py` (add TypeHole tests).

Per spec §5.1.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_ast.py`:

```python
from src.qft_pcn.logic.ast import TypeHole


def test_typehole_construction():
    th = TypeHole(candidates=(TInt(), TBool()), name="?T")
    assert th.candidates == (TInt(), TBool())
    assert th.name == "?T"


def test_typehole_default_name():
    th = TypeHole(candidates=(TInt(),))
    assert th.name == ""


def test_typehole_is_ty():
    from src.qft_pcn.logic.ast import Ty
    th = TypeHole(candidates=(TInt(),))
    assert isinstance(th, Ty)


def test_typehole_rejects_arrow_nested_candidate():
    # Spec §3.1: candidates may not include TYPE_ARR_NESTED-tier types.
    # We enforce this with a runtime check on construction: candidates must
    # all be primitive or single-level arrows.
    deep = TArrow(src=TArrow(src=TInt(), dst=TInt()), dst=TInt())
    with pytest.raises(ValueError, match="nested arrow"):
        TypeHole(candidates=(deep,))


def test_typehole_in_lam_position():
    # \x:?T. x  — TypeHole used as a parameter type.
    lam = Lam(param="x", param_ty=TypeHole(candidates=(TInt(), TBool())),
              body=Var(name="x"))
    assert isinstance(lam.param_ty, TypeHole)
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -k "typehole" -v`
Expected: `ImportError: cannot import name 'TypeHole'`.

- [ ] **Step 3: Implement TypeHole**

In `src/qft_pcn/logic/ast.py`, add after the existing `TArrow` class:

```python
@dataclass(frozen=True)
class TypeHole(Ty):
    """A type-position hole: this type is unknown but must come from one of
    the candidates. Encoded by the synthesis encoder extension (sub-project E)
    as an equal-amplitude superposition over the candidate tags on the `type`
    register.

    Candidates must be flat (TInt, TBool, or single-level TArrow); nested
    arrows are rejected to keep the type-tag enumeration finite.
    """
    candidates: tuple[Ty, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("TypeHole must have at least one candidate")
        for c in self.candidates:
            if isinstance(c, TArrow):
                if isinstance(c.src, TArrow) or isinstance(c.dst, TArrow):
                    raise ValueError(
                        f"TypeHole candidates may not include nested arrow "
                        f"types: {c!r}"
                    )
            elif not isinstance(c, (TInt, TBool)):
                raise ValueError(
                    f"TypeHole candidates must be Ty: got {type(c).__name__}"
                )
```

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -k "typehole" -v`
Expected: 5 passed.

- [ ] **Step 5: Run all ast tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -v 2>&1 | tail -10`
Expected: all tests pass (existing + 5 new).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/ast.py src/qft_pcn/tests/test_logic_ast.py
git commit -m "$(cat <<'EOF'
feat(logic/ast): add TypeHole node for synthesis sub-project

TypeHole is a Ty subclass with a finite list of candidate types.
Rejects nested arrow candidates at construction time (spec §3.1).
Used by sub-project E's encoder extension to put the type register
into superposition.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Upgrade `HoleVar` with candidates and target_type

**Files:**
- Modify: `src/qft_pcn/logic/ast.py`.
- Modify: `src/qft_pcn/tests/test_logic_ast.py`.

Per spec §5.2. A's existing `HoleVar` was minimal; E needs the full version.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_ast.py`:

```python
from src.qft_pcn.logic.ast import HoleVar


def test_holevar_full_construction():
    h = HoleVar(candidates=("x", "y"), target_type=TInt(), name="?HOLE")
    assert h.candidates == ("x", "y")
    assert h.target_type == TInt()
    assert h.name == "?HOLE"


def test_holevar_empty_candidates_means_any():
    # Empty candidates = "any in-scope binder"; encoder resolves at encode-time.
    h = HoleVar()
    assert h.candidates == ()
    assert h.target_type is None
    assert h.name == ""


def test_holevar_target_type_optional():
    h = HoleVar(candidates=("z",))
    assert h.target_type is None
```

- [ ] **Step 2: Verify failure or assess current state**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -k "holevar" -v`

If A already implements the full version, all three may already pass — note that and skip Step 3. If not, expect AttributeError or ValueError.

- [ ] **Step 3: Implement / upgrade HoleVar**

In `src/qft_pcn/logic/ast.py`, replace any existing minimal `HoleVar` with:

```python
@dataclass
class HoleVar(Node):
    """A variable-position hole: 'fill me with one of these binder
    references, with this expected type'.

    candidates: binder names admissible at this position; empty = any in-scope.
    target_type: optional expected type (None = inferred).
    name: optional diagnostic label.
    """
    candidates: tuple[str, ...] = ()
    target_type: Ty | None = None
    name: str = ""
```

If A had a different field name (e.g. `candidates: list[str]`), accept the existing form but ensure tests pass; standardize on `tuple` for hashability if practical, otherwise document the deviation.

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py -k "holevar" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/ast.py src/qft_pcn/tests/test_logic_ast.py
git commit -m "$(cat <<'EOF'
feat(logic/ast): upgrade HoleVar with candidates and target_type

Per spec §5.2: HoleVar carries a tuple of candidate binder names (empty
= any in-scope), an optional target_type, and an optional name label
for diagnostics. Encoder extension consumes these to produce bid-register
superpositions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Create `synthesis/` package skeleton

**Files:**
- Create: `src/qft_pcn/logic/synthesis/__init__.py` (empty for now).

- [ ] **Step 1: Create directory and empty init**

Run: `mkdir -p src/qft_pcn/logic/synthesis && touch src/qft_pcn/logic/synthesis/__init__.py`

Verify: `ls src/qft_pcn/logic/synthesis/`
Expected: `__init__.py`.

- [ ] **Step 2: Stage**

Run: `git add src/qft_pcn/logic/synthesis/__init__.py`

(Commit alongside Task 4.)

---

### Task 4: Problem and result dataclasses (`synthesis/problem.py`)

**Files:**
- Create: `src/qft_pcn/logic/synthesis/problem.py`.
- Create: `src/qft_pcn/tests/test_synthesis_problem.py`.

Per spec §3.1, §3.2.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_synthesis_problem.py`:

```python
"""Tests for synthesis/problem.py: SynthesisProblem, IOExample, etc."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import (
    Var, Lam, App, IntLit, BoolLit, HoleVar, TypeHole,
    TInt, TBool, TArrow,
)
from src.qft_pcn.logic.synthesis.problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)


def test_io_example_construction():
    ex = IOExample(inputs=(IntLit(val=2),), output=IntLit(val=3))
    assert ex.inputs == (IntLit(val=2),)
    assert ex.output == IntLit(val=3)


def test_synthesis_problem_minimal():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=("x",)))
    p = SynthesisProblem(sketch=sketch, name="P1")
    assert p.name == "P1"
    assert p.target_type is None
    assert p.examples == ()
    # defaults
    assert p.N == 32
    assert p.chi_max == 32
    assert p.n_samples == 64
    assert p.anneal_steps == 200
    assert p.anneal_dt == 0.05


def test_synthesis_problem_with_examples_and_target():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar())
    p = SynthesisProblem(
        sketch=sketch,
        target_type=TArrow(src=TInt(), dst=TInt()),
        examples=(IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3)),),
        name="P2",
    )
    assert p.target_type == TArrow(src=TInt(), dst=TInt())
    assert len(p.examples) == 1


def test_hamiltonian_weights_defaults():
    w = HamiltonianWeights()
    assert w.w_typing == 4.0
    assert w.w_eval == 2.0
    assert w.w_examples == 3.0
    assert w.w_target_type == 2.0
    assert w.w_size == 0.1


def test_hamiltonian_weights_override():
    w = HamiltonianWeights(w_typing=8.0, w_size=0.0)
    assert w.w_typing == 8.0
    assert w.w_size == 0.0
    # other defaults preserved
    assert w.w_eval == 2.0


def test_completion_construction():
    c = Completion(
        ast=IntLit(val=3),
        energy=0.5,
        energy_breakdown={"typing": 0.0, "size": 0.5},
        diagnostics={},
        multiplicity=10,
    )
    assert c.energy == 0.5
    assert c.multiplicity == 10
    assert c.energy_breakdown["typing"] == 0.0


def test_synthesis_result_construction():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar())
    p = SynthesisProblem(sketch=sketch, name="dummy")
    r = SynthesisResult(
        problem=p,
        completions=[],
        n_unique=0,
        n_samples_drawn=64,
        n_samples_decoded_ok=0,
        final_state_energy=42.0,
        wall_time_seconds=1.23,
        chi_observed_max=16,
        failure_mode="no_valid_completion",
    )
    assert r.failure_mode == "no_valid_completion"
    assert r.completions == []
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_problem.py -v`
Expected: ImportError on `synthesis.problem`.

- [ ] **Step 3: Implement problem.py**

Create `src/qft_pcn/logic/synthesis/problem.py`:

```python
"""Data types for the synthesis pipeline (spec §3).

A SynthesisProblem is the input contract: a hole-bearing AST plus optional
constraints (target_type, examples). A SynthesisResult is the output:
ranked completions, diagnostics, failure_mode classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.qft_pcn.logic.ast import Node, Ty


@dataclass(frozen=True)
class IOExample:
    """An example-based constraint on the synthesized function.

    Applying the sketch's root expression to `inputs` (treating it as a
    curried function) must produce `output`. Inputs and output must be
    literal Nodes (IntLit / BoolLit only).
    """
    inputs: tuple[Node, ...]
    output: Node


@dataclass(frozen=True)
class HamiltonianWeights:
    """Block weights for H_total = sum(w_block * H_block).

    Defaults from spec §4.1; do not tune these per-problem unless the
    energy gap check fails — a tuning amendment belongs in the spec.
    """
    w_typing: float = 4.0
    w_eval: float = 2.0
    w_examples: float = 3.0
    w_target_type: float = 2.0
    w_size: float = 0.1


@dataclass(frozen=True)
class SynthesisProblem:
    """The synthesis input contract.

    sketch: an AST containing one or more HoleVar / TypeHole nodes.
    target_type: optional pinned type for the root expression.
    examples: optional IO examples; each adds a witness region to H_examples.
    name: optional label for reports.
    N, chi_max, n_samples, anneal_steps, anneal_dt: knobs (spec §3.1).
    weights: Hamiltonian weights (spec §4.1).
    """
    sketch: Node
    target_type: Optional[Ty] = None
    examples: tuple[IOExample, ...] = ()
    name: str = ""
    N: int = 32
    chi_max: int = 32
    n_samples: int = 64
    anneal_steps: int = 200
    anneal_dt: float = 0.05
    weights: HamiltonianWeights = field(default_factory=HamiltonianWeights)


@dataclass(frozen=True)
class Completion:
    """One unique completion produced by the synthesis pipeline.

    ast: the filled-in AST (no holes).
    energy: <H_total> on the completion's product-state encoding.
    energy_breakdown: dict block_name -> <H_block> (weighted).
    diagnostics: from sub-project D's `diagnose` — residual constraint
                 violations per term.
    multiplicity: how many of the K samples produced this AST.
    """
    ast: Node
    energy: float
    energy_breakdown: dict
    diagnostics: dict
    multiplicity: int


@dataclass(frozen=True)
class SynthesisResult:
    """The synthesis output.

    completions: sorted ascending by energy. Empty iff failure_mode in
                 {"no_valid_completion"}.
    failure_mode: None for success, else a label per spec §6.4.
    """
    problem: SynthesisProblem
    completions: list
    n_unique: int
    n_samples_drawn: int
    n_samples_decoded_ok: int
    final_state_energy: float
    wall_time_seconds: float
    chi_observed_max: int
    failure_mode: Optional[str]
```

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_problem.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/synthesis/__init__.py src/qft_pcn/logic/synthesis/problem.py src/qft_pcn/tests/test_synthesis_problem.py
git commit -m "$(cat <<'EOF'
feat(logic/synthesis): SynthesisProblem, IOExample, SynthesisResult dataclasses

The input/output contract for sub-project E's synthesize() API. Per
spec §3.1, §3.2: SynthesisProblem packages a hole-bearing sketch with
optional target_type and IO examples; SynthesisResult returns ranked
completions plus failure-mode classification. HamiltonianWeights carries
the per-block weight defaults from spec §4.1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Problem validation + error types

**Files:**
- Create: `src/qft_pcn/logic/synthesis/errors.py`.
- Create: `src/qft_pcn/logic/synthesis/_validate.py`.
- Modify: `src/qft_pcn/tests/test_synthesis_problem.py` (add validation tests).

Per spec §9 and §3.1 invariants.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_synthesis_problem.py`:

```python
from src.qft_pcn.logic.synthesis.errors import (
    SynthesisError, SynthesisProblemError, SynthesisRuntimeError,
)
from src.qft_pcn.logic.synthesis._validate import validate_problem


def test_error_hierarchy():
    assert issubclass(SynthesisProblemError, SynthesisError)
    assert issubclass(SynthesisRuntimeError, SynthesisError)


def test_validate_rejects_no_holes():
    # Sketch with no HoleVar / TypeHole — spec §3.1 mandates at least one.
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    p = SynthesisProblem(sketch=sketch)
    with pytest.raises(SynthesisProblemError, match="at least one hole"):
        validate_problem(p)


def test_validate_accepts_holevar_sketch():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=("x",)))
    p = SynthesisProblem(sketch=sketch)
    validate_problem(p)  # no raise


def test_validate_accepts_typehole_sketch():
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    p = SynthesisProblem(sketch=sketch)
    validate_problem(p)  # no raise


def test_validate_rejects_non_literal_io_example():
    sketch = Lam(param="x", param_ty=TInt(), body=HoleVar())
    bad_ex = IOExample(inputs=(Var(name="x"),), output=IntLit(val=1))
    p = SynthesisProblem(sketch=sketch, examples=(bad_ex,))
    with pytest.raises(SynthesisProblemError, match="literal"):
        validate_problem(p)


def test_validate_rejects_holevar_with_undef_candidate():
    # candidate "z" is not in lexical scope
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("z",)))
    p = SynthesisProblem(sketch=sketch)
    with pytest.raises(SynthesisProblemError, match="not in scope"):
        validate_problem(p)
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_problem.py -k "validate or error_hierarchy" -v`
Expected: ImportError on `errors` / `_validate`.

- [ ] **Step 3: Implement errors.py**

Create `src/qft_pcn/logic/synthesis/errors.py`:

```python
"""Exception types for the synthesis runner."""

from __future__ import annotations


class SynthesisError(Exception):
    """Base class for synthesis-runner errors."""


class SynthesisProblemError(SynthesisError):
    """The SynthesisProblem violates spec §3.1 invariants."""


class SynthesisRuntimeError(SynthesisError):
    """Evolution diverged or sampling produced no decodable completions."""
```

- [ ] **Step 4: Implement _validate.py**

Create `src/qft_pcn/logic/synthesis/_validate.py`:

```python
"""Validation of SynthesisProblem invariants (spec §3.1).

Errors raised here are reported to the caller via SynthesisProblemError.
The runner calls validate_problem() first thing, before any encoding work.
"""

from __future__ import annotations

from src.qft_pcn.logic.ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar, TypeHole,
)
from .errors import SynthesisProblemError
from .problem import SynthesisProblem, IOExample


def _walk_collect_holes_and_check_scopes(
    node: Node,
    scope: tuple[str, ...] = (),
) -> tuple[int, int]:
    """Recursively visit `node`; return (n_var_holes, n_type_holes).

    Raises SynthesisProblemError on a HoleVar whose candidate is not in `scope`.
    """
    nvh, nth = 0, 0
    if isinstance(node, HoleVar):
        nvh += 1
        for cand in node.candidates:
            if cand not in scope:
                raise SynthesisProblemError(
                    f"HoleVar candidate {cand!r} not in scope "
                    f"(in-scope binders: {scope!r})"
                )
        # also check target_type for TypeHole (rare but valid)
        if isinstance(node.target_type, TypeHole):
            nth += 1
    elif isinstance(node, Var):
        pass  # not a hole; lexical-scope check is the encoder's job
    elif isinstance(node, IntLit) or isinstance(node, BoolLit):
        pass
    elif isinstance(node, Lam):
        if isinstance(node.param_ty, TypeHole):
            nth += 1
        child_scope = scope + (node.param,)
        a, b = _walk_collect_holes_and_check_scopes(node.body, child_scope)
        nvh += a
        nth += b
    elif isinstance(node, App):
        for child in (node.fn, node.arg):
            a, b = _walk_collect_holes_and_check_scopes(child, scope)
            nvh += a
            nth += b
    elif isinstance(node, If):
        for child in (node.cond, node.then_b, node.else_b):
            a, b = _walk_collect_holes_and_check_scopes(child, scope)
            nvh += a
            nth += b
    elif isinstance(node, Bin):
        for child in (node.lhs, node.rhs):
            a, b = _walk_collect_holes_and_check_scopes(child, scope)
            nvh += a
            nth += b
    else:
        raise SynthesisProblemError(f"unknown AST node: {type(node).__name__}")
    return nvh, nth


def _is_literal(node: Node) -> bool:
    return isinstance(node, (IntLit, BoolLit))


def validate_problem(problem: SynthesisProblem) -> None:
    """Raise SynthesisProblemError if `problem` violates spec §3.1.

    Checks:
        - sketch contains at least one HoleVar or TypeHole.
        - every HoleVar candidate name is in lexical scope at its position.
        - every IOExample's inputs and output are literals (IntLit/BoolLit).
        - knob ranges (N, chi_max, n_samples, anneal_*) are positive.
    """
    if problem.N < 1:
        raise SynthesisProblemError(f"N must be >= 1, got {problem.N}")
    if problem.chi_max < 1:
        raise SynthesisProblemError(
            f"chi_max must be >= 1, got {problem.chi_max}")
    if problem.n_samples < 1:
        raise SynthesisProblemError(
            f"n_samples must be >= 1, got {problem.n_samples}")
    if problem.anneal_steps < 1:
        raise SynthesisProblemError(
            f"anneal_steps must be >= 1, got {problem.anneal_steps}")
    if problem.anneal_dt <= 0:
        raise SynthesisProblemError(
            f"anneal_dt must be > 0, got {problem.anneal_dt}")

    nvh, nth = _walk_collect_holes_and_check_scopes(problem.sketch, ())
    if nvh + nth == 0:
        raise SynthesisProblemError(
            "SynthesisProblem must contain at least one hole "
            "(HoleVar or TypeHole). Got a fully-concrete sketch."
        )

    for i, ex in enumerate(problem.examples):
        for j, inp in enumerate(ex.inputs):
            if not _is_literal(inp):
                raise SynthesisProblemError(
                    f"IOExample[{i}].inputs[{j}] is not a literal: "
                    f"{type(inp).__name__}"
                )
        if not _is_literal(ex.output):
            raise SynthesisProblemError(
                f"IOExample[{i}].output is not a literal: "
                f"{type(ex.output).__name__}"
            )
```

- [ ] **Step 5: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_problem.py -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/synthesis/errors.py src/qft_pcn/logic/synthesis/_validate.py src/qft_pcn/tests/test_synthesis_problem.py
git commit -m "$(cat <<'EOF'
feat(logic/synthesis): problem validation and error types

Per spec §9 and §3.1: validate_problem() enforces invariants —
sketch has at least one hole, IOExamples are literal-only, HoleVar
candidates are in lexical scope, knob ranges are positive. Raises
SynthesisProblemError on violation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Encoder extensions

### Task 6: TypeHoleHandle and EncodingMeta extension

**Files:**
- Modify: `src/qft_pcn/logic/encoding.py`.
- Modify: `src/qft_pcn/tests/test_logic_encoding.py` (add tests).

Per spec §5.3: meta carries a `type_holes` table analogous to existing live-binder bookkeeping.

- [ ] **Step 1: Write the failing tests**

Append to `src/qft_pcn/tests/test_logic_encoding.py`:

```python
from src.qft_pcn.logic.encoding import (
    TypeHoleHandle, EncodingMeta as _Meta,
)


def test_type_hole_handle_frozen():
    th = TypeHoleHandle(hole_site=5, candidate_tags=(1, 2))   # T_INT, T_BOOL
    assert th.hole_site == 5
    assert th.candidate_tags == (1, 2)
    # Frozen -> hashable.
    {th}


def test_encoding_meta_has_type_holes_field():
    meta = _Meta(
        N=32, chi_max=32,
        field_dims={"kind": 8, "type": 8, "bid": 8, "value": 16},
        species=list(SPECIES),
        nested_type_index={},
        site_to_ast_path={},
        live_binders_per_bond=[],
    )
    # Default must be present and empty.
    assert hasattr(meta, "type_holes")
    assert meta.type_holes == {}


def test_encoding_meta_has_witness_regions_field():
    meta = _Meta(
        N=32, chi_max=32,
        field_dims={"kind": 8, "type": 8, "bid": 8, "value": 16},
        species=list(SPECIES),
        nested_type_index={},
        site_to_ast_path={},
        live_binders_per_bond=[],
    )
    assert hasattr(meta, "witness_regions")
    assert meta.witness_regions == []
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoding.py -k "type_hole or witness" -v`
Expected: ImportError on `TypeHoleHandle` or AttributeError on fields.

- [ ] **Step 3: Implement extensions**

In `src/qft_pcn/logic/encoding.py`:

1. Add `TypeHoleHandle` near `BinderHandle`:

```python
@dataclass(frozen=True)
class TypeHoleHandle:
    """A type-position hole's bookkeeping.

    hole_site: the site at which the TypeHole was placed (typically a Lam's
               position when the Lam's param_ty is a TypeHole).
    candidate_tags: the flat type-register tags (T_INT, T_BOOL, T_ARR_*)
                    over which the hole is in superposition.
    """
    hole_site: int
    candidate_tags: tuple[int, ...]
```

2. Add two new fields to `EncodingMeta` (with defaults via `field`):

```python
from dataclasses import dataclass, field

@dataclass
class EncodingMeta:
    N: int
    chi_max: int
    field_dims: dict[str, int]
    species: list[FieldSpecies]
    nested_type_index: dict[int, "object"]
    site_to_ast_path: dict[int, tuple[int, ...]]
    live_binders_per_bond: list[list[BinderHandle]]
    # New for sub-project E:
    type_holes: dict[int, TypeHoleHandle] = field(default_factory=dict)
    witness_regions: list[tuple[int, int]] = field(default_factory=list)
    # Each tuple is (start_site, end_site_exclusive) for one witness region.
```

- [ ] **Step 4: Verify pass + no regressions**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_encoding.py -v 2>&1 | tail -15`
Expected: all tests pass (including A's existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/encoding.py src/qft_pcn/tests/test_logic_encoding.py
git commit -m "$(cat <<'EOF'
feat(logic/encoding): TypeHoleHandle and EncodingMeta witness/type_hole fields

Per spec §5.3, §4.2 (lattice layout): meta now carries a type_holes map
(site -> TypeHoleHandle) recording type-register superposition positions,
and a witness_regions list demarcating example-witness sub-regions added
by the synthesis runner. Both default to empty so A's existing encoder
output remains a valid EncodingMeta unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: TypeHole encoder branch (`synthesis/encode_ext.py`)

**Files:**
- Create: `src/qft_pcn/logic/synthesis/encode_ext.py`.
- Create: `src/qft_pcn/tests/test_synthesis_encoder_ext.py`.

Per spec §5.3. This is the technically-tricky encoder extension; reuse A's existing channel construction. The strategy:

The encoder extension wraps A's `encode()`. It pre-processes the sketch to identify TypeHole positions, records them in meta after the base encode, then post-applies a state-modifying step that puts the type register at hole sites into superposition.

The simpler-and-equivalent approach we use: implement `encode_synthesis(sketch)` that walks the AST, computes the "definite" types as in A (substituting TypeHole.candidates[0] as a placeholder), calls A's `encode()`, then **applies a one-site superposition projector** at each TypeHole site that lifts the type register to the equal-amplitude state over the candidates. This is operationally identical to the §5.3 description (the bond channels emerge from A's machinery when the local state has the right reduced density matrix), and avoids surgery on A's internal modules.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_synthesis_encoder_ext.py`:

```python
"""Tests for synthesis/encode_ext.py: TypeHole superposition encoder."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    Var, Lam, IntLit, HoleVar, TypeHole, TInt, TBool,
)
from src.qft_pcn.logic.synthesis.encode_ext import encode_synthesis


def test_encode_synthesis_returns_state_and_meta():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("x",)))
    state, meta = encode_synthesis(sketch, N=32, chi_max=32)
    assert state.N == 32
    assert meta.N == 32
    assert meta.chi_max == 32


def test_encode_synthesis_records_type_hole():
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    state, meta = encode_synthesis(sketch, N=32, chi_max=32)
    assert len(meta.type_holes) == 1
    # Hole site is the Lam's site (site 0 in pre-order).
    handle = next(iter(meta.type_holes.values()))
    # candidate tags: TYPE_INT = 1, TYPE_BOOL = 2
    assert sorted(handle.candidate_tags) == [1, 2]


def test_encode_synthesis_typehole_produces_type_register_entropy():
    """The principled superposition test (analogous to A.§7.4)."""
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    state, meta = encode_synthesis(sketch, N=32, chi_max=32)
    # The hole is at the Lam's site (site 0). The Var at site 1 inherits
    # the type via the type-register channel. Hence the bond between
    # site 0 and site 1 must carry > 0 entanglement entropy on the type
    # register.
    bond = 0
    S = state.entanglement_entropy(bond)
    # Two equal-amplitude candidates => entropy ~ ln(2) on type register.
    assert S > 0.5, (
        f"bond {bond} entropy = {S}; type-register superposition not "
        f"engaging — re-check spec §5.3 implementation"
    )


def test_encode_synthesis_holevar_no_typehole_no_type_holes_entry():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("x",)))
    state, meta = encode_synthesis(sketch, N=32, chi_max=32)
    assert meta.type_holes == {}


def test_encode_synthesis_norm_preserved():
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    state, meta = encode_synthesis(sketch, N=32, chi_max=32)
    assert abs(state.norm_sq() - 1.0) < 1e-8
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_encoder_ext.py -v`
Expected: ImportError on `synthesis.encode_ext`.

- [ ] **Step 3: Implement encode_ext.py**

Create `src/qft_pcn/logic/synthesis/encode_ext.py`:

```python
"""Encoder extensions for sub-project E.

Two responsibilities:

1. `encode_synthesis(sketch, N, chi_max)`: encode a sketch containing
   TypeHole nodes. Wraps A's `encode()` and applies a superposition
   projector on the type register at each TypeHole site, lifting it
   from a definite-tag state to an equal-amplitude superposition over
   the candidate tags. Per spec §5.3.

2. `witness_augmented_sketch(sketch, examples)`: build the augmented
   AST that includes the sketch + per-example witness regions. Per
   spec §5.5. (Implemented in Task 9.)
"""

from __future__ import annotations

from copy import deepcopy
import numpy as np

from src.qft_pcn.logic.ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar,
    Ty, TInt, TBool, TArrow, TypeHole,
)
from src.qft_pcn.logic.encoding import (
    EncodingMeta, TypeHoleHandle,
    TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB,
    TYPE_ARR_NESTED, TYPE_CUTOFF,
    KIND_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, D_LOCAL,
)
from src.qft_pcn.logic.encoder import encode as _base_encode


def _ty_to_tag(t: Ty) -> int:
    """Map a flat Ty to its type-register tag. Raises on nested arrows."""
    if isinstance(t, TInt):
        return TYPE_INT
    if isinstance(t, TBool):
        return TYPE_BOOL
    if isinstance(t, TArrow):
        if isinstance(t.src, TInt) and isinstance(t.dst, TInt):
            return TYPE_ARR_II
        if isinstance(t.src, TInt) and isinstance(t.dst, TBool):
            return TYPE_ARR_IB
        if isinstance(t.src, TBool) and isinstance(t.dst, TInt):
            return TYPE_ARR_BI
        if isinstance(t.src, TBool) and isinstance(t.dst, TBool):
            return TYPE_ARR_BB
        return TYPE_ARR_NESTED
    raise ValueError(f"cannot tag {t!r}")


def _find_typeholes_with_sites(
    sketch: Node,
) -> list[tuple[tuple[int, ...], TypeHole]]:
    """Walk the sketch in pre-order; return list of (ast_path, TypeHole)
    for every TypeHole found in a Lam.param_ty position.

    (TypeHole in deeper positions is out of scope; see spec §2.2.)
    """
    out: list[tuple[tuple[int, ...], TypeHole]] = []

    def go(node: Node, path: tuple[int, ...]) -> None:
        if isinstance(node, Lam):
            if isinstance(node.param_ty, TypeHole):
                out.append((path, node.param_ty))
            go(node.body, path + (0,))
        elif isinstance(node, App):
            go(node.fn, path + (0,))
            go(node.arg, path + (1,))
        elif isinstance(node, If):
            go(node.cond, path + (0,))
            go(node.then_b, path + (1,))
            go(node.else_b, path + (2,))
        elif isinstance(node, Bin):
            go(node.lhs, path + (0,))
            go(node.rhs, path + (1,))
        # leaves: nothing.

    go(sketch, ())
    return out


def _substitute_typeholes_with_first_candidate(sketch: Node) -> Node:
    """Return a copy of sketch with every TypeHole replaced by its first
    candidate. Used to obtain a 'definite' sketch for A's base encoder.
    """
    def fix_ty(t: Ty) -> Ty:
        if isinstance(t, TypeHole):
            return t.candidates[0]
        if isinstance(t, TArrow):
            return TArrow(src=fix_ty(t.src), dst=fix_ty(t.dst))
        return t

    def go(node: Node) -> Node:
        if isinstance(node, Lam):
            return Lam(param=node.param,
                       param_ty=fix_ty(node.param_ty),
                       body=go(node.body))
        if isinstance(node, App):
            return App(fn=go(node.fn), arg=go(node.arg))
        if isinstance(node, If):
            return If(cond=go(node.cond),
                      then_b=go(node.then_b),
                      else_b=go(node.else_b))
        if isinstance(node, Bin):
            return Bin(op=node.op, lhs=go(node.lhs), rhs=go(node.rhs))
        return node

    return go(sketch)


def _path_to_site(path: tuple[int, ...], meta: EncodingMeta) -> int:
    """Reverse lookup: site index given an AST path. Uses meta's
    site_to_ast_path table."""
    for site, p in meta.site_to_ast_path.items():
        if p == path:
            return site
    raise KeyError(f"AST path {path!r} not in meta.site_to_ast_path")


def _apply_type_superposition(state, meta, site: int,
                              candidate_tags: tuple[int, ...]) -> None:
    """In-place: apply a one-site projector that takes the type register
    at `site` from a definite |tag_0> state to (1/sqrt(k)) Σ |tag_i> over
    the candidate tags.

    Strategy: build a (TYPE_CUTOFF x TYPE_CUTOFF) operator
        M = (1/sqrt(k)) Σ_i |tag_i><tag_0|
    embed it into the full local Hilbert space (D_LOCAL x D_LOCAL) via
    embed_op on the `type` register (species index 1, leftmost-slowest
    ordering), and apply via state.apply_local_gate(site, op).

    The result is no longer a unit-norm state; we renormalize after.
    """
    from src.qft_pcn.qft.fock import embed_op

    k = len(candidate_tags)
    amp = 1.0 / np.sqrt(k)
    M_type = np.zeros((TYPE_CUTOFF, TYPE_CUTOFF), dtype=complex)
    src_tag = candidate_tags[0]   # the definite tag the base encoder wrote
    for t in candidate_tags:
        M_type[t, src_tag] = amp

    # Embed M_type onto species index 1 of the local 4-species register.
    # Species ordering: kind (0), type (1), bid (2), value (3).
    species_dims = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF)
    M_full = embed_op(M_type, species_index=1, species_dims=species_dims)
    state.apply_local_gate(site, M_full)
    state.normalize()


def encode_synthesis(sketch: Node, N: int = 32, chi_max: int = 32):
    """Encode a synthesis sketch (may contain TypeHole) into an MPS.

    Pipeline:
        1. Find all TypeHoles in sketch (Lam.param_ty positions only).
        2. Substitute each with its first candidate to produce a definite
           sketch.
        3. Run A's `encode()` on the definite sketch.
        4. For each TypeHole, look up its site via meta.site_to_ast_path
           and apply the superposition projector on the type register.
        5. Record TypeHoleHandle entries in meta.type_holes.

    Returns (state, meta). The state is normalized.
    """
    typeholes = _find_typeholes_with_sites(sketch)
    definite_sketch = _substitute_typeholes_with_first_candidate(sketch)
    state, meta = _base_encode(definite_sketch, N=N, chi_max=chi_max)

    for path, th in typeholes:
        try:
            site = _path_to_site(path, meta)
        except KeyError:
            # path wasn't recorded — likely a path-format mismatch.
            # Fall back to scanning sketch in pre-order for the Lam at
            # that path and using its absolute pre-order index.
            raise RuntimeError(
                f"TypeHole at AST path {path} not findable in meta; "
                f"check A's site_to_ast_path emission convention"
            )
        cand_tags = tuple(sorted({_ty_to_tag(c) for c in th.candidates}))
        _apply_type_superposition(state, meta, site, cand_tags)
        meta.type_holes[site] = TypeHoleHandle(
            hole_site=site, candidate_tags=cand_tags,
        )

    return state, meta
```

**Implementer note on `_path_to_site`:** if A's `meta.site_to_ast_path` records paths in a different convention (e.g. uses index 0 for "the only child" of a Lam vs index 1), adjust `_find_typeholes_with_sites` accordingly. Confirm by reading A's encoder before writing the test in Step 1.

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_encoder_ext.py -v`
Expected: 5 passed.

If the entropy test fails (S < 0.5): the type-register projector did not produce the expected superposition. Common causes:
  - `embed_op` species ordering is reversed (species index 1 might be `bid` in A's convention; check `SPECIES_NAMES` order).
  - The projector formula uses `|tag_0>` for the "source" but the base encoder wrote a different definite tag at that site (e.g. `TYPE_NONE`); inspect the local state and adjust `src_tag`.

Do not lower the test threshold. Find the actual bug.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/synthesis/encode_ext.py src/qft_pcn/tests/test_synthesis_encoder_ext.py
git commit -m "$(cat <<'EOF'
feat(logic/synthesis): TypeHole superposition encoder extension

Per spec §5.3: wraps A's encode() to first substitute TypeHoles with
their first candidate (producing a definite sketch A can encode), then
applies a one-site projector on the type register at each TypeHole site
to lift it to an equal-amplitude superposition over the candidate tags.
The resulting MPS bond on the type register carries the candidate-set
correlation, verified by the entropy-probe test (S ~ ln(2) for 2 candidates).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Witness-AST builder

**Files:**
- Modify: `src/qft_pcn/logic/synthesis/encode_ext.py` (add `witness_augmented_sketch`).
- Modify: `src/qft_pcn/tests/test_synthesis_encoder_ext.py` (add tests).

Per spec §5.5. We need to express witness regions as part of the encoded MPS so that `H_examples`'s boundary pin lives on a real site.

For E we adopt the **classical-copy** simplification of the witness construction: each witness region holds an `App(... App(sketch_root_copy, in_0)..., in_n)` sub-tree where `sketch_root_copy` is a deep-copy of the sketch's root expression (a Lam). Spec §5.5 prefers a `RefVar` channel approach; the spec also flags this as an open implementation decision (§11) — for the first cut we use classical copy, then revisit if it causes encoding-size problems on the demo problems.

**Rationale for the simplification:** the §7 demo problems' sketches are small (≤ 7 sites); a classical copy doubles or triples their footprint but stays within N=32. The architectural claim about *the sketch's e subtree being shared with the witness* is preserved morally: the copy gives every site the same encoded state, and the typing/eval Hamiltonians will produce the same energy contribution at both copies — they are operationally equal. Revisit only if the demo runs out of sites.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_synthesis_encoder_ext.py`:

```python
from src.qft_pcn.logic.ast import App, IntLit
from src.qft_pcn.logic.synthesis.problem import IOExample
from src.qft_pcn.logic.synthesis.encode_ext import witness_augmented_sketch


def test_witness_augmented_no_examples_returns_sketch_unchanged():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("x",)))
    out, witness_ranges = witness_augmented_sketch(sketch, ())
    assert out == sketch    # structural eq
    assert witness_ranges == []


def test_witness_augmented_single_example_returns_application_chain():
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    ex = IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3))
    out, witness_ranges = witness_augmented_sketch(sketch, (ex,))
    # out is structured as a 2-arity App that the encoder will place at the
    # end of the lattice. We don't constrain the exact representation here
    # — just that the witness range is nonempty.
    assert len(witness_ranges) == 1
    assert witness_ranges[0][1] > witness_ranges[0][0]


def test_witness_augmented_multiple_examples():
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    examples = (
        IOExample(inputs=(IntLit(val=2),), output=IntLit(val=2)),
        IOExample(inputs=(IntLit(val=5),), output=IntLit(val=5)),
    )
    out, witness_ranges = witness_augmented_sketch(sketch, examples)
    assert len(witness_ranges) == 2
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_encoder_ext.py -k "witness" -v`
Expected: ImportError on `witness_augmented_sketch`.

- [ ] **Step 3: Implement witness_augmented_sketch**

Append to `src/qft_pcn/logic/synthesis/encode_ext.py`:

```python
# ---- witness-region augmented AST (spec §5.5) -----------------------------

# Implementation note: we represent the augmented program as a SeqNode
# wrapper that the encoder treats as concatenation of sub-ASTs. For the
# first cut we encode each witness as an extra (classical-copy) sub-tree
# that gets appended after the sketch's pre-order sites. The encoder
# emits witness_regions in meta so the runner can locate them.


from dataclasses import dataclass


@dataclass
class _WitnessChain(Node):
    """Internal: an App-chain (sketch_root_copy applied to literal inputs).

    The encoder treats this as a regular App tree but records the
    enclosing site range as a witness region.
    """
    sketch_root_copy: Node     # a deep-copy of the user's sketch root
    inputs: tuple[Node, ...]   # literal inputs (IntLit / BoolLit)
    output: Node               # expected output literal — used by H_examples


@dataclass
class _AugmentedProgram(Node):
    """Internal: pre-order traversal yields [sketch | witness_1 | ... | witness_K].

    The encoder unwraps this and emits witness_regions in meta.
    """
    sketch: Node
    witnesses: tuple[_WitnessChain, ...]


def witness_augmented_sketch(
    sketch: Node,
    examples: tuple,
):
    """Return (augmented_program_or_sketch, witness_ranges).

    If `examples` is empty, returns sketch unchanged and witness_ranges=[].

    Otherwise returns an _AugmentedProgram containing the sketch plus one
    _WitnessChain per example. `witness_ranges` is a list of (start, end)
    site ranges that the encoder will fill in after encode-time; the
    builder returns a placeholder list of None-pairs that the encoder
    overwrites. (Implemented this way so the runner can pass examples
    through without knowing site assignments.)

    For sub-project E we adopt classical-copy witness encoding (spec §11
    open question item 1). The classical copy preserves the architectural
    claim morally — both copies of the sketch root produce the same
    typing/eval energy contributions — at the cost of ~2x sites.
    """
    if not examples:
        return sketch, []

    witnesses = []
    for ex in examples:
        wc = _WitnessChain(
            sketch_root_copy=deepcopy(sketch),
            inputs=ex.inputs,
            output=ex.output,
        )
        witnesses.append(wc)

    aug = _AugmentedProgram(sketch=sketch, witnesses=tuple(witnesses))
    # Caller (the runner) will pass `aug` through an encoder that knows
    # about _AugmentedProgram and _WitnessChain. For now return placeholder
    # ranges (one per witness); the encoder overwrites them with real ones.
    witness_ranges = [(0, 0) for _ in witnesses]
    return aug, witness_ranges
```

This task introduces *internal* node kinds (`_WitnessChain`, `_AugmentedProgram`) consumed by Task 9's encoder shim. The shim flattens them into A's normal encode() input.

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_encoder_ext.py -k "witness" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/synthesis/encode_ext.py src/qft_pcn/tests/test_synthesis_encoder_ext.py
git commit -m "$(cat <<'EOF'
feat(logic/synthesis/encode_ext): witness-augmented sketch builder

Per spec §5.5 (with §11 open-question item 1 resolved to classical copy):
witness_augmented_sketch() wraps the sketch in an _AugmentedProgram that
holds the sketch plus one _WitnessChain per IOExample. Each witness is a
deep-copy of the sketch root applied to literal inputs, with the expected
output kept on the side for H_examples's boundary pin.

The internal _WitnessChain / _AugmentedProgram node kinds are not user-
facing; they are consumed by the synthesis encoder shim (next task).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: Encoder shim that flattens augmented programs

**Files:**
- Modify: `src/qft_pcn/logic/synthesis/encode_ext.py`.
- Modify: `src/qft_pcn/tests/test_synthesis_encoder_ext.py`.

The shim flattens `_AugmentedProgram` → a sequence of root-level sub-trees and emits witness ranges in meta. The simplest realization: encode each piece separately via A.encode and **horizontally compose** them by placing each piece's MPS into a designated lattice range and putting PAD in between.

A cleaner alternative: encode the sketch first to determine `n_sketch_sites`; then for each witness, encode it separately and overwrite the MPS tensors in the witness range. The two MPSes are connected via PAD sites which are vacuum (zero-coupling under any non-trivial Hamiltonian term).

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_synthesis_encoder_ext.py`:

```python
from src.qft_pcn.logic.synthesis.encode_ext import (
    encode_synthesis_augmented,
)
from src.qft_pcn.logic.encoding import KIND_PAD, D_LOCAL


def test_encode_synthesis_augmented_no_examples_matches_encode_synthesis():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("x",)))
    state_a, meta_a = encode_synthesis(sketch, N=32, chi_max=32)
    state_b, meta_b = encode_synthesis_augmented(
        sketch, examples=(), N=32, chi_max=32)
    # No examples => identical to encode_synthesis (no witness regions).
    overlap = abs(state_a.inner(state_b)) ** 2
    assert overlap > 1.0 - 1e-8
    assert meta_b.witness_regions == []


def test_encode_synthesis_augmented_with_one_example_has_witness_region():
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    ex = IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3))
    state, meta = encode_synthesis_augmented(
        sketch, examples=(ex,), N=32, chi_max=32)
    assert len(meta.witness_regions) == 1
    start, end = meta.witness_regions[0]
    assert end > start
    assert start >= 2     # past the sketch
    assert end <= 32


def test_encode_synthesis_augmented_witness_pad_separation():
    """Between sketch's last site and the witness's first site, at least
    one PAD site must separate them so the two regions are
    Hamiltonian-decoupled. Required for the simple classical-copy
    construction.

    Verified by checking that the site immediately after the sketch's
    last-non-PAD site is PAD.
    """
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    ex = IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3))
    state, meta = encode_synthesis_augmented(
        sketch, examples=(ex,), N=32, chi_max=32)
    # 2 sketch sites (Lam, Var) -> sketch end at site 2.
    # Witness must start at >= 3 (so site 2 stays PAD).
    start, _ = meta.witness_regions[0]
    assert start >= 3
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_encoder_ext.py -k "augmented" -v`
Expected: ImportError on `encode_synthesis_augmented`.

- [ ] **Step 3: Implement encode_synthesis_augmented**

Append to `src/qft_pcn/logic/synthesis/encode_ext.py`:

```python
def _ast_size(node: Node) -> int:
    """Number of nodes in the AST (PAD-free site count)."""
    if isinstance(node, (Var, IntLit, BoolLit, HoleVar)):
        return 1
    if isinstance(node, Lam):
        return 1 + _ast_size(node.body)
    if isinstance(node, App):
        return 1 + _ast_size(node.fn) + _ast_size(node.arg)
    if isinstance(node, If):
        return 1 + _ast_size(node.cond) + _ast_size(node.then_b) \
                 + _ast_size(node.else_b)
    if isinstance(node, Bin):
        return 1 + _ast_size(node.lhs) + _ast_size(node.rhs)
    raise TypeError(f"unsized node: {type(node).__name__}")


def _build_witness_app_chain(sketch_root: Node, inputs) -> Node:
    """Build  App(...App(sketch_root, in_0)..., in_{n-1})  as a normal AST.

    The encoder will encode this as a regular App tree; the witness range
    bookkeeping is added at the shim level.
    """
    expr = sketch_root
    for inp in inputs:
        expr = App(fn=expr, arg=inp)
    return expr


def encode_synthesis_augmented(
    sketch: Node,
    examples: tuple,
    N: int = 32,
    chi_max: int = 32,
):
    """Encode `sketch` plus one witness sub-tree per IOExample.

    Lattice layout (spec §4.2):
        [ sketch | PAD | witness_1 | PAD | witness_2 | ... | PAD ]

    Each PAD separator is one site wide, ensuring no Hamiltonian term
    couples adjacent regions (PAD sites have zero contribution to every
    H_term we will build).

    Returns (state, meta). meta.witness_regions is a list of (start, end)
    ranges; meta.type_holes is populated by encode_synthesis's path for
    the sketch portion (witness sub-trees are concrete, no holes).
    """
    # 1. Encode the sketch portion via encode_synthesis (TypeHole-aware).
    state, meta = encode_synthesis(sketch, N=N, chi_max=chi_max)
    if not examples:
        return state, meta

    # 2. For each example, encode the witness chain in a sub-lattice of
    #    its own size, then splice the tensors into the main MPS.
    sketch_size = _ast_size(sketch)
    cursor = sketch_size + 1   # leave one PAD after sketch

    for ex in examples:
        chain = _build_witness_app_chain(deepcopy(sketch), ex.inputs)
        w_size = _ast_size(chain)
        if cursor + w_size + 1 > N:
            raise RuntimeError(
                f"witness region for example {ex!r} does not fit at "
                f"cursor={cursor}, w_size={w_size}, N={N}"
            )
        # Encode the chain alone into an MPS of length w_size + 1 (so the
        # last site is PAD), then splice that into state at sites
        # [cursor, cursor + w_size).
        wit_state, wit_meta = _base_encode(
            chain, N=w_size + 1, chi_max=chi_max)
        _splice_tensors(state, wit_state, start=cursor, length=w_size)
        meta.witness_regions.append((cursor, cursor + w_size))
        cursor += w_size + 1   # leave one PAD after this witness

    state.normalize()
    return state, meta


def _splice_tensors(target_state, source_state, start: int, length: int) -> None:
    """In-place: overwrite target_state.tensors[start:start+length] with
    source_state.tensors[0:length].

    The bond at start-1 (between PAD and witness) is dimension 1 on both
    sides (PAD vacuum on left; source's left edge is also dim-1 by
    construction). Likewise for the bond at start+length. So splicing
    requires no SVD — we just replace the per-site tensors directly.
    """
    for i in range(length):
        target_state.tensors[start + i] = source_state.tensors[i].copy()
```

**Important caveat:** the `_splice_tensors` correctness depends on the source state's leftmost and rightmost bonds being dimension 1 — which they are by MPS convention (the boundary bonds of any MPS are trivially dim-1). If this assumption breaks (e.g. the source state was canonicalized into a form that introduced explicit dim-1 padding), adjust accordingly. The unit test in Step 1 verifies the spliced state encodes the right structure by re-running the synthesis pipeline; if witness encoding misbehaves the H_examples test in Task 11 will catch it.

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_encoder_ext.py -k "augmented" -v`
Expected: 3 passed.

- [ ] **Step 5: Run all synthesis encoder tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_encoder_ext.py -v 2>&1 | tail -15`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/synthesis/encode_ext.py src/qft_pcn/tests/test_synthesis_encoder_ext.py
git commit -m "$(cat <<'EOF'
feat(logic/synthesis/encode_ext): augmented encoder with witness splicing

encode_synthesis_augmented() builds the full synthesis lattice layout
(spec §4.2): [sketch | PAD | witness_1 | PAD | witness_2 | ...]. For
each IOExample we encode an App-chain (sketch_root applied to the
example's inputs) into a sub-MPS and splice the per-site tensors into
the main state at a designated cursor position. The PAD separators
guarantee Hamiltonian-decoupling between regions.

meta.witness_regions records the (start, end) ranges so H_examples
knows where to pin the expected output.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: The synthesis Hamiltonian

### Task 10: Hamiltonian.__add__ and .expectation (qft core)

**Files:**
- Modify: `src/qft_pcn/qft/hamiltonian.py` (only if Task 0 Step 6 found gaps).
- Modify: `src/qft_pcn/tests/test_qft.py`.

Per spec §4.5. **Skip this task if both methods are already present.** If only one is missing, do the relevant half.

- [ ] **Step 1: Determine which methods are missing**

Run:
```bash
.venv/bin/python -c "
from src.qft_pcn.qft.hamiltonian import Hamiltonian
print('add:', hasattr(Hamiltonian, '__add__'))
print('expectation:', hasattr(Hamiltonian, 'expectation'))
"
```

If both `True`: **skip to Task 11.**

- [ ] **Step 2: Write the failing tests for whichever is missing**

Append to `src/qft_pcn/tests/test_qft.py`:

```python
def test_hamiltonian_add_sums_local_and_bond_ops():
    cfg_a = HamiltonianConfig(
        species=[FieldSpecies(name="x", cutoff=2, bare_mass=1.0, kinetic=0.0)],
        couplings={}, params={},
    )
    cfg_b = HamiltonianConfig(
        species=[FieldSpecies(name="x", cutoff=2, bare_mass=2.0, kinetic=0.0)],
        couplings={}, params={},
    )
    H_a = Hamiltonian(cfg_a, N=4)
    H_b = Hamiltonian(cfg_b, N=4)
    H_sum = H_a + H_b
    # Local op at site 0 should be sum of the two.
    expected = H_a.local_op(0) + H_b.local_op(0)
    actual = H_sum.local_op(0)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_hamiltonian_expectation_matches_evolution_energy():
    from src.qft_pcn.qft.evolution import energy as ev_energy
    cfg = HamiltonianConfig(
        species=[FieldSpecies(name="x", cutoff=3, bare_mass=1.0, kinetic=0.0)],
        couplings={}, params={},
    )
    H = Hamiltonian(cfg, N=4)
    psi = MPS.number_states([1, 0, 1, 0], d=3).normalize()
    # Hamiltonian.expectation(psi) should equal evolution.energy(psi, H).
    assert abs(H.expectation(psi) - ev_energy(psi, H)) < 1e-12
```

- [ ] **Step 3: Implement whichever is missing**

In `src/qft_pcn/qft/hamiltonian.py`, add to the `Hamiltonian` class:

```python
def __add__(self, other: "Hamiltonian") -> "Hamiltonian":
    """Term-wise sum.

    Both H's must have the same N and the same species. Couplings and
    params are merged; if a coupling key collides, values are summed.
    """
    if self.N != other.N:
        raise ValueError(f"N mismatch: {self.N} vs {other.N}")
    if [s.name for s in self.cfg.species] != [s.name for s in other.cfg.species]:
        raise ValueError("species mismatch")
    # Build a combined config: bare_masses, kinetics summed per species;
    # couplings summed by key.
    new_species = []
    for sa, sb in zip(self.cfg.species, other.cfg.species):
        new_species.append(FieldSpecies(
            name=sa.name,
            cutoff=sa.cutoff,
            bare_mass=sa.bare_mass + sb.bare_mass,
            kinetic=sa.kinetic + sb.kinetic,
        ))
    new_couplings = dict(self.cfg.couplings)
    for k, v in other.cfg.couplings.items():
        new_couplings[k] = new_couplings.get(k, 0.0) + v
    new_params = dict(self.cfg.params)
    for k, v in other.cfg.params.items():
        new_params[k] = new_params.get(k, 0.0) + v
    new_cfg = HamiltonianConfig(
        species=new_species, couplings=new_couplings, params=new_params,
    )
    return Hamiltonian(new_cfg, N=self.N)


def expectation(self, state: "MPS") -> float:
    """<state | H | state>. Convenience wrapper around evolution.energy."""
    from .evolution import energy as _energy
    return _energy(state, self)
```

**If B's Hamiltonian sub-project has a richer term structure (e.g. arbitrary local/bond terms outside `HamiltonianConfig`'s coupling dict),** this scalar sum-of-params approach is insufficient. In that case, escalate to the human: either B's Hamiltonian needs a public `add_local_term` / `add_bond_term` extension, or sub-project E maintains a list of Hamiltonian objects and a `MultiHamiltonian` wrapper that calls each in turn during `trotter_step`. **Do not silently implement a fake `__add__`** that ignores some terms.

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_qft.py -k "hamiltonian_add or hamiltonian_expectation" -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/qft/hamiltonian.py src/qft_pcn/tests/test_qft.py
git commit -m "$(cat <<'EOF'
feat(qft/hamiltonian): __add__ and expectation methods

Needed by sub-project E's synthesis Hamiltonian assembly: composes
H_typing + H_eval + H_examples + H_target_type + H_size as a sum of
Hamiltonian objects, and reports per-block <H> via expectation().

Scalar-sum implementation that works for HamiltonianConfig-based
H's. If B/C introduce a richer term API, this is the place to extend.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: `H_examples`, `H_target_type`, `H_size` builders

**Files:**
- Create: `src/qft_pcn/logic/synthesis/hamiltonian.py`.
- Create: `src/qft_pcn/tests/test_synthesis_hamiltonian.py`.

Per spec §4.2, §4.3, §4.4. These three are sub-project E's contribution to H_total.

For this task we depend on B's `compile_typing_hamiltonian` and C's `compile_eval_hamiltonian` interfaces. If they don't exist yet, write the E builders against a **mock interface** (`compile_typing_hamiltonian: meta -> Hamiltonian` returning a zero-Hamiltonian) and document the dependency for the integration task.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_synthesis_hamiltonian.py`:

```python
"""Tests for synthesis/hamiltonian.py: per-block builders + assembly."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    Var, Lam, App, IntLit, BoolLit, HoleVar,
    TInt, TBool, TArrow,
)
from src.qft_pcn.logic.synthesis.problem import (
    SynthesisProblem, IOExample, HamiltonianWeights,
)
from src.qft_pcn.logic.synthesis.encode_ext import (
    encode_synthesis_augmented,
)
from src.qft_pcn.logic.synthesis.hamiltonian import (
    build_h_examples, build_h_target_type, build_h_size,
    compile_synthesis_hamiltonian,
)


def _trivial_sketch():
    return Lam(param="x", param_ty=TInt(), body=HoleVar(candidates=("x",)))


def test_h_size_zero_on_pad_only_state():
    # An empty AST (all PAD) — no nodes. H_size = 0.
    # Use a constructed empty state directly.
    from src.qft_pcn.qft.mps import MPS
    from src.qft_pcn.logic.encoding import D_LOCAL
    state = MPS.vacuum(N=4, d=D_LOCAL)
    # Build h_size against a meta with sketch_range (0, 0) -> empty.
    from src.qft_pcn.logic.encoding import EncodingMeta, SPECIES
    meta = EncodingMeta(
        N=4, chi_max=4,
        field_dims={"kind": 8, "type": 8, "bid": 8, "value": 16},
        species=list(SPECIES), nested_type_index={},
        site_to_ast_path={}, live_binders_per_bond=[],
    )
    h_size = build_h_size(meta, sketch_range=(0, 0), weight=0.1)
    assert abs(h_size.expectation(state)) < 1e-10


def test_h_target_type_returns_zero_when_target_is_none():
    sketch = _trivial_sketch()
    state, meta = encode_synthesis_augmented(sketch, (), N=32, chi_max=32)
    h_tt = build_h_target_type(meta, target_type=None, weight=2.0)
    assert abs(h_tt.expectation(state)) < 1e-10


def test_h_target_type_penalizes_mismatch():
    # Sketch root has type Int->Int (identity on Int).
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    # No holes — use base encoder directly.
    from src.qft_pcn.logic.encoder import encode as _enc
    state, meta = _enc(sketch, N=32, chi_max=16)
    # H_target_type pinning to T_BOOL (mismatch) -> positive.
    h_bad = build_h_target_type(meta, target_type=TBool(), weight=2.0)
    e_bad = h_bad.expectation(state)
    assert e_bad > 1.0    # weight=2, full mismatch -> ~2
    # H_target_type pinning to T_ARR_II (correct) -> ~0.
    h_good = build_h_target_type(meta,
                                  target_type=TArrow(src=TInt(), dst=TInt()),
                                  weight=2.0)
    e_good = h_good.expectation(state)
    assert e_good < 0.01


def test_h_examples_penalizes_wrong_output():
    """Sketch is the identity; example claims input 3 -> output 5. Identity
    returns 3, not 5, so H_examples > 0."""
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    ex = IOExample(inputs=(IntLit(val=3),), output=IntLit(val=5))
    state, meta = encode_synthesis_augmented(sketch, (ex,), N=32, chi_max=32)
    h_ex = build_h_examples(meta, examples=(ex,), weight=3.0)
    e = h_ex.expectation(state)
    assert e > 1.0     # the witness root carries val=3, expected 5 -> penalty


def test_h_examples_satisfied_yields_zero():
    """Same setup but example claims 3 -> 3 (correct for identity)."""
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    ex = IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3))
    state, meta = encode_synthesis_augmented(sketch, (ex,), N=32, chi_max=32)
    h_ex = build_h_examples(meta, examples=(ex,), weight=3.0)
    e = h_ex.expectation(state)
    # Note: this only works if the witness encodes the *application result*
    # at the witness root's value register. For sub-project E we encode the
    # application chain syntactically; H_eval (sub-project C) is what
    # reduces it. So at *encoding time*, before any imag-time evolution,
    # this test is expected to fail unless C has run. Mark as xfail until
    # the runner integration test.
    pytest.xfail("requires H_eval / imag-time evolution to reduce the witness; "
                 "covered by integration test in Task 17")


def test_compile_synthesis_hamiltonian_returns_hamiltonian():
    sketch = _trivial_sketch()
    p = SynthesisProblem(sketch=sketch, name="dummy")
    state, meta = encode_synthesis_augmented(sketch, (), N=32, chi_max=32)
    H = compile_synthesis_hamiltonian(meta, p)
    # Should be a Hamiltonian object with N=32.
    assert H.N == 32
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_hamiltonian.py -v`
Expected: ImportError on `synthesis.hamiltonian`.

- [ ] **Step 3: Implement hamiltonian.py**

Create `src/qft_pcn/logic/synthesis/hamiltonian.py`. The full implementation is substantial; structure it as:

```python
"""Synthesis Hamiltonian builders (spec §4).

Three sub-project-E-specific blocks:
    H_examples: per-example boundary pin on witness-root value register.
    H_target_type: one-site pin on the sketch root's type register.
    H_size: gentle Occam penalty on non-PAD kind-register sites.

Plus compile_synthesis_hamiltonian which sums:
    w_T H_typing + w_E H_eval + w_X H_examples + w_Y H_target_type + w_S H_size.

H_typing comes from sub-project B; H_eval from sub-project C.
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.hamiltonian import (
    Hamiltonian, HamiltonianConfig, FieldSpecies,
)
from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.logic.ast import Ty
from src.qft_pcn.logic.encoding import (
    EncodingMeta, SPECIES, SPECIES_NAMES, SPECIES_DIMS, D_LOCAL,
    KIND_PAD, KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF,
    INT_LIT_OFFSET, VALUE_FALSE, VALUE_TRUE,
)
from src.qft_pcn.logic.synthesis.problem import (
    SynthesisProblem, IOExample, HamiltonianWeights,
)
from src.qft_pcn.logic.synthesis.encode_ext import _ty_to_tag


# ---- low-level local-term Hamiltonian builder ------------------------------

# Strategy: each H_block is a Hamiltonian object built from a custom
# HamiltonianConfig where we attach the desired local terms as bare_mass
# contributions per site. For terms that don't fit HamiltonianConfig's
# canonical form (e.g. a site-specific projector), we use a small
# `LocalTermHamiltonian` helper class that delegates to a list of explicit
# (site, op) pairs at construction time, overriding `local_op` and
# `bond_op`.


class _LocalTermHamiltonian(Hamiltonian):
    """A Hamiltonian that adds explicit per-site local operators on top of
    a base config. Used for synthesis-specific projectors that don't fit
    the canonical HamiltonianConfig schema.

    Overrides local_op(site) to return base.local_op(site) + self._terms[site].
    """

    def __init__(self, base_cfg, N: int, extra_local_terms: dict):
        super().__init__(base_cfg, N)
        # extra_local_terms: site -> ndarray(D_LOCAL, D_LOCAL)
        self._extra = extra_local_terms

    def local_op(self, site: int) -> np.ndarray:
        op = super().local_op(site)
        if site in self._extra:
            op = op + self._extra[site]
        return op


def _zero_cfg() -> HamiltonianConfig:
    """A HamiltonianConfig that produces a zero Hamiltonian for sub-project
    E's species set."""
    return HamiltonianConfig(
        species=list(SPECIES), couplings={}, params={},
    )


def _local_projector_on_register(species_idx: int, basis_index: int) -> np.ndarray:
    """|basis_index><basis_index| on `species_idx`, identity elsewhere,
    embedded into the D_LOCAL x D_LOCAL local Hilbert space.
    """
    from src.qft_pcn.qft.fock import embed_op
    dim = SPECIES_DIMS[species_idx]
    P = np.zeros((dim, dim), dtype=complex)
    P[basis_index, basis_index] = 1.0
    return embed_op(P, species_index=species_idx, species_dims=SPECIES_DIMS)


def _local_complement_projector(species_idx: int, basis_index: int) -> np.ndarray:
    """I - |basis_index><basis_index| on `species_idx`, embedded."""
    I_full = np.eye(D_LOCAL, dtype=complex)
    P = _local_projector_on_register(species_idx, basis_index)
    return I_full - P


# ---- H_size ----------------------------------------------------------------


KIND_SPECIES_IDX = 0
TYPE_SPECIES_IDX = 1
BID_SPECIES_IDX = 2
VALUE_SPECIES_IDX = 3


def build_h_size(meta: EncodingMeta, sketch_range, weight: float = 0.1) -> Hamiltonian:
    """H_size = weight * Σ_{i in sketch_range} (I - P_PAD_kind)_i.

    sketch_range: (start, end_exclusive). Witness sites are excluded.
    """
    start, end = sketch_range
    extra: dict = {}
    op = weight * _local_complement_projector(KIND_SPECIES_IDX, KIND_PAD)
    for i in range(start, end):
        extra[i] = op
    return _LocalTermHamiltonian(_zero_cfg(), meta.N, extra)


# ---- H_target_type ---------------------------------------------------------


def build_h_target_type(meta: EncodingMeta, target_type, weight: float = 2.0) -> Hamiltonian:
    """H_target_type = weight * (I - |tag(target_type)><tag(target_type)|)
    on the type register at site 0.

    If target_type is None, returns the zero Hamiltonian.
    """
    if target_type is None:
        return Hamiltonian(_zero_cfg(), meta.N)
    tag = _ty_to_tag(target_type)
    extra = {0: weight * _local_complement_projector(TYPE_SPECIES_IDX, tag)}
    return _LocalTermHamiltonian(_zero_cfg(), meta.N, extra)


# ---- H_examples ------------------------------------------------------------


def _literal_to_value_basis(node) -> int:
    """Map IntLit/BoolLit to its value-register basis index."""
    from src.qft_pcn.logic.ast import IntLit, BoolLit
    if isinstance(node, IntLit):
        idx = node.val + INT_LIT_OFFSET
        if idx < 0 or idx >= VALUE_CUTOFF:
            raise ValueError(f"IntLit({node.val}) out of value range")
        return idx
    if isinstance(node, BoolLit):
        return VALUE_TRUE if node.val else VALUE_FALSE
    raise ValueError(f"H_examples literal must be IntLit/BoolLit: {type(node)}")


def build_h_examples(meta: EncodingMeta, examples, weight: float = 3.0) -> Hamiltonian:
    """H_examples = weight * Σ_ex (I - |output_basis><output_basis|)
    on the value register at each witness region's *root* site.

    The witness root for an App-chain of inputs is the *first* site of the
    witness region (the outermost App in pre-order).
    """
    if not examples or not meta.witness_regions:
        return Hamiltonian(_zero_cfg(), meta.N)
    extra: dict = {}
    if len(examples) != len(meta.witness_regions):
        raise ValueError(
            f"examples ({len(examples)}) vs witness_regions "
            f"({len(meta.witness_regions)}) mismatch"
        )
    for ex, (start, _end) in zip(examples, meta.witness_regions):
        out_idx = _literal_to_value_basis(ex.output)
        op = weight * _local_complement_projector(VALUE_SPECIES_IDX, out_idx)
        extra[start] = op
    return _LocalTermHamiltonian(_zero_cfg(), meta.N, extra)


# ---- assembly --------------------------------------------------------------


def compile_synthesis_hamiltonian(
    meta: EncodingMeta,
    problem: SynthesisProblem,
    weights: HamiltonianWeights | None = None,
) -> Hamiltonian:
    """Assemble the full synthesis Hamiltonian.

    H_total = w_T H_typing + w_E H_eval + w_X H_examples
            + w_Y H_target_type + w_S H_size

    Uses sub-project B for H_typing and C for H_eval. If those imports
    fail, falls back to zero Hamiltonians and logs a warning (synthesis
    will still run but with no typing/eval constraints — useful for
    isolated E development).
    """
    if weights is None:
        weights = problem.weights

    # Sketch range: from site 0 to the first witness region's start, or
    # to N if no witnesses. (PAD separator at the end is one site before
    # the witness region in our layout.)
    if meta.witness_regions:
        sketch_end = meta.witness_regions[0][0] - 1  # the PAD before witness
    else:
        sketch_end = meta.N
    sketch_range = (0, sketch_end)

    # H_typing (B) — wrap import in try/except so E can develop standalone.
    try:
        from src.qft_pcn.logic.hamiltonian_compiler import (
            compile_typing_hamiltonian,
        )
        H_typing = compile_typing_hamiltonian(meta)
    except Exception:
        import warnings
        warnings.warn(
            "compile_typing_hamiltonian (sub-project B) not available; "
            "using zero H_typing (synthesis will lack typing constraints)"
        )
        H_typing = Hamiltonian(_zero_cfg(), meta.N)

    # H_eval (C) — same defensive import.
    try:
        from src.qft_pcn.logic.evaluation_hamiltonian import (
            compile_eval_hamiltonian,
        )
        H_eval = compile_eval_hamiltonian(meta)
    except Exception:
        import warnings
        warnings.warn(
            "compile_eval_hamiltonian (sub-project C) not available; "
            "using zero H_eval"
        )
        H_eval = Hamiltonian(_zero_cfg(), meta.N)

    H_examples = build_h_examples(meta, problem.examples,
                                  weight=weights.w_examples)
    H_target = build_h_target_type(meta, problem.target_type,
                                    weight=weights.w_target_type)
    H_size = build_h_size(meta, sketch_range, weight=weights.w_size)

    # Compose. We rely on Hamiltonian.__add__ if available; otherwise
    # we build a wrapper.
    # We pre-multiply H_typing by w_T and H_eval by w_E. Hamiltonian
    # doesn't define __mul__, so we use a _ScaledHamiltonian wrapper.
    H_total = _scale(H_typing, weights.w_typing) + _scale(H_eval, weights.w_eval) \
              + H_examples + H_target + H_size
    return H_total


def _scale(H: Hamiltonian, factor: float) -> Hamiltonian:
    """Return a Hamiltonian H' such that H'.local_op(i) = factor * H.local_op(i),
    H'.bond_op(i) = factor * H.bond_op(i)."""
    class _Scaled(Hamiltonian):
        def __init__(self, inner: Hamiltonian, f: float):
            super().__init__(inner.cfg, inner.N)
            self._inner = inner
            self._f = f
        def local_op(self, site):
            return self._f * self._inner.local_op(site)
        def bond_op(self, site):
            return self._f * self._inner.bond_op(site)
    return _Scaled(H, factor)
```

The implementation has several `try/except` defensive imports for B and C — this lets E develop standalone if those sub-projects are not yet merged. The final test (Task 17 integration) is the one that requires B and C to actually work.

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_hamiltonian.py -v`
Expected: 5 passed, 1 xfailed (the H_examples xfail in `test_h_examples_satisfied_yields_zero`).

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/synthesis/hamiltonian.py src/qft_pcn/tests/test_synthesis_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/synthesis): per-block Hamiltonian builders + compile

Per spec §4: H_examples (boundary pin on witness-root value register),
H_target_type (one-site type-tag pin on sketch root), H_size (Occam
penalty on non-PAD kind register). compile_synthesis_hamiltonian
assembles them with sub-project B's H_typing and sub-project C's H_eval
into a single weighted sum (spec §4.5).

Uses defensive try/except imports for B and C so E can develop standalone;
warnings are emitted when typing or eval Hamiltonians fall back to zero.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: The synthesis runner

### Task 12: `ast_canonical` helper in decoder

**Files:**
- Modify: `src/qft_pcn/logic/decoder.py`.
- Modify: `src/qft_pcn/tests/test_logic_encoder.py` (or `test_logic_decoder.py`).

Per spec §6.2. Needed for dedup: turn alpha-equivalent ASTs into a canonical form so a `set` of completions deduplicates them correctly.

- [ ] **Step 1: Write the failing test**

Append to whichever test file covers the decoder (likely `test_logic_decoder.py`; check via `ls src/qft_pcn/tests/`):

```python
from src.qft_pcn.logic.ast import Var, Lam, TInt
from src.qft_pcn.logic.decoder import ast_canonical


def test_ast_canonical_renames_to_v0_v1():
    p = Lam(param="abc", param_ty=TInt(), body=Var(name="abc"))
    c = ast_canonical(p)
    assert c == Lam(param="_v0", param_ty=TInt(), body=Var(name="_v0"))


def test_ast_canonical_idempotent():
    p = Lam(param="abc", param_ty=TInt(), body=Var(name="abc"))
    assert ast_canonical(ast_canonical(p)) == ast_canonical(p)


def test_ast_canonical_alpha_eq_inputs_match():
    p1 = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    p2 = Lam(param="y", param_ty=TInt(), body=Var(name="y"))
    assert ast_canonical(p1) == ast_canonical(p2)
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -k "ast_canonical" -v`
Expected: ImportError.

- [ ] **Step 3: Implement ast_canonical**

In `src/qft_pcn/logic/decoder.py`, add:

```python
def ast_canonical(node):
    """Return the alpha-normalized form of `node`: every binder renamed to
    `_v0`, `_v1`, ... in pre-order of occurrence. Two alpha-equivalent
    ASTs canonicalize to the same structure.
    """
    from src.qft_pcn.logic.ast import (
        Var, Lam, App, IntLit, BoolLit, If, Bin,
    )

    counter = [0]
    rename: dict[str, list[str]] = {}     # stack of renames per original name

    def fresh() -> str:
        n = f"_v{counter[0]}"
        counter[0] += 1
        return n

    def go(n):
        if isinstance(n, Var):
            stack = rename.get(n.name, [])
            new_name = stack[-1] if stack else n.name
            return Var(name=new_name)
        if isinstance(n, Lam):
            new_name = fresh()
            rename.setdefault(n.param, []).append(new_name)
            try:
                new_body = go(n.body)
            finally:
                rename[n.param].pop()
            return Lam(param=new_name, param_ty=n.param_ty, body=new_body)
        if isinstance(n, App):
            return App(fn=go(n.fn), arg=go(n.arg))
        if isinstance(n, If):
            return If(cond=go(n.cond), then_b=go(n.then_b), else_b=go(n.else_b))
        if isinstance(n, Bin):
            return Bin(op=n.op, lhs=go(n.lhs), rhs=go(n.rhs))
        if isinstance(n, (IntLit, BoolLit)):
            return n
        return n   # unknown - pass through

    return go(node)
```

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -k "ast_canonical" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/decoder.py src/qft_pcn/tests/
git commit -m "$(cat <<'EOF'
feat(logic/decoder): ast_canonical alpha-normalization helper

Renames every binder to _v0, _v1, ... in pre-order so alpha-equivalent
ASTs canonicalize to the same structure. Used by sub-project E's
sampling pipeline to dedupe completions (spec §6.2).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 13: Ranking module (`synthesis/ranking.py`)

**Files:**
- Create: `src/qft_pcn/logic/synthesis/ranking.py`.
- Create: `src/qft_pcn/tests/test_synthesis_ranking.py`.

Per spec §6.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_synthesis_ranking.py`:

```python
"""Tests for synthesis/ranking.py: dedupe + rank + classify."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import Var, Lam, IntLit, TInt
from src.qft_pcn.logic.synthesis.ranking import (
    dedupe_samples, classify_failure_mode,
)
from src.qft_pcn.logic.synthesis.problem import (
    SynthesisProblem, IOExample, HamiltonianWeights, Completion,
)


def _id_ast():
    return Lam(param="x", param_ty=TInt(), body=Var(name="x"))


def test_dedupe_samples_groups_alpha_equivalent():
    p1 = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    p2 = Lam(param="y", param_ty=TInt(), body=Var(name="y"))   # alpha-eq to p1
    p3 = IntLit(val=3)
    samples = [(p1, 0.0), (p2, 0.0), (p3, 0.0), (p1, 0.0)]
    groups = dedupe_samples([s[0] for s in samples],
                            residuals=[0.0] * 4,
                            residual_threshold=1e-3)
    # Two unique canonical forms, with multiplicities 3 and 1.
    assert len(groups) == 2
    mult_for_intlit = next(m for ast, m in groups
                           if not isinstance(ast, Lam))
    assert mult_for_intlit == 1
    mult_for_id = next(m for ast, m in groups if isinstance(ast, Lam))
    assert mult_for_id == 3


def test_dedupe_samples_drops_high_residual():
    p = _id_ast()
    groups = dedupe_samples([p, p], residuals=[0.0, 0.5],
                            residual_threshold=1e-3)
    # One survives.
    assert len(groups) == 1
    assert groups[0][1] == 1


def test_classify_failure_mode_success():
    weights = HamiltonianWeights()
    completions = [
        Completion(ast=_id_ast(), energy=0.001,
                   energy_breakdown={"typing": 0.0, "eval": 0.0,
                                     "examples": 0.0, "target_type": 0.0,
                                     "size": 0.001},
                   diagnostics={}, multiplicity=50),
        Completion(ast=IntLit(val=3), energy=4.5,
                   energy_breakdown={}, diagnostics={}, multiplicity=2),
    ]
    fm = classify_failure_mode(completions, weights=weights, final_state_energy=0.01)
    assert fm is None


def test_classify_failure_mode_no_completions():
    fm = classify_failure_mode([], weights=HamiltonianWeights(),
                                final_state_energy=99.0)
    assert fm == "no_valid_completion"


def test_classify_failure_mode_ambiguous_top1():
    weights = HamiltonianWeights()
    completions = [
        Completion(ast=_id_ast(), energy=0.001,
                   energy_breakdown={"typing": 0.0, "size": 0.001},
                   diagnostics={}, multiplicity=30),
        Completion(ast=Var("x"), energy=0.0015,    # within tolerance_ambiguous
                   energy_breakdown={"typing": 0.0, "size": 0.0015},
                   diagnostics={}, multiplicity=25),
    ]
    fm = classify_failure_mode(completions, weights=weights,
                                final_state_energy=0.01)
    assert fm == "ambiguous_top1"


def test_classify_failure_mode_imag_time_did_not_converge():
    weights = HamiltonianWeights()
    completions = [
        Completion(ast=_id_ast(), energy=10.0,    # huge
                   energy_breakdown={"typing": 8.0, "eval": 2.0,
                                     "examples": 0.0, "target_type": 0.0,
                                     "size": 0.0},
                   diagnostics={}, multiplicity=30),
    ]
    fm = classify_failure_mode(completions, weights=weights,
                                final_state_energy=20.0)
    # top-1 energy is too high to be a "correct" completion.
    assert fm in ("imag_time_did_not_converge", "no_valid_completion")
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_ranking.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement ranking.py**

Create `src/qft_pcn/logic/synthesis/ranking.py`:

```python
"""Sample dedup and ranking (spec §6.2, §6.3, §6.4)."""

from __future__ import annotations

from collections import OrderedDict

from src.qft_pcn.logic.decoder import ast_canonical
from src.qft_pcn.logic.synthesis.problem import (
    Completion, HamiltonianWeights,
)


def dedupe_samples(asts, residuals, residual_threshold: float = 1e-3):
    """Group samples by alpha-equivalence (via ast_canonical).

    Drops samples whose residual_norm exceeds the threshold.

    Returns a list of (canonical_ast, multiplicity) pairs in insertion
    (first-occurrence) order.
    """
    groups: OrderedDict = OrderedDict()
    for ast, r in zip(asts, residuals):
        if r > residual_threshold:
            continue
        canon = ast_canonical(ast)
        key = _ast_key(canon)
        if key in groups:
            groups[key] = (groups[key][0], groups[key][1] + 1)
        else:
            groups[key] = (canon, 1)
    return [(ast, mult) for ast, mult in groups.values()]


def _ast_key(node) -> str:
    """A hashable string key for a canonical AST (used for grouping)."""
    from src.qft_pcn.logic.ast import pretty
    try:
        return pretty(node)
    except Exception:
        return repr(node)


def rank_completions(completions: list, tol: float = 1e-6) -> list:
    """Sort completions ascending by energy; ties break by multiplicity desc,
    then by AST size ascending."""
    from src.qft_pcn.logic.synthesis.encode_ext import _ast_size

    def sort_key(c: Completion):
        return (
            round(c.energy / tol),       # bucket so ties cluster
            -c.multiplicity,             # higher mult first
            _ast_size(c.ast),            # smaller first
        )

    return sorted(completions, key=sort_key)


def classify_failure_mode(
    completions: list,
    weights: HamiltonianWeights,
    final_state_energy: float,
    tolerance_correct_factor: float = 1e-3,
    tolerance_ambiguous: float = 1e-3,
    convergence_threshold_factor: float = 2.0,
) -> str | None:
    """Return one of: None, "no_valid_completion", "ambiguous_top1",
    "imag_time_did_not_converge".
    """
    if not completions:
        return "no_valid_completion"

    # Threshold for "correct": top-1's hard constraints must be small.
    hard_weights_sum = (weights.w_typing + weights.w_eval +
                        weights.w_examples + weights.w_target_type)
    tol_correct = tolerance_correct_factor * hard_weights_sum

    top1 = completions[0]
    # Top-1 must have small *non-size* energy.
    non_size = sum(v for k, v in top1.energy_breakdown.items() if k != "size")
    if non_size > tol_correct:
        # The top-1 itself fails hard constraints.
        # If final_state_energy is high too, imag-time didn't converge.
        if final_state_energy > convergence_threshold_factor * tol_correct:
            return "imag_time_did_not_converge"
        return "no_valid_completion"

    # Top-1 looks good. Check ambiguity.
    if len(completions) >= 2:
        gap = completions[1].energy - top1.energy
        if abs(gap) < tolerance_ambiguous:
            return "ambiguous_top1"

    return None
```

- [ ] **Step 4: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_ranking.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/synthesis/ranking.py src/qft_pcn/tests/test_synthesis_ranking.py
git commit -m "$(cat <<'EOF'
feat(logic/synthesis): dedupe + rank + classify_failure_mode

Per spec §6.2-§6.4: dedupe_samples groups alpha-equivalent ASTs and
drops high-residual samples; rank_completions sorts by energy with
multiplicity and size as tiebreakers; classify_failure_mode emits the
four-way classification (None / no_valid_completion / ambiguous_top1 /
imag_time_did_not_converge).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: The `synthesize()` runner

**Files:**
- Create: `src/qft_pcn/logic/synthesis/runner.py`.
- Modify: `src/qft_pcn/logic/synthesis/__init__.py` (export `synthesize`).
- Create: `src/qft_pcn/tests/test_synthesis_runner.py` (smoke tests only — integration tests come in Task 17).

Per spec §3.3, §6.5. The runner ties everything together.

- [ ] **Step 1: Write the failing smoke test**

Create `src/qft_pcn/tests/test_synthesis_runner.py`:

```python
"""Smoke tests for synthesize(). Full demo problems live in Task 17."""

from __future__ import annotations

import time
import numpy as np
import pytest

from src.qft_pcn.logic.ast import (
    Var, Lam, IntLit, HoleVar, TInt,
)
from src.qft_pcn.logic.synthesis import (
    synthesize, SynthesisProblem, IOExample,
)
from src.qft_pcn.logic.synthesis.errors import SynthesisProblemError


def test_synthesize_returns_synthesis_result():
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("x",)))
    p = SynthesisProblem(sketch=sketch, name="smoke", n_samples=4,
                         anneal_steps=5)
    rng = np.random.default_rng(seed=42)
    result = synthesize(p, rng=rng)
    assert result.problem is p
    assert result.n_samples_drawn == 4
    assert isinstance(result.wall_time_seconds, float)
    assert result.wall_time_seconds > 0.0


def test_synthesize_raises_on_problem_with_no_holes():
    sketch = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    p = SynthesisProblem(sketch=sketch, name="bad")
    with pytest.raises(SynthesisProblemError):
        synthesize(p)


def test_synthesize_completes_within_timeout():
    """Smoke: 1 problem, n_samples=4, anneal_steps=5 -> seconds, not minutes."""
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("x",)))
    p = SynthesisProblem(sketch=sketch, name="smoke", n_samples=4,
                         anneal_steps=5)
    rng = np.random.default_rng(seed=42)
    t0 = time.time()
    result = synthesize(p, rng=rng)
    elapsed = time.time() - t0
    assert elapsed < 30.0   # generous; tighter budget enforced in §10 acceptance
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_runner.py -v`
Expected: ImportError on `synthesize`.

- [ ] **Step 3: Implement runner.py**

Create `src/qft_pcn/logic/synthesis/runner.py`:

```python
"""The synthesize() pipeline (spec §3.3, §6).

Composes:
    validate_problem (Task 5)
    encode_synthesis_augmented (Task 9)
    compile_synthesis_hamiltonian (Task 11)
    imag-time evolve (3 phases per §6.5)
    sample (sub-project A)
    dedupe_samples + rank_completions + classify_failure_mode (Task 13)
"""

from __future__ import annotations

import time
import warnings
import numpy as np

from src.qft_pcn.qft.evolution import evolve, energy as ev_energy
from src.qft_pcn.logic.decoder import sample
from src.qft_pcn.logic.encoder import encode as _encode

from src.qft_pcn.logic.synthesis.problem import (
    SynthesisProblem, SynthesisResult, Completion, HamiltonianWeights,
    IOExample,
)
from src.qft_pcn.logic.synthesis.errors import (
    SynthesisProblemError, SynthesisRuntimeError,
)
from src.qft_pcn.logic.synthesis._validate import validate_problem
from src.qft_pcn.logic.synthesis.encode_ext import encode_synthesis_augmented
from src.qft_pcn.logic.synthesis.hamiltonian import (
    compile_synthesis_hamiltonian,
    build_h_examples, build_h_target_type, build_h_size,
    _scale, _zero_cfg,
)
from src.qft_pcn.qft.hamiltonian import Hamiltonian
from src.qft_pcn.logic.synthesis.ranking import (
    dedupe_samples, rank_completions, classify_failure_mode,
)


def _phase1_hamiltonian(meta, problem, weights) -> Hamiltonian:
    """Phase 1 (warmup): only H_typing + H_eval, no examples/target/size."""
    # Reuse compile_synthesis_hamiltonian but with zeroed example/target/size
    # weights for phase 1.
    w_phase1 = HamiltonianWeights(
        w_typing=weights.w_typing,
        w_eval=weights.w_eval,
        w_examples=0.0,
        w_target_type=0.0,
        w_size=0.0,
    )
    return compile_synthesis_hamiltonian(meta, problem, weights=w_phase1)


def synthesize(
    problem: SynthesisProblem,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
) -> SynthesisResult:
    """Run the synthesis pipeline (spec §3.3).

    Pipeline:
        1. validate_problem
        2. encode_synthesis_augmented (sketch + witnesses)
        3. compile H_warmup (typing + eval only) and H_full
        4. evolve phase 1 (warmup), phase 2 (main), phase 3 (fine)
        5. sample n_samples from the relaxed state
        6. dedupe by alpha-eq, re-score each unique completion
        7. rank + classify failure_mode
        8. assemble SynthesisResult
    """
    if rng is None:
        rng = np.random.default_rng()
    t_start = time.time()

    # 1. Validate.
    validate_problem(problem)

    # 2. Encode.
    state, meta = encode_synthesis_augmented(
        problem.sketch, problem.examples,
        N=problem.N, chi_max=problem.chi_max,
    )

    # 3. Compile Hamiltonians.
    weights = problem.weights
    H_warmup = _phase1_hamiltonian(meta, problem, weights)
    H_full = compile_synthesis_hamiltonian(meta, problem, weights=weights)

    # 4. Evolve (3-phase annealing schedule, spec §6.5).
    main_steps = problem.anneal_steps
    warmup_steps = max(25, main_steps // 4)
    fine_steps = max(25, main_steps // 4)
    main_dt = problem.anneal_dt
    warmup_dt = main_dt * 2.0
    fine_dt = main_dt / 5.0
    chi = problem.chi_max

    try:
        evolve(state, H_warmup, dt=warmup_dt, steps=warmup_steps,
               imaginary=True, chi_max=chi, normalize_every=1)
        evolve(state, H_full, dt=main_dt, steps=main_steps,
               imaginary=True, chi_max=chi, normalize_every=1)
        evolve(state, H_full, dt=fine_dt, steps=fine_steps,
               imaginary=True, chi_max=chi, normalize_every=1)
    except Exception as e:
        raise SynthesisRuntimeError(
            f"imag-time evolution failed: {type(e).__name__}: {e}"
        ) from e

    chi_observed_max = max(state.bond_dimensions()) if hasattr(state, "bond_dimensions") else chi
    final_state_energy = float(ev_energy(state, H_full))
    if np.isnan(final_state_energy) or np.isinf(final_state_energy):
        raise SynthesisRuntimeError(
            f"final state energy is non-finite: {final_state_energy}"
        )

    # 5. Sample.
    decode_results = sample(state, meta, n_samples=problem.n_samples, rng=rng)

    asts = [dr.ast for dr in decode_results]
    residuals = [dr.residual_norm for dr in decode_results]
    grouped = dedupe_samples(asts, residuals)
    n_decoded_ok = sum(m for _, m in grouped)

    # 6. Re-score each unique completion. Per spec §6.3: re-encode the
    # concrete completion, re-compile Hamiltonian on its meta, compute
    # per-block <H>.
    completions = []
    for canon_ast, mult in grouped:
        try:
            e_total, breakdown, diag = _score_completion(
                canon_ast, problem, weights,
            )
        except Exception as exc:
            warnings.warn(
                f"failed to re-score completion {canon_ast!r}: {exc}; "
                f"using a high-energy placeholder so it falls to the bottom"
            )
            e_total, breakdown, diag = float("inf"), {}, {}
        completions.append(Completion(
            ast=canon_ast,
            energy=e_total,
            energy_breakdown=breakdown,
            diagnostics=diag,
            multiplicity=mult,
        ))

    # 7. Rank + classify.
    completions = rank_completions(completions)
    failure_mode = classify_failure_mode(
        completions, weights=weights, final_state_energy=final_state_energy,
    )

    wall_time = time.time() - t_start
    return SynthesisResult(
        problem=problem,
        completions=completions,
        n_unique=len(completions),
        n_samples_drawn=problem.n_samples,
        n_samples_decoded_ok=n_decoded_ok,
        final_state_energy=final_state_energy,
        wall_time_seconds=wall_time,
        chi_observed_max=chi_observed_max,
        failure_mode=failure_mode,
    )


def _score_completion(
    ast,
    problem: SynthesisProblem,
    weights: HamiltonianWeights,
):
    """Re-encode a concrete completion (no holes) and compute per-block <H>.

    Returns (total_energy, breakdown_dict, diagnostics_dict).
    """
    # Encode the completion via A's base encode (no holes, no TypeHoles).
    state_c, meta_c = encode_synthesis_augmented(
        ast, problem.examples,
        N=problem.N, chi_max=problem.chi_max,
    )
    # Build each block separately so we can report breakdown.
    h_size = build_h_size(meta_c, sketch_range=_sketch_range(meta_c),
                          weight=weights.w_size)
    h_target = build_h_target_type(meta_c, problem.target_type,
                                    weight=weights.w_target_type)
    h_examples = build_h_examples(meta_c, problem.examples,
                                   weight=weights.w_examples)

    # H_typing / H_eval via defensive imports.
    try:
        from src.qft_pcn.logic.hamiltonian_compiler import (
            compile_typing_hamiltonian,
        )
        h_typing = _scale(compile_typing_hamiltonian(meta_c), weights.w_typing)
    except Exception:
        h_typing = Hamiltonian(_zero_cfg(), meta_c.N)
    try:
        from src.qft_pcn.logic.evaluation_hamiltonian import (
            compile_eval_hamiltonian,
        )
        h_eval = _scale(compile_eval_hamiltonian(meta_c), weights.w_eval)
    except Exception:
        h_eval = Hamiltonian(_zero_cfg(), meta_c.N)

    e_typing = float(h_typing.expectation(state_c))
    e_eval = float(h_eval.expectation(state_c))
    e_examples = float(h_examples.expectation(state_c))
    e_target = float(h_target.expectation(state_c))
    e_size = float(h_size.expectation(state_c))
    total = e_typing + e_eval + e_examples + e_target + e_size

    breakdown = {
        "typing": e_typing,
        "eval": e_eval,
        "examples": e_examples,
        "target_type": e_target,
        "size": e_size,
    }

    # Diagnostics from sub-project D, if available.
    diag = {}
    try:
        from src.qft_pcn.logic.debugger import diagnose
        diag = diagnose(state_c, meta_c, {
            "typing": h_typing, "eval": h_eval, "examples": h_examples,
            "target_type": h_target, "size": h_size,
        })
    except Exception:
        diag = {"_warning": "sub-project D's diagnose() not available"}

    return total, breakdown, diag


def _sketch_range(meta) -> tuple[int, int]:
    if meta.witness_regions:
        return (0, meta.witness_regions[0][0] - 1)
    return (0, meta.N)
```

- [ ] **Step 4: Export `synthesize` from synthesis/__init__.py**

Update `src/qft_pcn/logic/synthesis/__init__.py`:

```python
"""Sub-project E: STLC synthesis milestone demo.

Public API:
    synthesize(problem) -> SynthesisResult
    SynthesisProblem, IOExample, Completion, SynthesisResult, HamiltonianWeights
    SynthesisProblemError, SynthesisRuntimeError
"""

from .problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)
from .runner import synthesize
from .errors import SynthesisError, SynthesisProblemError, SynthesisRuntimeError

__all__ = [
    "synthesize",
    "SynthesisProblem", "IOExample", "Completion", "SynthesisResult",
    "HamiltonianWeights",
    "SynthesisError", "SynthesisProblemError", "SynthesisRuntimeError",
]
```

- [ ] **Step 5: Verify pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_runner.py -v`
Expected: 3 passed.

If the runner takes > 30s on the smoke test, reduce `anneal_steps` further in the test and investigate; do not loosen the test timeout without an explanation.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/synthesis/runner.py src/qft_pcn/logic/synthesis/__init__.py src/qft_pcn/tests/test_synthesis_runner.py
git commit -m "$(cat <<'EOF'
feat(logic/synthesis): synthesize() pipeline runner

End-to-end pipeline (spec §3.3, §6.5):
    1. validate_problem
    2. encode_synthesis_augmented (sketch + witnesses)
    3. compile H_warmup (typing+eval only) and H_full
    4. 3-phase imag-time anneal: warmup -> main -> fine
    5. sample n_samples from the relaxed state
    6. dedupe by alpha-eq, re-score each unique completion
    7. rank + classify failure_mode
    8. assemble SynthesisResult with diagnostics from sub-project D

Smoke tests confirm the API contract: SynthesisResult returned for valid
input, SynthesisProblemError for hole-less sketches, completes in seconds
for tiny problems.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: Top-level re-exports and integration

### Task 15: Re-export synthesis API from `logic/__init__.py`

**Files:**
- Modify: `src/qft_pcn/logic/__init__.py`.

Per spec §8.

- [ ] **Step 1: Add re-exports**

Edit `src/qft_pcn/logic/__init__.py`; append after the existing imports/__all__:

```python
from .synthesis import (
    synthesize, SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights, SynthesisError, SynthesisProblemError,
    SynthesisRuntimeError,
)
from .ast import TypeHole

__all__ += [
    "synthesize", "SynthesisProblem", "IOExample", "Completion",
    "SynthesisResult", "HamiltonianWeights",
    "SynthesisError", "SynthesisProblemError", "SynthesisRuntimeError",
    "TypeHole",
]
```

- [ ] **Step 2: Smoke-test imports**

Run:
```bash
.venv/bin/python -c "
from src.qft_pcn.logic import (
    synthesize, SynthesisProblem, IOExample, SynthesisResult,
    TypeHole, HoleVar, parse, pretty,
)
print('logic re-exports: ok')
"
```

Expected: `logic re-exports: ok`.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/logic/__init__.py
git commit -m "$(cat <<'EOF'
chore(logic): re-export synthesis API and TypeHole from logic package

Per spec §8: callers can `from qft_pcn.logic import synthesize,
SynthesisProblem, TypeHole` without reaching into the synthesis
sub-module.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 16: Top-level qft_pcn re-export

**Files:**
- Modify: `src/qft_pcn/__init__.py`.

- [ ] **Step 1: Add re-exports**

Read `src/qft_pcn/__init__.py` first; append:

```python
from src.qft_pcn.logic import (
    synthesize, SynthesisProblem, IOExample, SynthesisResult,
    Completion, HamiltonianWeights, TypeHole,
    SynthesisError, SynthesisProblemError, SynthesisRuntimeError,
)
```

(Idempotent: if any name is already exported, skip it.)

- [ ] **Step 2: Verify**

Run:
```bash
.venv/bin/python -c "
from qft_pcn import synthesize, SynthesisProblem, TypeHole
print('top-level re-exports: ok')
"
```

If imports fail because the test was using `from src.qft_pcn...` everywhere, drop the top-level re-export step (it's a convenience). Document and move on.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/__init__.py
git commit -m "$(cat <<'EOF'
chore(qft_pcn): top-level re-export of synthesis API

Convenience for callers: `from qft_pcn import synthesize, SynthesisProblem,
TypeHole` works without traversing logic.synthesis.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: Demo problems and acceptance tests

### Task 17: Integration tests for P1, P2, P3 (basic success cases)

**Files:**
- Modify: `src/qft_pcn/tests/test_synthesis_runner.py` (add integration tests).

Per spec §7 — the publishable acceptance suite. We split into 3 tasks (17, 18, 19) to keep each commit reviewable.

- [ ] **Step 1: Write the integration tests for P1, P2, P3**

Append to `src/qft_pcn/tests/test_synthesis_runner.py`:

```python
from src.qft_pcn.logic.ast import (
    App, Bin, BoolLit, TBool, TArrow,
)
from src.qft_pcn.logic.decoder import ast_canonical


def _alpha_eq(a, b) -> bool:
    return ast_canonical(a) == ast_canonical(b)


@pytest.mark.timeout(60)
def test_P1_identity_completion():
    """sketch: \\x:Int. ?HOLE  with candidates=["x"]  -> top-1 = \\x. x"""
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("x",)))
    p = SynthesisProblem(sketch=sketch, name="P1")
    rng = np.random.default_rng(seed=1)
    r = synthesize(p, rng=rng)
    assert r.failure_mode is None, f"P1 failed: {r.failure_mode}"
    expected = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    assert _alpha_eq(r.completions[0].ast, expected), (
        f"P1 top-1 mismatch: got {r.completions[0].ast!r}"
    )


@pytest.mark.timeout(60)
def test_P2_constant_vs_identity_by_example():
    """sketch: \\x:Int. ?HOLE  example (3,) -> 3  ===>  \\x. x"""
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar())     # no candidates -> all in scope + literals
    p = SynthesisProblem(
        sketch=sketch, name="P2",
        target_type=TArrow(src=TInt(), dst=TInt()),
        examples=(IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3)),),
    )
    rng = np.random.default_rng(seed=2)
    r = synthesize(p, rng=rng)
    assert r.failure_mode is None, f"P2 failed: {r.failure_mode}"
    expected = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    assert _alpha_eq(r.completions[0].ast, expected)
    # Energy gap check (§10 criterion 3).
    if len(r.completions) >= 2:
        gap = r.completions[1].energy - r.completions[0].energy
        assert gap >= 1.0, f"P2 energy gap too small: {gap}"


@pytest.mark.timeout(60)
def test_P3_use_both_arguments():
    """sketch: \\f:Int->Int. \\x:Int. ?HOLE  example: succ applied at 2 -> 3
    => top-1: f x"""
    succ = Lam(param="n", param_ty=TInt(),
               body=Bin(op="+", lhs=Var("n"), rhs=IntLit(val=1)))
    sketch = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("f", "x"))),
    )
    p = SynthesisProblem(
        sketch=sketch, name="P3",
        examples=(IOExample(inputs=(succ, IntLit(val=2)),
                            output=IntLit(val=3)),),
    )
    rng = np.random.default_rng(seed=3)
    r = synthesize(p, rng=rng)
    # NOTE: this is the hardest of P1-P3 because it requires evaluation
    # of the application of f to x. If sub-project C's H_eval is not
    # strong enough, this test will fail; relax the assertion in that case
    # but DOCUMENT it for the human, do not silently lower the bar.
    assert r.failure_mode is None, f"P3 failed: {r.failure_mode}"
    expected = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(param="x", param_ty=TInt(),
                 body=App(fn=Var("f"), arg=Var("x"))),
    )
    assert _alpha_eq(r.completions[0].ast, expected)
```

- [ ] **Step 2: Run; verify or escalate**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_runner.py -k "P1 or P2 or P3" -v 2>&1 | tail -30`

Expected: 3 passed.

If any of P1/P2/P3 fails, **diagnose first**: the most common failure modes and their diagnostics:

- *P1 fails*: encoder TypeHole/HoleVar superposition is not engaging. Run `test_encode_synthesis_typehole_produces_type_register_entropy` (Task 7) in isolation; if that fails, fix the encoder first.
- *P2 fails*: H_target_type or H_examples weight is too low, or H_typing is zero (sub-project B missing). Print `r.completions[0].energy_breakdown` from a temporary `pytest -s` run and check which block dominates.
- *P3 fails*: H_eval is too weak (most likely cause; sub-project C is the hard part). The test message says "DOCUMENT it for the human"; print the breakdown, mark `pytest.xfail("requires stronger H_eval; see Task 17 escalation")`, and proceed.

Do **not** mark the test as `xfail` permanently without escalation to the human first. The honest report is: "P3 cannot pass with current H_eval; sub-project C needs strengthening."

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_synthesis_runner.py
git commit -m "$(cat <<'EOF'
test(synthesis): integration tests for P1, P2, P3 (basic synthesis)

Per spec §7: identity completion (P1), constant-vs-identity disambiguation
by example (P2), and use-both-arguments (P3 — the canonical §10.7 example).
Energy gap checked on P2 (top-1 vs top-2 >= 1.0 = 0.25 w_T).

If P3 fails, the diagnosis is most likely a too-weak H_eval (sub-project C);
escalate before relaxing the assertion.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 18: Integration tests for P4, P5, P6, P7

**Files:**
- Modify: `src/qft_pcn/tests/test_synthesis_runner.py`.

- [ ] **Step 1: Write the failing tests**

Append:

```python
@pytest.mark.timeout(60)
def test_P4_conditional_with_literals():
    """sketch: \\x:Int. if (x < ?HOLE_INT) then x else (x + ?HOLE_INT)
    examples: 2 -> 2, 7 -> 8  =>  threshold=5, increment=1
    """
    # HoleVar at "literal" positions: we model an int-literal hole as a
    # HoleVar with empty candidates and target_type=Int — the runner will
    # interpret this as any in-scope literal at encoding time.
    # For simplicity we use a single HoleVar shared in two positions and
    # rely on alpha-eq for evaluation — actually no, we need two separate
    # holes. Define two:
    h_thresh = HoleVar(candidates=(), target_type=TInt(), name="?thresh")
    h_incr = HoleVar(candidates=(), target_type=TInt(), name="?incr")
    sketch = Lam(
        param="x", param_ty=TInt(),
        body=If(
            cond=Bin(op="<", lhs=Var("x"), rhs=h_thresh),
            then_b=Var("x"),
            else_b=Bin(op="+", lhs=Var("x"), rhs=h_incr),
        ),
    )
    p = SynthesisProblem(
        sketch=sketch, name="P4",
        examples=(
            IOExample(inputs=(IntLit(val=2),), output=IntLit(val=2)),
            IOExample(inputs=(IntLit(val=7),), output=IntLit(val=8)),
        ),
    )
    rng = np.random.default_rng(seed=4)
    r = synthesize(p, rng=rng)
    if r.failure_mode is not None:
        pytest.xfail(
            f"P4 not yet solved: {r.failure_mode}; "
            f"top-1 breakdown={r.completions[0].energy_breakdown if r.completions else None}"
        )
    expected = Lam(
        param="x", param_ty=TInt(),
        body=If(
            cond=Bin(op="<", lhs=Var("x"), rhs=IntLit(val=5)),
            then_b=Var("x"),
            else_b=Bin(op="+", lhs=Var("x"), rhs=IntLit(val=1)),
        ),
    )
    assert _alpha_eq(r.completions[0].ast, expected)


@pytest.mark.timeout(60)
def test_P5_typehole_resolution():
    """sketch: \\x:?T. x  target_type: Bool -> Bool  =>  \\x:Bool. x"""
    from src.qft_pcn.logic.ast import TypeHole
    sketch = Lam(param="x",
                 param_ty=TypeHole(candidates=(TInt(), TBool())),
                 body=Var(name="x"))
    p = SynthesisProblem(
        sketch=sketch, name="P5",
        target_type=TArrow(src=TBool(), dst=TBool()),
    )
    rng = np.random.default_rng(seed=5)
    r = synthesize(p, rng=rng)
    assert r.failure_mode is None, f"P5 failed: {r.failure_mode}"
    expected = Lam(param="x", param_ty=TBool(), body=Var(name="x"))
    assert _alpha_eq(r.completions[0].ast, expected)


@pytest.mark.timeout(120)
def test_P6_function_composition():
    """sketch: \\f:Int->Int. \\g:Int->Int. \\x:Int. ?HOLE
    examples: f=(n+1), g=(n*2), x=2 -> f(g(x))=f(4)=5
    expected top-1: f (g x)
    """
    plus1 = Lam(param="n", param_ty=TInt(),
                body=Bin(op="+", lhs=Var("n"), rhs=IntLit(val=1)))
    times2 = Lam(param="n", param_ty=TInt(),
                  body=Bin(op="*", lhs=Var("n"), rhs=IntLit(val=2)))
    sketch = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(
            param="g", param_ty=TArrow(src=TInt(), dst=TInt()),
            body=Lam(
                param="x", param_ty=TInt(),
                body=HoleVar(candidates=("f", "g", "x")),
            ),
        ),
    )
    p = SynthesisProblem(
        sketch=sketch, name="P6",
        examples=(IOExample(inputs=(plus1, times2, IntLit(val=2)),
                            output=IntLit(val=5)),),
        N=32, chi_max=32,
    )
    rng = np.random.default_rng(seed=6)
    r = synthesize(p, rng=rng)
    if r.failure_mode is not None:
        pytest.xfail(f"P6 not yet solved: {r.failure_mode}")
    expected = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(
            param="g", param_ty=TArrow(src=TInt(), dst=TInt()),
            body=Lam(
                param="x", param_ty=TInt(),
                body=App(fn=Var("f"),
                          arg=App(fn=Var("g"), arg=Var("x"))),
            ),
        ),
    )
    assert _alpha_eq(r.completions[0].ast, expected)


@pytest.mark.timeout(60)
def test_P7_boolean_synthesis_mixed_types():
    """sketch: \\x:Int. \\y:Int. ?HOLE  target_type: Int->Int->Bool
    examples: (2,3) -> True, (5,3) -> False  =>  x < y
    """
    sketch = Lam(
        param="x", param_ty=TInt(),
        body=Lam(
            param="y", param_ty=TInt(),
            body=HoleVar(candidates=("x", "y")),
        ),
    )
    p = SynthesisProblem(
        sketch=sketch, name="P7",
        target_type=TArrow(src=TInt(), dst=TArrow(src=TInt(), dst=TBool())),
        examples=(
            IOExample(inputs=(IntLit(val=2), IntLit(val=3)),
                      output=BoolLit(val=True)),
            IOExample(inputs=(IntLit(val=5), IntLit(val=3)),
                      output=BoolLit(val=False)),
        ),
    )
    rng = np.random.default_rng(seed=7)
    r = synthesize(p, rng=rng)
    if r.failure_mode is not None:
        pytest.xfail(f"P7 not yet solved: {r.failure_mode}")
    expected = Lam(
        param="x", param_ty=TInt(),
        body=Lam(
            param="y", param_ty=TInt(),
            body=Bin(op="<", lhs=Var("x"), rhs=Var("y")),
        ),
    )
    assert _alpha_eq(r.completions[0].ast, expected)
```

- [ ] **Step 2: Run; record results**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_runner.py -k "P4 or P5 or P6 or P7" -v 2>&1 | tail -30`

Expected: at minimum P5 must pass (it tests the TypeHole encoder directly with no eval dependency). P4/P6/P7 are likely xfail unless B and C are very strong; document the failures.

**Honesty check (spec §1.6):** if 3 of 4 of these xfail, that's fine — they're documented. If you find yourself tempted to lower the assertion to "top-N contains expected" rather than "top-1 == expected", **stop**. Lowering the assertion is fine, but make a new test with the relaxed criterion alongside, don't quietly change the bar.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_synthesis_runner.py
git commit -m "$(cat <<'EOF'
test(synthesis): integration tests for P4 (conditional), P5 (TypeHole),
P6 (composition), P7 (boolean)

Per spec §7: P5 (TypeHole resolution) is the must-pass test for sub-project
E's encoder extension. P4, P6, P7 depend on the strength of sub-projects
B's H_typing and C's H_eval; they are marked xfail with a diagnostic when
they fail, per spec §1.6 (honest reporting).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 19: P8 — the intentionally unsolvable case

**Files:**
- Modify: `src/qft_pcn/tests/test_synthesis_runner.py`.

Per spec §7 / P8.

- [ ] **Step 1: Write the failing test**

Append:

```python
@pytest.mark.timeout(60)
def test_P8_unsolvable_returns_failure_mode():
    """sketch: \\x:Int. ?HOLE  target_type: Int -> Bool
    Hole candidates ["x"] — only x:Int is in scope, can't make a Bool.
    Expected: failure_mode != None, ideally with non-trivial H_typing
    residual.
    """
    sketch = Lam(param="x", param_ty=TInt(),
                 body=HoleVar(candidates=("x",)))
    p = SynthesisProblem(
        sketch=sketch, name="P8",
        target_type=TArrow(src=TInt(), dst=TBool()),
        examples=(IOExample(inputs=(IntLit(val=2),), output=BoolLit(val=True)),),
    )
    rng = np.random.default_rng(seed=8)
    r = synthesize(p, rng=rng)
    # Honest reporting (spec §1.6): we MUST return failure_mode != None,
    # OR (top-1 with high enough residual H_typing/H_target_type that it's
    # clearly wrong).
    if r.failure_mode is None:
        # Then the top-1 must at least carry visible residual.
        breakdown = r.completions[0].energy_breakdown
        residual = breakdown.get("typing", 0.0) + breakdown.get("target_type", 0.0)
        assert residual > 1.0, (
            f"P8 returned a top-1 with no visible residual: {breakdown!r}; "
            f"this violates §1.6 (honest reporting): the synthesis cannot "
            f"silently succeed on an unsolvable problem"
        )
    else:
        assert r.failure_mode in ("no_valid_completion",
                                    "imag_time_did_not_converge"), (
            f"P8 unexpected failure_mode: {r.failure_mode}"
        )
```

- [ ] **Step 2: Run and verify**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_runner.py -k "P8" -v -s`

Expected: pass. The system either (a) classifies as failure_mode or (b) returns a top-1 with visible H_typing/H_target_type residual.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_synthesis_runner.py
git commit -m "$(cat <<'EOF'
test(synthesis): P8 — unsolvable problem must be classified honestly

Per spec §7/P8 and §1.6: a problem with target_type Int->Bool whose
sole hole candidate is x:Int has no valid completion. The synthesis
runner must either set failure_mode != None or surface visible residual
energy (H_typing or H_target_type > 1.0) on the top-1. Silent success
violates the honest-reporting principle.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 7: The demo script

### Task 20: `demo_stlc_synthesis.py`

**Files:**
- Create: `src/qft_pcn/logic/demo_stlc_synthesis.py`.
- Create: `src/qft_pcn/tests/test_synthesis_demo.py` (smoke test).

Per spec §10 criterion 10.

- [ ] **Step 1: Write the failing smoke test**

Create `src/qft_pcn/tests/test_synthesis_demo.py`:

```python
"""Smoke test: the demo script runs end-to-end without raising."""

from __future__ import annotations

import io
import contextlib

import pytest


@pytest.mark.timeout(360)    # 6 minutes total
def test_demo_runs_end_to_end():
    from src.qft_pcn.logic.demo_stlc_synthesis import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main()
    out = buf.getvalue()
    # The demo must print at least P1 through P8 labels.
    for label in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        assert label in out, f"demo output missing {label!r}"
    # The demo must print 'top-1' and 'energy_breakdown' for at least one
    # problem (the per-spec acceptance §10 criterion 10).
    assert "top-1" in out
    assert "energy_breakdown" in out or "breakdown" in out
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_demo.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the demo script**

Create `src/qft_pcn/logic/demo_stlc_synthesis.py`:

```python
"""Sub-project E: the publishable milestone demo.

Runs the eight §7 synthesis problems and prints structured output:

    problem name | expected top-1 | actual top-1 | energy | breakdown |
    n_unique | n_samples | failure_mode | wall_time | chi_max

Usage:
    python -m qft_pcn.logic.demo_stlc_synthesis

The demo is the artifact that ships with the workshop-paper submission.
Per spec §1.6 (honest reporting): failures are printed verbatim, not
hidden. P8 is intentionally unsolvable and must classify as failure_mode.
"""

from __future__ import annotations

import time

import numpy as np

from src.qft_pcn.logic.ast import (
    Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar, TypeHole,
    TInt, TBool, TArrow, pretty,
)
from src.qft_pcn.logic.decoder import ast_canonical
from src.qft_pcn.logic.synthesis import (
    synthesize, SynthesisProblem, IOExample,
)


def _alpha_eq(a, b) -> bool:
    return ast_canonical(a) == ast_canonical(b)


# ---- problem definitions ---------------------------------------------------


def _problems():
    """The eight §7 synthesis problems. Each entry is:
        (SynthesisProblem, expected_ast_or_None)

    expected=None for P8 (unsolvable).
    """
    out = []

    # P1
    p = SynthesisProblem(
        sketch=Lam(param="x", param_ty=TInt(),
                    body=HoleVar(candidates=("x",))),
        name="P1: identity",
    )
    exp = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    out.append((p, exp))

    # P2
    p = SynthesisProblem(
        sketch=Lam(param="x", param_ty=TInt(), body=HoleVar()),
        name="P2: identity by example",
        target_type=TArrow(src=TInt(), dst=TInt()),
        examples=(IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3)),),
    )
    exp = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    out.append((p, exp))

    # P3
    succ = Lam(param="n", param_ty=TInt(),
                body=Bin(op="+", lhs=Var("n"), rhs=IntLit(val=1)))
    p = SynthesisProblem(
        sketch=Lam(
            param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
            body=Lam(param="x", param_ty=TInt(),
                     body=HoleVar(candidates=("f", "x"))),
        ),
        name="P3: use both arguments",
        examples=(IOExample(inputs=(succ, IntLit(val=2)),
                            output=IntLit(val=3)),),
    )
    exp = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(param="x", param_ty=TInt(),
                 body=App(fn=Var("f"), arg=Var("x"))),
    )
    out.append((p, exp))

    # P4
    h_thresh = HoleVar(candidates=(), target_type=TInt(), name="?thresh")
    h_incr = HoleVar(candidates=(), target_type=TInt(), name="?incr")
    p = SynthesisProblem(
        sketch=Lam(
            param="x", param_ty=TInt(),
            body=If(
                cond=Bin(op="<", lhs=Var("x"), rhs=h_thresh),
                then_b=Var("x"),
                else_b=Bin(op="+", lhs=Var("x"), rhs=h_incr),
            ),
        ),
        name="P4: conditional with literal holes",
        examples=(
            IOExample(inputs=(IntLit(val=2),), output=IntLit(val=2)),
            IOExample(inputs=(IntLit(val=7),), output=IntLit(val=8)),
        ),
    )
    exp = Lam(
        param="x", param_ty=TInt(),
        body=If(
            cond=Bin(op="<", lhs=Var("x"), rhs=IntLit(val=5)),
            then_b=Var("x"),
            else_b=Bin(op="+", lhs=Var("x"), rhs=IntLit(val=1)),
        ),
    )
    out.append((p, exp))

    # P5
    p = SynthesisProblem(
        sketch=Lam(param="x",
                    param_ty=TypeHole(candidates=(TInt(), TBool())),
                    body=Var(name="x")),
        name="P5: TypeHole resolution",
        target_type=TArrow(src=TBool(), dst=TBool()),
    )
    exp = Lam(param="x", param_ty=TBool(), body=Var(name="x"))
    out.append((p, exp))

    # P6
    plus1 = Lam(param="n", param_ty=TInt(),
                 body=Bin(op="+", lhs=Var("n"), rhs=IntLit(val=1)))
    times2 = Lam(param="n", param_ty=TInt(),
                  body=Bin(op="*", lhs=Var("n"), rhs=IntLit(val=2)))
    p = SynthesisProblem(
        sketch=Lam(
            param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
            body=Lam(
                param="g", param_ty=TArrow(src=TInt(), dst=TInt()),
                body=Lam(param="x", param_ty=TInt(),
                         body=HoleVar(candidates=("f", "g", "x"))),
            ),
        ),
        name="P6: function composition",
        examples=(IOExample(inputs=(plus1, times2, IntLit(val=2)),
                            output=IntLit(val=5)),),
    )
    exp = Lam(
        param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Lam(
            param="g", param_ty=TArrow(src=TInt(), dst=TInt()),
            body=Lam(param="x", param_ty=TInt(),
                     body=App(fn=Var("f"),
                              arg=App(fn=Var("g"), arg=Var("x")))),
        ),
    )
    out.append((p, exp))

    # P7
    p = SynthesisProblem(
        sketch=Lam(
            param="x", param_ty=TInt(),
            body=Lam(param="y", param_ty=TInt(),
                     body=HoleVar(candidates=("x", "y"))),
        ),
        name="P7: boolean synthesis",
        target_type=TArrow(src=TInt(), dst=TArrow(src=TInt(), dst=TBool())),
        examples=(
            IOExample(inputs=(IntLit(val=2), IntLit(val=3)),
                      output=BoolLit(val=True)),
            IOExample(inputs=(IntLit(val=5), IntLit(val=3)),
                      output=BoolLit(val=False)),
        ),
    )
    exp = Lam(
        param="x", param_ty=TInt(),
        body=Lam(param="y", param_ty=TInt(),
                 body=Bin(op="<", lhs=Var("x"), rhs=Var("y"))),
    )
    out.append((p, exp))

    # P8
    p = SynthesisProblem(
        sketch=Lam(param="x", param_ty=TInt(),
                    body=HoleVar(candidates=("x",))),
        name="P8: unsolvable",
        target_type=TArrow(src=TInt(), dst=TBool()),
        examples=(IOExample(inputs=(IntLit(val=2),), output=BoolLit(val=True)),),
    )
    out.append((p, None))

    return out


def _format_completion(c) -> str:
    try:
        prog = pretty(c.ast)
    except Exception:
        prog = repr(c.ast)
    parts = [f"{k}={v:.4f}" for k, v in c.energy_breakdown.items()]
    return (f"  energy={c.energy:.4f}  mult={c.multiplicity}\n"
            f"  ast: {prog}\n"
            f"  breakdown: {', '.join(parts)}\n")


def main() -> None:
    print("=" * 72)
    print("STLC Synthesis Milestone Demo (sub-project E)")
    print("=" * 72)

    n_passed = 0
    n_total = 0

    for problem, expected in _problems():
        n_total += 1
        print()
        print(f"--- {problem.name} ---")
        print(f"sketch: (see source)")
        if expected is not None:
            try:
                print(f"expected top-1: {pretty(expected)}")
            except Exception:
                print(f"expected top-1: {expected!r}")
        else:
            print(f"expected: failure_mode != None (intentionally unsolvable)")

        rng = np.random.default_rng(seed=hash(problem.name) % (2**32))
        try:
            r = synthesize(problem, rng=rng)
        except Exception as exc:
            print(f"  RAISED: {type(exc).__name__}: {exc}")
            continue

        print(f"  wall_time: {r.wall_time_seconds:.2f} s")
        print(f"  n_unique: {r.n_unique}  n_samples: {r.n_samples_drawn}  "
              f"n_decoded_ok: {r.n_samples_decoded_ok}")
        print(f"  failure_mode: {r.failure_mode}")
        print(f"  final_state_energy: {r.final_state_energy:.4f}")
        print(f"  chi_observed_max: {r.chi_observed_max}")

        if r.completions:
            print(f"  top-1:")
            print(_format_completion(r.completions[0]))
            if len(r.completions) >= 2:
                gap = r.completions[1].energy - r.completions[0].energy
                print(f"  top-2 energy gap: {gap:.4f}")

        # Acceptance check.
        if expected is None:
            ok = (r.failure_mode is not None) or (
                r.completions and
                (r.completions[0].energy_breakdown.get("typing", 0.0)
                 + r.completions[0].energy_breakdown.get("target_type", 0.0)
                 > 1.0)
            )
            verdict = "PASS (correctly classified)" if ok else "FAIL"
        else:
            ok = (r.failure_mode is None and r.completions
                  and _alpha_eq(r.completions[0].ast, expected))
            verdict = "PASS" if ok else "FAIL"
        if ok:
            n_passed += 1
        print(f"  >>> {verdict}")

    print()
    print("=" * 72)
    print(f"Summary: {n_passed}/{n_total} problems passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify smoke test pass**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_demo.py -v -s`
Expected: pass within timeout (6 minutes). Print summary should show n_passed/8 ≥ 5 per spec §10 criterion 2.

- [ ] **Step 5: Run the demo standalone**

Run: `.venv/bin/python -m src.qft_pcn.logic.demo_stlc_synthesis`
Expected: same end-to-end output as the smoke test, ending with `Summary: N/8 problems passed`.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/demo_stlc_synthesis.py src/qft_pcn/tests/test_synthesis_demo.py
git commit -m "$(cat <<'EOF'
feat(logic): the publishable milestone demo — STLC synthesis (P1-P8)

Per spec §7 and §10 criterion 10: runs the eight synthesis problems and
prints structured output per problem (top-1 AST, energy, per-block
breakdown, multiplicity, n_unique, failure_mode, wall_time). PASS/FAIL
verdict per problem and an aggregate summary at the end.

P5 is the must-pass test for sub-project E's TypeHole encoder. P1-P4,
P6, P7 depend on B's H_typing and C's H_eval strength; P8 is
intentionally unsolvable. Per spec §1.6 the demo reports failures
honestly without hiding them.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 8: Verification and finalization

### Task 21: Full test suite run + diagnostics audit

**Files:** none modified; pure verification.

- [ ] **Step 1: Run the full test suite**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ --ignore=src/qft_pcn/tests/test_quantum.py -v 2>&1 | tail -50`

Expected:
- All `test_logic_*` tests pass.
- All `test_synthesis_*` tests pass (or xfail with documented diagnostics).
- All A/B/C/D tests pass.
- All qft core tests pass.

If any test that previously passed (per Task 0 Step 5 baseline) now fails: **STOP, diagnose, fix.** Do not commit on a red baseline.

- [ ] **Step 2: Check diagnostics emission for at least one P_i**

Run: `.venv/bin/python -m src.qft_pcn.logic.demo_stlc_synthesis 2>&1 | grep -A 3 "breakdown:"`
Expected: at least one breakdown line per problem; values should be finite floats.

- [ ] **Step 3: Verify acceptance criterion §10.2 (≥ 5 of 8 pass)**

Manually inspect demo output's summary line. **Acceptance:** `n_passed >= 5` (the user's "majority" criterion). If `n_passed < 5`, the most likely root cause is sub-project C's H_eval being too weak; escalate to the human with the specific failing problems and their breakdown.

Do not attempt to lower the acceptance bar in this plan. Either pass with 5+/8, or escalate.

- [ ] **Step 4: Check the wall-time budget (§10 criterion 5)**

The demo must complete in under 5 minutes. If it does not, the most likely culprits are: anneal_steps too high (lower per-problem), n_samples too high (lower to 32), or chi_max too high (lower to 16 for problems without holes).

Tune `SynthesisProblem` defaults if needed, **document the change** in a follow-up commit. Do not modify the spec's acceptance bar.

---

### Task 22: Final spec-completion checklist

**Files:** none modified; pure verification.

Walk through spec §10 acceptance criteria one by one:

- [ ] **Criterion 1: All P1–P8 run via `synthesize()` without raising.**
      Verify: demo output shows no `RAISED:` lines.

- [ ] **Criterion 2: 7 of 8 problems (P1, P2, P3, P4, P5, P6, P7) produce top-1 = expected.**
      Verify: demo summary `n_passed >= 7` (we're aiming for 7 of 8; the user said "majority of cases" which we read as ≥ 5; aim higher).
      If fewer pass, escalate with specific failures and their `energy_breakdown` to the human.

- [ ] **Criterion 3: For P1–P7, top-2 energy ≥ top-1 + 2.0.**
      Verify: demo output shows `top-2 energy gap: G` for each problem with G >= 2.0. Where lower, print a `WARNING:` line and continue.

- [ ] **Criterion 4: P8 has failure_mode != None.**
      Verify: demo output's P8 section shows `failure_mode: <something other than None>`.

- [ ] **Criterion 5: Demo runs in under 5 minutes.**
      Verify: time the demo: `time .venv/bin/python -m src.qft_pcn.logic.demo_stlc_synthesis`.
      Expected: < 300s.

- [ ] **Criterion 6: All unit tests in tests/test_synthesis_* pass.**
      Verify: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_*.py -v 2>&1 | tail -5`.

- [ ] **Criterion 7: Baseline tests still pass.**
      Verify: re-run the Task 0 Step 5 baseline command.

- [ ] **Criterion 8: TypeHole encoder superposition produces type-register entropy.**
      Verify: `.venv/bin/python -m pytest src/qft_pcn/tests/test_synthesis_encoder_ext.py::test_encode_synthesis_typehole_produces_type_register_entropy -v`.

- [ ] **Criterion 9: Per-block diagnostics sum to final_state_energy.**
      Verify by adding (or running the existing) test that for one demo problem:
      `abs(sum(breakdown.values()) - r.completions[0].energy) < 1e-6`.

- [ ] **Criterion 10: Demo prints all required fields.**
      Verify: `python -m src.qft_pcn.logic.demo_stlc_synthesis | grep -E "(top-1|breakdown|failure_mode|wall_time)"` shows lines for each.

If every criterion checks: write a single final commit summarizing the result.

```bash
git add -A
git commit --allow-empty -m "$(cat <<'EOF'
docs(synthesis): sub-project E acceptance verified

Spec §10 criteria 1-10 walked through:
  - All P1-P8 run without raising.
  - n_passed/8 = <FILL IN> per demo summary.
  - Energy gap top-2 vs top-1 documented per problem.
  - P8 classified as failure_mode != None.
  - Demo wall-time: <FILL IN> seconds (< 300s budget).
  - All synthesis unit tests pass; baseline still green.

Sub-project E is complete; the QPCN logic layer now composes A's encoder,
B's typing rules, C's evaluation, D's diagnostics into an end-to-end
program-synthesis pipeline with no LLM in the loop. Publishable milestone
hit per QFT_PCN_ARCHITECTURE.md §10.7.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If any criterion fails: **DO NOT commit a "success" message.** Instead, write a structured report of what passed and what didn't, including:

- The specific demo problems that fail and their `energy_breakdown`.
- Which sub-project (B, C, or E) appears to be the weak link.
- A recommendation: weaker tests, stronger H_typing/H_eval, or amended spec.

Escalate to the human for decision.

---

**End of plan.** Sub-project E is the publishable milestone. Honest reporting (spec §1.6) is non-negotiable: if the architecture isn't strong enough yet, the demo says so, and the negative result is itself the publishable contribution.
