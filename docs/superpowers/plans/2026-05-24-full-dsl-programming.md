# Full §9.2 DSL + Programming-in-DSL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the architecture's §9.2 constraint-expression DSL end-to-end — bridge schema extensions, four new constraint kinds, MERA routing, lemma-library wiring, §10.10 cross-level decomposition, predicate compiler, visualizer DSL rewrite, four programming presets — so the LLM frontend emits constraint-DSL and the QPCN runs program synthesis per §15.

**Architecture:** Extend the existing `src/qft_pcn/bridge/` (already ships `local`/`two_site` + `imag_time` MPS) rather than greenfield-rewrite it. Add four constraint kinds + `argmax` op + MERA routing + `version`/`decomposition`. Greenfield-rewrite only the visualizer's DSL surface (`viz/dsl.py`, `viz/llm.py`) and add new panels (LemmaLibraryPanel, GoalGraphPanel). Wire to the already-existing composition layer (`dispatcher.py`, `result_integrator.py`, `lemma_library_adapter.py`, `promoter.py`). Pythonic-restricted AST predicate language replaces the existing term grammar for the new kinds.

**Tech Stack:** Python 3.12 + uv, numpy/scipy, jsonschema, FastAPI (bridge), pytest, React + TypeScript + Monaco + Vite + vitest (visualizer web), Ollama (gemma4:31b for DSL emission).

**Spec:** `docs/superpowers/specs/2026-05-24-full-dsl-programming-design.md`

**Architecture authority:** `QFT_PCN_ARCHITECTURE.md` — citations §X.Y throughout.

---

## Workstream Map

| Workstream | Scope | Depends on |
|---|---|---|
| W1 | Bridge schema v1: version, `mera` runtime, `argmax` op, decomposition top-level | none |
| W2 | Pythonic-restricted predicate compiler (`viz/dsl_predicates.py` + bridge wiring) | none |
| W3 | Four new constraint-kind compilers in `bridge/runtime/hamiltonian_compiler.py` | W1, W2 |
| W4 | MERA runtime routing + `use_lemma` clamp wiring + new `/dsl/v1/run` endpoint | W1, W3 |
| W7 | `bridge/decomposition.py` — child DAG dispatch + integration | W4 |
| W5 | `viz/llm.py` rewrite + three `llm_examples/*.json` + Ollama regression test | W1 |
| W6a | Visualizer panels: DslEditorPanel rewire, LemmaLibraryPanel, GoalGraphPanel, run-decoder | W4, W7 |
| W6b | Acceptance tests: length-synthesis, use_lemma, decomposition, dsl_v1 edge cases | W3, W4, W7 |

Tasks below are numbered within each workstream. Dispatch one task per subagent.

---

## W1 — Bridge schema v1 extension

**Files:**
- Modify: `src/qft_pcn/bridge/dsl/schema.py`
- Modify: `src/qft_pcn/bridge/tests/test_schema.py` (or create if absent — verify first)
- Test: `src/qft_pcn/bridge/tests/test_schema_v1.py` (NEW)

### W1.T1: Add `version: "1"` required top-level field

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/bridge/tests/test_schema_v1.py`:

```python
"""§9.2 v1 schema extensions: version, mera runtime, argmax, decomposition."""
import pytest
from qft_pcn.bridge.dsl.schema import parse_and_validate
from qft_pcn.bridge.errors import BadSchemaError


def _minimal_v1() -> str:
    return """
    {
      "version": "1",
      "fields": [{"name": "n", "cutoff": 4}],
      "sites": 2,
      "constraints": [{"kind": "local", "site": 0, "term": "n == 1", "weight": 1.0}],
      "observables": [{"site": 1, "field": "n", "op": "n"}],
      "search": {"method": "imag_time", "runtime": "mps", "steps": 10, "chi_max": 8, "dt": 0.05}
    }
    """


def test_v1_minimal_parses():
    dsl = parse_and_validate(_minimal_v1())
    assert dsl["version"] == "1"


def test_missing_version_rejects():
    spec = _minimal_v1().replace('"version": "1",', "")
    with pytest.raises(BadSchemaError, match="version"):
        parse_and_validate(spec)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/qft_pcn/bridge/tests/test_schema_v1.py::test_missing_version_rejects -v`
Expected: FAIL (no `version` field in current schema → rejection message will differ or pass spuriously)

- [ ] **Step 3: Add `version` to schema**

In `src/qft_pcn/bridge/dsl/schema.py`, extend `SCHEMA["required"]` and `SCHEMA["properties"]`:

```python
SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QPCN Bridge DSL v1",
    "type": "object",
    "required": ["version", "fields", "sites", "constraints", "observables"],
    "additionalProperties": False,
    "properties": {
        "version": {"const": "1"},
        # ... existing fields, search, constraints, observables, boundary remain ...
    },
}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest src/qft_pcn/bridge/tests/test_schema_v1.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/schema.py src/qft_pcn/bridge/tests/test_schema_v1.py
git -c commit.gpgsign=false commit -m "feat(bridge/dsl): require version: '1' per §9.2 schema v1"
```

### W1.T2: Add `runtime: 'mps' | 'mera'` to search

- [ ] **Step 1: Write the failing test**

Append to `test_schema_v1.py`:

```python
def test_search_runtime_mera_accepted():
    spec = _minimal_v1().replace('"runtime": "mps"', '"runtime": "mera"')
    dsl = parse_and_validate(spec)
    assert dsl["search"]["runtime"] == "mera"


def test_search_runtime_invalid_rejects():
    spec = _minimal_v1().replace('"runtime": "mps"', '"runtime": "tebd"')
    with pytest.raises(BadSchemaError, match="runtime"):
        parse_and_validate(spec)


def test_search_runtime_default_mps_when_absent():
    spec = _minimal_v1().replace(', "runtime": "mps"', "")
    dsl = parse_and_validate(spec)
    assert dsl["search"]["runtime"] == "mps"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest src/qft_pcn/bridge/tests/test_schema_v1.py -v`
Expected: 3 FAIL

- [ ] **Step 3: Extend the search subschema**

In `schema.py`, the `search` properties block becomes:

```python
"search": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "method":  {"enum": ["imag_time"]},
        "runtime": {"enum": ["mps", "mera"], "default": "mps"},
        "steps":   {"type": "integer", "minimum": 1, "maximum": 1000, "default": 50},
        "chi_max": {"type": "integer", "minimum": 2, "maximum": 64, "default": 32},
        "dt":      {"type": "number", "exclusiveMinimum": 0, "maximum": 0.5, "default": 0.05},
    },
    "default": {**_DEFAULT_SEARCH, "runtime": "mps"},
},
```

And update `_DEFAULT_SEARCH`:

```python
_DEFAULT_SEARCH = {
    "method": "imag_time",
    "runtime": "mps",
    "steps": 50,
    "chi_max": 32,
    "dt": 0.05,
}
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest src/qft_pcn/bridge/tests/test_schema_v1.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/schema.py src/qft_pcn/bridge/tests/test_schema_v1.py
git -c commit.gpgsign=false commit -m "feat(bridge/dsl): add search.runtime 'mps'|'mera' per spec §2.6"
```

### W1.T3: Add `argmax` observable op

- [ ] **Step 1: Write the failing test**

Append to `test_schema_v1.py`:

```python
def test_observable_argmax_accepted():
    spec = _minimal_v1().replace('"op": "n"', '"op": "argmax"')
    dsl = parse_and_validate(spec)
    assert dsl["observables"][0]["op"] == "argmax"
```

- [ ] **Step 2: Run to verify failure**
Run: `uv run pytest src/qft_pcn/bridge/tests/test_schema_v1.py::test_observable_argmax_accepted -v`
Expected: FAIL (`argmax` not in current enum)

- [ ] **Step 3: Extend observable op enum**

In `schema.py`:

```python
"op":    {"enum": ["n", "phi", "pi", "a", "adag", "identity", "argmax"]},
```

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/schema.py src/qft_pcn/bridge/tests/test_schema_v1.py
git -c commit.gpgsign=false commit -m "feat(bridge/dsl): add 'argmax' observable op per §15.7"
```

### W1.T4: Add four new constraint kinds to schema (well_typed_subtree, example, vocabulary, use_lemma)

- [ ] **Step 1: Write the failing tests**

Append to `test_schema_v1.py`:

```python
def _v1_with_constraint(c: str) -> str:
    return _minimal_v1().replace(
        '"constraints": [{"kind": "local", "site": 0, "term": "n == 1", "weight": 1.0}]',
        f'"constraints": [{c}]'
    )


def test_well_typed_subtree_accepted():
    dsl = parse_and_validate(_v1_with_constraint(
        '{"kind": "well_typed_subtree", "root": 0, "weight": 10.0}'
    ))
    assert dsl["constraints"][0]["kind"] == "well_typed_subtree"


def test_example_constraint_accepted():
    dsl = parse_and_validate(_v1_with_constraint(
        '{"kind": "example", "input": "[]", "output": "0", "weight": 5.0}'
    ))
    assert dsl["constraints"][0]["kind"] == "example"


def test_vocabulary_constraint_accepted():
    dsl = parse_and_validate(_v1_with_constraint(
        '{"kind": "vocabulary", "primitives": ["Match", "Cons", "Nil"]}'
    ))
    assert dsl["constraints"][0]["primitives"] == ["Match", "Cons", "Nil"]


def test_use_lemma_constraint_accepted():
    dsl = parse_and_validate(_v1_with_constraint(
        '{"kind": "use_lemma", "lemma_id": "length-base", "sites": [0, 1], "weight": 8.0}'
    ))
    assert dsl["constraints"][0]["lemma_id"] == "length-base"


def test_use_lemma_requires_nonempty_sites():
    with pytest.raises(BadSchemaError):
        parse_and_validate(_v1_with_constraint(
            '{"kind": "use_lemma", "lemma_id": "x", "sites": [], "weight": 1.0}'
        ))
```

- [ ] **Step 2: Run to verify failures**
Expected: 5 FAIL (kinds not in `oneOf`)

- [ ] **Step 3: Add the four kinds to the constraints `oneOf`**

In `schema.py`, extend the constraints `oneOf` list with:

```python
{
    "type": "object",
    "required": ["kind", "root"],
    "additionalProperties": False,
    "properties": {
        "kind":   {"const": "well_typed_subtree"},
        "root":   {"type": "integer", "minimum": 0},
        "weight": {"type": "number", "minimum": 0, "default": 1.0},
    },
},
{
    "type": "object",
    "required": ["kind", "input", "output"],
    "additionalProperties": False,
    "properties": {
        "kind":   {"const": "example"},
        "input":  {"type": "string"},
        "output": {"type": "string"},
        "weight": {"type": "number", "minimum": 0, "default": 1.0},
    },
},
{
    "type": "object",
    "required": ["kind", "primitives"],
    "additionalProperties": False,
    "properties": {
        "kind":       {"const": "vocabulary"},
        "primitives": {"type": "array", "items": {"type": "string"},
                       "minItems": 1, "uniqueItems": True},
        "weight":     {"type": "number", "minimum": 0, "default": 1.0},
    },
},
{
    "type": "object",
    "required": ["kind", "lemma_id", "sites"],
    "additionalProperties": False,
    "properties": {
        "kind":     {"const": "use_lemma"},
        "lemma_id": {"type": "string", "minLength": 1},
        "sites":    {"type": "array", "items": {"type": "integer", "minimum": 0},
                     "minItems": 1, "uniqueItems": True},
        "weight":   {"type": "number", "minimum": 0, "default": 1.0},
    },
},
```

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/schema.py src/qft_pcn/bridge/tests/test_schema_v1.py
git -c commit.gpgsign=false commit -m "feat(bridge/dsl): four new constraint kinds (well_typed_subtree, example, vocabulary, use_lemma) per §2.4"
```

### W1.T5: Add optional top-level `decomposition` (§10.10)

- [ ] **Step 1: Write the failing tests**

Append to `test_schema_v1.py`:

```python
def _v1_with_decomposition(decomp: str) -> str:
    spec = _minimal_v1()
    return spec[:-1] + f', "decomposition": {decomp}' + spec[-1:]


def test_decomposition_accepted_with_children():
    child = """{
      "id": "child-1",
      "spec": {
        "version": "1",
        "fields": [{"name": "n", "cutoff": 4}],
        "sites": 2,
        "constraints": [{"kind": "local", "site": 0, "term": "n == 0", "weight": 1.0}],
        "observables": [{"site": 0, "field": "n", "op": "n"}],
        "search": {"method": "imag_time", "runtime": "mps", "steps": 10, "chi_max": 8, "dt": 0.05}
      },
      "integrates_at_sites": [0],
      "weight": 8.0
    }"""
    dsl = parse_and_validate(_v1_with_decomposition(
        f'{{"children": [{child}], "execution": "parallel"}}'
    ))
    assert dsl["decomposition"]["children"][0]["id"] == "child-1"


def test_decomposition_omitted_is_ok():
    dsl = parse_and_validate(_minimal_v1())
    assert "decomposition" not in dsl or dsl.get("decomposition") is None


def test_decomposition_rejects_empty_children():
    with pytest.raises(BadSchemaError):
        parse_and_validate(_v1_with_decomposition('{"children": []}'))
```

- [ ] **Step 2: Run failures**
Expected: 2 FAIL (children-accept and reject-empty)

- [ ] **Step 3: Add decomposition subschema**

In `schema.py`, add to top-level `properties`:

```python
"decomposition": {
    "type": "object",
    "additionalProperties": False,
    "required": ["children"],
    "properties": {
        "children": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "spec", "integrates_at_sites"],
                "additionalProperties": False,
                "properties": {
                    "id":     {"type": "string", "pattern": "^[a-zA-Z][a-zA-Z0-9_-]{0,63}$"},
                    "spec":   {"$ref": "#"},      # recursive: child is a full DSL
                    "integrates_at_sites": {"type": "array",
                                            "items": {"type": "integer", "minimum": 0},
                                            "minItems": 1, "uniqueItems": True},
                    "weight": {"type": "number", "minimum": 0, "default": 1.0},
                    "dependsOn": {"type": "array",
                                  "items": {"type": "string"},
                                  "uniqueItems": True, "default": []},
                },
            },
        },
        "execution": {"enum": ["parallel", "sequential"], "default": "parallel"},
    },
},
```

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/schema.py src/qft_pcn/bridge/tests/test_schema_v1.py
git -c commit.gpgsign=false commit -m "feat(bridge/dsl): optional top-level decomposition per §10.10/§2.10"
```

---

## W2 — Pythonic-restricted predicate compiler

**Files:**
- Create: `src/qft_pcn/bridge/dsl/predicate_ast.py`
- Test: `src/qft_pcn/bridge/tests/test_predicate_ast.py`

### W2.T1: AllowedNodeVisitor with closed allowlist

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/bridge/tests/test_predicate_ast.py`:

```python
"""§2.7 Pythonic-restricted predicate AST visitor."""
import pytest
from qft_pcn.bridge.dsl.predicate_ast import parse_predicate, PredicateRejected


@pytest.mark.parametrize("term", [
    "expr == 'Lambda'",
    "type(expr) == 'Nat'",
    "type(arg1) == type(arg2)",
    "binder(expr) == 1",
    "node_kind == 'App' and type(expr) != 'unknown'",
    "value(expr) in [0, 1, 2]",
    "not (expr == 'Nil')",
])
def test_allowed_terms_parse(term):
    parse_predicate(term)   # no exception


@pytest.mark.parametrize("term", [
    "lambda x: x",                  # Lambda banned
    "[x for x in xs]",              # ListComp banned
    "__import__('os')",             # Call to non-allowlisted
    "expr.__class__",               # Attribute on bound name banned
    "x = 1",                        # Assign banned (also: parse mode rejects)
    "f'{expr}'",                    # FormattedValue banned
    "x if y else z",                # IfExp banned
])
def test_disallowed_terms_reject(term):
    with pytest.raises(PredicateRejected):
        parse_predicate(term)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest src/qft_pcn/bridge/tests/test_predicate_ast.py -v`
Expected: FAIL (module doesn't exist)

- [ ] **Step 3: Implement `predicate_ast.py`**

Create `src/qft_pcn/bridge/dsl/predicate_ast.py`:

```python
"""§2.7 — Pythonic-restricted predicate parser.

Compiles a predicate string into a validated AST whose evaluation is safe
to delegate to a small interpreter. The grammar is a strict subset of
Python expressions: see spec §2.7 for the closed allowlist of node types
and helper names.
"""
from __future__ import annotations
import ast
from dataclasses import dataclass


HELPER_NAMES = frozenset({"type", "value", "binder", "kind", "int", "str", "bool", "len"})
BOUND_NAMES_LOCAL = frozenset({"expr"})   # the site, plus user-declared field names
BOUND_NAMES_TWO_SITE = frozenset({"arg1", "arg2", "expr"})


ALLOWED_NODES: frozenset = frozenset({
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult,
    ast.Name, ast.Constant, ast.Load,
    ast.Call, ast.Subscript, ast.Tuple, ast.List, ast.Index,
})


class PredicateRejected(ValueError):
    """Raised when a predicate uses banned syntax or names."""


@dataclass(frozen=True)
class ParsedPredicate:
    tree: ast.Expression
    free_names: frozenset[str]


def parse_predicate(term: str, *, two_site: bool = False,
                    user_field_names: frozenset[str] = frozenset()) -> ParsedPredicate:
    """Parse `term` as a §2.7-allowed predicate. Returns the AST + free names."""
    try:
        tree = ast.parse(term, mode="eval")
    except SyntaxError as exc:
        raise PredicateRejected(f"syntax error: {exc.msg}") from exc

    bound = (BOUND_NAMES_TWO_SITE if two_site else BOUND_NAMES_LOCAL) | user_field_names
    visitor = _Visitor(allowed_callees=HELPER_NAMES, bound_names=bound)
    visitor.visit(tree)
    return ParsedPredicate(tree=tree, free_names=visitor.free_names)


class _Visitor(ast.NodeVisitor):
    def __init__(self, allowed_callees: frozenset[str], bound_names: frozenset[str]):
        self.allowed_callees = allowed_callees
        self.bound_names = bound_names
        self.free_names: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if type(node) not in ALLOWED_NODES:
            raise PredicateRejected(
                f"banned AST node: {type(node).__name__}"
            )
        super().generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name):
            raise PredicateRejected(
                "calls must target a bare identifier from the helper allowlist"
            )
        if node.func.id not in self.allowed_callees:
            raise PredicateRejected(
                f"call to non-allowlisted helper '{node.func.id}'"
            )
        for arg in node.args:
            self.visit(arg)
        if node.keywords:
            raise PredicateRejected("keyword arguments are not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        raise PredicateRejected("attribute access is not allowed")

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in self.allowed_callees or node.id in self.bound_names:
            return
        self.free_names.add(node.id)
```

Note on `free_names`: bare identifiers that are neither helpers nor explicitly bound are recorded — the compiler in W3 will validate them against the spec's declared field names.

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/predicate_ast.py src/qft_pcn/bridge/tests/test_predicate_ast.py
git -c commit.gpgsign=false commit -m "feat(bridge/dsl): Pythonic-restricted predicate AST visitor per §2.7"
```

### W2.T2: Predicate enumeration → satisfying-label sets

The compiler needs to evaluate a parsed predicate against every basis-label assignment to find which assignments satisfy it. Implementation: a tiny interpreter over the parsed AST.

- [ ] **Step 1: Write the failing test**

Append to `test_predicate_ast.py`:

```python
from qft_pcn.bridge.dsl.predicate_ast import evaluate_predicate


def test_evaluate_simple_equality():
    p = parse_predicate("expr == 'Lambda'")
    assert evaluate_predicate(p, {"expr": "Lambda"}) is True
    assert evaluate_predicate(p, {"expr": "App"}) is False


def test_evaluate_helper_call():
    p = parse_predicate("type(expr) == 'Nat'")
    # The interpreter resolves type(x) by looking up "type" in the env's helpers
    env = {"expr": "Zero", "_helpers": {"type": lambda x: "Nat" if x == "Zero" else "?"}}
    assert evaluate_predicate(p, env) is True


def test_evaluate_two_site_helper():
    p = parse_predicate("type(arg1) == type(arg2)", two_site=True)
    env = {"arg1": "Zero", "arg2": "Succ",
           "_helpers": {"type": lambda x: "Nat"}}
    assert evaluate_predicate(p, env) is True


def test_evaluate_boolop_and_compare():
    p = parse_predicate("node_kind == 'App' and value(expr) in [0, 1, 2]",
                        user_field_names=frozenset({"node_kind"}))
    env = {"expr": "x", "node_kind": "App",
           "_helpers": {"value": lambda _: 1}}
    assert evaluate_predicate(p, env) is True
    env2 = {**env, "node_kind": "Var"}
    assert evaluate_predicate(p, env2) is False
```

- [ ] **Step 2: Run to verify failure**
Expected: FAIL (no `evaluate_predicate`)

- [ ] **Step 3: Implement `evaluate_predicate`**

Append to `src/qft_pcn/bridge/dsl/predicate_ast.py`:

```python
def evaluate_predicate(parsed: ParsedPredicate, env: dict) -> bool:
    """Evaluate a parsed predicate against `env`.

    `env` is a dict of bound names → values. Helper functions are looked up
    from `env["_helpers"]` (or the empty dict if absent). Returns a bool.
    """
    helpers = env.get("_helpers", {})
    return bool(_eval(parsed.tree.body, env, helpers))


def _eval(node: ast.AST, env: dict, helpers: dict):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in helpers:
            return helpers[node.id]
        if node.id in env:
            return env[node.id]
        raise PredicateRejected(f"unbound name during eval: {node.id}")
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, env, helpers) for v in node.values]
        return all(vals) if isinstance(node.op, ast.And) else any(vals)
    if isinstance(node, ast.UnaryOp):
        v = _eval(node.operand, env, helpers)
        if isinstance(node.op, ast.Not):
            return not v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
    if isinstance(node, ast.BinOp):
        l, r = _eval(node.left, env, helpers), _eval(node.right, env, helpers)
        if isinstance(node.op, ast.Add): return l + r
        if isinstance(node.op, ast.Sub): return l - r
        if isinstance(node.op, ast.Mult): return l * r
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env, helpers)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval(comp, env, helpers)
            ok = _cmp(op, left, right)
            if not ok:
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        fn = _eval(node.func, env, helpers)
        args = [_eval(a, env, helpers) for a in node.args]
        return fn(*args)
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_eval(e, env, helpers) for e in node.elts]
    raise PredicateRejected(f"unsupported at eval time: {type(node).__name__}")


def _cmp(op: ast.cmpop, left, right) -> bool:
    if isinstance(op, ast.Eq):    return left == right
    if isinstance(op, ast.NotEq): return left != right
    if isinstance(op, ast.Lt):    return left < right
    if isinstance(op, ast.LtE):   return left <= right
    if isinstance(op, ast.Gt):    return left > right
    if isinstance(op, ast.GtE):   return left >= right
    if isinstance(op, ast.In):    return left in right
    if isinstance(op, ast.NotIn): return left not in right
    raise PredicateRejected(f"unsupported comparator: {type(op).__name__}")
```

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/predicate_ast.py src/qft_pcn/bridge/tests/test_predicate_ast.py
git -c commit.gpgsign=false commit -m "feat(bridge/dsl): predicate evaluator (AST → bool given env) per §2.7"
```

---

## W3 — Four new constraint-kind compilers

**Files:**
- Create: `src/qft_pcn/bridge/runtime/hamiltonian_compiler.py`
- Test: `src/qft_pcn/bridge/tests/test_hamiltonian_compiler.py`

Each compiler returns a Hermitian, positive-semidefinite operator. Tests assert (a) Hermiticity, (b) ground-state energy = 0 on a satisfying state, (c) energy > 0 on a violating state (§13.2).

Before writing W3, inspect the existing `bridge/dsl/compiler.py` and `bridge/runtime/hamiltonian.py` to understand how `local`/`two_site` constraints are currently compiled — the new module reuses the local-Hilbert-space conventions established there.

### W3.T1: `vocabulary` compiler

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/bridge/tests/test_hamiltonian_compiler.py`:

```python
"""§2.4 four new constraint-kind compilers — Hermiticity + ground-state checks."""
import numpy as np
import pytest

from qft_pcn.bridge.runtime.hamiltonian_compiler import (
    compile_vocabulary, compile_well_typed_subtree, compile_example, compile_use_lemma,
)


def test_vocabulary_hermitian_psd():
    # 4 sites, node_kind cutoff 8, allowed primitives = first 3 labels
    H = compile_vocabulary(
        primitives=["a", "b", "c"],
        vocab=["a", "b", "c", "d", "e", "f", "g", "h"],
        n_sites=4, kind_cutoff=8, weight=1.0,
    )
    assert H.shape == (8**4, 8**4)
    assert np.allclose(H, H.conj().T)            # Hermitian
    eigs = np.linalg.eigvalsh(H)
    assert eigs[0] >= -1e-10                     # PSD
    assert eigs[0] < 1e-10                       # has a zero eigenstate
```

- [ ] **Step 2: Run to verify failure**
Run: `uv run pytest src/qft_pcn/bridge/tests/test_hamiltonian_compiler.py::test_vocabulary_hermitian_psd -v`
Expected: FAIL (module doesn't exist)

- [ ] **Step 3: Implement vocabulary compiler**

Create `src/qft_pcn/bridge/runtime/hamiltonian_compiler.py`:

```python
"""§2.4 constraint-kind compilers.

Each returns a Hermitian, positive-semidefinite operator on the appropriate
local Hilbert space. Energy = 0 iff the constraint is satisfied (§13.2).

Conventions:
- Local Hilbert space for a single site of `kind_cutoff` labels = a (cutoff x
  cutoff) computational basis indexed by label position in the spec's vocab.
- Multi-site operators are built via numpy tensor products; the encoder
  cap D_LOCAL=65536 applies (validated upstream in dsl/schema.py).
"""
from __future__ import annotations
from typing import Sequence
import numpy as np


def compile_vocabulary(*, primitives: Sequence[str], vocab: Sequence[str],
                       n_sites: int, kind_cutoff: int, weight: float) -> np.ndarray:
    """Sum of per-site projectors onto disallowed labels.

    P_site = diag( 0 if label in primitives else 1 )
    H = weight * Σ_site (P_site ⊗ I_rest)
    """
    if kind_cutoff != len(vocab):
        raise ValueError(f"kind_cutoff={kind_cutoff} but vocab has {len(vocab)} labels")
    allowed = {vocab.index(p) for p in primitives}
    diag1 = np.array([0.0 if i in allowed else 1.0 for i in range(kind_cutoff)])
    proj_site = np.diag(diag1)
    I = np.eye(kind_cutoff)
    dim = kind_cutoff ** n_sites
    H = np.zeros((dim, dim), dtype=float)
    for k in range(n_sites):
        ops = [I] * n_sites
        ops[k] = proj_site
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H = H + weight * term
    return H
```

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/runtime/hamiltonian_compiler.py src/qft_pcn/bridge/tests/test_hamiltonian_compiler.py
git -c commit.gpgsign=false commit -m "feat(bridge/runtime): vocabulary constraint compiler (PSD per §13.2)"
```

### W3.T2: `well_typed_subtree` compiler

This compiles to the §10.2 typing-rule projectors (T-Var, T-App, T-Abs) rooted at `root`. The architecture lays out the exact rules in §10.2; consult that section for the projector forms. Output is a sum of two-site projectors penalizing each rule violation by `weight`.

- [ ] **Step 1: Write the failing test**

Append to `test_hamiltonian_compiler.py`:

```python
def test_well_typed_subtree_hermitian_psd():
    # Minimal 3-site subtree; fields = node_kind(4), type(4), with one
    # known-satisfying assignment in the spec
    H = compile_well_typed_subtree(
        root=0, n_sites=3,
        kind_cutoff=4, type_cutoff=4,
        vocab=["Var", "App", "Lambda", "Const"],
        types=["Nat", "Bool", "NatToNat", "BoolToBool"],
        weight=10.0,
    )
    d = (4 * 4) ** 3
    assert H.shape == (d, d)
    assert np.allclose(H, H.conj().T)
    eigs = np.linalg.eigvalsh(H)
    assert eigs[0] >= -1e-9
    # At least one well-typed configuration exists → ground energy is zero
    assert eigs[0] < 1e-9
```

- [ ] **Step 2: Run to verify failure**
Expected: FAIL (function not defined)

- [ ] **Step 3: Implement `compile_well_typed_subtree`**

The §10.2 rules in operator form (paraphrased here, full math in architecture §10.2):

- **T-Var**: at a site with `node_kind = Var`, the site's `type` field must match the type of its binder. Compile as a two-site projector between the Var site and the bound site.
- **T-App**: at a site with `node_kind = App`, the function child's `type` must be `A → B` and the argument child's `type` must be `A`; the App site's `type` must be `B`. Compile as a three-site projector reduced to two two-site couplings (App↔fn-child, App↔arg-child) plus a one-site consistency.
- **T-Abs**: at a site with `node_kind = Lambda`, the body's `type` must combine with the binder's `type` to form the Lambda's `type`. Compile analogously.

For the first cut, implement a *simplified* well-typed-subtree: a one-site projector that fires only on `node_kind ∈ {Var, App, Lambda}` requiring `type ≠ "unknown"`. This is enough to satisfy the test's PSD + ground-state property; the full §10.2 elaboration lands in W3.T2b.

```python
def compile_well_typed_subtree(*, root: int, n_sites: int,
                               kind_cutoff: int, type_cutoff: int,
                               vocab: Sequence[str], types: Sequence[str],
                               weight: float) -> np.ndarray:
    """First-cut well-typed-subtree: one-site projector requiring type≠'unknown'
    at every typed-node site under `root`. Full §10.2 T-Var/T-App/T-Abs
    elaboration is W3.T2b. PSD by construction (§13.2)."""
    if "unknown" not in types:
        unknown_idx = -1   # no penalty if 'unknown' isn't in the type vocab
    else:
        unknown_idx = types.index("unknown")
    typed_kinds = {vocab.index(k) for k in ("Var", "App", "Lambda") if k in vocab}
    d_site = kind_cutoff * type_cutoff
    dim = d_site ** n_sites
    H = np.zeros((dim, dim), dtype=float)
    if unknown_idx < 0 or not typed_kinds:
        return H
    # local projector: |kind in typed_kinds⟩ ⊗ |type == unknown⟩
    proj_kind = np.zeros((kind_cutoff, kind_cutoff))
    for k in typed_kinds:
        proj_kind[k, k] = 1.0
    proj_type = np.zeros((type_cutoff, type_cutoff))
    proj_type[unknown_idx, unknown_idx] = 1.0
    p_site = np.kron(proj_kind, proj_type)
    I = np.eye(d_site)
    for s in range(n_sites):
        ops = [I] * n_sites
        ops[s] = p_site
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        H = H + weight * term
    return H
```

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/runtime/hamiltonian_compiler.py src/qft_pcn/bridge/tests/test_hamiltonian_compiler.py
git -c commit.gpgsign=false commit -m "feat(bridge/runtime): well_typed_subtree first-cut compiler (§10.2 simplified, PSD)"
```

### W3.T2b: Full §10.2 T-Var/T-App/T-Abs elaboration

After W3.T2's first-cut lands, dispatch a follow-on subagent with the architecture §10.2 typing rules in full as the brief. Acceptance: a corpus of 5 hand-encoded well-typed ASTs evaluate to ⟨H⟩ ≈ 0, and 5 hand-encoded ill-typed ASTs evaluate to ⟨H⟩ > weight·0.5. Tests live in `test_hamiltonian_compiler.py`. Commit message: `feat(bridge/runtime): full §10.2 T-Var/T-App/T-Abs in well_typed_subtree`.

### W3.T3: `example` compiler (delegates to §10.3 evaluation Hamiltonian)

The architecture's §10.3 evaluation Hamiltonian already encodes beta-reduction as Trotter steps and is the substrate for example constraints. Verify whether `src/qft_pcn/logic/evaluation_hamiltonian.py` exists; if so, this compiler is a thin adapter that takes `(input, output, weight)` and instantiates the eval Ham on auxiliary sites.

- [ ] **Step 1: Audit step**

Run: `ls src/qft_pcn/logic/ 2>/dev/null && grep -l evaluation_hamiltonian src/qft_pcn/logic/*.py 2>/dev/null`

If the file exists, proceed to step 2. If not, implement a first-cut `compile_example` that returns a zero operator + warns once via `logging.warning`, and record the gap in `EXTENSIONS.md` under "example constraint: needs §10.3 eval Ham".

- [ ] **Step 2: Write the failing test**

Append to `test_hamiltonian_compiler.py`:

```python
def test_example_compiler_hermitian():
    H = compile_example(
        input_expr="[]", output_expr="0",
        n_sites=4, fields=[("node_kind", 4), ("value", 4)],
        weight=5.0,
    )
    d = (4 * 4) ** 4
    assert H.shape == (d, d)
    assert np.allclose(H, H.conj().T)
```

- [ ] **Step 3: Implement adapter (or first-cut zero op + EXTENSIONS entry)**

```python
def compile_example(*, input_expr: str, output_expr: str,
                    n_sites: int, fields: Sequence[tuple[str, int]],
                    weight: float) -> np.ndarray:
    """Adapter to §10.3 evaluation Hamiltonian.

    If src/qft_pcn/logic/evaluation_hamiltonian.py is present, instantiate
    it on auxiliary sites; otherwise return a zero op and record the gap
    in EXTENSIONS.md (no placeholders allowed).
    """
    try:
        from qft_pcn.logic.evaluation_hamiltonian import (
            instantiate_example_term,
        )
        return instantiate_example_term(
            input_expr=input_expr, output_expr=output_expr,
            n_sites=n_sites, fields=fields, weight=weight,
        )
    except ImportError:
        import logging
        logging.warning(
            "compile_example: §10.3 evaluation_hamiltonian not yet wired; "
            "see EXTENSIONS.md 'example constraint: needs §10.3 eval Ham'."
        )
        d = 1
        for _, c in fields:
            d *= c
        dim = d ** n_sites
        return np.zeros((dim, dim), dtype=float)
```

- [ ] **Step 4: Run tests**
Expected: PASS (zero op is Hermitian)

- [ ] **Step 5: Update `EXTENSIONS.md` if the import failed**

Add an entry to `EXTENSIONS.md` under a "DSL constraint gaps" heading:

```markdown
## DSL constraint gaps (2026-05-24)

### `example` constraint depends on §10.3 evaluation Hamiltonian

- **Status:** stub returns zero op + WARN log
- **Needed:** `src/qft_pcn/logic/evaluation_hamiltonian.py::instantiate_example_term(input_expr, output_expr, n_sites, fields, weight)`
- **Spec:** architecture §10.3 (constraint-based: evaluation IS ground-state finding)
- **Affected presets:** `length-synthesis` (§15), `list-reverse-length`
```

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/runtime/hamiltonian_compiler.py src/qft_pcn/bridge/tests/test_hamiltonian_compiler.py EXTENSIONS.md
git -c commit.gpgsign=false commit -m "feat(bridge/runtime): example constraint adapter to §10.3 eval Ham (graceful fallback)"
```

### W3.T4: `use_lemma` compiler (clamp via composition/promoter)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_use_lemma_compiler_hermitian():
    # Mock lemma: 2-site MPS in product form |0⟩⊗|0⟩
    import numpy as np
    n_sites, kind_cutoff = 4, 4
    sites = [1, 2]
    fake_lemma_state = np.zeros(kind_cutoff ** len(sites), dtype=complex)
    fake_lemma_state[0] = 1.0
    H = compile_use_lemma(
        lemma_id="test-lemma",
        sites=sites, n_sites=n_sites, kind_cutoff=kind_cutoff,
        weight=8.0,
        lemma_lookup=lambda _id: fake_lemma_state,
    )
    d = kind_cutoff ** n_sites
    assert H.shape == (d, d)
    assert np.allclose(H, H.conj().T)
    # Negative-projector form: -W|Ψ⟩⟨Ψ| → ground energy is -W
    eigs = np.linalg.eigvalsh(H)
    assert eigs[0] < -8.0 + 1e-9
```

- [ ] **Step 2: Run to verify failure**
Expected: FAIL

- [ ] **Step 3: Implement `compile_use_lemma`**

```python
def compile_use_lemma(*, lemma_id: str, sites: Sequence[int],
                      n_sites: int, kind_cutoff: int, weight: float,
                      lemma_lookup) -> np.ndarray:
    """§10.8 clamp via strong projector −W|Ψ_L⟩⟨Ψ_L| on the listed sites.

    `lemma_lookup` is injected (in production: composition/lemma_library_adapter
    .get_state). It returns a flattened MPS state vector on the lemma's sites.
    """
    state = lemma_lookup(lemma_id)
    expected_dim = kind_cutoff ** len(sites)
    if state.shape != (expected_dim,):
        raise ValueError(
            f"lemma {lemma_id!r} dim {state.shape} != expected {(expected_dim,)}"
        )
    proj_local = np.outer(state, state.conj())
    I = np.eye(kind_cutoff)
    site_set = set(sites)
    ops = []
    lemma_dims_inserted = False
    for s in range(n_sites):
        if s in site_set:
            if not lemma_dims_inserted:
                ops.append(proj_local)
                lemma_dims_inserted = True
            # subsequent lemma sites already inside proj_local; skip
        else:
            ops.append(I)
    term = ops[0]
    for op in ops[1:]:
        term = np.kron(term, op)
    return -weight * term
```

NOTE: above assumes the lemma sites are contiguous. For non-contiguous lemma sites, the implementation needs a permutation step — record that as a follow-on if W6b's acceptance test hits it.

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/runtime/hamiltonian_compiler.py src/qft_pcn/bridge/tests/test_hamiltonian_compiler.py
git -c commit.gpgsign=false commit -m "feat(bridge/runtime): use_lemma constraint compiler (−W projector clamp per §10.8)"
```

---

## W4 — MERA routing + new `/dsl/v1/run` endpoint

**Files:**
- Modify: `src/qft_pcn/bridge/runtime/evolution.py`
- Modify: `src/qft_pcn/bridge/api.py`
- Modify: `src/qft_pcn/bridge/runtime/__init__.py` (or `pipeline.py` — verify which is the entrypoint)
- Test: `src/qft_pcn/bridge/tests/test_api_v1.py` (NEW)

### W4.T1: MERA routing in evolution

Inspect `src/qft_pcn/bridge/runtime/evolution.py` to see how it currently routes to flat-MPS TEBD. Add a branch on `search["runtime"]`:

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/bridge/tests/test_evolution_runtime.py`:

```python
"""§2.6 search.runtime dispatches mps vs mera."""
import pytest
from qft_pcn.bridge.runtime.evolution import evolve_for_search


def test_runtime_mps_path():
    res = evolve_for_search(
        # use the same builders as the existing test fixture
        runtime="mps", n_sites=2, kind_cutoff=2,
        hamiltonian=None,  # placeholder — real fixture below
        steps=2, chi_max=4, dt=0.05,
    )
    assert res is not None


def test_runtime_mera_path():
    res = evolve_for_search(
        runtime="mera", n_sites=4, kind_cutoff=2,
        hamiltonian=None,
        steps=2, chi_max=4, dt=0.05,
    )
    assert res is not None


def test_runtime_invalid_raises():
    with pytest.raises(ValueError, match="runtime"):
        evolve_for_search(
            runtime="bogus", n_sites=2, kind_cutoff=2,
            hamiltonian=None, steps=2, chi_max=4, dt=0.05,
        )
```

Update with real fixtures from the existing test suite before running. Read `src/qft_pcn/bridge/tests/test_runtime.py` (or equivalent) to find the standard builder.

- [ ] **Step 2: Run to verify failure**
Expected: FAIL (`evolve_for_search` or branch not present)

- [ ] **Step 3: Add routing**

In `bridge/runtime/evolution.py`, surface an `evolve_for_search(runtime, ...)` wrapper that dispatches:
- `runtime == "mps"` → existing flat-MPS TEBD path
- `runtime == "mera"` → `from qft_pcn.qft.mera_evolution import mera_trotter_step` (verify path)
- else → `raise ValueError(f"unknown search.runtime: {runtime!r}")`

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/runtime/evolution.py src/qft_pcn/bridge/tests/test_evolution_runtime.py
git -c commit.gpgsign=false commit -m "feat(bridge/runtime): route search.runtime mps|mera per §2.6"
```

### W4.T2: `use_lemma` constraint compilation hooked into the bridge runtime pipeline

Wire `compile_use_lemma` from W3.T4 into the pipeline that builds the Hamiltonian from a DSL. The lemma_lookup parameter resolves to `composition/lemma_library_adapter.py::get_state(lemma_id)`.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/bridge/tests/test_use_lemma_pipeline.py`:

```python
"""§10.8 use_lemma end-to-end through the bridge pipeline."""
import numpy as np
import pytest

from qft_pcn.bridge.runtime.pipeline import compile_dsl_to_hamiltonian


def test_use_lemma_compiles_via_lemma_library(monkeypatch):
    # Stub the lemma library adapter to return a known state
    state = np.zeros(4, dtype=complex); state[0] = 1.0
    monkeypatch.setattr(
        "qft_pcn.composition.lemma_library_adapter.get_state",
        lambda _id: state,
    )
    dsl = {
        "version": "1",
        "fields": [{"name": "n", "cutoff": 2}],
        "sites": 4,
        "constraints": [
            {"kind": "use_lemma", "lemma_id": "x", "sites": [1, 2], "weight": 8.0},
        ],
        "observables": [{"site": 0, "field": "n", "op": "n"}],
        "search": {"method": "imag_time", "runtime": "mps",
                   "steps": 1, "chi_max": 4, "dt": 0.05},
    }
    H = compile_dsl_to_hamiltonian(dsl)
    assert H.shape == (2**4, 2**4)
    assert np.allclose(H, H.conj().T)
```

- [ ] **Step 2: Run to verify failure**
Expected: FAIL

- [ ] **Step 3: Wire into the pipeline**

In `src/qft_pcn/bridge/runtime/pipeline.py` (or whichever module currently builds Hamiltonians from a parsed DSL — audit before editing), extend the per-constraint dispatch to include the four new kinds. For `use_lemma`, import `from qft_pcn.composition.lemma_library_adapter import get_state` and pass `get_state` as `lemma_lookup`.

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/runtime/pipeline.py src/qft_pcn/bridge/tests/test_use_lemma_pipeline.py
git -c commit.gpgsign=false commit -m "feat(bridge/runtime): wire use_lemma → composition/lemma_library_adapter"
```

### W4.T3: `POST /dsl/v1/run` endpoint

Add a new FastAPI route in `src/qft_pcn/bridge/api.py`:

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/bridge/tests/test_api_v1.py`:

```python
"""§3.2 /dsl/v1/run endpoint accepts v1 schema and returns §2.8 shape."""
from fastapi.testclient import TestClient
from qft_pcn.bridge.api import app


client = TestClient(app)


def test_v1_run_returns_2_8_shape():
    body = {
        "version": "1",
        "fields": [{"name": "n", "cutoff": 2}],
        "sites": 2,
        "constraints": [{"kind": "local", "site": 0, "term": "n == 0", "weight": 1.0}],
        "observables": [{"site": 0, "field": "n", "op": "n"}],
        "search": {"method": "imag_time", "runtime": "mps",
                   "steps": 4, "chi_max": 4, "dt": 0.05},
    }
    r = client.post("/dsl/v1/run", json=body)
    assert r.status_code == 200
    out = r.json()
    assert set(out.keys()) >= {"observable_values",
                               "residual_constraint_energies", "diagnostics"}
    assert set(out["diagnostics"].keys()) >= {
        "final_energy", "step_history", "converged",
    }
```

- [ ] **Step 2: Run to verify failure**
Expected: FAIL (404)

- [ ] **Step 3: Add the route**

In `bridge/api.py`:

```python
from fastapi import HTTPException
from .dsl.schema import parse_and_validate
from .errors import BadJsonError, BadSchemaError
import json


@app.post("/dsl/v1/run")
def dsl_v1_run(spec: dict):
    try:
        validated = parse_and_validate(json.dumps(spec))
    except (BadJsonError, BadSchemaError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    from .runtime.pipeline import run_dsl_v1
    return run_dsl_v1(validated)
```

And in `bridge/runtime/pipeline.py`, add `run_dsl_v1(dsl)` that builds the Hamiltonian, runs `evolve_for_search`, and returns the §2.8 shape.

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/api.py src/qft_pcn/bridge/runtime/pipeline.py src/qft_pcn/bridge/tests/test_api_v1.py
git -c commit.gpgsign=false commit -m "feat(bridge/api): POST /dsl/v1/run returns §2.8 shape"
```

---

## W7 — Decomposition (§10.10)

**Files:**
- Create: `src/qft_pcn/bridge/decomposition.py`
- Test: `src/qft_pcn/bridge/tests/test_decomposition.py`

### W7.T1: Child DAG validator (cycle detection)

- [ ] **Step 1: Write the failing test**

```python
"""§10.10 decomposition: cycle detection, parallel/sequential ordering."""
import pytest
from qft_pcn.bridge.decomposition import validate_dag, CycleError


def test_acyclic_passes():
    children = [
        {"id": "a", "dependsOn": [], "spec": {}, "integrates_at_sites": [0]},
        {"id": "b", "dependsOn": ["a"], "spec": {}, "integrates_at_sites": [1]},
    ]
    validate_dag(children)


def test_cycle_rejects():
    children = [
        {"id": "a", "dependsOn": ["b"], "spec": {}, "integrates_at_sites": [0]},
        {"id": "b", "dependsOn": ["a"], "spec": {}, "integrates_at_sites": [1]},
    ]
    with pytest.raises(CycleError):
        validate_dag(children)


def test_unknown_dep_rejects():
    children = [
        {"id": "a", "dependsOn": ["ghost"], "spec": {}, "integrates_at_sites": [0]},
    ]
    with pytest.raises(ValueError, match="unknown"):
        validate_dag(children)
```

- [ ] **Step 2: Run failure**
Expected: FAIL

- [ ] **Step 3: Implement**

```python
"""§10.10 cross-level decomposition. Parses the `decomposition` field of a
v1 DSL, validates the child DAG, runs children via composition/dispatcher
in topological order (parallel within layers), integrates each ground
state via composition/result_integrator, registers as transient lemmas,
and rewrites the parent's constraints to add auto-generated use_lemma
clamps."""
from __future__ import annotations
from typing import Any


class CycleError(ValueError): ...


def validate_dag(children: list[dict[str, Any]]) -> list[list[str]]:
    """Kahn topological sort. Returns layers (lists of ids) for parallel
    execution. Raises CycleError on cycles, ValueError on unknown deps."""
    ids = [c["id"] for c in children]
    id_set = set(ids)
    deps = {c["id"]: set(c.get("dependsOn") or []) for c in children}
    for cid, ds in deps.items():
        unknown = ds - id_set
        if unknown:
            raise ValueError(f"child {cid!r} dependsOn unknown: {sorted(unknown)}")
    layers: list[list[str]] = []
    remaining = dict(deps)
    while remaining:
        free = [cid for cid, ds in remaining.items() if not ds]
        if not free:
            raise CycleError(f"cycle in dependsOn among: {sorted(remaining)}")
        layers.append(sorted(free))
        for cid in free:
            remaining.pop(cid)
        for ds in remaining.values():
            ds -= set(free)
    return layers
```

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/decomposition.py src/qft_pcn/bridge/tests/test_decomposition.py
git -c commit.gpgsign=false commit -m "feat(bridge/decomposition): child-DAG cycle detection per §10.10"
```

### W7.T2: Child dispatch + integration

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_children_dispatch_and_integrate(monkeypatch):
    from qft_pcn.bridge.decomposition import run_decomposition

    # Stub the per-child runner to return a known ground-state vector
    import numpy as np
    fake_state = np.array([1.0+0j, 0.0])
    runs = []
    def fake_run(spec):
        runs.append(spec)
        return {"observable_values": {}, "residual_constraint_energies": {0: 0.0},
                "diagnostics": {"final_energy": 0.0, "step_history": [],
                                "converged": True},
                "_ground_state": fake_state}
    monkeypatch.setattr("qft_pcn.bridge.runtime.pipeline.run_dsl_v1", fake_run)

    decomp = {
        "execution": "parallel",
        "children": [
            {"id": "c1", "spec": {"version": "1"}, "integrates_at_sites": [0],
             "weight": 1.0, "dependsOn": []},
            {"id": "c2", "spec": {"version": "1"}, "integrates_at_sites": [1],
             "weight": 1.0, "dependsOn": ["c1"]},
        ],
    }
    children_results, augmented_constraints = run_decomposition(decomp)
    assert len(runs) == 2
    assert children_results["c1"]["diagnostics"]["converged"]
    # Two auto-generated use_lemma constraints appended
    assert sum(1 for c in augmented_constraints if c["kind"] == "use_lemma") == 2
```

- [ ] **Step 2: Run failure**
Expected: FAIL

- [ ] **Step 3: Implement `run_decomposition`**

```python
def run_decomposition(decomp: dict[str, Any]) -> tuple[dict[str, dict], list[dict]]:
    """Run children in topological layers, return (per-child results, list of
    auto-generated use_lemma constraints to append to the parent's Hamiltonian).

    Failed children (residual ≥ ε_register) propagate as exceptions; the API
    layer translates to a 400/422 surfaced to the caller (§10.10).
    """
    from .runtime.pipeline import run_dsl_v1
    from qft_pcn.composition.lemma_library_adapter import register_transient

    layers = validate_dag(decomp["children"])
    children_by_id = {c["id"]: c for c in decomp["children"]}
    results: dict[str, dict] = {}
    augmented: list[dict] = []
    for layer in layers:
        for cid in layer:
            child = children_by_id[cid]
            r = run_dsl_v1(child["spec"])
            results[cid] = r
            # Register the child's ground state as a transient lemma
            transient_id = f"_decomp_{cid}"
            register_transient(transient_id, r.get("_ground_state"))
            augmented.append({
                "kind": "use_lemma",
                "lemma_id": transient_id,
                "sites": list(child["integrates_at_sites"]),
                "weight": float(child.get("weight", 1.0)),
            })
    return results, augmented
```

If `register_transient` doesn't exist in `lemma_library_adapter`, add it as a thin wrapper that stores into an in-memory session dict; surface that dict via a sibling `get_state` lookup that falls through transients first.

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/decomposition.py src/qft_pcn/bridge/tests/test_decomposition.py src/qft_pcn/composition/lemma_library_adapter.py
git -c commit.gpgsign=false commit -m "feat(bridge/decomposition): dispatch children + auto-generate use_lemma per §10.10"
```

### W7.T3: Wire decomposition into `run_dsl_v1`

- [ ] **Step 1: Write the failing test**

Append to `test_api_v1.py`:

```python
def test_v1_run_with_decomposition_dispatches_children(monkeypatch):
    # Stub child runs to return ground-state vectors
    import numpy as np
    fake_state = np.array([1.0+0j, 0.0])
    def fake_run_child(spec):
        return {"observable_values": {}, "residual_constraint_energies": {0: 0.0},
                "diagnostics": {"final_energy": 0.0, "step_history": [],
                                "converged": True},
                "_ground_state": fake_state}
    # ... wire monkeypatch for run_dsl_v1 call recursion guard ...

    body = { ...parent spec with decomposition... }
    r = client.post("/dsl/v1/run", json=body)
    assert r.status_code == 200
    out = r.json()
    assert "children" in out["diagnostics"]
    assert len(out["diagnostics"]["children"]) == 2
```

- [ ] **Step 2: Run failure**
Expected: FAIL

- [ ] **Step 3: Wire `run_dsl_v1` to call decomposition before parent**

In `bridge/runtime/pipeline.py::run_dsl_v1`, if the validated DSL has a `decomposition` field, call `run_decomposition()` first, append the returned `use_lemma` constraints to the parent's constraint list, then proceed with the normal compile + evolve. Return shape extended with `diagnostics.children`.

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/runtime/pipeline.py src/qft_pcn/bridge/tests/test_api_v1.py
git -c commit.gpgsign=false commit -m "feat(bridge): wire decomposition into /dsl/v1/run (§10.10 end-to-end)"
```

---

## W5 — Visualizer LLM rewrite + few-shot examples

**Files:**
- Modify: `src/qft_pcn/viz/llm.py`
- Create: `src/qft_pcn/viz/llm_examples/length_synthesis.json`
- Create: `src/qft_pcn/viz/llm_examples/peano_zero_axiom.json`
- Create: `src/qft_pcn/viz/llm_examples/stlc_id_function.json`
- Test: `src/qft_pcn/viz/tests/test_llm_v1.py`

### W5.T1: Author the three few-shot DSL examples

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/viz/tests/test_llm_v1.py`:

```python
"""Few-shot DSL examples must validate against the §9.2 v1 schema."""
import json
import pathlib
import pytest
from qft_pcn.bridge.dsl.schema import parse_and_validate


EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "llm_examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES_DIR.glob("*.json")))
def test_example_validates(path: pathlib.Path):
    parse_and_validate(path.read_text())
```

- [ ] **Step 2: Run failure**
Expected: FAIL (no examples)

- [ ] **Step 3: Author the three examples**

**`viz/llm_examples/length_synthesis.json`** — full §15.3 canonical:

```json
{
  "version": "1",
  "fields": [
    {"name": "node_kind", "cutoff": 12},
    {"name": "type",      "cutoff": 8},
    {"name": "binder_id", "cutoff": 16},
    {"name": "value",     "cutoff": 8}
  ],
  "sites": 12,
  "boundary": {
    "0": {"node_kind": "Lambda", "type": "List_a_to_Nat"}
  },
  "constraints": [
    {"kind": "well_typed_subtree", "root": 0, "weight": 10.0},
    {"kind": "example", "input": "[]",        "output": "0", "weight": 5.0},
    {"kind": "example", "input": "[x]",       "output": "1", "weight": 5.0},
    {"kind": "example", "input": "[x, y, z]", "output": "3", "weight": 5.0},
    {"kind": "vocabulary",
     "primitives": ["Match", "Cons", "Nil", "Succ", "Zero", "Var", "App"]}
  ],
  "observables": [
    {"site": 0,  "field": "node_kind", "op": "argmax"},
    {"site": 1,  "field": "node_kind", "op": "argmax"},
    {"site": 2,  "field": "node_kind", "op": "argmax"},
    {"site": 3,  "field": "node_kind", "op": "argmax"},
    {"site": 4,  "field": "node_kind", "op": "argmax"},
    {"site": 5,  "field": "node_kind", "op": "argmax"},
    {"site": 6,  "field": "node_kind", "op": "argmax"},
    {"site": 7,  "field": "node_kind", "op": "argmax"},
    {"site": 8,  "field": "node_kind", "op": "argmax"},
    {"site": 9,  "field": "node_kind", "op": "argmax"},
    {"site": 10, "field": "node_kind", "op": "argmax"},
    {"site": 11, "field": "node_kind", "op": "argmax"}
  ],
  "search": {"method": "imag_time", "runtime": "mera",
             "steps": 100, "chi_max": 32, "dt": 0.05}
}
```

**`viz/llm_examples/stlc_id_function.json`** — minimal smoke:

```json
{
  "version": "1",
  "fields": [
    {"name": "node_kind", "cutoff": 4},
    {"name": "type",      "cutoff": 4}
  ],
  "sites": 4,
  "boundary": {"0": {"node_kind": "Lambda"}},
  "constraints": [
    {"kind": "well_typed_subtree", "root": 0, "weight": 10.0},
    {"kind": "vocabulary", "primitives": ["Lambda", "Var", "App", "Const"]}
  ],
  "observables": [
    {"site": 0, "field": "node_kind", "op": "argmax"},
    {"site": 1, "field": "node_kind", "op": "argmax"},
    {"site": 2, "field": "node_kind", "op": "argmax"},
    {"site": 3, "field": "node_kind", "op": "argmax"}
  ],
  "search": {"method": "imag_time", "runtime": "mps",
             "steps": 30, "chi_max": 8, "dt": 0.05}
}
```

**`viz/llm_examples/peano_zero_axiom.json`** — §10.8 lemma demo, stage 2 only (stage 1 is implicit lemma registration):

```json
{
  "version": "1",
  "fields": [
    {"name": "node_kind", "cutoff": 8},
    {"name": "type",      "cutoff": 4}
  ],
  "sites": 8,
  "boundary": {"0": {"node_kind": "App"}},
  "constraints": [
    {"kind": "well_typed_subtree", "root": 0, "weight": 10.0},
    {"kind": "vocabulary",
     "primitives": ["App", "Var", "Zero", "Succ", "Add", "Eq", "Forall", "Const"]},
    {"kind": "use_lemma",
     "lemma_id": "peano-x-plus-0-eq-x",
     "sites": [2, 3, 4],
     "weight": 8.0}
  ],
  "observables": [
    {"site": 0, "field": "node_kind", "op": "argmax"},
    {"site": 7, "field": "node_kind", "op": "argmax"}
  ],
  "search": {"method": "imag_time", "runtime": "mera",
             "steps": 50, "chi_max": 16, "dt": 0.05}
}
```

- [ ] **Step 4: Run tests**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/llm_examples/ src/qft_pcn/viz/tests/test_llm_v1.py
git -c commit.gpgsign=false commit -m "feat(viz/llm): three §9.2 v1 few-shot examples (length, stlc-id, peano)"
```

### W5.T2: Rewrite `viz/llm.py` system prompt around §9.2

- [ ] **Step 1: Audit existing**

Read `src/qft_pcn/viz/llm.py`. Identify the system prompt constant. Identify any tests asserting the old prompt format.

- [ ] **Step 2: Write the failing test**

Append to `test_llm_v1.py`:

```python
def test_system_prompt_contains_v1_schema_keywords():
    from qft_pcn.viz.llm import build_system_prompt
    prompt = build_system_prompt()
    for kw in ["version", "fields", "sites", "constraints", "observables",
               "search", "runtime", "well_typed_subtree", "example",
               "vocabulary", "use_lemma", "argmax"]:
        assert kw in prompt, f"system prompt missing keyword: {kw}"


def test_system_prompt_includes_length_synthesis_few_shot():
    from qft_pcn.viz.llm import build_system_prompt
    prompt = build_system_prompt()
    assert '"runtime": "mera"' in prompt
    assert '"List_a_to_Nat"' in prompt
```

- [ ] **Step 3: Run failure**
Expected: FAIL

- [ ] **Step 4: Implement**

Rewrite `src/qft_pcn/viz/llm.py::build_system_prompt()` to:

1. Describe the §9.2 schema (a short prose preamble, ~10 lines).
2. Inline the three example JSONs from `llm_examples/` as few-shot.
3. Replace any physics-tuning language with constraint-DSL language.

Preserve the public API (`generate_dsl`, `verbalize`, `list_models`) and Ollama-host configuration. Only the prompt content changes.

- [ ] **Step 5: Run tests**
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/viz/llm.py src/qft_pcn/viz/tests/test_llm_v1.py
git -c commit.gpgsign=false commit -m "refactor(viz/llm): rewrite system prompt around §9.2 v1 DSL + three few-shot examples"
```

### W5.T3: Ollama regression test (skippable when Ollama is unreachable)

- [ ] **Step 1: Write the test**

Append:

```python
def test_ollama_emits_validating_dsl_for_length_prompt():
    pytest.importorskip("ollama")
    try:
        from qft_pcn.viz.llm import generate_dsl
        emitted = generate_dsl(
            "Write a function called length that returns the number of "
            "elements in a list. length [] = 0, length [x] = 1.",
            model="gemma4:31b",
        )
    except Exception as exc:
        pytest.skip(f"Ollama unreachable: {exc}")
    parse_and_validate(emitted)
```

- [ ] **Step 2: Run**

Run: `uv run pytest src/qft_pcn/viz/tests/test_llm_v1.py::test_ollama_emits_validating_dsl_for_length_prompt -v`
Expected: PASS or SKIP (the test is informational — never blocks CI).

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/viz/tests/test_llm_v1.py
git -c commit.gpgsign=false commit -m "test(viz/llm): Ollama regression for length-synthesis prompt (skip if unreachable)"
```

---

## W6a — Visualizer panels

**Files:**
- Modify: `src/qft_pcn/viz/web/src/lib/dsl-client.ts`
- Modify: `src/qft_pcn/viz/web/src/panels/DslEditorPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/LemmaLibraryPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/GoalGraphPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/lib/run-decoder.ts`
- Test: `src/qft_pcn/viz/web/src/lib/run-decoder.test.ts`
- Test: `src/qft_pcn/viz/web/src/panels/GoalGraphPanel.test.tsx`

### W6a.T1: Typed DSL client (TypeScript types matching §2.1)

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/viz/web/src/lib/dsl-client.test.ts`:

```typescript
import { describe, it, expect } from 'vitest';
import { type DslV1Spec, runDslV1 } from './dsl-client';


describe('DslV1Spec', () => {
  it('compiles a minimal spec literal', () => {
    const spec: DslV1Spec = {
      version: '1',
      fields: [{ name: 'n', cutoff: 4 }],
      sites: 2,
      constraints: [{ kind: 'local', site: 0, term: 'n == 0', weight: 1.0 }],
      observables: [{ site: 0, field: 'n', op: 'n' }],
      search: { method: 'imag_time', runtime: 'mps', steps: 10, chi_max: 8, dt: 0.05 },
    };
    expect(spec.version).toBe('1');
  });
});


describe('runDslV1', () => {
  it('POSTs to /dsl/v1/run', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true, status: 200,
      json: async () => ({
        observable_values: {}, residual_constraint_energies: {},
        diagnostics: { final_energy: 0, step_history: [], converged: true },
      }),
    });
    globalThis.fetch = fetchMock as any;
    await runDslV1({ version: '1' /* etc */ } as any);
    expect(fetchMock).toHaveBeenCalledWith('/dsl/v1/run', expect.objectContaining({
      method: 'POST',
    }));
  });
});
```

- [ ] **Step 2: Run failure**
Run: `cd src/qft_pcn/viz/web && npm test -- dsl-client`
Expected: FAIL

- [ ] **Step 3: Implement `dsl-client.ts`**

Rewrite `src/qft_pcn/viz/web/src/lib/dsl-client.ts` to export TypeScript types covering §2.1-§2.6 (Constraint as a discriminated union of the six kinds + the `decomposition` field), plus `runDslV1(spec): Promise<DslV1Result>` calling `POST /dsl/v1/run`.

- [ ] **Step 4: Run**
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/dsl-client.ts src/qft_pcn/viz/web/src/lib/dsl-client.test.ts
git -c commit.gpgsign=false commit -m "feat(viz/web): typed DslV1 client per §2.1-§2.6"
```

### W6a.T2: Monaco JSON-Schema wiring in DslEditorPanel

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/viz/web/src/panels/DslEditorPanel.test.tsx`:

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DslEditorPanel } from './DslEditorPanel';


describe('DslEditorPanel', () => {
  it('renders Monaco with the v1 schema wired', () => {
    render(<DslEditorPanel />);
    // Editor mounts, the schema URI surfaces in a known data attribute
    const el = screen.getByTestId('dsl-editor');
    expect(el).toBeDefined();
    expect(el.getAttribute('data-schema-uri')).toBe('inmemory://dsl/v1');
  });
});
```

- [ ] **Step 2-4:** modify `DslEditorPanel.tsx` to register the v1 JSON Schema (mirrored from `bridge/dsl/schema.py::SCHEMA` — exported via a new endpoint `GET /dsl/v1/schema` to avoid duplication; or inlined as a constant if the round-trip is too heavy). Add the `data-testid="dsl-editor"` + `data-schema-uri` for the test.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/panels/DslEditorPanel.tsx src/qft_pcn/viz/web/src/panels/DslEditorPanel.test.tsx
git -c commit.gpgsign=false commit -m "feat(viz/web): Monaco wired to §9.2 v1 schema"
```

### W6a.T3: `LemmaLibraryPanel.tsx`

Surfaces `GET /lemmas` (existing or to be added in bridge) and a "Register from current run" button on run-complete.

- [ ] **Step 1: Audit** — check `bridge/api.py` for an existing `/lemmas` endpoint. If absent, add it in W4 follow-on or a brief bridge task here:

```python
@app.get("/lemmas")
def list_lemmas():
    from qft_pcn.composition.lemma_library_adapter import list_lemmas as _ls
    return {"lemmas": _ls()}
```

- [ ] **Step 2-5:** TDD a `LemmaLibraryPanel` component with vitest+RTL: lists the lemmas table (id, type signature, derivation cost), shows a "register from last run" button when a recent run has residual < ε_register. Commit message: `feat(viz/web): LemmaLibraryPanel reads/registers via bridge per §10.8`.

### W6a.T4: `GoalGraphPanel.tsx` (§10.10 DAG view)

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { GoalGraphPanel } from './GoalGraphPanel';


describe('GoalGraphPanel', () => {
  it('renders one node per child with residual badge', () => {
    const children = [
      { id: 'c1', residual: 0.0, status: 'solved' as const, dependsOn: [] },
      { id: 'c2', residual: 1.2, status: 'failed' as const, dependsOn: ['c1'] },
    ];
    render(<GoalGraphPanel children={children} />);
    expect(screen.getByText('c1')).toBeDefined();
    expect(screen.getByText('c2')).toBeDefined();
    expect(screen.getByText('failed')).toBeDefined();
  });
});
```

- [ ] **Step 2-4:** implement the panel using an existing graph rendering primitive (the codebase uses D3 elsewhere — check `viz/web/src/lib/` for an existing renderer; if none, use a simple SVG layout with nodes positioned by topological layer). Wire to the `diagnostics.children` slice of the §2.8 return shape.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/panels/GoalGraphPanel.tsx src/qft_pcn/viz/web/src/panels/GoalGraphPanel.test.tsx
git -c commit.gpgsign=false commit -m "feat(viz/web): GoalGraphPanel renders §10.10 child DAG with residuals"
```

### W6a.T5: `run-decoder.ts` — §2.8 → AST view

- [ ] **Step 1: Write the failing test**

```typescript
import { describe, it, expect } from 'vitest';
import { decodeRunToAst } from './run-decoder';


describe('decodeRunToAst', () => {
  it('reconstructs the AST from per-site argmax observables (§15.8)', () => {
    const result = {
      observable_values: {
        '0': 'Lambda', '1': 'Var', '2': 'Match', '3': 'PatternNil',
        '4': 'Zero', '5': 'PatternCons', '7': 'Succ', '8': 'App',
        '9': 'Var', '10': 'Var',
      },
      residual_constraint_energies: {}, diagnostics: { final_energy: 0,
        step_history: [], converged: true },
    };
    const tree = decodeRunToAst(result);
    expect(tree.root.kind).toBe('Lambda');
    // The full §15.8 decode is the eventual target; first cut: top-level only
  });
});
```

- [ ] **Step 2-4:** implement a first-cut decoder that maps per-site argmax observables to a flat AST node list. Full structural decode (parent/child wiring per §15.8) is a follow-on noted in EXTENSIONS.md.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/run-decoder.ts src/qft_pcn/viz/web/src/lib/run-decoder.test.ts
git -c commit.gpgsign=false commit -m "feat(viz/web): run-decoder first-cut maps argmax observables to AST nodes"
```

### W6a.T6: Wire all three panels into the visualizer's main layout

- [ ] Touch the panel-registry / main route in `viz/web/src/` (audit `viz/web/src/App.tsx` or `viz/web/src/main.tsx` to find the layout). Add LemmaLibraryPanel, GoalGraphPanel, and the existing DslEditorPanel to the panel grid. Smoke-test by `npm run dev` and visiting `/`.

- [ ] Commit: `feat(viz/web): register LemmaLibrary + GoalGraph panels in main layout`.

---

## W6b — Acceptance tests

Each acceptance test consumes the full pipeline end-to-end. They live in `bridge/tests/` because they exercise the bridge API; they may take many seconds and should be marked `@pytest.mark.slow` so CI can opt out.

### W6b.T1: `test_length_synthesis.py` (§15 canonical)

- [ ] **Step 1: Write the test**

```python
"""§15 worked example — full pipeline acceptance."""
import json
import pathlib
import pytest
from fastapi.testclient import TestClient
from qft_pcn.bridge.api import app


client = TestClient(app)
EXAMPLE = pathlib.Path("src/qft_pcn/viz/llm_examples/length_synthesis.json")


@pytest.mark.slow
def test_length_synthesis_converges():
    spec = json.loads(EXAMPLE.read_text())
    r = client.post("/dsl/v1/run", json=spec)
    assert r.status_code == 200
    out = r.json()
    assert out["diagnostics"]["final_energy"] < 0.01, (
        f"§15 acceptance: ⟨H⟩ {out['diagnostics']['final_energy']} ≥ 0.01"
    )
    # §15.7: per-site argmax observables should decode to valid AST node kinds
    kinds = {k: v for k, v in out["observable_values"].items()}
    assert kinds[str(0)] == "Lambda"
```

- [ ] **Step 2-4:** run; if fails, the failure is the signal to elaborate W3.T2b (full typing rules) and W3.T3 (full eval Hamiltonian). Iterate. Commit each elaboration with explicit messages.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/tests/test_length_synthesis.py
git -c commit.gpgsign=false commit -m "test(bridge): §15 length-synthesis acceptance (⟨H⟩<0.01, Lambda at site 0)"
```

### W6b.T2: `test_use_lemma.py` (§10.8 acceptance)

- [ ] Two-stage test:
  - Stage 1: run a small DSL that proves `∀x. x+0 = x` (Peano), registers the result as lemma `peano-x-plus-0-eq-x`, record Trotter step count.
  - Stage 2: run a parent DSL that uses `use_lemma` to prove `∀x. (x+0)+0 = x`, record step count.
  - Assert: stage 2 step count ≤ stage 1 / 3.

- [ ] Commit: `test(bridge): §10.8 use_lemma acceptance (≥3× Trotter reduction)`.

### W6b.T3: `test_decomposition.py` (§10.10 acceptance)

- [ ] End-to-end DSL with two children and a parent that integrates both via auto-generated `use_lemma`. Assert: parent run returns 200, `diagnostics.children` has 2 entries with `converged: true`, parent `final_energy < threshold`.

- [ ] Commit: `test(bridge): §10.10 decomposition acceptance (parent + 2 children DAG)`.

### W6b.T4: `test_dsl_v1.py` edge cases

- [ ] Aggregated tests covering: unknown top-level key → reject; constraint without `weight` → default 1.0; `boundary` with negative site → reject; `decomposition` self-reference (cycle) → reject; `use_lemma` with empty sites → reject (already covered in W1.T4 but re-asserted here as integration); predicate with banned AST node in a `local` constraint → reject at compile time.

- [ ] Commit: `test(bridge/dsl): v1 edge-case battery (negative inputs reject cleanly)`.

---

## Self-review notes

After all workstreams land:

- **Spec coverage:** §1 flow → W4.T3 endpoint + W5 prompt; §2.1-§2.6 → W1.T1-T4; §2.7 → W2.T1-T2; §2.8 return shape → W4.T3; §2.9 lemma registration → W6a.T3; §2.10 decomposition → W7 + W6a.T4; §3.4 presets → W5.T1 + W6a.T6; §3.5 acceptance → W6b.
- **Architecture coverage:** §9.2 (schema) ✓, §9.3 (flow) ✓, §10.2 (typing rules) W3.T2 + W3.T2b, §10.3 (eval Ham) W3.T3 (gap recorded in EXTENSIONS.md), §10.4 (MERA routing) W4.T1, §10.5 (bridge) extended throughout, §10.8 (lemma promotion) W3.T4 + W4.T2 + W6b.T2, §10.10 (cross-level passing) W7 + W6a.T4 + W6b.T3, §15 (worked example) W5.T1 + W6b.T1, §13.2 (PSD correctness) every W3 test.
- **EXTENSIONS.md entries to expect:**
  - "DSL constraint gaps: example needs §10.3 eval Ham" (W3.T3)
  - "run-decoder: full §15.8 structural decode" (W6a.T5)
  - "lemma-library: composition/lemma_library_adapter.register_transient" if absent (W7.T2)

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-24-full-dsl-programming.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Best for a plan this large; the subagent-driven workflow already used for the previous viz workstreams keeps context lean and produces per-task spec+code reviews per your QPCN completion protocol.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Faster for short plans; risks context exhaustion here given the 30+ tasks.

Which approach?
