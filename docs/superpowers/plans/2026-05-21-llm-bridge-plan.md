# LLM Bridge / DSL Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement sub-project G from `docs/superpowers/specs/2026-05-21-llm-bridge-design.md`: a thin stdio JSON-RPC bridge between an LLM frontend and the QPCN reasoning core. The LLM emits DSL specs; the bridge compiles them into Hamiltonians via B/C, runs imaginary-time evolution, measures observables, returns results.

**Architecture:** Three layers under `src/qft_pcn/bridge/`:
1. **DSL** (`bridge/dsl/`): JSON Schema + expression parser + term compiler.
2. **Runtime** (`bridge/runtime.py`): `run_problem`, `diagnose_problem` — transport-independent.
3. **Protocol** (`bridge/api.py`): stdio JSON-RPC server. Plus `bridge/templates.py` LLM helpers and `bridge/llm.py` mock/real-LLM shims.

**Tech Stack:** Python 3.11, numpy, scipy, pytest, **jsonschema** (new required dep), **anthropic** (optional). Built on existing `src/qft_pcn/qft/` and `src/qft_pcn/logic/`.

**Driving principles (from spec §1, non-negotiable):**

1. The DSL is the entire contract. No NL in, no MPSes out.
2. The bridge is THIN. Reuse B/C/A/qft; do not reimplement.
3. The QPCN core knows nothing of natural language. No `anthropic`/`openai` import outside `bridge/`.
4. Errors are structured (`BridgeError` subclasses with stable codes).
5. Stdio JSON-RPC, synchronous, one request at a time.
6. No LLM call on any test path.
7. **No `eval()` / `exec()` / `compile()` / `pickle.loads` on user-supplied strings, ever.** The constraint-expression parser is hand-written; the expression compiler is structural.
8. When B/C aren't ready, use the stub factory (§4.3 of spec) — every term not in the stub set raises `BRIDGE_E_TERM_UNSUPPORTED`.

**Reference:** The spec at `docs/superpowers/specs/2026-05-21-llm-bridge-design.md` is the authoritative contract. When in doubt, re-read it. If something here disagrees with the spec, the spec wins.

**Test command throughout:** `.venv/bin/python -m pytest <path> -v`.

---

## Task 1: Add jsonschema dependency and verify environment

**Files:**
- Modify: `pyproject.toml` (add `jsonschema>=4.0,<5`).

- [ ] **Step 1: Check current state of jsonschema availability**

Run: `.venv/bin/python -c "import jsonschema; print(jsonschema.__version__)" 2>&1`
Either prints a version or `ModuleNotFoundError`. Both are acceptable starting states.

- [ ] **Step 2: Install jsonschema directly into the venv**

Run: `.venv/bin/python -m pip install 'jsonschema>=4.0,<5'`
Expected: success.

- [ ] **Step 3: Add jsonschema to pyproject.toml**

Open `pyproject.toml`. Locate the `[project]` table's `dependencies` (or equivalent). Add `"jsonschema>=4.0,<5"`. Leave other entries unchanged.

- [ ] **Step 4: Verify the baseline test suite still passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ --ignore=src/qft_pcn/tests/test_quantum.py -q 2>&1 | tail -5`
Expected: existing tests pass; no regression.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "$(cat <<'EOF'
deps: add jsonschema for the LLM-bridge DSL validator

Sub-project G needs JSON Schema validation on every incoming DSL spec.
jsonschema is small, pure Python, MIT-licensed. anthropic remains an
optional/lazy import inside bridge/llm.py only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Bridge package skeleton + BridgeError hierarchy

**Files:**
- Create: `src/qft_pcn/bridge/__init__.py`, `bridge/errors.py`, `bridge/dsl/__init__.py`, `bridge/runtime/__init__.py`.
- Create: `src/qft_pcn/tests/test_bridge_errors.py`.

This task lays down the package layout and the typed-error hierarchy from spec §10. Every later task layers on top.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_bridge_errors.py`:

```python
"""Tests for BridgeError hierarchy and code stability."""

from __future__ import annotations

import pytest

from src.qft_pcn.bridge.errors import (
    BridgeError,
    BadJsonError, BadSchemaError, BadReferenceError, BadTermError,
    TypeError as BridgeTypeError, TermUnsupportedError,
    UnsupportedMethodError, SitesOutOfRangeError, TooLargeError,
    ConstraintNotAdjacentError, NumericFailureError, InternalError,
    ALL_CODES,
)


def test_bridge_error_has_code_and_details():
    e = BadJsonError(message="bad json at pos 5", details={"position": 5})
    assert e.code == "BRIDGE_E_BAD_JSON"
    assert e.details == {"position": 5}
    assert "bad json at pos 5" in str(e)


def test_every_subclass_has_distinct_code():
    codes = set()
    for cls in (BadJsonError, BadSchemaError, BadReferenceError, BadTermError,
                BridgeTypeError, TermUnsupportedError, UnsupportedMethodError,
                SitesOutOfRangeError, TooLargeError, ConstraintNotAdjacentError,
                NumericFailureError, InternalError):
        codes.add(cls.code)
    assert len(codes) == 12


def test_all_codes_set_is_stable():
    # If you add a code, snapshot must be updated deliberately.
    expected = {
        "BRIDGE_E_BAD_JSON",
        "BRIDGE_E_BAD_SCHEMA",
        "BRIDGE_E_BAD_REFERENCE",
        "BRIDGE_E_BAD_TERM",
        "BRIDGE_E_TYPE_ERROR",
        "BRIDGE_E_TERM_UNSUPPORTED",
        "BRIDGE_E_UNSUPPORTED_METHOD",
        "BRIDGE_E_SITES_OUT_OF_RANGE",
        "BRIDGE_E_TOO_LARGE",
        "BRIDGE_E_CONSTRAINT_NOT_ADJACENT",
        "BRIDGE_E_NUMERIC_FAILURE",
        "BRIDGE_E_INTERNAL",
    }
    assert set(ALL_CODES) == expected


def test_bridge_error_to_dict():
    e = BadSchemaError(message="missing 'sites'", details={"pointer": "/sites"})
    d = e.to_dict()
    assert d == {
        "code": "BRIDGE_E_BAD_SCHEMA",
        "message": "missing 'sites'",
        "details": {"pointer": "/sites"},
    }


def test_default_details_is_empty_dict():
    e = InternalError(message="oops")
    assert e.details == {}
```

- [ ] **Step 2: Verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_errors.py -v`
Expected: ImportError on `src.qft_pcn.bridge.errors`.

- [ ] **Step 3: Create the package skeleton**

```bash
mkdir -p src/qft_pcn/bridge/dsl src/qft_pcn/bridge/runtime
touch src/qft_pcn/bridge/__init__.py
touch src/qft_pcn/bridge/dsl/__init__.py
touch src/qft_pcn/bridge/runtime/__init__.py
```

- [ ] **Step 4: Implement `bridge/errors.py`**

Create `src/qft_pcn/bridge/errors.py`:

```python
"""Typed error hierarchy for the LLM bridge.

Every error has a stable string code (BRIDGE_E_*), a human-readable message,
and an optional machine-readable details dict. The LLM consumes codes and
details, not free-form prose.

See spec §10 of docs/superpowers/specs/2026-05-21-llm-bridge-design.md.
"""

from __future__ import annotations

from typing import Any


class BridgeError(Exception):
    """Base class. All bridge errors inherit; all carry a stable `code`."""
    code: str = "BRIDGE_E_INTERNAL"

    def __init__(self, message: str = "", details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(f"{self.code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "details": self.details}


class BadJsonError(BridgeError):              code = "BRIDGE_E_BAD_JSON"
class BadSchemaError(BridgeError):            code = "BRIDGE_E_BAD_SCHEMA"
class BadReferenceError(BridgeError):         code = "BRIDGE_E_BAD_REFERENCE"
class BadTermError(BridgeError):              code = "BRIDGE_E_BAD_TERM"
class TypeError(BridgeError):                 code = "BRIDGE_E_TYPE_ERROR"
class TermUnsupportedError(BridgeError):      code = "BRIDGE_E_TERM_UNSUPPORTED"
class UnsupportedMethodError(BridgeError):    code = "BRIDGE_E_UNSUPPORTED_METHOD"
class SitesOutOfRangeError(BridgeError):      code = "BRIDGE_E_SITES_OUT_OF_RANGE"
class TooLargeError(BridgeError):             code = "BRIDGE_E_TOO_LARGE"
class ConstraintNotAdjacentError(BridgeError):code = "BRIDGE_E_CONSTRAINT_NOT_ADJACENT"
class NumericFailureError(BridgeError):       code = "BRIDGE_E_NUMERIC_FAILURE"
class InternalError(BridgeError):             code = "BRIDGE_E_INTERNAL"


ALL_CODES: tuple[str, ...] = (
    BadJsonError.code, BadSchemaError.code, BadReferenceError.code,
    BadTermError.code, TypeError.code, TermUnsupportedError.code,
    UnsupportedMethodError.code, SitesOutOfRangeError.code,
    TooLargeError.code, ConstraintNotAdjacentError.code,
    NumericFailureError.code, InternalError.code,
)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_errors.py -v`
Expected: all 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/ src/qft_pcn/tests/test_bridge_errors.py
git commit -m "$(cat <<'EOF'
feat(bridge/errors): typed exception hierarchy with stable codes

Twelve BRIDGE_E_* codes per spec §10. The LLM consumes codes and a
machine-readable details dict; the message is for humans. ALL_CODES
is snapshot-tested so adding a code is a deliberate change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: DSL JSON Schema + phase-1/2 validation

**Files:**
- Create: `src/qft_pcn/bridge/dsl/schema.py`, `bridge/dsl/dsl_schema.json` (generated by schema.py).
- Create: `src/qft_pcn/tests/test_bridge_schema.py`.

Implements phases 1 and 2 of spec §3.8: JSON parse + JSON Schema validation. Phases 3 (cross-field) and 4 (constraint-term parsing) come in later tasks.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_bridge_schema.py`:

```python
"""Schema-validation tests for the bridge DSL (spec §3, §11.1)."""

from __future__ import annotations

import pytest

from src.qft_pcn.bridge.dsl.schema import validate_schema, parse_and_validate
from src.qft_pcn.bridge.errors import BadJsonError, BadSchemaError


MINIMAL_DSL = {
    "fields": [{"name": "expr", "cutoff": 16}],
    "sites": 4,
    "constraints": [],
    "observables": [{"site": 0, "field": "expr", "op": "n"}],
}


CANONICAL_STLC_DSL = {
    "fields": [
        {"name": "kind",  "cutoff": 8},
        {"name": "type",  "cutoff": 8},
        {"name": "bid",   "cutoff": 8},
        {"name": "value", "cutoff": 16},
    ],
    "sites": 32,
    "constraints": [
        {"kind": "local",    "site": 0, "term": "kind == LAM", "weight": 1.0},
        {"kind": "two_site", "sites": [0, 1],
         "term": "type(arg1) == T_INT", "weight": 0.5},
    ],
    "boundary": {"0": {"kind": "KIND_LAM"}},
    "observables": [{"site": 1, "field": "kind", "op": "n"}],
    "search": {"method": "imag_time", "steps": 50, "chi_max": 32},
}


def test_minimal_dsl_validates():
    validate_schema(MINIMAL_DSL)   # does not raise


def test_canonical_stlc_dsl_validates():
    validate_schema(CANONICAL_STLC_DSL)


def test_parse_and_validate_returns_dict():
    import json
    out = parse_and_validate(json.dumps(MINIMAL_DSL))
    assert out["sites"] == 4


def test_bad_json_raises_bad_json_error():
    with pytest.raises(BadJsonError) as ei:
        parse_and_validate("not valid json {")
    assert ei.value.code == "BRIDGE_E_BAD_JSON"


def test_missing_required_field_raises_bad_schema():
    bad = dict(MINIMAL_DSL)
    del bad["sites"]
    with pytest.raises(BadSchemaError) as ei:
        validate_schema(bad)
    assert ei.value.code == "BRIDGE_E_BAD_SCHEMA"
    # The JSON Pointer should point to the missing field or the root.
    assert "sites" in ei.value.message or "/sites" in str(ei.value.details)


def test_negative_sites_raises_bad_schema():
    bad = dict(MINIMAL_DSL)
    bad["sites"] = -1
    with pytest.raises(BadSchemaError):
        validate_schema(bad)


def test_too_large_sites_raises_bad_schema():
    bad = dict(MINIMAL_DSL)
    bad["sites"] = 1024
    with pytest.raises(BadSchemaError):
        validate_schema(bad)


def test_invalid_field_name_raises_bad_schema():
    bad = dict(MINIMAL_DSL)
    bad["fields"] = [{"name": "ExprUpper", "cutoff": 16}]
    with pytest.raises(BadSchemaError):
        validate_schema(bad)


def test_invalid_op_raises_bad_schema():
    bad = dict(MINIMAL_DSL)
    bad["observables"] = [{"site": 0, "field": "expr", "op": "BOGUS"}]
    with pytest.raises(BadSchemaError):
        validate_schema(bad)


def test_constraint_kind_local_requires_site():
    bad = {
        **MINIMAL_DSL,
        "constraints": [{"kind": "local", "term": "kind == LAM", "weight": 1.0}],
    }
    with pytest.raises(BadSchemaError):
        validate_schema(bad)


def test_constraint_kind_two_site_requires_sites_array():
    bad = {
        **MINIMAL_DSL,
        "constraints": [{"kind": "two_site", "site": 0,
                         "term": "kind == LAM", "weight": 1.0}],
    }
    with pytest.raises(BadSchemaError):
        validate_schema(bad)


def test_search_defaults_are_filled_when_omitted():
    # validate_schema does not mutate; defaults are applied by parse_and_validate.
    out = parse_and_validate("""{"fields":[{"name":"x","cutoff":4}],"sites":2,
                                  "constraints":[],
                                  "observables":[{"site":0,"field":"x","op":"n"}]}""")
    assert out["search"]["method"] == "imag_time"
    assert out["search"]["steps"] == 50
    assert out["search"]["chi_max"] == 32
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_schema.py -v`
Expected: ImportError on `bridge.dsl.schema`.

- [ ] **Step 3: Implement `bridge/dsl/schema.py`**

Create `src/qft_pcn/bridge/dsl/schema.py`:

```python
"""JSON Schema for the bridge DSL plus phase-1/2 validation.

See spec §3 and §3.8 phases 1-2 in
docs/superpowers/specs/2026-05-21-llm-bridge-design.md.
"""

from __future__ import annotations

import json
from typing import Any

import jsonschema
from jsonschema import Draft202012Validator

from ..errors import BadJsonError, BadSchemaError


_DEFAULT_SEARCH = {
    "method": "imag_time",
    "steps": 50,
    "chi_max": 32,
    "dt": 0.05,
}


SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "QPCN Bridge DSL",
    "type": "object",
    "required": ["fields", "sites", "constraints", "observables"],
    "additionalProperties": False,
    "properties": {
        "fields": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {
                "type": "object",
                "required": ["name", "cutoff"],
                "additionalProperties": False,
                "properties": {
                    "name":   {"type": "string",
                               "pattern": "^[a-z][a-z0-9_]{0,31}$"},
                    "cutoff": {"type": "integer", "minimum": 2, "maximum": 32},
                },
            },
        },
        "sites": {"type": "integer", "minimum": 1, "maximum": 256},
        "constraints": {
            "type": "array",
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "required": ["kind", "site", "term"],
                        "additionalProperties": False,
                        "properties": {
                            "kind":   {"const": "local"},
                            "site":   {"type": "integer", "minimum": 0},
                            "term":   {"type": "string", "minLength": 1},
                            "weight": {"type": "number", "minimum": 0,
                                       "default": 1.0},
                        },
                    },
                    {
                        "type": "object",
                        "required": ["kind", "sites", "term"],
                        "additionalProperties": False,
                        "properties": {
                            "kind":  {"const": "two_site"},
                            "sites": {"type": "array",
                                      "items": {"type": "integer", "minimum": 0},
                                      "minItems": 2, "maxItems": 2,
                                      "uniqueItems": True},
                            "term":  {"type": "string", "minLength": 1},
                            "weight":{"type": "number", "minimum": 0,
                                      "default": 1.0},
                        },
                    },
                ]
            },
        },
        "boundary": {
            "type": "object",
            "patternProperties": {
                "^[0-9]+$": {
                    "type": "object",
                    "minProperties": 1,
                    "additionalProperties": {
                        "type": ["string", "number", "integer", "boolean"]
                    },
                },
            },
            "additionalProperties": False,
            "default": {},
        },
        "observables": {
            "type": "array", "minItems": 1,
            "items": {
                "type": "object",
                "required": ["site", "field", "op"],
                "additionalProperties": False,
                "properties": {
                    "site":  {"type": "integer", "minimum": 0},
                    "field": {"type": "string"},
                    "op":    {"enum": ["n", "phi", "pi", "a", "adag",
                                       "identity"]},
                },
            },
        },
        "search": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "method":  {"enum": ["imag_time"]},
                "steps":   {"type": "integer", "minimum": 1, "maximum": 1000,
                            "default": 50},
                "chi_max": {"type": "integer", "minimum": 2, "maximum": 64,
                            "default": 32},
                "dt":      {"type": "number", "exclusiveMinimum": 0,
                            "maximum": 0.5, "default": 0.05},
            },
            "default": _DEFAULT_SEARCH,
        },
    },
}


_VALIDATOR = Draft202012Validator(SCHEMA)


def validate_schema(dsl: dict[str, Any]) -> None:
    """Phase 2 of the validation pipeline: JSON Schema check.

    Raises BadSchemaError on any failure; on success returns None.
    """
    try:
        errs = sorted(_VALIDATOR.iter_errors(dsl), key=lambda e: list(e.path))
    except Exception as exc:
        raise BadSchemaError(message=f"schema validator internal: {exc}") from exc
    if errs:
        e = errs[0]
        pointer = "/" + "/".join(str(p) for p in e.absolute_path)
        raise BadSchemaError(
            message=f"{pointer or '/'}: {e.message}",
            details={"pointer": pointer or "/", "constraint": e.validator,
                     "value": e.instance},
        )


def _apply_defaults(dsl: dict[str, Any]) -> dict[str, Any]:
    """Fill in defaults for `boundary`, `search`, and `weight`. Non-mutating."""
    out = dict(dsl)
    out.setdefault("boundary", {})
    search = dict(out.get("search") or {})
    for k, v in _DEFAULT_SEARCH.items():
        search.setdefault(k, v)
    out["search"] = search
    out["constraints"] = [
        {**c, "weight": c.get("weight", 1.0)} for c in out.get("constraints", [])
    ]
    return out


def parse_and_validate(raw: str) -> dict[str, Any]:
    """Phase 1 (parse) + phase 2 (schema) + default-fill.

    Raises BadJsonError on parse failure, BadSchemaError on schema failure.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BadJsonError(
            message=f"JSON parse error at line {exc.lineno} col {exc.colno}: "
                    f"{exc.msg}",
            details={"line": exc.lineno, "col": exc.colno},
        ) from exc
    if not isinstance(data, dict):
        raise BadSchemaError(
            message="top-level DSL must be a JSON object",
            details={"pointer": "/", "got": type(data).__name__},
        )
    validate_schema(data)
    return _apply_defaults(data)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_schema.py -v`
Expected: all 12 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/schema.py src/qft_pcn/tests/test_bridge_schema.py
git commit -m "$(cat <<'EOF'
feat(bridge/dsl/schema): JSON Schema and parse_and_validate

Phases 1-2 of the four-phase validation pipeline (spec §3.8). Uses
Draft 2020-12 via the jsonschema package; emits BadJsonError /
BadSchemaError with JSON-Pointer details for the LLM to fix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Cross-field validation (phase 3)

**Files:**
- Create: `src/qft_pcn/bridge/dsl/cross_validate.py`.
- Modify: `src/qft_pcn/tests/test_bridge_schema.py` (extend with cross-validation tests).

Phase 3 of spec §3.8: assert site indices are in range, field names in `boundary` and `observables` are declared, etc.

- [ ] **Step 1: Append failing tests**

Append to `src/qft_pcn/tests/test_bridge_schema.py`:

```python
from src.qft_pcn.bridge.dsl.cross_validate import cross_validate
from src.qft_pcn.bridge.errors import BadReferenceError


def test_cross_validate_accepts_canonical_dsl():
    cross_validate(CANONICAL_STLC_DSL)


def test_cross_validate_rejects_oob_constraint_site():
    bad = {**MINIMAL_DSL,
           "sites": 4,
           "constraints": [{"kind": "local", "site": 99,
                            "term": "kind == LAM", "weight": 1.0}]}
    with pytest.raises(BadReferenceError, match="constraints/0/site"):
        cross_validate(bad)


def test_cross_validate_rejects_oob_two_site():
    bad = {**MINIMAL_DSL,
           "sites": 4,
           "constraints": [{"kind": "two_site", "sites": [0, 99],
                            "term": "x", "weight": 1.0}]}
    with pytest.raises(BadReferenceError, match="constraints/0/sites/1"):
        cross_validate(bad)


def test_cross_validate_rejects_unknown_field_in_observable():
    bad = {**MINIMAL_DSL,
           "observables": [{"site": 0, "field": "bogus", "op": "n"}]}
    with pytest.raises(BadReferenceError, match="observables/0/field"):
        cross_validate(bad)


def test_cross_validate_rejects_oob_observable_site():
    bad = {**MINIMAL_DSL,
           "sites": 4,
           "observables": [{"site": 5, "field": "expr", "op": "n"}]}
    with pytest.raises(BadReferenceError, match="observables/0/site"):
        cross_validate(bad)


def test_cross_validate_rejects_unknown_field_in_boundary():
    bad = {**MINIMAL_DSL,
           "boundary": {"0": {"bogus": 1}}}
    with pytest.raises(BadReferenceError, match="boundary/0/bogus"):
        cross_validate(bad)


def test_cross_validate_rejects_oob_boundary_site():
    bad = {**MINIMAL_DSL,
           "sites": 4,
           "boundary": {"99": {"expr": 1}}}
    with pytest.raises(BadReferenceError, match="boundary/99"):
        cross_validate(bad)
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_schema.py -v -k cross_validate`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/dsl/cross_validate.py`**

Create `src/qft_pcn/bridge/dsl/cross_validate.py`:

```python
"""Phase 3 of validation: cross-field references.

Assumes the input has already passed phase-2 JSON Schema validation
(bridge/dsl/schema.py::validate_schema). Raises BadReferenceError with
a JSON Pointer naming the bad reference.
"""

from __future__ import annotations

from typing import Any

from ..errors import BadReferenceError


def cross_validate(dsl: dict[str, Any]) -> None:
    sites: int = dsl["sites"]
    field_names = {f["name"] for f in dsl["fields"]}

    for i, c in enumerate(dsl.get("constraints", [])):
        if c["kind"] == "local":
            if not 0 <= c["site"] < sites:
                raise BadReferenceError(
                    message=f"constraint {i} site {c['site']} out of [0, {sites-1}]",
                    details={"pointer": f"/constraints/{i}/site",
                             "value": c["site"], "sites": sites},
                )
        else:  # two_site
            for j, s in enumerate(c["sites"]):
                if not 0 <= s < sites:
                    raise BadReferenceError(
                        message=f"constraint {i} sites[{j}]={s} out of [0, {sites-1}]",
                        details={"pointer": f"/constraints/{i}/sites/{j}",
                                 "value": s, "sites": sites},
                    )

    for i, o in enumerate(dsl["observables"]):
        if not 0 <= o["site"] < sites:
            raise BadReferenceError(
                message=f"observable {i} site {o['site']} out of [0, {sites-1}]",
                details={"pointer": f"/observables/{i}/site",
                         "value": o["site"], "sites": sites},
            )
        if o["field"] not in field_names:
            raise BadReferenceError(
                message=f"observable {i} field {o['field']!r} not declared",
                details={"pointer": f"/observables/{i}/field",
                         "value": o["field"],
                         "known_fields": sorted(field_names)},
            )

    for site_key, fmap in (dsl.get("boundary") or {}).items():
        try:
            site = int(site_key)
        except ValueError:
            raise BadReferenceError(
                message=f"boundary key {site_key!r} is not an integer",
                details={"pointer": f"/boundary/{site_key}"},
            )
        if not 0 <= site < sites:
            raise BadReferenceError(
                message=f"boundary site {site} out of [0, {sites-1}]",
                details={"pointer": f"/boundary/{site_key}",
                         "value": site, "sites": sites},
            )
        for fname in fmap:
            if fname not in field_names:
                raise BadReferenceError(
                    message=f"boundary site {site} field {fname!r} not declared",
                    details={"pointer": f"/boundary/{site_key}/{fname}",
                             "value": fname,
                             "known_fields": sorted(field_names)},
                )
```

- [ ] **Step 4: Run all schema tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_schema.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/cross_validate.py src/qft_pcn/tests/test_bridge_schema.py
git commit -m "$(cat <<'EOF'
feat(bridge/dsl/cross_validate): phase-3 reference checking

Catches OOB site indices, undeclared field names in observables and
boundary, and non-integer boundary keys. Every error carries a JSON
Pointer in details so the LLM can pinpoint the bad reference.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Constraint-expression parser (no eval!)

**Files:**
- Create: `src/qft_pcn/bridge/dsl/expr_ast.py`, `bridge/dsl/expr_parser.py`.
- Create: `src/qft_pcn/tests/test_bridge_expr_parser.py`.

Hand-written recursive-descent parser for the constraint-expression sub-grammar (spec §4.1). **This module must never call `eval/exec/compile/literal_eval` on user input.**

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_bridge_expr_parser.py`:

```python
"""Tests for the constraint-expression parser (spec §4.1, §4.2, §11.2)."""

from __future__ import annotations

import builtins
import pytest

from src.qft_pcn.bridge.dsl.expr_parser import parse_expr
from src.qft_pcn.bridge.dsl.expr_ast import Lit, Ident, Call, Unary, Binary
from src.qft_pcn.bridge.errors import BadTermError


def test_parse_simple_ident():
    assert parse_expr("KIND_LAM") == Ident(name="KIND_LAM")


def test_parse_int_literal():
    assert parse_expr("42") == Lit(value=42)


def test_parse_string_literal():
    assert parse_expr("'+'") == Lit(value="+")


def test_parse_call_one_arg():
    e = parse_expr("type(arg1)")
    assert e == Call(fn="type", args=[Ident(name="arg1")])


def test_parse_call_two_args():
    e = parse_expr("arr(T_INT, T_BOOL)")
    assert e == Call(fn="arr",
                     args=[Ident(name="T_INT"), Ident(name="T_BOOL")])


def test_parse_eq_with_calls():
    e = parse_expr("type(arg1) == type(arg2)")
    assert e == Binary(op="==",
                       lhs=Call(fn="type", args=[Ident("arg1")]),
                       rhs=Call(fn="type", args=[Ident("arg2")]))


def test_parse_kind_equals_lit():
    e = parse_expr("kind == LAM")
    assert e == Binary(op="==", lhs=Ident("kind"), rhs=Ident("LAM"))


def test_parse_and_left_assoc():
    e = parse_expr("a and b and c")
    assert e == Binary("and",
                       lhs=Binary("and", lhs=Ident("a"), rhs=Ident("b")),
                       rhs=Ident("c"))


def test_parse_or_lower_prec_than_and():
    # "a or b and c" == "a or (b and c)"
    e = parse_expr("a or b and c")
    assert e == Binary("or", lhs=Ident("a"),
                       rhs=Binary("and", lhs=Ident("b"), rhs=Ident("c")))


def test_parse_not_higher_prec_than_and():
    e = parse_expr("not a and b")
    assert e == Binary("and",
                       lhs=Unary(op="not", operand=Ident("a")),
                       rhs=Ident("b"))


def test_parse_arithmetic():
    e = parse_expr("n(arg1) + 1")
    assert e == Binary("+", lhs=Call("n", [Ident("arg1")]), rhs=Lit(1))


def test_parse_parens_override_precedence():
    e = parse_expr("(a or b) and c")
    assert e == Binary("and",
                       lhs=Binary("or", lhs=Ident("a"), rhs=Ident("b")),
                       rhs=Ident("c"))


def test_parse_cmp_non_associative():
    # a == b == c is a parse error.
    with pytest.raises(BadTermError):
        parse_expr("a == b == c")


def test_parse_bool_literals():
    assert parse_expr("true") == Lit(value=True)
    assert parse_expr("false") == Lit(value=False)


def test_parse_unary_minus():
    assert parse_expr("-3") == Unary(op="-", operand=Lit(value=3))


def test_parse_trailing_garbage_raises():
    with pytest.raises(BadTermError, match="unexpected"):
        parse_expr("a == b extra")


def test_parse_empty_raises():
    with pytest.raises(BadTermError):
        parse_expr("")


def test_parse_unclosed_paren_raises():
    with pytest.raises(BadTermError):
        parse_expr("(a == b")


def test_parser_does_not_call_eval(monkeypatch):
    """Audit: parser must never invoke eval/exec/compile."""
    def boom(*a, **kw):
        raise AssertionError("parser invoked eval/exec/compile!")
    monkeypatch.setattr(builtins, "eval", boom)
    monkeypatch.setattr(builtins, "exec", boom)
    monkeypatch.setattr(builtins, "compile", boom)
    # Parse a bunch of expressions; none should trip the trap.
    for s in ["a", "type(arg1) == type(arg2)", "a and b", "n(x) + 1",
              "(a or b) and not c", "kind == '+'"]:
        parse_expr(s)


def test_bad_term_error_carries_position():
    try:
        parse_expr("a + + b")
    except BadTermError as e:
        assert "position" in e.details or "pos" in e.details
    else:
        pytest.fail("expected BadTermError")
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_expr_parser.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/dsl/expr_ast.py`**

Create `src/qft_pcn/bridge/dsl/expr_ast.py`:

```python
"""AST for the constraint-expression DSL (spec §4.1, §4.2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Lit:
    value: object         # int | bool | str

@dataclass(frozen=True)
class Ident:
    name: str

@dataclass(frozen=True)
class Call:
    fn: str
    args: tuple            # tuple[Expr, ...]; tuple for hashability

    def __init__(self, fn, args):
        # accept list or tuple
        object.__setattr__(self, "fn", fn)
        object.__setattr__(self, "args", tuple(args))

@dataclass(frozen=True)
class Unary:
    op: str               # "not" | "-"
    operand: "Expr"

@dataclass(frozen=True)
class Binary:
    op: str               # "or" | "and" | "==" | "!=" | "<" | "<=" | ">" | ">="
                          # | "+" | "-" | "*"
    lhs: "Expr"
    rhs: "Expr"


Expr = Union[Lit, Ident, Call, Unary, Binary]
```

- [ ] **Step 4: Implement `bridge/dsl/expr_parser.py`**

Create `src/qft_pcn/bridge/dsl/expr_parser.py`:

```python
"""Recursive-descent parser for the constraint-expression sub-DSL (spec §4.1).

Hand-written, never calls eval/exec/compile/literal_eval. Returns an Expr AST.
"""

from __future__ import annotations

import re
from typing import Iterator

from .expr_ast import Lit, Ident, Call, Unary, Binary, Expr
from ..errors import BadTermError


_TOKEN_RE = re.compile(
    r"\s+"
    r"|(?P<le><=)"
    r"|(?P<ge>>=)"
    r"|(?P<eq>==)"
    r"|(?P<ne>!=)"
    r"|(?P<lt><)"
    r"|(?P<gt>>)"
    r"|(?P<lp>\()"
    r"|(?P<rp>\))"
    r"|(?P<comma>,)"
    r"|(?P<plus>\+)"
    r"|(?P<minus>-)"
    r"|(?P<times>\*)"
    r"|(?P<int>\d+)"
    r"|(?P<str>'[^']*')"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)"
)


_KEYWORDS = {"and", "or", "not", "true", "false"}


class _Tok:
    __slots__ = ("kind", "value", "pos")
    def __init__(self, kind: str, value: str, pos: int):
        self.kind, self.value, self.pos = kind, value, pos


def _tokenize(src: str) -> list[_Tok]:
    pos = 0
    out: list[_Tok] = []
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise BadTermError(
                message=f"unexpected character at position {pos}: {src[pos]!r}",
                details={"pos": pos, "char": src[pos]},
            )
        kind = m.lastgroup
        if kind is None:
            pos = m.end()
            continue
        value = m.group(kind)
        if kind == "ident" and value in _KEYWORDS:
            kind = value
        out.append(_Tok(kind, value, pos))
        pos = m.end()
    out.append(_Tok("eof", "", len(src)))
    return out


class _Parser:
    def __init__(self, toks: list[_Tok], src: str):
        self.toks = toks
        self.i = 0
        self.src = src

    def peek(self) -> _Tok:
        return self.toks[self.i]

    def eat(self, kind: str) -> _Tok:
        t = self.peek()
        if t.kind != kind:
            raise BadTermError(
                message=f"unexpected token {t.kind}={t.value!r} at position "
                        f"{t.pos}; expected {kind}",
                details={"pos": t.pos, "expected": kind, "found": t.kind},
            )
        self.i += 1
        return t

    # expr ::= or_expr
    def parse_or(self) -> Expr:
        left = self.parse_and()
        while self.peek().kind == "or":
            self.i += 1
            right = self.parse_and()
            left = Binary(op="or", lhs=left, rhs=right)
        return left

    def parse_and(self) -> Expr:
        left = self.parse_not()
        while self.peek().kind == "and":
            self.i += 1
            right = self.parse_not()
            left = Binary(op="and", lhs=left, rhs=right)
        return left

    def parse_not(self) -> Expr:
        if self.peek().kind == "not":
            self.i += 1
            return Unary(op="not", operand=self.parse_not())
        return self.parse_cmp()

    _CMP_KINDS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=",
                  "gt": ">", "ge": ">="}

    def parse_cmp(self) -> Expr:
        left = self.parse_add()
        t = self.peek()
        if t.kind in self._CMP_KINDS:
            op = self._CMP_KINDS[t.kind]
            self.i += 1
            right = self.parse_add()
            # Non-associative: another cmp is a parse error.
            if self.peek().kind in self._CMP_KINDS:
                bad = self.peek()
                raise BadTermError(
                    message=f"comparison is non-associative; unexpected "
                            f"{bad.value!r} at position {bad.pos}",
                    details={"pos": bad.pos, "found": bad.value},
                )
            return Binary(op=op, lhs=left, rhs=right)
        return left

    def parse_add(self) -> Expr:
        left = self.parse_mul()
        while self.peek().kind in ("plus", "minus"):
            op = "+" if self.peek().kind == "plus" else "-"
            self.i += 1
            right = self.parse_mul()
            left = Binary(op=op, lhs=left, rhs=right)
        return left

    def parse_mul(self) -> Expr:
        left = self.parse_unary()
        while self.peek().kind == "times":
            self.i += 1
            right = self.parse_unary()
            left = Binary(op="*", lhs=left, rhs=right)
        return left

    def parse_unary(self) -> Expr:
        if self.peek().kind == "minus":
            self.i += 1
            return Unary(op="-", operand=self.parse_unary())
        return self.parse_call()

    def parse_call(self) -> Expr:
        atom = self.parse_atom()
        # Only an Ident can be followed by '(' to form a Call.
        if isinstance(atom, Ident) and self.peek().kind == "lp":
            self.i += 1
            args: list[Expr] = []
            if self.peek().kind != "rp":
                args.append(self.parse_or())
                while self.peek().kind == "comma":
                    self.i += 1
                    args.append(self.parse_or())
            self.eat("rp")
            return Call(fn=atom.name, args=args)
        return atom

    def parse_atom(self) -> Expr:
        t = self.peek()
        if t.kind == "int":
            self.i += 1
            return Lit(value=int(t.value))
        if t.kind == "str":
            self.i += 1
            return Lit(value=t.value[1:-1])    # strip the surrounding quotes
        if t.kind == "true":
            self.i += 1
            return Lit(value=True)
        if t.kind == "false":
            self.i += 1
            return Lit(value=False)
        if t.kind == "ident":
            self.i += 1
            return Ident(name=t.value)
        if t.kind == "lp":
            self.i += 1
            e = self.parse_or()
            self.eat("rp")
            return e
        raise BadTermError(
            message=f"unexpected token {t.kind}={t.value!r} at position {t.pos}",
            details={"pos": t.pos, "found": t.kind},
        )


def parse_expr(src: str) -> Expr:
    """Parse a constraint expression (spec §4.1) into an Expr AST."""
    if not src.strip():
        raise BadTermError(message="empty expression", details={"pos": 0})
    toks = _tokenize(src)
    p = _Parser(toks, src)
    expr = p.parse_or()
    if p.peek().kind != "eof":
        t = p.peek()
        raise BadTermError(
            message=f"unexpected trailing token {t.kind}={t.value!r} "
                    f"at position {t.pos}",
            details={"pos": t.pos, "found": t.kind},
        )
    return expr
```

- [ ] **Step 5: Run parser tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_expr_parser.py -v`
Expected: all tests pass (~18-20 tests).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/dsl/expr_ast.py src/qft_pcn/bridge/dsl/expr_parser.py src/qft_pcn/tests/test_bridge_expr_parser.py
git commit -m "$(cat <<'EOF'
feat(bridge/dsl/expr_parser): hand-written constraint-expression parser

Implements spec §4.1's grammar via recursive descent. Closed grammar,
no eval/exec/compile invocation (explicitly audited by test_parser_
does_not_call_eval). Errors carry position for the LLM to repair.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Field vocabulary (basis-name <-> integer)

**Files:**
- Create: `src/qft_pcn/bridge/dsl/vocab.py`.
- Create: `src/qft_pcn/tests/test_bridge_vocab.py`.

The vocab maps human-readable basis names (`KIND_LAM`, `T_INT`, `V_TRUE`, etc.) to integer indices, mirroring the constants from `logic/encoding.py`. The DSL's `boundary` entries use names; the compiler resolves them to ints.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_bridge_vocab.py`:

```python
"""Tests for bridge/dsl/vocab.py (spec §3.5, §4.1)."""

from __future__ import annotations

import pytest

from src.qft_pcn.bridge.dsl.vocab import resolve_basis, CANONICAL_FIELDS


def test_canonical_field_set():
    assert CANONICAL_FIELDS == ("kind", "type", "bid", "value")


def test_resolve_kind_lam():
    assert resolve_basis("kind", "KIND_LAM") == 2


def test_resolve_type_int():
    assert resolve_basis("type", "T_INT") == 1


def test_resolve_value_true():
    assert resolve_basis("value", "V_TRUE") == 1


def test_resolve_bid_b0():
    assert resolve_basis("bid", "B_0") == 1


def test_resolve_int_literal_value():
    # V_INT(3) -> 3 + INT_LIT_OFFSET = 10
    assert resolve_basis("value", "V_INT(3)") == 10


def test_resolve_int_literal_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        resolve_basis("value", "V_INT(99)")


def test_resolve_passes_through_ints():
    assert resolve_basis("kind", 3) == 3
    assert resolve_basis("kind", 7) == 7


def test_resolve_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown basis name"):
        resolve_basis("kind", "NOT_A_THING")


def test_resolve_non_canonical_field_only_accepts_ints():
    # A user-defined field not in CANONICAL_FIELDS only accepts integer values.
    assert resolve_basis("custom_field", 4) == 4
    with pytest.raises(ValueError, match="non-canonical field"):
        resolve_basis("custom_field", "KIND_LAM")
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_vocab.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/dsl/vocab.py`**

Create `src/qft_pcn/bridge/dsl/vocab.py`:

```python
"""Vocabulary mapping basis names to integer indices for the canonical
kind/type/bid/value fields.

Mirrors constants from src/qft_pcn/logic/encoding.py. The DSL accepts either
integer values or these named constants in boundary entries and constraint
expression literals. Non-canonical fields (chemistry, lattice models, etc.)
accept only integer values.
"""

from __future__ import annotations

import re
from typing import Any

# Import canonical constants from sub-project A. If logic/encoding.py is not
# yet present, fall back to hard-coded equivalents — but the canonical source
# is sub-project A's spec §4.
try:
    from src.qft_pcn.logic.encoding import (
        KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
        KIND_IF, KIND_BIN,
        TYPE_NONE, TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB,
        TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
        BID_NONE, BID_0, BID_1, BID_2, BID_3, BID_4, BID_5, BID_6,
        VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
        VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
        INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX, VALUE_CUTOFF,
    )
except ImportError:  # pragma: no cover - sub-project A is in this repo
    raise


CANONICAL_FIELDS: tuple[str, ...] = ("kind", "type", "bid", "value")


_VOCAB: dict[str, dict[str, int]] = {
    "kind": {
        "KIND_PAD": KIND_PAD, "KIND_VAR": KIND_VAR, "KIND_LAM": KIND_LAM,
        "KIND_APP": KIND_APP, "KIND_INT": KIND_INT, "KIND_BOOL": KIND_BOOL,
        "KIND_IF":  KIND_IF,  "KIND_BIN": KIND_BIN,
        # Bareword form (LLM-friendly): "LAM" alias for "KIND_LAM"
        "PAD": KIND_PAD, "VAR": KIND_VAR, "LAM": KIND_LAM, "APP": KIND_APP,
        "INT": KIND_INT, "BOOL": KIND_BOOL, "IF": KIND_IF, "BIN": KIND_BIN,
    },
    "type": {
        "T_NONE": TYPE_NONE, "T_INT": TYPE_INT, "T_BOOL": TYPE_BOOL,
        "T_ARR_II": TYPE_ARR_II, "T_ARR_IB": TYPE_ARR_IB,
        "T_ARR_BI": TYPE_ARR_BI, "T_ARR_BB": TYPE_ARR_BB,
        "T_ARR_NESTED": TYPE_ARR_NESTED,
    },
    "bid": {
        "B_NONE": BID_NONE,
        "B_0": BID_0, "B_1": BID_1, "B_2": BID_2, "B_3": BID_3,
        "B_4": BID_4, "B_5": BID_5, "B_6": BID_6,
    },
    "value": {
        "V_NONE": VALUE_NONE, "V_FALSE": VALUE_FALSE, "V_TRUE": VALUE_TRUE,
        "V_PLUS": VALUE_PLUS, "V_MINUS": VALUE_MINUS, "V_TIMES": VALUE_TIMES,
        "V_LT": VALUE_LT, "V_EQ": VALUE_EQ,
    },
}

# Operator-literal aliases on the value register (for `kind == '+'`-style terms).
_VALUE_OP_ALIAS: dict[str, int] = {
    "+": VALUE_PLUS, "-": VALUE_MINUS, "*": VALUE_TIMES,
    "<": VALUE_LT, "==": VALUE_EQ,
}


_V_INT_RE = re.compile(r"^V_INT\((-?\d+)\)$")


def resolve_basis(field: str, value: Any) -> int:
    """Resolve a basis name (or integer / V_INT(n)) to an integer index.

    For canonical fields (kind/type/bid/value), looks the name up in the
    vocabulary. For non-canonical fields, only accepts integers.
    """
    if isinstance(value, bool):
        # bool is an int subclass in Python; map to integer index explicitly.
        return 1 if value else 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(
            f"basis value for field {field!r} must be an integer; got {value}"
        )
    if not isinstance(value, str):
        raise ValueError(
            f"basis value for field {field!r} has unsupported type {type(value).__name__}"
        )
    if field not in CANONICAL_FIELDS:
        raise ValueError(
            f"non-canonical field {field!r} only accepts integer basis values; "
            f"got {value!r}"
        )
    if field == "value":
        m = _V_INT_RE.match(value)
        if m:
            n = int(m.group(1))
            if not (INT_LIT_MIN <= n <= INT_LIT_MAX):
                raise ValueError(
                    f"V_INT({n}) out of range [{INT_LIT_MIN}, {INT_LIT_MAX}]"
                )
            return n + INT_LIT_OFFSET
        if value in _VALUE_OP_ALIAS:
            return _VALUE_OP_ALIAS[value]
    table = _VOCAB[field]
    if value in table:
        return table[value]
    raise ValueError(
        f"unknown basis name {value!r} on field {field!r}; "
        f"known: {sorted(table)[:6]}..."
    )
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_vocab.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/bridge/dsl/vocab.py src/qft_pcn/tests/test_bridge_vocab.py
git commit -m "$(cat <<'EOF'
feat(bridge/dsl/vocab): basis-name to integer resolution

Mirrors logic/encoding.py constants so DSL boundary entries and
constraint-expression literals can use human-readable names
(KIND_LAM, T_INT, V_TRUE, V_INT(3)). Non-canonical fields accept
only integers. Bareword aliases ("LAM" == "KIND_LAM") improve
LLM ergonomics.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Stub term compiler (Expr -> Hamiltonian terms)

**Files:**
- Create: `src/qft_pcn/bridge/dsl/term.py`, `bridge/dsl/compiler_stub.py`, `bridge/dsl/compiler.py`.
- Create: `src/qft_pcn/tests/test_bridge_compiler.py`.

`compiler.py` is the dispatch entry; `compiler_stub.py` implements term factories until sub-project B/C land. The stub supports exactly the patterns in spec §3.9 plus a handful of obvious extensions; anything else raises `TermUnsupportedError`.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_bridge_compiler.py`:

```python
"""Tests for the constraint-term compiler (spec §4.3, §11.3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.bridge.dsl.expr_parser import parse_expr
from src.qft_pcn.bridge.dsl.compiler import compile_constraint
from src.qft_pcn.bridge.dsl.term import LocalTerm, TwoSiteTerm, FieldSpec
from src.qft_pcn.bridge.errors import (
    TermUnsupportedError, TypeError as BridgeTypeError,
)


def _canon_fields() -> list[FieldSpec]:
    return [
        FieldSpec(name="kind",  cutoff=8),
        FieldSpec(name="type",  cutoff=8),
        FieldSpec(name="bid",   cutoff=8),
        FieldSpec(name="value", cutoff=16),
    ]


def test_kind_equals_LAM_yields_local_projector():
    expr = parse_expr("kind == LAM")
    terms = compile_constraint(expr, kind="local", site=0,
                               fields=_canon_fields(), weight=1.0)
    assert len(terms) == 1
    t = terms[0]
    assert isinstance(t, LocalTerm)
    assert t.site == 0
    # Hermitian and projector-shaped: trace == d_local - dim(LAM subspace).
    # The term is w * (I - P_LAM); P_LAM has trace = (cutoff_type * cutoff_bid
    # * cutoff_value) = 8 * 8 * 16 = 1024. d_local = 8192. Trace(I - P) = 7168.
    assert np.allclose(t.operator.T.conj(), t.operator)   # hermitian
    assert np.isclose(np.trace(t.operator).real, 8192 - 1024)


def test_two_site_type_equality_yields_two_site_term():
    expr = parse_expr("type(arg1) == type(arg2)")
    terms = compile_constraint(expr, kind="two_site", sites=(5, 7),
                               fields=_canon_fields(), weight=1.0)
    assert len(terms) == 1
    t = terms[0]
    assert isinstance(t, TwoSiteTerm)
    assert t.sites == (5, 7)
    # Operator acts on d_local^2 = 8192^2 space; verify Hermitian.
    assert np.allclose(t.operator.T.conj(), t.operator)


def test_and_combines_to_term_list():
    expr = parse_expr("kind == LAM and type(arg1) == T_INT")
    terms = compile_constraint(expr, kind="local", site=3,
                               fields=_canon_fields(), weight=2.0)
    # AND -> list of LocalTerms summed by the runtime; each at the same site.
    assert len(terms) == 2
    assert all(isinstance(t, LocalTerm) and t.site == 3 for t in terms)


def test_unsupported_pattern_raises():
    # kind(arg1) * value(arg1) is not a supported term shape.
    expr = parse_expr("kind(arg1) * value(arg1)")
    with pytest.raises(TermUnsupportedError):
        compile_constraint(expr, kind="local", site=0,
                           fields=_canon_fields(), weight=1.0)


def test_local_constraint_referencing_arg2_raises_type_error():
    expr = parse_expr("type(arg2) == T_INT")
    with pytest.raises(BridgeTypeError, match="arg2"):
        compile_constraint(expr, kind="local", site=0,
                           fields=_canon_fields(), weight=1.0)


def test_unknown_field_in_term_raises_type_error():
    expr = parse_expr("zzz == LAM")
    with pytest.raises(BridgeTypeError, match="zzz"):
        compile_constraint(expr, kind="local", site=0,
                           fields=_canon_fields(), weight=1.0)


def test_weight_scales_operator():
    e1 = parse_expr("kind == LAM")
    t_unit = compile_constraint(e1, kind="local", site=0,
                                fields=_canon_fields(), weight=1.0)[0]
    t_two  = compile_constraint(e1, kind="local", site=0,
                                fields=_canon_fields(), weight=2.0)[0]
    assert np.allclose(t_two.operator, 2.0 * t_unit.operator)


def test_true_lit_yields_no_terms():
    expr = parse_expr("true")
    terms = compile_constraint(expr, kind="local", site=0,
                               fields=_canon_fields(), weight=1.0)
    assert terms == []
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_compiler.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/dsl/term.py`**

Create `src/qft_pcn/bridge/dsl/term.py`:

```python
"""Term data classes for the bridge compiler.

A LocalTerm is a (site, operator) pair where `operator` acts on the local
d_local Hilbert space (a d_local x d_local Hermitian matrix). A TwoSiteTerm
is a (sites, operator) pair where `operator` acts on d_local^2.

These wrappers exist so the bridge can collect terms produced by B/C's
factory functions and pass them to a Hamiltonian constructor. When B/C
land, this file may be replaced by their canonical Term types — but the
shape (one-site or two-site Hermitian operator + site/sites + weight) is
the contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    """Mirror of DSL FieldSpec (name + cutoff). Independent of B's
    FieldSpecies so the bridge can compile terms even when B isn't fully
    wired. Becomes interchangeable with FieldSpecies in Task 11."""
    name: str
    cutoff: int


@dataclass
class LocalTerm:
    site: int
    operator: np.ndarray   # (d_local, d_local), Hermitian


@dataclass
class TwoSiteTerm:
    sites: tuple[int, int]
    operator: np.ndarray   # (d_local^2, d_local^2), Hermitian
```

- [ ] **Step 4: Implement `bridge/dsl/compiler_stub.py`**

Create `src/qft_pcn/bridge/dsl/compiler_stub.py`:

```python
"""Stub term factories used until sub-project B/C land.

Supports the patterns in spec §3.9 plus a handful of obvious extensions.
Any unsupported shape raises TermUnsupportedError.

When sub-project B/C lands, the bridge's compiler should call B/C's named
factories instead of these stubs; this file is retained as a fallback for
when B/C's mapping is incomplete.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .term import LocalTerm, TwoSiteTerm, FieldSpec
from .vocab import resolve_basis
from ..errors import TermUnsupportedError


def _species_dims(fields: list[FieldSpec]) -> tuple[int, ...]:
    return tuple(f.cutoff for f in fields)


def _embed_diag(field_idx: int, diag_per_field: np.ndarray,
                dims: tuple[int, ...]) -> np.ndarray:
    """Build a diagonal d_local x d_local operator whose `field_idx`'th
    species contributes `diag_per_field` and other species contribute
    identity. The product order matches the row-major ordering used by
    qft/fock.embed_op (leftmost species changes slowest)."""
    d_local = int(np.prod(dims))
    # Build the full diagonal by Kronecker structure.
    diag = np.array([1.0])
    for i, d in enumerate(dims):
        if i == field_idx:
            diag = np.kron(diag, diag_per_field)
        else:
            diag = np.kron(diag, np.ones(d))
    assert diag.shape == (d_local,)
    return np.diag(diag).astype(complex)


def projector_on_field_value(fields: list[FieldSpec], field_name: str,
                              basis_index: int) -> np.ndarray:
    """One-site projector onto |field_name = basis_index> in the full
    d_local Hilbert space."""
    names = [f.name for f in fields]
    if field_name not in names:
        raise TermUnsupportedError(
            message=f"field {field_name!r} not declared",
            details={"known": names},
        )
    idx = names.index(field_name)
    cutoff = fields[idx].cutoff
    if not 0 <= basis_index < cutoff:
        raise TermUnsupportedError(
            message=f"basis index {basis_index} out of [0, {cutoff-1}] "
                    f"for field {field_name!r}",
            details={"field": field_name, "basis": basis_index,
                     "cutoff": cutoff},
        )
    diag = np.zeros(cutoff)
    diag[basis_index] = 1.0
    return _embed_diag(idx, diag, _species_dims(fields))


def term_field_equals_constant(fields: list[FieldSpec], site: int,
                               field_name: str, basis_index: int,
                               weight: float) -> LocalTerm:
    """w * (I - P_{field=basis}) at `site`."""
    dims = _species_dims(fields)
    d_local = int(np.prod(dims))
    P = projector_on_field_value(fields, field_name, basis_index)
    op = weight * (np.eye(d_local, dtype=complex) - P)
    return LocalTerm(site=site, operator=op)


def term_field_equality_two_site(fields: list[FieldSpec],
                                 sites: tuple[int, int],
                                 field_name: str,
                                 weight: float) -> TwoSiteTerm:
    """w * (I - sum_t P_t @ A x P_t @ B) on a two-site bond.

    Penalizes configurations where the two sites disagree on `field_name`.
    """
    names = [f.name for f in fields]
    if field_name not in names:
        raise TermUnsupportedError(
            message=f"field {field_name!r} not declared",
            details={"known": names},
        )
    idx = names.index(field_name)
    cutoff = fields[idx].cutoff
    dims = _species_dims(fields)
    d_local = int(np.prod(dims))
    agree = np.zeros((d_local * d_local, d_local * d_local), dtype=complex)
    for t in range(cutoff):
        P_t = projector_on_field_value(fields, field_name, t)
        agree += np.kron(P_t, P_t)
    I2 = np.eye(d_local * d_local, dtype=complex)
    op = weight * (I2 - agree)
    return TwoSiteTerm(sites=sites, operator=op)
```

- [ ] **Step 5: Implement `bridge/dsl/compiler.py`**

Create `src/qft_pcn/bridge/dsl/compiler.py`:

```python
"""Compile constraint-expression ASTs into Hamiltonian terms.

Walks Expr trees (from bridge/dsl/expr_parser.py) and emits LocalTerm /
TwoSiteTerm objects via the stub factories in compiler_stub.py.

When sub-project B/C land, the relevant branches dispatch to their named
factory functions instead of the stubs. The dispatch table is intentionally
small so the swap is mechanical.
"""

from __future__ import annotations

from typing import Iterable

from .expr_ast import Expr, Lit, Ident, Call, Unary, Binary
from .term import LocalTerm, TwoSiteTerm, FieldSpec
from .vocab import resolve_basis, CANONICAL_FIELDS
from .compiler_stub import (
    term_field_equals_constant,
    term_field_equality_two_site,
)
from ..errors import TermUnsupportedError, TypeError as BridgeTypeError


# Reserved identifiers we recognize as "function-like" in this compiler.
_FIELD_ACCESSORS = {"kind", "type", "bid", "value", "n", "phi", "pi"}


def _expect_arg(name: str, kind: str) -> None:
    """`arg1` is valid in both local and two_site; `arg2` is only valid in
    two_site; `self` is an alias for `arg1` in local."""
    if name in ("arg1", "self"):
        return
    if name == "arg2" and kind == "two_site":
        return
    if name == "arg2" and kind == "local":
        raise BridgeTypeError(
            message=f"local constraint cannot reference arg2",
            details={"identifier": "arg2", "kind": kind},
        )


def _resolve_lit_to_basis(field_name: str, lit: Expr,
                          fields: list[FieldSpec]) -> int:
    """Turn a Lit or Ident on the rhs of a `==` into a basis index for
    `field_name`."""
    names = [f.name for f in fields]
    if field_name not in names:
        raise BridgeTypeError(
            message=f"unknown field {field_name!r}",
            details={"field": field_name, "known": names},
        )
    if isinstance(lit, Lit):
        return resolve_basis(field_name, lit.value)
    if isinstance(lit, Ident):
        return resolve_basis(field_name, lit.name)
    raise TermUnsupportedError(
        message=f"rhs of == must be a literal or named basis constant",
        details={"got": type(lit).__name__},
    )


def compile_constraint(expr: Expr, *, kind: str,
                       site: int | None = None,
                       sites: tuple[int, int] | None = None,
                       fields: list[FieldSpec],
                       weight: float) -> list[LocalTerm | TwoSiteTerm]:
    """Compile one constraint into a list of Hamiltonian terms.

    `kind` is "local" or "two_site"; one of `site` or `sites` is provided.
    Returns a list because AND-of-N composes into N terms summed by the runtime.
    """
    if kind == "local":
        return _compile_local(expr, site=site, fields=fields, weight=weight)
    if kind == "two_site":
        return _compile_two_site(expr, sites=sites, fields=fields,
                                 weight=weight)
    raise BridgeTypeError(
        message=f"unknown constraint kind {kind!r}",
        details={"kind": kind},
    )


def _compile_local(expr: Expr, *, site: int, fields: list[FieldSpec],
                   weight: float) -> list[LocalTerm]:
    # true -> no terms.
    if isinstance(expr, Lit) and expr.value is True:
        return []
    # AND of locals -> recurse.
    if isinstance(expr, Binary) and expr.op == "and":
        return (_compile_local(expr.lhs, site=site, fields=fields, weight=weight)
                + _compile_local(expr.rhs, site=site, fields=fields, weight=weight))
    # field == LIT     (where field is one of fields[i].name or a field accessor)
    if isinstance(expr, Binary) and expr.op == "==":
        field_name = _accessor_field_name(expr.lhs, kind="local")
        basis = _resolve_lit_to_basis(field_name, expr.rhs, fields)
        return [term_field_equals_constant(fields, site, field_name, basis,
                                            weight)]
    raise TermUnsupportedError(
        message=f"unsupported local-constraint shape: {type(expr).__name__}",
        details={"expr": repr(expr)},
    )


def _compile_two_site(expr: Expr, *, sites: tuple[int, int],
                      fields: list[FieldSpec], weight: float
                      ) -> list[LocalTerm | TwoSiteTerm]:
    if isinstance(expr, Lit) and expr.value is True:
        return []
    if isinstance(expr, Binary) and expr.op == "and":
        return (_compile_two_site(expr.lhs, sites=sites, fields=fields,
                                  weight=weight)
                + _compile_two_site(expr.rhs, sites=sites, fields=fields,
                                    weight=weight))
    if isinstance(expr, Binary) and expr.op == "==":
        # type(arg1) == type(arg2) form
        if (isinstance(expr.lhs, Call) and isinstance(expr.rhs, Call)
                and expr.lhs.fn == expr.rhs.fn
                and len(expr.lhs.args) == 1 and len(expr.rhs.args) == 1
                and isinstance(expr.lhs.args[0], Ident)
                and isinstance(expr.rhs.args[0], Ident)
                and {expr.lhs.args[0].name, expr.rhs.args[0].name}
                    == {"arg1", "arg2"}):
            field_name = _accessor_to_field(expr.lhs.fn)
            return [term_field_equality_two_site(fields, sites, field_name,
                                                  weight)]
        # type(argN) == LIT form -> attach to that site as a local term.
        if isinstance(expr.lhs, Call) and len(expr.lhs.args) == 1 \
                and isinstance(expr.lhs.args[0], Ident):
            arg_ident = expr.lhs.args[0].name
            _expect_arg(arg_ident, kind="two_site")
            target_site = sites[0] if arg_ident in ("arg1", "self") else sites[1]
            field_name = _accessor_to_field(expr.lhs.fn)
            basis = _resolve_lit_to_basis(field_name, expr.rhs, fields)
            return [term_field_equals_constant(fields, target_site,
                                                field_name, basis, weight)]
    raise TermUnsupportedError(
        message=f"unsupported two_site-constraint shape: {type(expr).__name__}",
        details={"expr": repr(expr)},
    )


def _accessor_field_name(lhs: Expr, kind: str) -> str:
    """Recognize `<field>` (Ident), `<field>(self)`, `<field>(arg1)` patterns
    and return the field name."""
    if isinstance(lhs, Ident):
        return lhs.name
    if isinstance(lhs, Call) and len(lhs.args) == 1 \
            and isinstance(lhs.args[0], Ident):
        _expect_arg(lhs.args[0].name, kind=kind)
        return _accessor_to_field(lhs.fn)
    raise TermUnsupportedError(
        message=f"lhs of == must be a field name or field accessor call",
        details={"got": type(lhs).__name__},
    )


def _accessor_to_field(fn: str) -> str:
    """`type(...)` -> "type"; `kind(...)` -> "kind"; etc."""
    if fn in _FIELD_ACCESSORS:
        # n/phi/pi are operator accessors, not field accessors per se, but for
        # constraint compilation we treat them as the same field (the local
        # operator differs but the field selection doesn't).
        return fn if fn in ("kind", "type", "bid", "value") else fn
    raise TermUnsupportedError(
        message=f"unsupported accessor {fn!r}",
        details={"fn": fn, "known": sorted(_FIELD_ACCESSORS)},
    )
```

- [ ] **Step 6: Run compiler tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_compiler.py -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/qft_pcn/bridge/dsl/term.py src/qft_pcn/bridge/dsl/compiler_stub.py src/qft_pcn/bridge/dsl/compiler.py src/qft_pcn/tests/test_bridge_compiler.py
git commit -m "$(cat <<'EOF'
feat(bridge/dsl/compiler): Expr -> Hamiltonian-term compiler with stubs

Walks parsed constraint expressions and emits LocalTerm / TwoSiteTerm
objects. The stub backend covers the spec §3.9 patterns (field == const,
field(arg1) == field(arg2), AND-of-supported); anything else raises
TermUnsupportedError. When B/C land, the dispatch swaps to their
canonical factories with no caller change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Top-level DSL pipeline (validate, compile, no run yet)

**Files:**
- Create: `src/qft_pcn/bridge/dsl/__init__.py` (re-exports), `bridge/dsl/pipeline.py`.
- Create: `src/qft_pcn/tests/test_bridge_dsl_pipeline.py`.

Wires phases 1-4 of spec §3.8: parse → schema validate → cross-validate → compile each constraint into terms. The result is a `CompiledDsl` ready for the runtime.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_bridge_dsl_pipeline.py`:

```python
"""Tests for the full DSL validate+compile pipeline."""

from __future__ import annotations

import json
import pytest

from src.qft_pcn.bridge.dsl.pipeline import validate_dsl, compile_dsl, CompiledDsl
from src.qft_pcn.bridge.errors import (
    BadJsonError, BadSchemaError, BadReferenceError, BadTermError,
    TermUnsupportedError,
)


CANONICAL_DSL = {
    "fields": [
        {"name": "kind",  "cutoff": 8},
        {"name": "type",  "cutoff": 8},
        {"name": "bid",   "cutoff": 8},
        {"name": "value", "cutoff": 16},
    ],
    "sites": 8,
    "constraints": [
        {"kind": "local", "site": 0, "term": "kind == LAM", "weight": 1.0},
        {"kind": "two_site", "sites": [0, 1],
         "term": "type(arg1) == type(arg2)", "weight": 0.5},
    ],
    "boundary": {"0": {"kind": "KIND_LAM"}},
    "observables": [{"site": 1, "field": "kind", "op": "n"}],
    "search": {"method": "imag_time", "steps": 10, "chi_max": 8},
}


def test_validate_dsl_accepts_canonical():
    result = validate_dsl(CANONICAL_DSL)
    assert result["valid"] is True
    assert result["errors"] == []


def test_validate_dsl_rejects_bad_json_at_string_phase():
    out = validate_dsl("garbage { json")
    assert out["valid"] is False
    assert any(e["code"] == "BRIDGE_E_BAD_JSON" for e in out["errors"])


def test_validate_dsl_rejects_bad_schema():
    bad = {**CANONICAL_DSL}
    bad["sites"] = -1
    out = validate_dsl(bad)
    assert out["valid"] is False
    assert any(e["code"] == "BRIDGE_E_BAD_SCHEMA" for e in out["errors"])


def test_validate_dsl_rejects_bad_reference():
    bad = {**CANONICAL_DSL,
           "observables": [{"site": 0, "field": "nope", "op": "n"}]}
    out = validate_dsl(bad)
    assert out["valid"] is False
    assert any(e["code"] == "BRIDGE_E_BAD_REFERENCE" for e in out["errors"])


def test_validate_dsl_rejects_bad_term():
    bad = {**CANONICAL_DSL,
           "constraints": [{"kind": "local", "site": 0,
                            "term": "kind == == LAM", "weight": 1.0}]}
    out = validate_dsl(bad)
    assert out["valid"] is False
    assert any(e["code"] == "BRIDGE_E_BAD_TERM" for e in out["errors"])


def test_compile_dsl_returns_compiled_dsl():
    cd = compile_dsl(CANONICAL_DSL)
    assert isinstance(cd, CompiledDsl)
    assert cd.sites == 8
    assert len(cd.terms) == 2          # two constraints; AND would split further
    assert len(cd.observables) == 1
    assert cd.boundary[0]["kind"] == 2  # KIND_LAM resolved to 2


def test_compile_dsl_propagates_term_unsupported():
    bad = {**CANONICAL_DSL,
           "constraints": [{"kind": "local", "site": 0,
                            "term": "kind(arg1) * value(arg1)",
                            "weight": 1.0}]}
    # validate_dsl phase 4 doesn't compile (phase 4 only parses); compile_dsl
    # is where TermUnsupportedError surfaces.
    with pytest.raises(TermUnsupportedError):
        compile_dsl(bad)
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_dsl_pipeline.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/dsl/pipeline.py`**

Create `src/qft_pcn/bridge/dsl/pipeline.py`:

```python
"""Top-level DSL pipeline: parse/validate/compile.

validate_dsl(spec) -> {"valid": bool, "errors": [<BridgeError-as-dict>]}
compile_dsl(spec)  -> CompiledDsl, ready for the runtime.

Phase 1: JSON parse  (if input is str)
Phase 2: JSON Schema validate
Phase 3: Cross-field reference validate
Phase 4: Constraint-expression parse  (only parses; compile happens in
         compile_dsl, not validate_dsl)
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Any

from .schema import parse_and_validate, validate_schema, _apply_defaults
from .cross_validate import cross_validate
from .expr_parser import parse_expr
from .compiler import compile_constraint
from .term import LocalTerm, TwoSiteTerm, FieldSpec
from .vocab import resolve_basis
from ..errors import (
    BridgeError, BadJsonError, BadSchemaError, BadReferenceError,
    BadTermError, TermUnsupportedError,
)


@dataclass
class CompiledDsl:
    fields: list[FieldSpec]
    sites: int
    terms: list[LocalTerm | TwoSiteTerm]
    boundary: dict[int, dict[str, int]]    # site -> {field: basis_index}
    observables: list[dict[str, Any]]
    search: dict[str, Any]


def _normalize_input(spec: Any) -> dict[str, Any]:
    """Accept either a dict (Python-side caller) or a JSON string (RPC side)."""
    if isinstance(spec, str):
        return parse_and_validate(spec)
    if not isinstance(spec, dict):
        raise BadSchemaError(
            message="DSL must be a JSON object or string",
            details={"got": type(spec).__name__},
        )
    validate_schema(spec)
    return _apply_defaults(spec)


def validate_dsl(spec: Any) -> dict[str, Any]:
    """Run phases 1-4 and return a structured report. Never raises;
    accumulates errors into a list (one per phase that fails)."""
    errors: list[dict[str, Any]] = []
    try:
        dsl = _normalize_input(spec)
    except BridgeError as e:
        errors.append(e.to_dict())
        return {"valid": False, "errors": errors}
    try:
        cross_validate(dsl)
    except BridgeError as e:
        errors.append(e.to_dict())
        return {"valid": False, "errors": errors}
    # Phase 4: parse every constraint term (no compile).
    for i, c in enumerate(dsl["constraints"]):
        try:
            parse_expr(c["term"])
        except BadTermError as e:
            e.details["constraint_index"] = i
            errors.append(e.to_dict())
    return {"valid": not errors, "errors": errors}


def compile_dsl(spec: Any) -> CompiledDsl:
    """Validate phases 1-3, parse and compile every constraint, resolve
    boundary basis names. Raises on any failure (single-error mode, unlike
    validate_dsl)."""
    dsl = _normalize_input(spec)
    cross_validate(dsl)
    fields = [FieldSpec(name=f["name"], cutoff=f["cutoff"])
              for f in dsl["fields"]]
    terms: list = []
    for i, c in enumerate(dsl["constraints"]):
        try:
            expr = parse_expr(c["term"])
        except BadTermError as e:
            e.details["constraint_index"] = i
            raise
        weight = float(c.get("weight", 1.0))
        if c["kind"] == "local":
            ts = compile_constraint(expr, kind="local", site=c["site"],
                                    fields=fields, weight=weight)
        else:
            ts = compile_constraint(expr, kind="two_site",
                                    sites=tuple(c["sites"]),
                                    fields=fields, weight=weight)
        terms.extend(ts)
    # Resolve boundary basis names to integers.
    boundary: dict[int, dict[str, int]] = {}
    for site_key, fmap in (dsl.get("boundary") or {}).items():
        site = int(site_key)
        boundary[site] = {fname: resolve_basis(fname, val)
                          for fname, val in fmap.items()}
    return CompiledDsl(
        fields=fields, sites=dsl["sites"], terms=terms, boundary=boundary,
        observables=list(dsl["observables"]), search=dict(dsl["search"]),
    )
```

- [ ] **Step 4: Update `bridge/dsl/__init__.py`**

Replace `src/qft_pcn/bridge/dsl/__init__.py` with:

```python
"""DSL subpackage: schema validation, expression parsing, term compilation."""

from .pipeline import validate_dsl, compile_dsl, CompiledDsl
from .term import LocalTerm, TwoSiteTerm, FieldSpec

__all__ = ["validate_dsl", "compile_dsl", "CompiledDsl",
           "LocalTerm", "TwoSiteTerm", "FieldSpec"]
```

- [ ] **Step 5: Run all bridge tests so far**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_*.py -v 2>&1 | tail -20`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/dsl/__init__.py src/qft_pcn/bridge/dsl/pipeline.py src/qft_pcn/tests/test_bridge_dsl_pipeline.py
git commit -m "$(cat <<'EOF'
feat(bridge/dsl/pipeline): validate_dsl + compile_dsl

Phases 1-4 of spec §3.8 in a single entry point. validate_dsl returns
a structured report and never raises; compile_dsl raises on first
failure and returns a CompiledDsl ready for the runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Runtime — initial state + boundary clamping

**Files:**
- Create: `src/qft_pcn/bridge/runtime/__init__.py` (placeholder), `bridge/runtime/clamp.py`, `bridge/runtime/initial_state.py`.
- Create: `src/qft_pcn/tests/test_bridge_runtime_clamp.py`.

Implements spec §5.4: building the initial MPS from boundary clamps, and the per-step projection that maintains them during evolution.

- [ ] **Step 1: Write failing tests**

Create `src/qft_pcn/tests/test_bridge_runtime_clamp.py`:

```python
"""Tests for runtime boundary-clamp construction and projection."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.bridge.dsl.term import FieldSpec
from src.qft_pcn.bridge.runtime.initial_state import build_initial_state
from src.qft_pcn.bridge.runtime.clamp import project_site, Clamp


def _canon_fields() -> list[FieldSpec]:
    return [
        FieldSpec(name="kind", cutoff=4),
        FieldSpec(name="value", cutoff=4),
    ]   # d_local = 16


def test_build_initial_state_no_boundary_is_vacuum():
    state = build_initial_state(fields=_canon_fields(), sites=3, boundary={})
    assert state.N == 3
    # Vacuum: all amplitude on local basis index 0.
    e0 = state.local_expectation(0, _proj_onto_basis(0, d=16))
    assert abs(e0.real - 1.0) < 1e-10


def test_build_initial_state_with_boundary_clamps_named_site():
    fields = _canon_fields()
    boundary = {1: {"kind": 2}}    # site 1 kind register = 2
    state = build_initial_state(fields=fields, sites=3, boundary=boundary)
    # Expected basis index at site 1: kind=2, value=0 -> index = 2 * 4 + 0 = 8
    e8 = state.local_expectation(1, _proj_onto_basis(8, d=16))
    assert abs(e8.real - 1.0) < 1e-10
    # Sites 0 and 2 are still vacuum (kind=0, value=0).
    e0_s0 = state.local_expectation(0, _proj_onto_basis(0, d=16))
    assert abs(e0_s0.real - 1.0) < 1e-10


def test_project_site_restores_clamp_after_disturbance():
    fields = _canon_fields()
    boundary = {0: {"kind": 1}}
    state = build_initial_state(fields=fields, sites=2, boundary=boundary)
    # Inject noise by applying a small random local gate at site 0.
    rng = np.random.default_rng(0)
    pert = np.eye(16) + 0.01 * (rng.standard_normal((16, 16))
                                + 1j * rng.standard_normal((16, 16)))
    state.apply_local_gate(0, pert)
    state.normalize()
    clamp = Clamp(site=0, field="kind", basis_index=1)
    project_site(state, clamp, fields=fields)
    state.normalize()
    # Expect restored basis index 4 (kind=1, value=0 -> 1*4+0=4).
    e4 = state.local_expectation(0, _proj_onto_basis(4, d=16))
    assert abs(e4.real - 1.0) < 1e-10


def _proj_onto_basis(idx: int, d: int) -> np.ndarray:
    p = np.zeros((d, d), dtype=complex)
    p[idx, idx] = 1.0
    return p
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_runtime_clamp.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/runtime/clamp.py`**

Create `src/qft_pcn/bridge/runtime/clamp.py`:

```python
"""Boundary-site projection (spec §5.4).

A Clamp pins one site's named field to a fixed basis index. After each
evolution step the runtime projects every clamped site back onto the
sub-Hilbert-space where that field has that value (and renormalizes).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsl.term import FieldSpec
from src.qft_pcn.qft.mps import MPS


@dataclass(frozen=True)
class Clamp:
    site: int
    field: str
    basis_index: int


def _field_projector(fields: list[FieldSpec], field_name: str,
                     basis_index: int) -> np.ndarray:
    names = [f.name for f in fields]
    idx = names.index(field_name)
    cutoff = fields[idx].cutoff
    # Build a diagonal d_local x d_local projector by Kron structure.
    diag = np.array([1.0])
    for i, f in enumerate(fields):
        if i == idx:
            d = np.zeros(f.cutoff)
            d[basis_index] = 1.0
            diag = np.kron(diag, d)
        else:
            diag = np.kron(diag, np.ones(f.cutoff))
    return np.diag(diag).astype(complex)


def project_site(state: MPS, clamp: Clamp, *,
                 fields: list[FieldSpec]) -> None:
    """Apply the projector for `clamp` to `state` in-place. Caller is
    responsible for re-normalizing after."""
    P = _field_projector(fields, clamp.field, clamp.basis_index)
    state.apply_local_gate(clamp.site, P)
```

- [ ] **Step 4: Implement `bridge/runtime/initial_state.py`**

Create `src/qft_pcn/bridge/runtime/initial_state.py`:

```python
"""Build the initial MPS from boundary clamps (spec §5.1 step 4)."""

from __future__ import annotations

import numpy as np

from ..dsl.term import FieldSpec
from src.qft_pcn.qft.mps import MPS


def build_initial_state(*, fields: list[FieldSpec], sites: int,
                        boundary: dict[int, dict[str, int]]) -> MPS:
    """Return a product MPS where each clamped site holds its clamped basis
    state on the named field (and vacuum on the others); unclamped sites are
    full vacuum (basis index 0)."""
    d_local = 1
    for f in fields:
        d_local *= f.cutoff
    cutoffs = [f.cutoff for f in fields]
    names = [f.name for f in fields]

    site_vecs: list[np.ndarray] = []
    for s in range(sites):
        # Per-species local index; vacuum on all unless overridden.
        per_species = [0] * len(fields)
        for fname, basis in (boundary.get(s) or {}).items():
            j = names.index(fname)
            per_species[j] = int(basis)
        # Build d_local-vector with a single 1 at the row-major index of
        # per_species under cutoffs.
        idx = 0
        for j, val in enumerate(per_species):
            idx = idx * cutoffs[j] + val
        v = np.zeros(d_local, dtype=complex)
        v[idx] = 1.0
        site_vecs.append(v)

    return MPS.from_product(site_vecs)
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_runtime_clamp.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/runtime/clamp.py src/qft_pcn/bridge/runtime/initial_state.py src/qft_pcn/tests/test_bridge_runtime_clamp.py
git commit -m "$(cat <<'EOF'
feat(bridge/runtime): initial-state builder and boundary projection

build_initial_state assembles a product MPS where clamped sites carry
the named basis state; project_site re-applies the clamp's field
projector after each evolution step. Together these implement the
post-selection treatment of boundary constraints from spec §5.4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Hamiltonian assembly + evolution wrapper

**Files:**
- Create: `src/qft_pcn/bridge/runtime/hamiltonian.py`, `bridge/runtime/evolution.py`.
- Create: `src/qft_pcn/tests/test_bridge_runtime_evolution.py`.

Builds a Hamiltonian from the compiled `LocalTerm` / `TwoSiteTerm` lists, suitable for the existing `qft/evolution.py::trotter_step`. This adapts the bridge's term shape to whatever the existing `Hamiltonian` class wants. To keep the bridge thin, this layer constructs a small `BridgeHamiltonian` wrapper that exposes `local_op(k)` and `bond_op(k)` — the only methods `trotter_step` needs.

- [ ] **Step 1: Write failing tests**

Create `src/qft_pcn/tests/test_bridge_runtime_evolution.py`:

```python
"""Tests for the runtime evolution wrapper (spec §5.5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.bridge.dsl.term import FieldSpec, LocalTerm, TwoSiteTerm
from src.qft_pcn.bridge.runtime.hamiltonian import BridgeHamiltonian
from src.qft_pcn.bridge.runtime.evolution import evolve_with_clamps
from src.qft_pcn.bridge.runtime.clamp import Clamp
from src.qft_pcn.bridge.runtime.initial_state import build_initial_state
from src.qft_pcn.qft.evolution import energy


def _two_field_setup():
    return [FieldSpec("kind", 4), FieldSpec("value", 4)]   # d_local = 16


def test_bridge_hamiltonian_local_op_is_hermitian():
    fields = _two_field_setup()
    op = np.zeros((16, 16), dtype=complex)
    op[3, 3] = 1.0
    op[7, 7] = 0.5
    H = BridgeHamiltonian(
        fields=fields, sites=3,
        terms=[LocalTerm(site=1, operator=op)],
    )
    L = H.local_op(1)
    assert L.shape == (16, 16)
    assert np.allclose(L, L.T.conj())


def test_bridge_hamiltonian_zero_op_when_site_unconstrained():
    fields = _two_field_setup()
    H = BridgeHamiltonian(fields=fields, sites=3, terms=[])
    assert np.allclose(H.local_op(2), 0.0)


def test_evolution_lowers_energy_toward_clamp():
    """Single-site local term penalizing kind != 1 at site 0; imag-time
    evolution from vacuum should drive site 0 toward kind=1 and lower energy."""
    fields = _two_field_setup()
    d_local = 16
    P_kind1 = np.zeros((d_local, d_local), dtype=complex)
    # kind=1 subspace: local indices [4..7]
    for i in range(4, 8):
        P_kind1[i, i] = 1.0
    op = np.eye(d_local, dtype=complex) - P_kind1   # penalty for kind != 1
    H = BridgeHamiltonian(fields=fields, sites=2,
                          terms=[LocalTerm(site=0, operator=op)])
    state = build_initial_state(fields=fields, sites=2, boundary={})
    E0 = energy(state, H)
    history = evolve_with_clamps(state, H, dt=0.1, steps=20,
                                 chi_max=8, clamps=[], fields=fields)
    E_final = history.energy_per_step[-1]
    assert E_final < E0 - 0.1


def test_clamp_persists_through_evolution():
    fields = _two_field_setup()
    H = BridgeHamiltonian(fields=fields, sites=2, terms=[])
    state = build_initial_state(fields=fields, sites=2,
                                boundary={0: {"kind": 2}})
    clamps = [Clamp(site=0, field="kind", basis_index=2)]
    evolve_with_clamps(state, H, dt=0.05, steps=10, chi_max=8,
                       clamps=clamps, fields=fields)
    # Site 0 should still have kind=2 (basis index 2 * 4 + 0 = 8).
    P = np.zeros((16, 16), dtype=complex)
    P[8, 8] = 1.0
    val = state.local_expectation(0, P)
    assert val.real > 1.0 - 1e-6
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_runtime_evolution.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/runtime/hamiltonian.py`**

Create `src/qft_pcn/bridge/runtime/hamiltonian.py`:

```python
"""BridgeHamiltonian: thin wrapper exposing local_op(k)/bond_op(k) from
a list of LocalTerm / TwoSiteTerm objects.

Compatible with qft/evolution.py:trotter_step's expectations: only
local_op(k), bond_op(k), N, and d_local are required.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsl.term import FieldSpec, LocalTerm, TwoSiteTerm


@dataclass
class BridgeHamiltonian:
    fields: list[FieldSpec]
    sites: int
    terms: list[LocalTerm | TwoSiteTerm]

    def __post_init__(self) -> None:
        self.N = self.sites
        self.d_local = 1
        for f in self.fields:
            self.d_local *= f.cutoff
        # Pre-aggregate per-site / per-bond operators.
        self._local: dict[int, np.ndarray] = {}
        self._bond: dict[int, np.ndarray] = {}
        for t in self.terms:
            if isinstance(t, LocalTerm):
                self._local.setdefault(t.site,
                    np.zeros((self.d_local, self.d_local), dtype=complex))
                self._local[t.site] = self._local[t.site] + t.operator
            elif isinstance(t, TwoSiteTerm):
                a, b = t.sites
                if abs(a - b) != 1:
                    # Non-adjacent two-site terms are not supported by TEBD.
                    # We surface this at evolve-time rather than here.
                    pass
                k = min(a, b)
                d2 = self.d_local * self.d_local
                self._bond.setdefault(k, np.zeros((d2, d2), dtype=complex))
                self._bond[k] = self._bond[k] + t.operator
            else:
                raise TypeError(f"unexpected term type {type(t).__name__}")

    def local_op(self, k: int) -> np.ndarray:
        if k in self._local:
            return self._local[k]
        return np.zeros((self.d_local, self.d_local), dtype=complex)

    def bond_op(self, k: int) -> np.ndarray:
        d2 = self.d_local * self.d_local
        if k in self._bond:
            return self._bond[k]
        return np.zeros((d2, d2), dtype=complex)
```

- [ ] **Step 4: Implement `bridge/runtime/evolution.py`**

Create `src/qft_pcn/bridge/runtime/evolution.py`:

```python
"""evolve_with_clamps — imag-time evolution with per-step boundary projection.

Reuses qft/evolution.py:trotter_step as-is; the only addition is the
post-step clamp loop and an energy-per-step trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hamiltonian import BridgeHamiltonian
from .clamp import Clamp, project_site
from ..dsl.term import FieldSpec
from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.qft.evolution import trotter_step, energy


@dataclass
class ConvergenceHistory:
    energy_per_step: list[float] = field(default_factory=list)
    trunc_error_per_step: list[float] = field(default_factory=list)


def evolve_with_clamps(state: MPS, H: BridgeHamiltonian, *,
                       dt: float, steps: int, chi_max: int,
                       clamps: list[Clamp], fields: list[FieldSpec]
                       ) -> ConvergenceHistory:
    hist = ConvergenceHistory()
    for _ in range(steps):
        err = trotter_step(state, H, dt, imaginary=True, chi_max=chi_max)
        state.normalize()
        for c in clamps:
            project_site(state, c, fields=fields)
            state.normalize()
        hist.energy_per_step.append(float(energy(state, H)))
        hist.trunc_error_per_step.append(float(err if err is not None else 0.0))
    return hist
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_runtime_evolution.py -v`
Expected: all pass. If `trotter_step`'s signature differs from `(state, H, dt, imaginary, chi_max)`, adapt accordingly by inspecting `src/qft_pcn/qft/evolution.py`.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/runtime/hamiltonian.py src/qft_pcn/bridge/runtime/evolution.py src/qft_pcn/tests/test_bridge_runtime_evolution.py
git commit -m "$(cat <<'EOF'
feat(bridge/runtime): BridgeHamiltonian and evolve_with_clamps

Aggregates LocalTerm/TwoSiteTerm into per-site/per-bond operators
exposed via local_op(k)/bond_op(k) for the existing TEBD trotter_step.
evolve_with_clamps drives imag-time evolution and re-projects clamped
sites after each step (spec §5.4, §5.5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `run_problem` and `RunResult`

**Files:**
- Create: `src/qft_pcn/bridge/runtime.py` (top-level — note: `runtime/` subdir already exists for clamp etc., so we'll create `runtime/api.py` instead and import from it).

Adjust to spec §9's layout: put the top-level `run_problem` / `diagnose_problem` in `bridge/runtime.py` (replacing the existing one-line module if any) while keeping subcomponents in `bridge/runtime/`. Python lets both coexist if `runtime/` is a package — pick one. **Decision:** use `bridge/runtime/` as a package and put `run_problem` / `diagnose_problem` in `bridge/runtime/__init__.py`.

**Files (corrected):**
- Modify: `src/qft_pcn/bridge/runtime/__init__.py`.
- Create: `src/qft_pcn/bridge/runtime/observables.py` (op factory for `n`, `phi`, etc.).
- Create: `src/qft_pcn/bridge/runtime/result.py` (RunResult, ObservableValue dataclasses).
- Create: `src/qft_pcn/tests/test_bridge_runtime_run.py`.

- [ ] **Step 1: Write failing tests**

Create `src/qft_pcn/tests/test_bridge_runtime_run.py`:

```python
"""Tests for run_problem / RunResult (spec §5, §11.4)."""

from __future__ import annotations

import math
import pytest

from src.qft_pcn.bridge.runtime import run_problem, diagnose_problem
from src.qft_pcn.bridge.runtime.result import RunResult, ObservableValue


MINIMAL = {
    "fields": [{"name": "x", "cutoff": 4}],
    "sites": 2,
    "constraints": [],
    "observables": [{"site": 0, "field": "x", "op": "n"}],
    "search": {"method": "imag_time", "steps": 5, "chi_max": 4},
}


def test_minimal_run_returns_run_result():
    res = run_problem(MINIMAL)
    assert isinstance(res, RunResult)
    assert len(res.observables) == 1
    ov = res.observables[0]
    assert isinstance(ov, ObservableValue)
    assert ov.site == 0 and ov.field == "x" and ov.op == "n"
    assert math.isfinite(ov.value)
    assert abs(ov.imag_part) < 1e-9


def test_boundary_clamp_persists_after_evolution():
    dsl = {
        "fields": [{"name": "x", "cutoff": 4}],
        "sites": 2,
        "constraints": [],
        "boundary": {"0": {"x": 2}},
        "observables": [{"site": 0, "field": "x", "op": "n"}],
        "search": {"method": "imag_time", "steps": 20, "chi_max": 4},
    }
    res = run_problem(dsl)
    # x=2 is occupation number 2 on the local Fock space, so <n_x@0> ≈ 2.
    assert abs(res.observables[0].value - 2.0) < 1e-6


def test_energy_per_term_sum_matches_total():
    dsl = {
        "fields": [
            {"name": "kind", "cutoff": 4},
            {"name": "value", "cutoff": 4},
        ],
        "sites": 2,
        "constraints": [
            {"kind": "local", "site": 0, "term": "kind == 1", "weight": 1.0},
            {"kind": "local", "site": 1, "term": "kind == 1", "weight": 1.0},
        ],
        "observables": [{"site": 0, "field": "kind", "op": "n"}],
        "search": {"method": "imag_time", "steps": 5, "chi_max": 4},
    }
    res = run_problem(dsl)
    assert abs(sum(res.energy_per_term) - res.energy) < 1e-8


def test_converged_flag_for_simple_problem():
    dsl = {
        "fields": [{"name": "x", "cutoff": 4}],
        "sites": 2,
        "constraints": [
            {"kind": "local", "site": 0, "term": "x == 0", "weight": 5.0},
        ],
        "observables": [{"site": 0, "field": "x", "op": "n"}],
        "search": {"method": "imag_time", "steps": 50, "chi_max": 4,
                   "dt": 0.05},
    }
    res = run_problem(dsl)
    assert res.converged is True


def test_diagnose_returns_none_debugger_report_when_d_absent():
    res = diagnose_problem(MINIMAL)
    # D not landed yet; debugger_report should be None.
    assert res.debugger_report is None
    assert res.result.observables[0].field == "x"
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_runtime_run.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/runtime/result.py`**

Create `src/qft_pcn/bridge/runtime/result.py`:

```python
"""Result dataclasses for the bridge runtime (spec §5.2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ObservableValue:
    site: int
    field: str
    op: str
    value: float       # real part
    imag_part: float   # imaginary part; should be ~0 for Hermitian ops


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
```

- [ ] **Step 4: Implement `bridge/runtime/observables.py`**

Create `src/qft_pcn/bridge/runtime/observables.py`:

```python
"""Construct local observable operators (n, phi, pi, a, adag, identity) on
the canonical d_local Hilbert space, embedded on the named field species.
"""

from __future__ import annotations

import numpy as np

from ..dsl.term import FieldSpec
from src.qft_pcn.qft.fock import (
    annihilation, creation, number, phi_op, pi_op, identity, embed_op,
)


_OPS = {
    "a":        annihilation,
    "adag":     creation,
    "n":        number,
    "phi":      phi_op,
    "pi":       pi_op,
    "identity": identity,
}


def build_observable_op(fields: list[FieldSpec], field_name: str,
                        op: str) -> np.ndarray:
    names = [f.name for f in fields]
    if field_name not in names:
        raise KeyError(f"unknown observable field {field_name!r}")
    if op not in _OPS:
        raise KeyError(f"unknown observable op {op!r}")
    idx = names.index(field_name)
    cutoff = fields[idx].cutoff
    op_local = _OPS[op](cutoff)
    dims = tuple(f.cutoff for f in fields)
    return embed_op(op_local, idx, dims)
```

- [ ] **Step 5: Implement `bridge/runtime/__init__.py`**

Replace `src/qft_pcn/bridge/runtime/__init__.py`:

```python
"""Bridge runtime: run_problem, diagnose_problem.

Spec §5. The runtime is transport-independent; it accepts a Python dict
(or JSON string) DSL and returns a RunResult / RunDiagnostic dataclass.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..dsl.pipeline import compile_dsl, CompiledDsl
from ..dsl.term import FieldSpec, LocalTerm, TwoSiteTerm
from .hamiltonian import BridgeHamiltonian
from .initial_state import build_initial_state
from .clamp import Clamp
from .evolution import evolve_with_clamps
from .observables import build_observable_op
from .result import (
    ObservableValue, ConvergenceHistorySummary, RunResult, RunDiagnostic,
)
from ..errors import (
    UnsupportedMethodError, NumericFailureError, InternalError,
)
from src.qft_pcn.qft.evolution import energy


def _clamps_from_boundary(cd: CompiledDsl) -> list[Clamp]:
    out: list[Clamp] = []
    for site, fmap in cd.boundary.items():
        for fname, basis in fmap.items():
            out.append(Clamp(site=site, field=fname, basis_index=basis))
    return out


def _energy_per_term(state, terms: list[LocalTerm | TwoSiteTerm]) -> list[float]:
    vals = []
    for t in terms:
        if isinstance(t, LocalTerm):
            v = state.local_expectation(t.site, t.operator)
        else:
            v = state.two_site_expectation(min(t.sites), t.operator)
        vals.append(float(v.real))
    return vals


def _converged(history: list[float], *, tol: float = 1e-6,
               window: int = 5) -> bool:
    if len(history) < window + 1:
        return False
    tail = history[-(window + 1):]
    deltas = [tail[i] - tail[i - 1] for i in range(1, len(tail))]
    monotonic = all(d <= 1e-10 for d in deltas)
    settled = all(abs(d) < tol for d in deltas)
    return monotonic and settled


def run_problem(dsl: Any) -> RunResult:
    """Validate, compile, evolve, measure (spec §5.1)."""
    cd = compile_dsl(dsl)
    if cd.search.get("method", "imag_time") != "imag_time":
        raise UnsupportedMethodError(
            message=f"method {cd.search.get('method')!r} not supported",
            details={"method": cd.search.get("method")},
        )
    state = build_initial_state(fields=cd.fields, sites=cd.sites,
                                boundary=cd.boundary)
    H = BridgeHamiltonian(fields=cd.fields, sites=cd.sites, terms=cd.terms)
    clamps = _clamps_from_boundary(cd)
    history = evolve_with_clamps(
        state, H,
        dt=float(cd.search.get("dt", 0.05)),
        steps=int(cd.search["steps"]),
        chi_max=int(cd.search["chi_max"]),
        clamps=clamps, fields=cd.fields,
    )
    # Observables.
    obs: list[ObservableValue] = []
    for o in cd.observables:
        op = build_observable_op(cd.fields, o["field"], o["op"])
        v = state.local_expectation(o["site"], op)
        if not (np.isfinite(v.real) and np.isfinite(v.imag)):
            raise NumericFailureError(
                message=f"observable at site {o['site']} returned non-finite",
                details={"site": o["site"], "field": o["field"], "op": o["op"]},
            )
        obs.append(ObservableValue(site=o["site"], field=o["field"],
                                    op=o["op"], value=float(v.real),
                                    imag_part=float(v.imag)))
    E = float(energy(state, H))
    e_per_term = _energy_per_term(state, cd.terms)
    return RunResult(
        observables=obs,
        energy=E,
        energy_per_term=e_per_term,
        truncation_error_sum=float(sum(history.trunc_error_per_step)),
        final_bond_dimensions=list(state.bond_dimensions()),
        converged=_converged(history.energy_per_step),
        convergence_history=ConvergenceHistorySummary(
            energy_per_step=list(history.energy_per_step)
        ),
    )


def diagnose_problem(dsl: Any) -> RunDiagnostic:
    """Same as run_problem plus sub-project D's diagnostic report (spec §5.3).

    Until D lands, debugger_report is None.
    """
    result = run_problem(dsl)
    try:
        from src.qft_pcn.logic.debugger import diagnose as _d_diagnose  # type: ignore
    except ImportError:
        return RunDiagnostic(result=result, debugger_report=None)
    try:
        report = _d_diagnose(...)  # type: ignore[call-arg]
    except Exception as exc:        # noqa: BLE001
        raise InternalError(
            message=f"D's diagnose() failed: {exc}",
            details={"exc": type(exc).__name__},
        ) from exc
    return RunDiagnostic(result=result, debugger_report=report)


__all__ = ["run_problem", "diagnose_problem",
           "Clamp", "BridgeHamiltonian", "build_initial_state",
           "evolve_with_clamps", "build_observable_op",
           "RunResult", "RunDiagnostic", "ObservableValue"]
```

- [ ] **Step 6: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_runtime_run.py -v`
Expected: all pass. If `energy_per_term sum` test fails, the BridgeHamiltonian aggregation may double-count; check that each term contributes once to `_energy_per_term`.

- [ ] **Step 7: Commit**

```bash
git add src/qft_pcn/bridge/runtime/__init__.py src/qft_pcn/bridge/runtime/result.py src/qft_pcn/bridge/runtime/observables.py src/qft_pcn/tests/test_bridge_runtime_run.py
git commit -m "$(cat <<'EOF'
feat(bridge/runtime): run_problem and diagnose_problem

Validate -> compile -> initial state -> imag-time evolve with clamps ->
measure observables -> return RunResult. diagnose_problem wraps and
attaches sub-project D's report when D is importable; None otherwise.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: stdio JSON-RPC server (`bridge/api.py`)

**Files:**
- Create: `src/qft_pcn/bridge/api.py`, `src/qft_pcn/bridge/__main__.py`.
- Create: `src/qft_pcn/tests/test_bridge_protocol.py`.

Synchronous JSON-RPC 2.0 over stdio (spec §6). Supports `problem.run`, `problem.diagnose`, `dsl.validate`, `health.check`.

- [ ] **Step 1: Write failing tests**

Create `src/qft_pcn/tests/test_bridge_protocol.py`:

```python
"""Tests for the JSON-RPC protocol layer (spec §6, §11.5)."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from src.qft_pcn.bridge.api import handle_request


CANONICAL = {
    "fields": [{"name": "x", "cutoff": 4}],
    "sites": 2,
    "constraints": [],
    "observables": [{"site": 0, "field": "x", "op": "n"}],
    "search": {"method": "imag_time", "steps": 5, "chi_max": 4},
}


def test_handle_health_check_in_proc():
    req = {"jsonrpc": "2.0", "id": 1, "method": "health.check", "params": {}}
    resp = handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert resp["result"]["ok"] is True


def test_handle_dsl_validate_in_proc():
    req = {"jsonrpc": "2.0", "id": 7, "method": "dsl.validate",
           "params": {"dsl": CANONICAL}}
    resp = handle_request(req)
    assert resp["id"] == 7
    assert resp["result"]["valid"] is True


def test_handle_problem_run_in_proc():
    req = {"jsonrpc": "2.0", "id": 2, "method": "problem.run",
           "params": {"dsl": CANONICAL}}
    resp = handle_request(req)
    assert resp["id"] == 2
    assert "result" in resp
    assert "observables" in resp["result"]


def test_handle_method_not_found_in_proc():
    req = {"jsonrpc": "2.0", "id": 3, "method": "nope", "params": {}}
    resp = handle_request(req)
    assert resp["id"] == 3
    assert resp["error"]["code"] == -32601


def test_handle_invalid_jsonrpc_version():
    req = {"jsonrpc": "1.0", "id": 4, "method": "health.check", "params": {}}
    resp = handle_request(req)
    assert resp["error"]["code"] == -32600


def test_handle_bridge_error_carries_typed_code():
    bad = {**CANONICAL, "sites": -3}
    req = {"jsonrpc": "2.0", "id": 5, "method": "problem.run",
           "params": {"dsl": bad}}
    resp = handle_request(req)
    assert resp["error"]["code"] == -32001
    assert resp["error"]["data"]["code"] == "BRIDGE_E_BAD_SCHEMA"


# --- subprocess tests ---


def _spawn_once(request_obj: dict) -> dict:
    """Run `python -m qft_pcn.bridge --once <json>` and parse response."""
    payload = json.dumps(request_obj)
    proc = subprocess.run(
        [sys.executable, "-m", "src.qft_pcn.bridge", "--once", payload],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_subprocess_health_check():
    resp = _spawn_once({"jsonrpc": "2.0", "id": 9, "method": "health.check",
                        "params": {}})
    assert resp["result"]["ok"] is True


def test_subprocess_problem_run():
    resp = _spawn_once({"jsonrpc": "2.0", "id": 10, "method": "problem.run",
                        "params": {"dsl": CANONICAL}})
    assert "result" in resp
    assert len(resp["result"]["observables"]) == 1
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_protocol.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/api.py`**

Create `src/qft_pcn/bridge/api.py`:

```python
"""Synchronous JSON-RPC 2.0 server over stdio for the QPCN bridge.

Methods (spec §6.1):
  problem.run       — run a DSL spec, return RunResult
  problem.diagnose  — run + D's diagnostic, return RunDiagnostic
  dsl.validate      — validate (no compile, no run), return {valid, errors}
  health.check      — return {ok, version, subprojects}

Errors map BridgeError -> JSON-RPC error.code = -32001 with the typed
BRIDGE_E_* code in error.data.code.

Stdout is the protocol channel; all logs go to stderr.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .dsl.pipeline import validate_dsl
from .errors import BridgeError, InternalError
from .runtime import run_problem, diagnose_problem


log = logging.getLogger("qft_pcn.bridge")


_VERSION = "0.1.0"


def _subproject_status() -> dict[str, str]:
    status: dict[str, str] = {}
    try:
        import src.qft_pcn.logic  # noqa: F401
        status["A"] = "ok"
    except Exception:               # noqa: BLE001
        status["A"] = "absent"
    for name in ("B", "C", "D", "E", "F"):
        status[name] = "stub"
    status["G"] = "ok"
    return status


def _ok(req_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str,
         data: dict[str, Any] | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _err_from_bridge(req_id: Any, exc: BridgeError) -> dict[str, Any]:
    return _err(req_id, -32001,
                f"{exc.code}: {exc.message}",
                data={"code": exc.code, "message": exc.message,
                      "details": exc.details})


def handle_request(req: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one JSON-RPC request and produce one response."""
    req_id = req.get("id")
    if req.get("jsonrpc") != "2.0":
        return _err(req_id, -32600, "jsonrpc field must be '2.0'")
    method = req.get("method")
    params = req.get("params") or {}
    if not isinstance(method, str):
        return _err(req_id, -32600, "method must be a string")
    if not isinstance(params, dict):
        return _err(req_id, -32602, "params must be an object")

    try:
        if method == "health.check":
            return _ok(req_id, {"ok": True, "version": _VERSION,
                                "subprojects": _subproject_status()})
        if method == "dsl.validate":
            if "dsl" not in params:
                return _err(req_id, -32602, "missing params.dsl")
            return _ok(req_id, validate_dsl(params["dsl"]))
        if method == "problem.run":
            if "dsl" not in params:
                return _err(req_id, -32602, "missing params.dsl")
            res = run_problem(params["dsl"])
            return _ok(req_id, res.to_dict())
        if method == "problem.diagnose":
            if "dsl" not in params:
                return _err(req_id, -32602, "missing params.dsl")
            res = diagnose_problem(params["dsl"])
            return _ok(req_id, res.to_dict())
        return _err(req_id, -32601, f"method not found: {method!r}")
    except BridgeError as e:
        log.warning("BridgeError on %s: %s", method, e)
        return _err_from_bridge(req_id, e)
    except Exception as e:          # noqa: BLE001
        log.exception("internal error on %s", method)
        return _err_from_bridge(req_id, InternalError(
            message=f"unexpected: {type(e).__name__}: {e}",
            details={"exc_type": type(e).__name__},
        ))


def serve_stdio(stream_in=sys.stdin, stream_out=sys.stdout) -> None:
    """Read newline-delimited JSON requests from `stream_in`; write
    newline-delimited JSON responses to `stream_out`."""
    for line in stream_in:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            resp = _err(None, -32700, f"JSON parse error: {exc.msg}")
        else:
            resp = handle_request(req)
        stream_out.write(json.dumps(resp) + "\n")
        stream_out.flush()


def serve_once(payload: str, stream_out=sys.stdout) -> None:
    """Handle exactly one request from a CLI string."""
    try:
        req = json.loads(payload)
    except json.JSONDecodeError as exc:
        resp = _err(None, -32700, f"JSON parse error: {exc.msg}")
    else:
        resp = handle_request(req)
    stream_out.write(json.dumps(resp) + "\n")
    stream_out.flush()
```

- [ ] **Step 4: Implement `bridge/__main__.py`**

Create `src/qft_pcn/bridge/__main__.py`:

```python
"""CLI entry point: `python -m qft_pcn.bridge`."""

from __future__ import annotations

import argparse
import logging
import sys

from .api import serve_stdio, serve_once


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="qft_pcn.bridge",
                                description="QPCN LLM bridge (stdio JSON-RPC)")
    p.add_argument("--once", metavar="JSON",
                   help="Process one request from this JSON string and exit.")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG logging on stderr.")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s qft_pcn.bridge %(message)s",
    )
    if args.once is not None:
        serve_once(args.once)
        return 0
    serve_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run protocol tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_protocol.py -v`
Expected: all pass (in-process and subprocess).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/api.py src/qft_pcn/bridge/__main__.py src/qft_pcn/tests/test_bridge_protocol.py
git commit -m "$(cat <<'EOF'
feat(bridge/api): stdio JSON-RPC 2.0 server

Four methods: problem.run, problem.diagnose, dsl.validate, health.check.
Stdout is the protocol channel, stderr is logs. --once flag handles a
single request from a CLI string for scripting. BridgeError subclasses
map to JSON-RPC -32001 with the typed BRIDGE_E_* code in error.data.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: LLM-side helpers (`templates.py` and `llm.py`)

**Files:**
- Create: `src/qft_pcn/bridge/templates.py`, `bridge/llm.py`.
- Create: `src/qft_pcn/tests/test_bridge_templates.py`.

`templates.py`: builder dataclasses that produce DSL dicts.
`llm.py`: `MockLLM` (tests + default demo) plus an `AnthropicLLM` shim that lazy-imports the `anthropic` SDK.

- [ ] **Step 1: Write failing tests**

Create `src/qft_pcn/tests/test_bridge_templates.py`:

```python
"""Tests for bridge/templates.py and bridge/llm.py (spec §7, §11.6)."""

from __future__ import annotations

import pytest

from src.qft_pcn.bridge.templates import (
    Problem, FieldSpec, ConstraintSpec, ObservableSpec, SearchSpec,
)
from src.qft_pcn.bridge.dsl.pipeline import validate_dsl
from src.qft_pcn.bridge.llm import MockLLM


def test_problem_builder_to_dsl_validates():
    p = Problem(
        fields=[FieldSpec(name="kind", cutoff=8)],
        sites=4,
        constraints=[
            ConstraintSpec(kind="local", site=0,
                            term="kind == LAM", weight=1.0)
        ],
        boundary={0: {"kind": "KIND_LAM"}},
        observables=[ObservableSpec(site=0, field="kind", op="n")],
    )
    out = validate_dsl(p.to_dsl())
    assert out["valid"] is True


def test_problem_builder_defaults_search():
    p = Problem(
        fields=[FieldSpec(name="x", cutoff=4)],
        sites=1,
        observables=[ObservableSpec(site=0, field="x", op="n")],
    )
    dsl = p.to_dsl()
    assert dsl["search"]["method"] == "imag_time"


def test_mock_llm_emits_canned_dsl():
    mock = MockLLM(responses={
        "double-int": {
            "fields": [{"name": "x", "cutoff": 4}],
            "sites": 1, "constraints": [],
            "observables": [{"site": 0, "field": "x", "op": "n"}],
        }
    })
    dsl = mock.emit_dsl("double-int")
    assert dsl["sites"] == 1


def test_mock_llm_raises_for_unknown_prompt():
    mock = MockLLM(responses={})
    with pytest.raises(KeyError):
        mock.emit_dsl("nope")


def test_mock_llm_verbalize_uses_template():
    mock = MockLLM(responses={})
    result = {"observables": [{"site": 0, "field": "x", "op": "n",
                                "value": 2.5, "imag_part": 1e-13}],
              "converged": True}
    text = mock.verbalize("anything", result)
    assert "2.5" in text or "2.50" in text
    assert "converged" in text.lower() or "converg" in text.lower()


def test_anthropic_llm_import_is_lazy():
    """Importing bridge.llm must succeed without the anthropic SDK installed."""
    import importlib, sys
    # Ensure bridge.llm imports cleanly even if anthropic is monkey-removed.
    if "anthropic" in sys.modules:
        del sys.modules["anthropic"]
    importlib.import_module("src.qft_pcn.bridge.llm")
    # Constructing AnthropicLLM is what actually requires the SDK; merely
    # importing the module must not.
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_templates.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `bridge/templates.py`**

Create `src/qft_pcn/bridge/templates.py`:

```python
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
    site: int | None = None         # required for kind == "local"
    sites: tuple[int, int] | None = None  # required for kind == "two_site"


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
    """Canonical builder for §10.7 STLC synthesis problems. Holes parameter
    is a placeholder until sub-project E lands; for now it is ignored except
    to inform the observable list when caller doesn't supply one."""
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
```

- [ ] **Step 4: Implement `bridge/llm.py`**

Create `src/qft_pcn/bridge/llm.py`:

```python
"""LLM shims. MockLLM is used by tests and the default demo.
AnthropicLLM is the optional real-LLM adapter — lazy-imports the SDK
so importing this module never requires `anthropic` to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class MockLLM:
    """Deterministic stub LLM: emits canned DSL specs and a fixed-template
    verbalization. Used by every test that involves an LLM-shaped path."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None):
        self.responses = responses or {}

    def emit_dsl(self, prompt: str) -> dict[str, Any]:
        if prompt not in self.responses:
            raise KeyError(f"MockLLM has no canned response for prompt {prompt!r}")
        return self.responses[prompt]

    def verbalize(self, prompt: str, result: dict[str, Any]) -> str:
        lines = [f"Result for: {prompt}"]
        for o in result.get("observables", []):
            lines.append(f"  - {o['field']}@{o['site']} ({o['op']}) = "
                         f"{o['value']:.4f}")
        if "converged" in result:
            lines.append("converged" if result["converged"]
                         else "(did not converge)")
        return "\n".join(lines)


class AnthropicLLM:
    """Real Anthropic-API-backed LLM. The SDK is imported lazily inside
    __init__; importing this module never requires the SDK to be installed.

    Use the claude-api skill patterns when implementing prompt caching and
    model selection. This class is not used by any test.
    """

    def __init__(self, *, model: str = "claude-opus-4-7-1m",
                 schema_text: str | None = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "anthropic SDK not installed. `pip install anthropic`."
            ) from exc
        self.model = model
        self._schema_text = schema_text
        # Lazy: actual client construction deferred to first call.
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def emit_dsl(self, prompt: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError(
            "AnthropicLLM.emit_dsl: implement using the claude-api skill's "
            "prompt-caching pattern. Out of scope for sub-project G tests."
        )

    def verbalize(self, prompt: str, result: dict[str, Any]) -> str:  # pragma: no cover
        raise NotImplementedError(
            "AnthropicLLM.verbalize: implement using the claude-api skill."
        )
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_templates.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/templates.py src/qft_pcn/bridge/llm.py src/qft_pcn/tests/test_bridge_templates.py
git commit -m "$(cat <<'EOF'
feat(bridge/templates,llm): Problem builder and Mock/Anthropic LLM shims

Problem.to_dsl produces validated JSON-compatible dicts from typed
templates. MockLLM serves the test path; AnthropicLLM lazy-imports
the anthropic SDK so bridge.llm imports cleanly without it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Bridge package public surface and import-graph test

**Files:**
- Modify: `src/qft_pcn/bridge/__init__.py`.
- Modify: `src/qft_pcn/__init__.py` (add bridge re-exports).
- Create: `src/qft_pcn/tests/test_bridge_dependency_graph.py`.

Enforce spec §1.3: nothing under `logic/` or `qft/` may import `bridge`. Audit by AST-walking each `.py` file.

- [ ] **Step 1: Write failing tests**

Create `src/qft_pcn/tests/test_bridge_dependency_graph.py`:

```python
"""Enforce the §1.3 dependency rule: bridge depends on logic/qft, not the
other way around. Also verify the public surface exists and that importing
bridge does not transitively load `anthropic`."""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
QFT_PCN = REPO_ROOT / "src" / "qft_pcn"


def _python_files_under(d: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in d.rglob("*.py") if "__pycache__" not in str(p)]


def _imports_in(path: pathlib.Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                out.append(alias.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                out.append(n.module)
    return out


def test_logic_does_not_import_bridge():
    bad: list[str] = []
    for p in _python_files_under(QFT_PCN / "logic"):
        for imp in _imports_in(p):
            if "bridge" in imp.split("."):
                bad.append(f"{p}: {imp}")
    assert not bad, "logic/ files importing bridge:\n" + "\n".join(bad)


def test_qft_does_not_import_bridge():
    bad: list[str] = []
    for p in _python_files_under(QFT_PCN / "qft"):
        for imp in _imports_in(p):
            if "bridge" in imp.split("."):
                bad.append(f"{p}: {imp}")
    assert not bad, "qft/ files importing bridge:\n" + "\n".join(bad)


def test_bridge_does_not_import_anthropic_eagerly():
    # Drop anthropic from sys.modules to detect eager import.
    sys.modules.pop("anthropic", None)
    # Drop bridge submodules to force fresh import.
    for k in list(sys.modules):
        if k.startswith("src.qft_pcn.bridge") or k == "src.qft_pcn.bridge":
            sys.modules.pop(k, None)
    importlib.import_module("src.qft_pcn.bridge")
    assert "anthropic" not in sys.modules, \
        "importing qft_pcn.bridge eagerly loaded anthropic"


def test_public_bridge_surface_present():
    import src.qft_pcn.bridge as br
    for name in ("run_problem", "diagnose_problem", "validate_dsl",
                 "Problem", "MockLLM", "BridgeError"):
        assert hasattr(br, name), f"bridge.{name} missing"
```

- [ ] **Step 2: Update `bridge/__init__.py`**

Replace `src/qft_pcn/bridge/__init__.py`:

```python
"""QPCN LLM bridge: stdio JSON-RPC + DSL runtime (spec §10.5).

Public surface:
  run_problem, diagnose_problem  — synchronous runtime API
  validate_dsl                   — validate without running
  Problem                        — DSL builder
  MockLLM, AnthropicLLM          — LLM shims
  BridgeError + subclasses       — typed error hierarchy
"""

from .errors import (
    BridgeError, BadJsonError, BadSchemaError, BadReferenceError,
    BadTermError, TermUnsupportedError, UnsupportedMethodError,
    SitesOutOfRangeError, TooLargeError, ConstraintNotAdjacentError,
    NumericFailureError, InternalError, ALL_CODES,
)
from .dsl.pipeline import validate_dsl, compile_dsl, CompiledDsl
from .runtime import (
    run_problem, diagnose_problem,
    RunResult, RunDiagnostic, ObservableValue,
)
from .templates import (
    Problem, FieldSpec, ConstraintSpec, ObservableSpec, SearchSpec,
    stlc_synthesis,
)
from .llm import MockLLM, AnthropicLLM


__all__ = [
    "run_problem", "diagnose_problem", "validate_dsl", "compile_dsl",
    "RunResult", "RunDiagnostic", "ObservableValue", "CompiledDsl",
    "Problem", "FieldSpec", "ConstraintSpec", "ObservableSpec",
    "SearchSpec", "stlc_synthesis",
    "MockLLM", "AnthropicLLM",
    "BridgeError", "BadJsonError", "BadSchemaError", "BadReferenceError",
    "BadTermError", "TermUnsupportedError", "UnsupportedMethodError",
    "SitesOutOfRangeError", "TooLargeError", "ConstraintNotAdjacentError",
    "NumericFailureError", "InternalError", "ALL_CODES",
]
```

- [ ] **Step 3: Update `src/qft_pcn/__init__.py`**

Read the current `src/qft_pcn/__init__.py`, then append at the bottom:

```python
# Sub-project G public surface (LLM bridge).
from .bridge import (
    run_problem, diagnose_problem, validate_dsl, Problem,
    MockLLM, BridgeError,
)
```

If the existing `__init__.py` has an `__all__`, extend it with the new names. Otherwise leave `__all__` untouched.

- [ ] **Step 4: Run dependency-graph tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_dependency_graph.py -v`
Expected: all pass.

- [ ] **Step 5: Run the complete bridge test surface**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_*.py -v 2>&1 | tail -30`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/__init__.py src/qft_pcn/__init__.py src/qft_pcn/tests/test_bridge_dependency_graph.py
git commit -m "$(cat <<'EOF'
feat(bridge): public surface and §1.3 import-graph enforcement

Re-exports run_problem, diagnose_problem, validate_dsl, Problem,
MockLLM, BridgeError at qft_pcn.bridge and qft_pcn top levels.
Adds AST-walking tests that fail loudly if logic/ or qft/ ever
imports bridge, and verifies importing bridge does not eagerly
load the anthropic SDK.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: End-to-end mock-LLM demo

**Files:**
- Create: `src/qft_pcn/bridge/demo.py`, `bridge/demo_protocol.py`.
- Create: `src/qft_pcn/tests/test_bridge_demo.py`.

A standalone script: `MockLLM` emits a canned DSL for a hardcoded prompt → runtime runs it → `MockLLM.verbalize` renders the answer. Plus a subprocess variant that runs through JSON-RPC.

- [ ] **Step 1: Write failing tests**

Create `src/qft_pcn/tests/test_bridge_demo.py`:

```python
"""End-to-end demo tests (spec §8, §11.8)."""

from __future__ import annotations

import json
import subprocess
import sys


def _run_module(module: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-m", module],
        capture_output=True, text=True, timeout=60,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_demo_mock_path_runs_to_completion():
    rc, out, err = _run_module("src.qft_pcn.bridge.demo")
    assert rc == 0, f"stderr: {err}"
    # Five-section transcript: prompt, DSL, observables, energy, verbalization.
    for label in ("PROMPT", "DSL", "OBSERVABLES", "ENERGY", "VERBALIZATION"):
        assert label in out, f"missing {label} section:\n{out}"


def test_demo_protocol_path_runs_to_completion():
    rc, out, err = _run_module("src.qft_pcn.bridge.demo_protocol")
    assert rc == 0, f"stderr: {err}"
    for label in ("PROMPT", "DSL", "OBSERVABLES", "ENERGY", "VERBALIZATION"):
        assert label in out, f"missing {label} section:\n{out}"
    # The protocol path must show JSON-RPC envelopes on the transcript.
    assert "jsonrpc" in out
```

- [ ] **Step 2: Verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_demo.py -v`
Expected: module not found.

- [ ] **Step 3: Implement `bridge/demo.py`**

Create `src/qft_pcn/bridge/demo.py`:

```python
"""End-to-end demo: mock LLM emits DSL -> runtime runs -> verbalize.

Run via `python -m src.qft_pcn.bridge.demo`.

A real-LLM path is available with --llm anthropic, but is not exercised
by tests.
"""

from __future__ import annotations

import argparse
import json
import sys

from .llm import MockLLM, AnthropicLLM
from .runtime import run_problem


CANNED_PROMPT = "find a configuration where site 0 holds occupation 2 on field x"

CANNED_DSL = {
    "fields": [{"name": "x", "cutoff": 4}],
    "sites": 2,
    "constraints": [],
    "boundary": {"0": {"x": 2}},
    "observables": [{"site": 0, "field": "x", "op": "n"}],
    "search": {"method": "imag_time", "steps": 20, "chi_max": 4},
}


def render_transcript(prompt: str, dsl: dict, result: dict,
                      verbalization: str, *, out=sys.stdout) -> None:
    out.write("=" * 60 + "\n")
    out.write("PROMPT\n")
    out.write("=" * 60 + "\n")
    out.write(prompt + "\n\n")

    out.write("=" * 60 + "\n")
    out.write("DSL\n")
    out.write("=" * 60 + "\n")
    out.write(json.dumps(dsl, indent=2) + "\n\n")

    out.write("=" * 60 + "\n")
    out.write("OBSERVABLES\n")
    out.write("=" * 60 + "\n")
    for o in result["observables"]:
        out.write(f"  site={o['site']:>3} field={o['field']:<8} "
                  f"op={o['op']:<8} value={o['value']:.6f} "
                  f"(imag={o['imag_part']:.2e})\n")
    out.write("\n")

    out.write("=" * 60 + "\n")
    out.write("ENERGY\n")
    out.write("=" * 60 + "\n")
    out.write(f"  total = {result['energy']:.6f}\n")
    out.write(f"  per-term = {result['energy_per_term']}\n")
    out.write(f"  converged = {result['converged']}\n\n")

    out.write("=" * 60 + "\n")
    out.write("VERBALIZATION\n")
    out.write("=" * 60 + "\n")
    out.write(verbalization + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--llm", choices=["mock", "anthropic"], default="mock")
    args = p.parse_args(argv)

    if args.llm == "mock":
        llm = MockLLM(responses={CANNED_PROMPT: CANNED_DSL})
    else:                                              # pragma: no cover
        llm = AnthropicLLM()

    prompt = CANNED_PROMPT
    dsl = llm.emit_dsl(prompt)
    result = run_problem(dsl).to_dict()
    verbalization = llm.verbalize(prompt, result)
    render_transcript(prompt, dsl, result, verbalization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement `bridge/demo_protocol.py`**

Create `src/qft_pcn/bridge/demo_protocol.py`:

```python
"""End-to-end demo via subprocess JSON-RPC.

Spawns `python -m src.qft_pcn.bridge --once <request>` and feeds the
response to MockLLM.verbalize.
"""

from __future__ import annotations

import json
import subprocess
import sys

from .llm import MockLLM
from .demo import CANNED_PROMPT, CANNED_DSL, render_transcript


def main() -> int:
    llm = MockLLM(responses={CANNED_PROMPT: CANNED_DSL})
    dsl = llm.emit_dsl(CANNED_PROMPT)

    request = {"jsonrpc": "2.0", "id": 1, "method": "problem.run",
               "params": {"dsl": dsl}}
    proc = subprocess.run(
        [sys.executable, "-m", "src.qft_pcn.bridge", "--once",
         json.dumps(request)],
        capture_output=True, text=True, timeout=60, check=True,
    )
    # Print envelope first so the test can detect "jsonrpc" in the output.
    sys.stdout.write("JSON-RPC REQUEST:  " + json.dumps(request) + "\n")
    sys.stdout.write("JSON-RPC RESPONSE: " + proc.stdout.strip() + "\n\n")
    response = json.loads(proc.stdout.strip().splitlines()[-1])
    result = response["result"]
    verbalization = llm.verbalize(CANNED_PROMPT, result)
    render_transcript(CANNED_PROMPT, dsl, result, verbalization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the demo tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_bridge_demo.py -v`
Expected: both pass.

Optional sanity check: run the demos directly and visually inspect:
```
.venv/bin/python -m src.qft_pcn.bridge.demo
.venv/bin/python -m src.qft_pcn.bridge.demo_protocol
```

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/bridge/demo.py src/qft_pcn/bridge/demo_protocol.py src/qft_pcn/tests/test_bridge_demo.py
git commit -m "$(cat <<'EOF'
feat(bridge/demo): end-to-end mock-LLM demos (in-proc and subprocess)

demo.py: MockLLM emits canned DSL, runtime solves, MockLLM verbalizes,
transcript is rendered in five labeled sections. demo_protocol.py does
the same but routes the DSL through `python -m qft_pcn.bridge --once`
to exercise the full JSON-RPC transport. Both are deterministic and
require no external API access.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Final acceptance sweep and verification

**Goal:** Run the full project test suite. Confirm spec §12's acceptance criteria. Iterate on any regression.

- [ ] **Step 1: Run the full test suite (excluding qiskit / non-installed deps)**

Run:
```
.venv/bin/python -m pytest src/qft_pcn/tests/ \
    --ignore=src/qft_pcn/tests/test_quantum.py -v 2>&1 | tail -40
```
Expected: every test passes, including:
- existing qft/multifield/qft_pcn/logic/* tests
- 8 new bridge test files

- [ ] **Step 2: Verify each acceptance criterion from spec §12**

For each, run and confirm:

1. `pytest src/qft_pcn/tests/test_bridge_*.py -v` — all green.
2. `python -m src.qft_pcn.bridge --once '{"jsonrpc":"2.0","id":1,"method":"health.check","params":{}}'` produces a JSON response with `"ok": true`. Run:
   ```
   .venv/bin/python -m src.qft_pcn.bridge --once '{"jsonrpc":"2.0","id":1,"method":"health.check","params":{}}'
   ```
3. `python -m src.qft_pcn.bridge.demo` ends in a non-empty verbalization line. Run and inspect.
4. `python -m src.qft_pcn.bridge.demo_protocol` produces the same transcript shape. Run and inspect.
5. Import-graph test passes (run from Task 14).
6. `ALL_CODES` snapshot (in `test_bridge_errors.py`) passes — verify with `pytest -k test_all_codes_set_is_stable -v`.
7. Existing tests still pass (covered by step 1).
8. `pyproject.toml` has exactly one new required dep (`jsonschema`); verify via `grep jsonschema pyproject.toml` and `grep anthropic pyproject.toml` (the latter should not appear in required deps; if it appears at all, it must be in an optional group).
9. Top-level re-exports verified by `test_public_bridge_surface_present` in Task 14.

- [ ] **Step 3: Inspect coverage gaps**

Run a fast count of bridge files vs bridge tests:
```
ls src/qft_pcn/bridge/**/*.py | wc -l
ls src/qft_pcn/tests/test_bridge_*.py | wc -l
```
Expect roughly: ~17 bridge source files, 9-10 bridge test files. Any source file completely uncovered should be inspected; document under "open questions" if intentional.

- [ ] **Step 4: Confirm no `eval`/`exec`/`compile` usage anywhere in the bridge**

Run:
```
grep -rn "\beval\b\|\bexec\b\|\bcompile\b\|literal_eval\|pickle.loads" src/qft_pcn/bridge/
```
Expected: zero results. Any hit is a §1.8 violation and must be fixed.

- [ ] **Step 5: Confirm no anthropic / openai import outside `bridge/llm.py`**

Run:
```
grep -rln "import anthropic\|from anthropic\|import openai\|from openai" src/qft_pcn/
```
Expected: only `src/qft_pcn/bridge/llm.py` (lazy-imported, gated by try/except inside `__init__`).

- [ ] **Step 6: Final commit (only if there are uncommitted polish changes)**

If any small fixes accumulated during step 1-5:
```bash
git add -p   # review what's about to commit
git commit -m "$(cat <<'EOF'
chore(bridge): final acceptance-sweep fixes

Final pass against spec §12 criteria: no eval/exec/compile in bridge,
no eager anthropic/openai import, full test suite green, demos produce
five-section transcripts.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

If clean, skip the commit.

- [ ] **Step 7: Report**

Print a final status summary:

```
=== SUB-PROJECT G STATUS ===
Spec: docs/superpowers/specs/2026-05-21-llm-bridge-design.md
Plan: docs/superpowers/plans/2026-05-21-llm-bridge-plan.md

Files created under src/qft_pcn/bridge/:
  - errors.py, __init__.py, __main__.py, api.py
  - dsl/{schema.py, cross_validate.py, expr_ast.py, expr_parser.py,
         vocab.py, term.py, compiler_stub.py, compiler.py,
         pipeline.py, __init__.py}
  - runtime/{__init__.py, clamp.py, initial_state.py, hamiltonian.py,
             evolution.py, observables.py, result.py}
  - templates.py, llm.py, demo.py, demo_protocol.py

Tests:
  - test_bridge_errors.py
  - test_bridge_schema.py
  - test_bridge_expr_parser.py
  - test_bridge_vocab.py
  - test_bridge_compiler.py
  - test_bridge_dsl_pipeline.py
  - test_bridge_runtime_clamp.py
  - test_bridge_runtime_evolution.py
  - test_bridge_runtime_run.py
  - test_bridge_protocol.py
  - test_bridge_templates.py
  - test_bridge_dependency_graph.py
  - test_bridge_demo.py

End-to-end demos:
  python -m src.qft_pcn.bridge.demo               # in-proc MockLLM
  python -m src.qft_pcn.bridge.demo_protocol      # subprocess JSON-RPC
  python -m src.qft_pcn.bridge                    # stdio JSON-RPC server

Sub-project G complete. The QPCN system is end-to-end functional with
a Mock LLM; the AnthropicLLM shim is in place for real-LLM demos.
```

---

**End of plan.** When sub-project B or C lands, swap `bridge/dsl/compiler_stub.py`'s factories for B/C's named entry points in `bridge/dsl/compiler.py`. When sub-project D lands, `bridge/runtime/__init__.py::diagnose_problem` will start returning real debugger reports automatically (it already attempts the import).
