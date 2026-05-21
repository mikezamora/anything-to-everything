"""Tests for the full DSL validate+compile pipeline."""

from __future__ import annotations

import pytest

from src.qft_pcn.bridge.dsl.pipeline import validate_dsl, compile_dsl, CompiledDsl
from src.qft_pcn.bridge.errors import TermUnsupportedError


CANONICAL_DSL = {
    "fields": [
        {"name": "kind",  "cutoff": 4},
        {"name": "type",  "cutoff": 4},
    ],
    "sites": 8,
    "constraints": [
        {"kind": "local", "site": 0, "term": "kind == 1", "weight": 1.0},
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
    bad = {**CANONICAL_DSL, "sites": -1}
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
    assert len(cd.terms) == 2
    assert len(cd.observables) == 1
    assert cd.boundary[0]["kind"] == 2  # KIND_LAM resolves to 2


def test_compile_dsl_propagates_term_unsupported():
    bad = {**CANONICAL_DSL,
           "constraints": [{"kind": "local", "site": 0,
                            "term": "kind(arg1) * value(arg1)",
                            "weight": 1.0}]}
    with pytest.raises(TermUnsupportedError):
        compile_dsl(bad)
