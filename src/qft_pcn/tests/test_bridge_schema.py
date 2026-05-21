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
    validate_schema(MINIMAL_DSL)


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
    out = parse_and_validate("""{"fields":[{"name":"x","cutoff":4}],"sites":2,
                                  "constraints":[],
                                  "observables":[{"site":0,"field":"x","op":"n"}]}""")
    assert out["search"]["method"] == "imag_time"
    assert out["search"]["steps"] == 50
    assert out["search"]["chi_max"] == 32


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
