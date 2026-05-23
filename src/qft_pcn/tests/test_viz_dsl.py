"""DSL schema, validator, and DSL<->RunSpec translator tests."""

import json
from pathlib import Path

import pytest

from src.qft_pcn.viz.dsl import (
    load_schema, validate, dsl_to_runspec, runspec_to_dsl, EXAMPLES,
)
from src.qft_pcn.viz.runs import RunSpec, run_simulation


def test_schema_loads_with_expected_id():
    schema = load_schema()
    assert schema["$id"].endswith("dsl/v1.json")
    assert "fields" in schema["properties"]


def test_validator_accepts_minimal_dsl():
    dsl = {
        "fields": [{"name": "A", "cutoff": 2}],
        "hamiltonian": {"terms": [
            {"kind": "mass", "species": "A", "coefficient": 1.0},
            {"kind": "kinetic", "species": "A", "coefficient": 0.5},
        ]},
        "observables": [
            {"operator": "n", "site": 0, "species": "A", "target": 0.25},
        ],
    }
    assert validate(dsl) == []


def test_validator_rejects_missing_required():
    bad = {"fields": [], "hamiltonian": {}}
    errors = validate(bad)
    assert any("observables" in e for e in errors)


def test_dsl_to_runspec_round_trips_through_simulation():
    dsl = EXAMPLES[0]
    spec = dsl_to_runspec(dsl)
    frames = list(run_simulation(spec))
    assert frames, "DSL-produced RunSpec yielded no frames"
    # The DSL declares a species, so qpcn should be present in the run.
    assert "qpcn" in frames[0].layer_states


def test_runspec_to_dsl_preserves_field_names():
    dsl = EXAMPLES[0]
    spec = dsl_to_runspec(dsl)
    back = runspec_to_dsl(spec)
    assert {f["name"] for f in back["fields"]} == {
        f["name"] for f in dsl["fields"]}


def test_examples_are_all_valid():
    for ex in EXAMPLES:
        assert validate(ex) == [], f"EXAMPLES contains invalid DSL: {ex}"
