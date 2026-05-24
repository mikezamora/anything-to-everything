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


def test_runspec_to_dsl_round_trips_examples_losslessly():
    # With the RunSpec.dsl companion, every EXAMPLE survives the
    # dsl -> RunSpec -> dsl trip byte-for-byte (no flat-params lossy reverse).
    for ex in EXAMPLES:
        spec = dsl_to_runspec(ex)
        back = runspec_to_dsl(spec)
        assert back == ex, f"round-trip lost data for: {ex}"


def test_runspec_to_dsl_falls_back_when_no_dsl_companion():
    # A hand-built RunSpec (no `dsl` companion) still gets a best-effort
    # reverse from the flat qpcn params; the result is valid DSL.
    spec = RunSpec(layers=["qpcn"], steps=20, grid=12, seed=0,
                   params={"qpcn": {"species": ["A"], "mass": 1.0,
                                    "kinetic": 0.5, "target_n0": 0.25}})
    back = runspec_to_dsl(spec)
    assert validate(back) == []
    assert {f["name"] for f in back["fields"]} == {"A"}


def test_lossless_roundtrip_preserves_unknown_keys():
    # Future/unknown DSL keys (top-level + nested) must survive the
    # dsl -> RunSpec -> dsl trip verbatim, since the companion stash is
    # opaque w.r.t. flat-params reconstruction.
    dsl = {
        "fields": [{"name": "A", "cutoff": 2}],
        "hamiltonian": {"terms": [
            {"kind": "mass", "species": "A", "coefficient": 1.0},
        ]},
        "observables": [
            {"operator": "n", "site": 0, "species": "A", "target": 0.25},
        ],
        "run": {"steps": 20, "seed": 0},
        # Hypothetical future top-level key that today's translator drops:
        "future_section": {"foo": [1, 2, 3], "bar": "baz"},
    }
    assert validate(dsl) == [], "fixture must validate under current schema"
    spec = dsl_to_runspec(dsl)
    back = runspec_to_dsl(spec)
    assert back == dsl
    assert "future_section" in back
    assert back["future_section"] == {"foo": [1, 2, 3], "bar": "baz"}


def test_fallback_reconstruction_when_dsl_absent():
    # Explicit no-companion path: drop `spec.dsl` to None and assert the
    # reverse falls back to flat-params reconstruction (not the stash).
    spec = dsl_to_runspec(EXAMPLES[0])
    spec.dsl = None  # force the fallback branch
    back = runspec_to_dsl(spec)
    assert validate(back) == []
    # Reconstruction recovers species + run metadata even without the stash.
    assert {f["name"] for f in back["fields"]} == {"A"}
    assert back["run"]["steps"] == spec.steps
