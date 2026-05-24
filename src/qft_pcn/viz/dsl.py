"""QPCN DSL: schema, validator, and DSL<->RunSpec translator.

The DSL is the JSON the LLM emits (architecture §1.3). It declares fields,
Hamiltonian terms, and target observables. This module:

  * loads `dsl-schema.json` (JSON Schema Draft 2020-12),
  * validates a DSL dict against it,
  * translates a validated DSL to the existing `RunSpec` shape so the
    standard `run_simulation` driver can execute it,
  * provides a best-effort reverse (`runspec_to_dsl`) for an Export button.

`EXAMPLES` are three canonical DSL dicts used by the LLM system prompt
few-shot AND asserted as schema-valid in the test suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from .runs import RunSpec

_SCHEMA_PATH = Path(__file__).with_name("dsl-schema.json")


def load_schema() -> dict:
    """Read and return the DSL schema dict."""
    with _SCHEMA_PATH.open() as f:
        return json.load(f)


_VALIDATOR = jsonschema.Draft202012Validator(load_schema())


def validate(dsl: dict) -> list[str]:
    """Return a list of validation error messages (empty = valid)."""
    return [str(e.message) for e in _VALIDATOR.iter_errors(dsl)]


def dsl_to_runspec(dsl: dict) -> RunSpec:
    """Translate a validated DSL to the existing RunSpec shape.

    Field names with their cutoffs become the qpcn species list. Hamiltonian
    `kind == "mass"` terms set the species mass (first match wins);
    `kind == "kinetic"` sets the kinetic coefficient. `observables` with a
    non-null `target` map to qpcn.target_n0 (for `operator == "n"` on
    `site == 0`).
    """
    errs = validate(dsl)
    if errs:
        raise ValueError(f"DSL validation failed: {errs}")

    species_names = [f["name"] for f in dsl["fields"]]
    qpcn_params: dict[str, Any] = {"species": species_names}

    for term in dsl["hamiltonian"].get("terms", []):
        if term["kind"] == "mass" and "coefficient" in term:
            qpcn_params.setdefault("mass", float(term["coefficient"]))
        elif term["kind"] == "kinetic" and "coefficient" in term:
            qpcn_params.setdefault("kinetic", float(term["coefficient"]))

    for obs in dsl.get("observables", []):
        if (obs.get("operator") == "n" and obs.get("site") == 0
                and obs.get("target") is not None):
            qpcn_params.setdefault("target_n0", float(obs["target"]))

    run_cfg = dsl.get("run", {}) or {}
    # Stash the original DSL verbatim on the spec so `runspec_to_dsl` can
    # round-trip losslessly. We deep-copy via json round-trip so later mutation
    # of either side cannot leak into the other.
    dsl_copy = json.loads(json.dumps(dsl))
    return RunSpec(
        layers=["qpcn", "mps", "hamiltonian"],
        steps=int(run_cfg.get("steps", 30)),
        grid=12,
        seed=run_cfg.get("seed"),
        params={"qpcn": qpcn_params},
        dsl=dsl_copy,
    )


def runspec_to_dsl(spec: RunSpec) -> dict:
    """Reverse-translate a RunSpec back to a DSL dict.

    If the spec carries its original DSL on `spec.dsl` (set by
    `dsl_to_runspec`), return that verbatim (deep-copied) for a lossless
    round-trip. Otherwise fall back to a best-effort reverse from the flat
    qpcn params -- which is intentionally lossy, since `RunSpec.params`
    doesn't preserve every DSL feature.
    """
    if spec.dsl is not None:
        return json.loads(json.dumps(spec.dsl))
    qp = (spec.params or {}).get("qpcn") or {}
    names = list(qp.get("species") or ["A"])
    dsl: dict[str, Any] = {
        "fields": [{"name": n, "cutoff": 2} for n in names],
        "hamiltonian": {"terms": []},
        "observables": [],
        "run": {"steps": spec.steps, "seed": spec.seed},
    }
    for name in names:
        if "mass" in qp:
            dsl["hamiltonian"]["terms"].append({
                "kind": "mass", "species": name,
                "coefficient": float(qp["mass"]),
            })
        if "kinetic" in qp:
            dsl["hamiltonian"]["terms"].append({
                "kind": "kinetic", "species": name,
                "coefficient": float(qp["kinetic"]),
            })
    if qp.get("target_n0") is not None:
        dsl["observables"].append({
            "operator": "n", "site": 0,
            "species": names[0], "target": float(qp["target_n0"]),
        })
    return dsl


EXAMPLES: list[dict] = [
    {
        "fields": [{"name": "A", "cutoff": 2,
                    "bare_mass": 1.0, "kinetic": 0.5}],
        "hamiltonian": {"terms": [
            {"kind": "mass",    "species": "A", "coefficient": 1.0},
            {"kind": "kinetic", "species": "A", "coefficient": 0.5},
        ]},
        "observables": [
            {"operator": "n", "site": 0, "species": "A", "target": 0.25},
        ],
        "run": {"steps": 20, "seed": 0},
    },
    {
        "fields": [{"name": "A", "cutoff": 2},
                   {"name": "B", "cutoff": 2}],
        "hamiltonian": {"terms": [
            {"kind": "mass",    "species": "A", "coefficient": 1.0},
            {"kind": "mass",    "species": "B", "coefficient": 1.5},
            {"kind": "kinetic", "species": "A", "coefficient": 0.5},
            {"kind": "kinetic", "species": "B", "coefficient": 0.5},
            {"kind": "yukawa",  "species": "A", "coefficient": 0.2},
        ]},
        "observables": [
            {"operator": "n", "site": 0, "species": "A", "target": 0.20},
            {"operator": "n", "site": 0, "species": "B", "target": 0.30},
        ],
        "run": {"steps": 30, "seed": 1},
    },
    {
        "fields": [{"name": "A", "cutoff": 2}],
        "hamiltonian": {"terms": [
            {"kind": "mass",    "species": "A", "coefficient": 3.0},
            {"kind": "kinetic", "species": "A", "coefficient": 0.5},
            {"kind": "quartic", "species": "A", "coefficient": 0.05},
        ]},
        "observables": [
            {"operator": "n", "site": 0, "species": "A", "target": 0.10},
        ],
        "run": {"steps": 40, "seed": 2},
    },
]
