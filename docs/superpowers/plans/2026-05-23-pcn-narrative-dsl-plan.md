# PCN Panels + Narrative Arc + DSL Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the QPCN visualizer's PCN-side fidelity up to the QFT side (3 new layer panels), add a narrative arc grouping layers into QFT / PCN / QPCN sections with intro panels, and expose a DSL editor + Ollama-driven LLM pipeline so users can run the full LLM → DSL → QPCN → DSL → LLM round trip.

**Architecture:** Five workstreams (A–E from spec §8) that can be implemented mostly in parallel. Backend additions read existing `QFTPCNNetwork` / `MultiFieldNetwork` substrate non-mutatingly; frontend additions sit on the existing 4-region layout via a new `RouteSwitcher`. The DSL pipeline goes through new `/dsl/*` FastAPI endpoints backed by a thin Ollama wrapper.

**Tech Stack:** Python 3.11+, FastAPI, asyncio, httpx, jsonschema; React 18 + Vite + TypeScript, Zustand, Monaco (`@monaco-editor/react`), Ollama (local HTTP API).

**Spec:** `docs/superpowers/specs/2026-05-23-pcn-narrative-dsl-design.md`

**Hard constraints:**
- Touch only `src/qft_pcn/viz/**` and `src/qft_pcn/tests/test_viz_*.py` (plus `pyproject.toml` for new deps and `web/package.json` / `web/vite.config.ts`). Never edit substrate modules.
- No placeholders / TODOs / fake data. Missing-substrate features go in `src/qft_pcn/viz/EXTENSIONS.md`.
- All snapshot reads must use the existing `_safe(lambda: …)` pattern.
- Python tests run with `--noconftest` (the repo `conftest.py` imports the Unix-only `resource` module). Tests live in `src/qft_pcn/tests/test_viz_*.py` and use `from src.qft_pcn...` imports (repo convention).
- Frontend tests run via `./node_modules/.bin/vitest --run <pattern>` (pnpm is sometimes blocked by a release-age policy; vitest direct works). Add `import '@testing-library/jest-dom/vitest';` at top of each new test file (project's `test-setup.ts` does not auto-register).
- **Commit hygiene:** Always `git add <explicit paths>`. Run `git diff --staged --name-only` before every commit. `git restore --staged <path>` any parallel-agent files that leak in. Never `git add -A` or `git add .`.

---

## File Map

**Created (backend)**
- `src/qft_pcn/viz/dsl.py` — schema loader, validator, DSL↔RunSpec translator, examples
- `src/qft_pcn/viz/dsl-schema.json` — JSON Schema (Draft 2020-12) for the DSL
- `src/qft_pcn/viz/llm.py` — Ollama wrapper (`list_models`, `generate_dsl`, `verbalize`)
- `src/qft_pcn/tests/test_viz_pcn_snapshots.py`
- `src/qft_pcn/tests/test_viz_dsl.py`
- `src/qft_pcn/tests/test_viz_llm.py`

**Modified (backend)**
- `src/qft_pcn/viz/snapshots.py` — add `snapshot_pcn_fields`, `snapshot_pcn_dynamics`, `snapshot_pcn_coupling`
- `src/qft_pcn/viz/runs.py` — add `pcn-fields`, `pcn-dynamics`, `pcn-coupling` to dispatch
- `src/qft_pcn/viz/schema.py` — `LAYER_KEYS` grows from 8 → 11
- `src/qft_pcn/viz/server.py` — six new `/dsl/*` endpoints
- `src/qft_pcn/viz/presets.py` — 3 PCN presets + `PARAM_SCHEMA` entries for the new keys
- `src/qft_pcn/viz/EXTENSIONS.md` — 3 new deferred items appended
- `src/qft_pcn/tests/test_viz_endpoints_extra.py` — DSL endpoint tests
- `pyproject.toml` — add `jsonschema`, `httpx` to the `viz` optional-dependency extra

**Created (frontend)** — under `src/qft_pcn/viz/web/src/`
- `lib/sections.ts`, `lib/dsl.ts`, `lib/llm.ts`
- `panels/PcnFieldsPanel.tsx` + `.test.tsx`
- `panels/PcnDynamicsPanel.tsx` + `.test.tsx`
- `panels/PcnCouplingPanel.tsx` + `.test.tsx`
- `panels/IntroQftPanel.tsx`, `IntroPcnPanel.tsx`, `IntroQpcnPanel.tsx`
- `panels/IntroPanel.test.tsx` (shared)
- `components/SectionedLayerSelector.tsx` + `.test.tsx`
- `components/RouteSwitcher.tsx` + `.test.tsx`
- `routes/DslRoute.tsx`
- `routes/dsl/ChatPane.tsx` + `.test.tsx`
- `routes/dsl/DslEditor.tsx` + `.test.tsx`
- `routes/dsl/RunOutputPane.tsx`
- `routes/dsl/SteppedFlowBar.tsx` + `.test.tsx`

**Modified (frontend)**
- `web/src/App.tsx` — mount `RouteSwitcher` + conditional route render
- `web/src/lib/types.ts` — `LAYER_KEYS` adds 3 + new `DslSpec`, `LlmModel`, `ChatTurn`, `Section`, `Route`
- `web/src/panels/index.ts` — register the 3 PCN panels + 3 intro pseudo-panels
- `web/src/lib/explainer.ts` — add ExplainerSpecs for the 3 PCN layers
- `web/src/App.css` — section rail styles + DSL route layout
- `web/src/store.ts` — add `route`, `chat`, `dslText`, `llmModel`, `steppedMode`
- `web/src/components/LayerSelector.tsx` — replaced by `SectionedLayerSelector` (delete OR keep + add new alongside; **delete** to avoid drift)
- `web/package.json` — add `@monaco-editor/react`, `monaco-editor`, `ajv`, `ajv-formats`
- `web/vite.config.ts` — proxy `/dsl` to backend port 8000

---

## Per-Task Process

For each task: spec-review the task description first (does it match the spec?), then run the TDD cycle, then run a code-review pass on the diff before commit.

**Spec review (before starting):** Re-read the relevant spec section. Confirm the task does not extend scope, does not require substrate edits, and respects the no-placeholders rule.

**Code review (before commit):** Re-read the diff. Check: (1) no TODO/stub/fake values, (2) no substrate files touched, (3) snapshot reads are defensive, (4) tests cover the failure mode you just fixed, (5) no unrelated drive-by changes.

---

## Workstream A — PCN snapshots + runs wiring

### Task 1: Add 3 new layer keys to schema + snapshots

**Files:**
- Modify: `src/qft_pcn/viz/schema.py`
- Modify: `src/qft_pcn/viz/snapshots.py`
- Create: `src/qft_pcn/tests/test_viz_pcn_snapshots.py`

- [ ] **Step 1: Write the failing test**

```python
# src/qft_pcn/tests/test_viz_pcn_snapshots.py
"""PCN-side snapshot extractor tests.

Builds a real two-layer QFTPCNNetwork (no substrate mocks) and verifies
snapshot_pcn_fields / snapshot_pcn_dynamics / snapshot_pcn_coupling return
the documented shape. Defensive paths covered via a NetworkConfig that
strips optional attributes.
"""

import numpy as np
import pytest

from src.qft_pcn.layer import LayerConfig
from src.qft_pcn.network import NetworkConfig, QFTPCNNetwork
from src.qft_pcn.viz.snapshots import (
    snapshot_pcn_fields,
    snapshot_pcn_dynamics,
    snapshot_pcn_coupling,
)


@pytest.fixture
def net():
    rng = np.random.default_rng(0)
    cfg = NetworkConfig(layers=[LayerConfig(channels=1),
                                LayerConfig(channels=1)])
    n = QFTPCNNetwork(8, 8, cfg, rng=rng)
    obs = np.zeros((1, 8, 8))
    n.step(obs, learn=True)
    return n


def test_pcn_fields_shape(net):
    s = snapshot_pcn_fields(net)
    assert isinstance(s["layers"], list) and len(s["layers"]) == 2
    for layer in s["layers"]:
        assert "phi" in layer and "E" in layer and "Pi" in layer
        assert layer["channels"] == 1


def test_pcn_dynamics_shape(net):
    s = snapshot_pcn_dynamics(net)
    assert isinstance(s["total_free_energy"], float)
    assert isinstance(s["per_layer_free_energy"], list)
    assert len(s["per_layer_free_energy"]) == 2


def test_pcn_coupling_shape(net):
    s = snapshot_pcn_coupling(net)
    # Stress-energy + curvature aggregates from layer 0's error field.
    assert isinstance(s["mean_abs_stress_energy"], float)
    assert isinstance(s["mean_abs_ricci"], float)
    assert isinstance(s["kappa_R"], float)


def test_extractors_defensive_with_missing_attrs():
    # A bare object without manifold/layers must not raise.
    class Empty: pass
    assert snapshot_pcn_fields(Empty()) == {"layers": [], "step": None}
    assert snapshot_pcn_dynamics(Empty())["total_free_energy"] is None
    assert snapshot_pcn_coupling(Empty())["kappa_R"] is None
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest src/qft_pcn/tests/test_viz_pcn_snapshots.py -v --noconftest
```
Expected: ImportError on `snapshot_pcn_fields` (the names don't exist yet).

- [ ] **Step 3: Add `LAYER_KEYS` entries to `schema.py`**

```python
LAYER_KEYS = (
    "manifold",
    "multifield",
    "mps",
    "hamiltonian",
    "qpcn",
    "mera",
    "vqc",
    "logic",
    "pcn-fields",
    "pcn-dynamics",
    "pcn-coupling",
)
```

- [ ] **Step 4: Add the three extractors to `snapshots.py`**

Insert at the end of the file (after the `snapshot_logic` block):

```python
# ---- PCN-side: hierarchical fields stack -----------------------------------

def snapshot_pcn_fields(net: Any) -> dict:
    """Snapshot the full PCN layer stack (Phi/E/Pi per layer)."""
    layers_out = []
    for layer in _safe(lambda: net.layers) or []:
        layers_out.append({
            "phi": _grid(_safe(lambda l=layer: l.phi.values)),
            "E":   _grid(_safe(lambda l=layer: l.error.values)),
            "Pi":  _grid(_safe(lambda l=layer: l.precision.pi)),
            "channels": _safe(lambda l=layer: l.phi.channels),
        })
    return {
        "layers": layers_out,
        "step": _safe(lambda: int(net._step)),
    }


# ---- PCN-side: free-energy / learning dynamics -----------------------------

def snapshot_pcn_dynamics(net: Any) -> dict:
    """Snapshot PCN free-energy aggregates.

    Reads the substrate's `QFTPCNLayer.free_energy(phi_below, kappa_R)`.
    For the bottom layer phi_below is a zero observation array (matching
    the run-time pattern); for higher layers it is the layer-below's phi.
    """
    layers = _safe(lambda: net.layers) or []
    manifold = _safe(lambda: net.manifold)
    kappa_R = _safe(lambda: float(net.cfg.kappa_R))

    per_layer_F: list[float | None] = []
    per_layer_e_norm: list[float | None] = []
    per_layer_pi_mean: list[float | None] = []

    if layers and manifold is not None and kappa_R is not None:
        zero_obs = None
        try:
            c0 = layers[0].phi.channels
            zero_obs = np.zeros((c0, manifold.nx, manifold.ny))
        except (AttributeError, IndexError):
            zero_obs = None
        below = zero_obs
        for layer in layers:
            f = _safe(lambda l=layer, b=below:
                      float(l.free_energy(b, kappa_R))) \
                if below is not None else None
            per_layer_F.append(f)
            per_layer_e_norm.append(_safe(
                lambda l=layer: float(np.linalg.norm(l.error.values))))
            per_layer_pi_mean.append(_safe(
                lambda l=layer: float(np.mean(l.precision.pi))))
            below = _safe(lambda l=layer: l.phi.values)

    total_F = None
    if per_layer_F and all(v is not None for v in per_layer_F):
        total_F = float(sum(per_layer_F))

    return {
        "total_free_energy": total_F,
        "per_layer_free_energy": per_layer_F,
        "per_layer_e_norm": per_layer_e_norm,
        "per_layer_pi_mean": per_layer_pi_mean,
        "n_layers": len(layers),
        "step": _safe(lambda: int(net._step)),
    }


# ---- PCN-side: QFT <-> PCN coupling bridge ---------------------------------

def snapshot_pcn_coupling(net: Any, qpcn: Any = None) -> dict:
    """Snapshot the PCN <-> QFT coupling state.

    PCN -> QFT: bottom-layer error field's stress-energy magnitude.
    QFT -> PCN: mean curvature (the geometry the metric is currently in).
    Optional qpcn argument lets the panel surface QFT operator expectations
    feeding back into PCN observations; if absent the QFT->PCN arrow is
    rendered with curvature only.
    """
    manifold = _safe(lambda: net.manifold)
    layers = _safe(lambda: net.layers) or []
    kappa_R = _safe(lambda: float(net.cfg.kappa_R))

    mean_abs_stress = None
    if manifold is not None and layers:
        e0 = _safe(lambda: layers[0].error.values)
        if e0 is not None:
            ses = _safe(lambda: manifold.stress_energy(e0))
            if ses is not None:
                t_xx, t_xy, t_yy = ses
                mean_abs_stress = float(np.mean(
                    np.abs(t_xx) + np.abs(t_xy) + np.abs(t_yy)) / 3.0)

    mean_abs_ricci = _safe(
        lambda: float(np.abs(net.manifold.ricci_scalar()).mean())
    )

    qpcn_observable_energy = None
    if qpcn is not None:
        qpcn_observable_energy = _safe(
            lambda: float(np.real(qpcn._last_energy)))

    return {
        "kappa_R": kappa_R,
        "mean_abs_stress_energy": mean_abs_stress,
        "mean_abs_ricci": mean_abs_ricci,
        "qpcn_observable_energy": qpcn_observable_energy,
        "step": _safe(lambda: int(net._step)),
    }
```

- [ ] **Step 5: Run test to verify it passes**

```
uv run pytest src/qft_pcn/tests/test_viz_pcn_snapshots.py -v --noconftest
```
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/viz/schema.py src/qft_pcn/viz/snapshots.py \
        src/qft_pcn/tests/test_viz_pcn_snapshots.py
git diff --staged --name-only
git commit -m "feat(viz/snapshots): PCN-side extractors (fields, dynamics, coupling)"
```

---

### Task 2: Wire PCN snapshots into `run_simulation`

**Files:**
- Modify: `src/qft_pcn/viz/runs.py`

- [ ] **Step 1: Add new want-flags + per-step calls**

In `run_simulation`, after the existing `want_logic`, `want_vqc` block, add:

```python
    want_pcn_fields = "pcn-fields" in requested
    want_pcn_dynamics = "pcn-dynamics" in requested
    want_pcn_coupling = "pcn-coupling" in requested
    want_any_pcn = want_pcn_fields or want_pcn_dynamics or want_pcn_coupling
```

Extend the "fall back" condition to include the new PCN flags so requesting any of them counts:

```python
    if not (want_network or want_multifield or want_qpcn or want_mera
            or want_logic or want_vqc or want_any_pcn):
        want_network = True
```

The PCN extractors need the manifold network — ensure it is built when any of the new flags is set:

```python
    if want_any_pcn and net is None:
        net = _build_network(spec)
        c = net.layers[0].phi.channels
        observation = np.zeros((c, net.manifold.nx, net.manifold.ny))
```

In the per-step loop, after the existing `manifold` snapshot block, add:

```python
        if net is not None:
            if want_pcn_fields:
                snaps["pcn-fields"] = snapshots.snapshot_pcn_fields(net)
            if want_pcn_dynamics:
                snaps["pcn-dynamics"] = snapshots.snapshot_pcn_dynamics(net)
            if want_pcn_coupling:
                snaps["pcn-coupling"] = snapshots.snapshot_pcn_coupling(
                    net, qpcn)
```

Update the `run_simulation` docstring to mention the three new keys.

- [ ] **Step 2: Add a smoke test**

Append to `src/qft_pcn/tests/test_viz_pcn_snapshots.py`:

```python
from src.qft_pcn.viz.runs import RunSpec, run_simulation


def test_run_simulation_emits_pcn_layer_states():
    spec = RunSpec(layers=["pcn-fields", "pcn-dynamics", "pcn-coupling"],
                   steps=2, grid=8)
    frames = list(run_simulation(spec))
    assert len(frames) == 2
    for key in ("pcn-fields", "pcn-dynamics", "pcn-coupling"):
        assert key in frames[0].layer_states
```

- [ ] **Step 3: Run tests**

```
uv run pytest src/qft_pcn/tests/test_viz_pcn_snapshots.py \
              src/qft_pcn/tests/test_viz_server.py -v --noconftest
```
Expected: all pass (no regression on existing server tests).

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/viz/runs.py \
        src/qft_pcn/tests/test_viz_pcn_snapshots.py
git diff --staged --name-only
git commit -m "feat(viz/runs): dispatch pcn-fields / pcn-dynamics / pcn-coupling layers"
```

---

### Task 3: PCN presets + param schema

**Files:**
- Modify: `src/qft_pcn/viz/presets.py`

- [ ] **Step 1: Append three presets**

```python
    # ---- PCN substrate ----------------------------------------------------
    Preset(
        id="pcn-fields.two-layer-default",
        layer="pcn-fields",
        label="Two-layer hierarchy (default)",
        description="Two-layer QFTPCNNetwork on a flat manifold. Watch "
                    "Phi/E/Pi propagate top-down + bottom-up.",
        spec_overrides={"steps": 30, "grid": 12, "seed": 0,
                        "params": {"manifold": {"source": "flat"}}},
    ),
    Preset(
        id="pcn-dynamics.free-energy-decay",
        layer="pcn-dynamics",
        label="Free-energy decay — quiescent observation",
        description="Quiescent observation; total free energy should decay "
                    "as beliefs settle toward the prior.",
        spec_overrides={"steps": 40, "grid": 12, "seed": 0,
                        "params": {"manifold": {"source": "flat"}}},
    ),
    Preset(
        id="pcn-coupling.bridge-snapshot",
        layer="pcn-coupling",
        label="PCN <-> QFT bridge — coupled run",
        description="PCN net plus a single-species QPCN, watching the "
                    "bidirectional coupling magnitudes.",
        spec_overrides={"steps": 30, "grid": 12, "seed": 0,
                        "layers": ["pcn-coupling", "qpcn"],
                        "params": {"manifold": {"source": "hot-spot"}}},
    ),
```

- [ ] **Step 2: Add `PARAM_SCHEMA` entries**

```python
    "pcn-fields":   {"type": "object", "properties": {}},
    "pcn-dynamics": {"type": "object", "properties": {}},
    "pcn-coupling": {"type": "object", "properties": {}},
```

The empty properties intentionally mirror the existing `mps`/`hamiltonian`
pattern — these layers do not own substrate params; they share `manifold`
and `qpcn` knobs.

- [ ] **Step 3: Run preset tests**

```
uv run pytest src/qft_pcn/tests/test_viz_presets.py -v --noconftest
```
Expected: PASS (the round-trip test will now parametrize over 17 presets).

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/viz/presets.py
git diff --staged --name-only
git commit -m "feat(viz/presets): PCN-side presets + param schema entries"
```

---

## Workstream D — DSL backend (in parallel with A from this point)

### Task 4: DSL schema + validator + translator

**Files:**
- Create: `src/qft_pcn/viz/dsl-schema.json`
- Create: `src/qft_pcn/viz/dsl.py`
- Create: `src/qft_pcn/tests/test_viz_dsl.py`
- Modify: `pyproject.toml` (add `jsonschema` to `viz` extra)

- [ ] **Step 1: Add `jsonschema` and `httpx` to viz extra**

In `pyproject.toml`, find `[project.optional-dependencies]` and update:

```toml
viz = [
    # ... existing entries ...
    "jsonschema>=4.0",
    "httpx>=0.27",
]
```

Run `uv sync --extra viz` to install.

- [ ] **Step 2: Write `dsl-schema.json`**

Create with the full JSON Schema body from spec §4.3 (`fields`, `hamiltonian.terms`, `observables`, optional `run`). Include `$id` of `https://qpcn.local/dsl/v1.json` and `$schema` of `https://json-schema.org/draft/2020-12/schema`.

- [ ] **Step 3: Write the failing test**

```python
# src/qft_pcn/tests/test_viz_dsl.py
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
```

- [ ] **Step 4: Run test to verify it fails**

```
uv run pytest src/qft_pcn/tests/test_viz_dsl.py -v --noconftest
```
Expected: ImportError on `dsl`.

- [ ] **Step 5: Implement `dsl.py`**

```python
# src/qft_pcn/viz/dsl.py
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
    return RunSpec(
        layers=["qpcn", "mps", "hamiltonian"],
        steps=int(run_cfg.get("steps", 30)),
        grid=12,
        seed=run_cfg.get("seed"),
        params={"qpcn": qpcn_params},
    )


def runspec_to_dsl(spec: RunSpec) -> dict:
    """Reverse-translate a RunSpec back to a DSL dict (best-effort).

    Documented in EXTENSIONS.md as lossy: RunSpec.params is flat, so any
    structure not captured by the standard qpcn keys is dropped.
    """
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
```

- [ ] **Step 6: Run tests to verify they pass**

```
uv run pytest src/qft_pcn/tests/test_viz_dsl.py -v --noconftest
```
Expected: 6 PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock \
        src/qft_pcn/viz/dsl-schema.json \
        src/qft_pcn/viz/dsl.py \
        src/qft_pcn/tests/test_viz_dsl.py
git diff --staged --name-only
git commit -m "feat(viz/dsl): schema + validator + RunSpec translator + 3 examples"
```

---

### Task 5: Ollama wrapper (`llm.py`)

**Files:**
- Create: `src/qft_pcn/viz/llm.py`
- Create: `src/qft_pcn/tests/test_viz_llm.py`

- [ ] **Step 1: Write the failing test**

```python
# src/qft_pcn/tests/test_viz_llm.py
"""Ollama wrapper tests (mocks httpx — no live Ollama needed)."""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.qft_pcn.viz.llm import (
    list_models, generate_dsl, verbalize, OLLAMA_URL,
)


def _mock_response(payload: dict, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status = MagicMock()
    return r


@patch("src.qft_pcn.viz.llm.httpx.get")
def test_list_models(mock_get):
    mock_get.return_value = _mock_response(
        {"models": [{"name": "gemma3:4b"}, {"name": "deepseek-r1:32b"}]})
    out = list_models()
    assert [m["name"] for m in out] == ["gemma3:4b", "deepseek-r1:32b"]


@patch("src.qft_pcn.viz.llm.httpx.post")
def test_generate_dsl_returns_validated(mock_post):
    dsl = {"fields": [{"name": "A", "cutoff": 2}],
           "hamiltonian": {"terms": [
               {"kind": "mass", "species": "A", "coefficient": 1.0}]},
           "observables": [
               {"operator": "n", "site": 0, "species": "A", "target": 0.25}]}
    mock_post.return_value = _mock_response(
        {"response": json.dumps(dsl)})
    from src.qft_pcn.viz.dsl import load_schema, EXAMPLES
    result = generate_dsl("simulate A at quarter density",
                          model="gemma3:4b",
                          schema=load_schema(), examples=EXAMPLES)
    assert "dsl" in result
    assert result["dsl"]["fields"][0]["name"] == "A"


@patch("src.qft_pcn.viz.llm.httpx.post")
def test_generate_dsl_surfaces_validation_errors(mock_post):
    mock_post.return_value = _mock_response(
        {"response": json.dumps({"fields": []})})  # missing required keys
    from src.qft_pcn.viz.dsl import load_schema, EXAMPLES
    result = generate_dsl("garbage prompt", model="gemma3:4b",
                          schema=load_schema(), examples=EXAMPLES)
    assert "error" in result
    assert "raw" in result
    assert isinstance(result["validation_errors"], list)


@patch("src.qft_pcn.viz.llm.httpx.post")
def test_verbalize_strips_think_blocks(mock_post):
    mock_post.return_value = _mock_response(
        {"response": "<think>some chain</think>The result is 0.25."})
    txt = verbalize({"obs": {"n_0": 0.25}}, model="deepseek-r1:32b",
                    original_prompt="what is n_0?")
    assert "<think>" not in txt
    assert "result is 0.25" in txt
```

- [ ] **Step 2: Run test to verify it fails**

```
uv run pytest src/qft_pcn/tests/test_viz_llm.py -v --noconftest
```
Expected: ImportError on `llm`.

- [ ] **Step 3: Implement `llm.py`**

```python
# src/qft_pcn/viz/llm.py
"""Thin wrapper around the local Ollama HTTP API.

The viz never talks to a managed inference API; the LLM lives on the user's
machine. We expose three entry points:

  * `list_models()` -> Ollama's installed models.
  * `generate_dsl(prompt, model, schema, examples)` -> validated DSL dict
    or {error, raw, validation_errors} on validation failure.
  * `verbalize(observations, model, original_prompt)` -> natural-language
    summary, with `<think>...</think>` chain-of-thought blocks stripped
    out so reasoning models (e.g. deepseek-r1) don't leak internals.

`OLLAMA_URL` is overridable via the `OLLAMA_HOST` env var (matches the
Ollama CLI's own convention).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

from .dsl import validate as dsl_validate

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def list_models() -> list[dict]:
    """Return the Ollama-installed model list (name + size + modified)."""
    r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=10.0)
    r.raise_for_status()
    return r.json().get("models", [])


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def _system_prompt_for_dsl(schema: dict, examples: list[dict]) -> str:
    ex_block = "\n\n".join(
        f"Example {i + 1}:\n```json\n{json.dumps(ex, indent=2)}\n```"
        for i, ex in enumerate(examples)
    )
    return (
        "You are a QPCN DSL emitter. Given a problem in natural language, "
        "you respond with ONLY a JSON object conforming to this schema:\n\n"
        f"```json\n{json.dumps(schema, indent=2)}\n```\n\n"
        "Do not include any prose, markdown, or explanation. Return raw JSON. "
        "Use the cutoff field to bound the per-site Fock truncation (default 2). "
        "Available hamiltonian term kinds: mass, kinetic, quartic, yukawa, "
        "density, curvature_coupling. Available observable operators: n, phi, phi2.\n\n"
        f"{ex_block}"
    )


def generate_dsl(prompt: str, *, model: str, schema: dict,
                 examples: list[dict]) -> dict:
    """Ask Ollama to emit a DSL for `prompt`. Validate before returning.

    Returns `{"dsl": <validated dict>}` on success, or
    `{"error": str, "raw": str, "validation_errors": list[str]}` on failure.
    """
    body = {
        "model": model,
        "prompt": prompt,
        "system": _system_prompt_for_dsl(schema, examples),
        "stream": False,
        "format": "json",
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=120.0)
    r.raise_for_status()
    raw = _strip_think(r.json().get("response", ""))

    try:
        candidate = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"error": f"LLM returned non-JSON: {exc}",
                "raw": raw, "validation_errors": []}

    errors = dsl_validate(candidate)
    if errors:
        return {"error": "DSL validation failed",
                "raw": raw, "validation_errors": errors}

    return {"dsl": candidate}


def verbalize(observations: Any, *, model: str,
              original_prompt: str) -> str:
    """Ask Ollama to explain the run's observations in plain language."""
    body = {
        "model": model,
        "prompt": (
            "Original question:\n"
            f"  {original_prompt}\n\n"
            "Final observables from the QPCN run:\n"
            f"  {json.dumps(observations, indent=2)}\n\n"
            "Briefly explain what the observables mean in the context of the "
            "original question. Keep it under 120 words. No preamble."
        ),
        "stream": False,
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=body, timeout=120.0)
    r.raise_for_status()
    return _strip_think(r.json().get("response", ""))
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest src/qft_pcn/tests/test_viz_llm.py -v --noconftest
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/llm.py src/qft_pcn/tests/test_viz_llm.py
git diff --staged --name-only
git commit -m "feat(viz/llm): Ollama wrapper for DSL generation + verbalization"
```

---

### Task 6: `/dsl/*` server endpoints

**Files:**
- Modify: `src/qft_pcn/viz/server.py`
- Modify: `src/qft_pcn/tests/test_viz_endpoints_extra.py`

- [ ] **Step 1: Add failing endpoint tests**

Append to `test_viz_endpoints_extra.py`:

```python
from unittest.mock import patch


def test_dsl_schema_endpoint(client):
    r = client.get("/dsl/schema")
    assert r.status_code == 200
    assert r.json()["$id"].endswith("dsl/v1.json")


def test_dsl_models_endpoint_503_when_ollama_down(client):
    # When httpx.get raises, the endpoint should surface 503, not 500.
    with patch("src.qft_pcn.viz.server.llm.list_models",
               side_effect=Exception("connection refused")):
        r = client.get("/dsl/models")
        assert r.status_code == 503


def test_dsl_translate_returns_dsl(client):
    fake = {"dsl": {"fields": [{"name": "A", "cutoff": 2}],
                    "hamiltonian": {"terms": [
                        {"kind": "mass", "species": "A", "coefficient": 1.0}]},
                    "observables": [
                        {"operator": "n", "site": 0,
                         "species": "A", "target": 0.25}]}}
    with patch("src.qft_pcn.viz.server.llm.generate_dsl",
               return_value=fake):
        r = client.post("/dsl/translate",
                        json={"prompt": "go", "model": "gemma3:4b"})
        assert r.status_code == 200
        assert r.json()["dsl"]["fields"][0]["name"] == "A"


def test_dsl_run_registers_runspec(client):
    dsl = {"fields": [{"name": "A", "cutoff": 2}],
           "hamiltonian": {"terms": [
               {"kind": "mass", "species": "A", "coefficient": 1.0},
               {"kind": "kinetic", "species": "A", "coefficient": 0.5}]},
           "observables": [
               {"operator": "n", "site": 0,
                "species": "A", "target": 0.25}]}
    r = client.post("/dsl/run", json={"dsl": dsl})
    assert r.status_code == 200
    assert "run_id" in r.json()


def test_dsl_run_validates(client):
    r = client.post("/dsl/run", json={"dsl": {"fields": []}})
    assert r.status_code == 400


def test_dsl_export_returns_dsl_for_known_run(client):
    run = client.post("/run", json={"layers": ["qpcn"], "steps": 2,
                                     "grid": 8}).json()
    r = client.get(f"/dsl/export/{run['run_id']}")
    assert r.status_code == 200
    body = r.json()
    assert "fields" in body and "hamiltonian" in body
```

- [ ] **Step 2: Add the endpoints to `server.py`**

Near the existing imports:

```python
from . import dsl as _dsl
from . import llm
```

Add (group with the other route definitions):

```python
@app.get("/dsl/schema")
def get_dsl_schema():
    return _dsl.load_schema()


@app.get("/dsl/models")
def get_dsl_models():
    try:
        return llm.list_models()
    except Exception as exc:  # noqa: BLE001  (Ollama down -> 503)
        raise HTTPException(status_code=503,
                            detail=f"ollama unreachable: {exc}")


@app.post("/dsl/translate")
def post_dsl_translate(body: dict):
    prompt = (body.get("prompt") or "").strip()
    model = (body.get("model") or "").strip()
    if not prompt or not model:
        raise HTTPException(status_code=400,
                            detail="prompt and model are required")
    try:
        return llm.generate_dsl(
            prompt, model=model,
            schema=_dsl.load_schema(), examples=_dsl.EXAMPLES,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503,
                            detail=f"ollama call failed: {exc}")


@app.post("/dsl/run")
def post_dsl_run(body: dict):
    dsl = body.get("dsl") or {}
    errors = _dsl.validate(dsl)
    if errors:
        raise HTTPException(status_code=400,
                            detail={"validation_errors": errors})
    spec = _dsl.dsl_to_runspec(dsl)
    run_id = uuid.uuid4().hex
    _registry.add(run_id, spec)
    return {"run_id": run_id}


@app.post("/dsl/verbalize")
def post_dsl_verbalize(body: dict):
    model = (body.get("model") or "").strip()
    prompt = body.get("original_prompt") or ""
    observations = body.get("observations") or {}
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    try:
        return {"text": llm.verbalize(observations, model=model,
                                       original_prompt=prompt)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503,
                            detail=f"ollama call failed: {exc}")


@app.get("/dsl/export/{run_id}")
def get_dsl_export(run_id: str):
    spec = _registry.get(run_id)
    if spec is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _dsl.runspec_to_dsl(spec)
```

- [ ] **Step 3: Run all endpoint tests**

```
uv run pytest src/qft_pcn/tests/test_viz_endpoints_extra.py -v --noconftest
```
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/viz/server.py \
        src/qft_pcn/tests/test_viz_endpoints_extra.py
git diff --staged --name-only
git commit -m "feat(viz/server): /dsl/{schema,models,translate,run,verbalize,export}"
```

---

### Task 7: EXTENSIONS.md additions + Vite proxy

**Files:**
- Modify: `src/qft_pcn/viz/EXTENSIONS.md`
- Modify: `src/qft_pcn/viz/web/vite.config.ts`

- [ ] **Step 1: Append three deferred items to `EXTENSIONS.md`**

```markdown
<a id="pcn-layer-kl-divergence"></a>
## PCN — per-layer KL divergence

- **What's needed:** A `QFTPCNLayer.kl_divergence() -> float` method exposing
  the per-layer KL term so `PcnDynamicsPanel` can decompose F = accuracy + KL.
- **Why deferred:** Substrate currently exposes only the aggregate
  `free_energy(phi_below, kappa_R)`.
- **Wire-up when ready:** Snapshot `per_layer_kl` next to `per_layer_free_energy`
  in `snapshot_pcn_dynamics`; the panel already has the chart slot.

<a id="lossless-dsl-runspec-round-trip"></a>
## DSL — lossless RunSpec round-trip

- **What's needed:** `RunSpec` to carry an optional `dsl: dict | None` companion
  so `runspec_to_dsl` can return the original DSL verbatim.
- **Why deferred:** `RunSpec.params` is intentionally flat for fast dict-merge
  semantics; structured DSL is currently dropped on `dsl_to_runspec`.
- **Wire-up when ready:** Store the original DSL on `RunSpec.dsl` in
  `dsl_to_runspec`; `runspec_to_dsl` returns it verbatim when present.

<a id="llm-retry-with-validation-feedback"></a>
## DSL — LLM retry on validation failure

- **What's needed:** A retry loop in `generate_dsl` that feeds the validation
  errors back into the LLM as a follow-up turn.
- **Why deferred:** v1 surfaces raw output to the chat pane for manual repair.
- **Wire-up when ready:** Add `max_retries: int = 0` param; on validation
  failure, append a "this DSL failed validation: <errors>; fix and re-emit"
  message and retry up to `max_retries` times.
```

- [ ] **Step 2: Extend the Vite proxy**

In `web/vite.config.ts`, add `/dsl` to the proxy map:

```ts
    proxy: {
      '/run': 'http://localhost:8000',
      '/runs': 'http://localhost:8000',
      '/presets': 'http://localhost:8000',
      '/params': 'http://localhost:8000',
      '/export': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/dsl': 'http://localhost:8000',
      '/ws': { target: 'http://localhost:8000', ws: true },
    },
```

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/viz/EXTENSIONS.md \
        src/qft_pcn/viz/web/vite.config.ts
git diff --staged --name-only
git commit -m "docs(viz)+chore(vite): record 3 new deferred items + proxy /dsl"
```

---

## Workstream C — Sectioned LayerSelector + Intro panels (independent — can start any time)

### Task 8: `sections.ts` content registry

**Files:**
- Create: `src/qft_pcn/viz/web/src/lib/sections.ts`

- [ ] **Step 1: Write the file**

```ts
// src/qft_pcn/viz/web/src/lib/sections.ts
/**
 * Three top-level narrative sections grouping the 11 layers into:
 * QFT Substrate, PCN Substrate, QPCN Fusion.
 *
 * Content sourced from QFT_PCN_ARCHITECTURE.md sections §2.1 / §2.2 / §2.3.
 * Used by SectionedLayerSelector + Intro panels.
 */

export interface SectionSpec {
  key: 'qft' | 'pcn' | 'qpcn';
  title: string;
  tagline: string;
  story: string[];
  layers: string[];
  layerSummaries: { layer: string; oneLine: string }[];
  references?: { label: string; href: string }[];
}

export const SECTIONS: SectionSpec[] = [
  {
    key: 'qft',
    title: 'QFT Substrate',
    tagline: 'Operator-valued fields on a tensor-network state.',
    story: [
      'The QFT side carries the system\'s quantum content. Each spatial point holds an operator-valued bosonic field with a truncated Fock space; the many-body state is a Matrix Product State (MPS) with controllable bond dimension.',
      'The Hamiltonian is built locally from one-site and two-site terms. Evolution uses second-order Suzuki-Trotter splitting with SVD-based bond truncation — real-time for unitary dynamics, imaginary-time for relaxation.',
      'MERA layers add multi-scale structure; the VQC panel shows a parameterized quantum circuit that can drop into any PCN layer as a generative map.',
    ],
    layers: ['mps', 'mera', 'vqc', 'hamiltonian'],
    layerSummaries: [
      { layer: 'mps', oneLine: 'The entanglement carrier — 1D chain of low-rank tensors.' },
      { layer: 'mera', oneLine: 'Multi-scale tree of disentanglers + isometries.' },
      { layer: 'vqc', oneLine: 'Parameterized quantum circuit trained via parameter-shift gradients.' },
      { layer: 'hamiltonian', oneLine: 'Local many-body operator — the QPCN\'s generative model.' },
    ],
    references: [{ label: 'Architecture §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
  {
    key: 'pcn',
    title: 'PCN Substrate',
    tagline: 'Hierarchical predictive coding on a dynamic Riemannian manifold.',
    story: [
      'The PCN side is the classical Friston/Bogacz-style predictive-coding network. Belief fields Φ, error fields E, and precision fields Π live on a 2D manifold whose metric is itself dynamic.',
      'Each hierarchical layer compares its downward prediction to the layer below; errors propagate up, predictions propagate down. The variational free energy F = ½ Π E² − ½ log Π + κ R minimizes jointly with respect to beliefs, precisions, and the metric.',
      'Multi-field setups let several field types share one manifold, coupled via learnable Yukawa-style couplings g_ij — correlated fields grow their coupling; uncorrelated fields don\'t.',
    ],
    layers: ['pcn-fields', 'pcn-dynamics', 'multifield'],
    layerSummaries: [
      { layer: 'pcn-fields', oneLine: 'Φ / E / Π surfaces stacked across the full layer hierarchy.' },
      { layer: 'pcn-dynamics', oneLine: 'Free energy F, per-layer trajectories, learning rates.' },
      { layer: 'multifield', oneLine: 'Multiple field species coupled via learnable g_ij.' },
    ],
    references: [{ label: 'Architecture §2.1, §2.2', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
  {
    key: 'qpcn',
    title: 'QPCN Fusion',
    tagline: 'How QFT + PCN compose into one learner.',
    story: [
      'The QPCN is the fusion layer: its belief state is the MPS, its generative model is the Hamiltonian, observations are target expectation values of local operators, and prediction errors drive gradient updates on Hamiltonian parameters.',
      'Stress-energy expectations from the QFT side feed back into the classical 2D manifold so geometry and quantum content are bidirectionally coupled — that\'s the load-bearing structural claim of the architecture.',
      'The Logic panel demonstrates the same machinery applied to symbolic reasoning: rule terms encoded as Hamiltonian costs whose ground state corresponds to a well-typed / fully-reduced program.',
    ],
    layers: ['pcn-coupling', 'qpcn', 'manifold', 'logic'],
    layerSummaries: [
      { layer: 'pcn-coupling', oneLine: 'The bidirectional bridge — stress-energy ↑, expectations ↓.' },
      { layer: 'qpcn', oneLine: 'Belief = MPS; generative model = H; errors drive parameter updates.' },
      { layer: 'manifold', oneLine: '2D Riemannian manifold whose metric is sourced by prediction error.' },
      { layer: 'logic', oneLine: 'Symbolic-reasoning Hamiltonian — relaxation = reduction.' },
    ],
    references: [{ label: 'Architecture §1.3, §2.3', href: '../../QFT_PCN_ARCHITECTURE.md' }],
  },
];

export function sectionForLayer(layer: string): SectionSpec | undefined {
  return SECTIONS.find((s) => s.layers.includes(layer));
}
```

No test for this static data file — coverage comes from the panels that consume it.

- [ ] **Step 2: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/sections.ts
git diff --staged --name-only
git commit -m "feat(viz/web): sections content registry (QFT/PCN/QPCN narrative)"
```

---

### Task 9: Intro panels (3 components + shared test)

**Files:**
- Create: `src/qft_pcn/viz/web/src/panels/IntroQftPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/IntroPcnPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/IntroQpcnPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/IntroPanel.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// src/qft_pcn/viz/web/src/panels/IntroPanel.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { IntroQftPanel } from './IntroQftPanel';
import { IntroPcnPanel } from './IntroPcnPanel';
import { IntroQpcnPanel } from './IntroQpcnPanel';
import { SECTIONS } from '../lib/sections';

const frame = { step: 0, layer_states: {} } as any;

describe('Intro panels', () => {
  it('IntroQftPanel renders the QFT section title', () => {
    render(<IntroQftPanel frame={frame} />);
    expect(screen.getByText('QFT Substrate')).toBeInTheDocument();
  });
  it('IntroPcnPanel renders the PCN section title', () => {
    render(<IntroPcnPanel frame={frame} />);
    expect(screen.getByText('PCN Substrate')).toBeInTheDocument();
  });
  it('IntroQpcnPanel renders the QPCN section title', () => {
    render(<IntroQpcnPanel frame={frame} />);
    expect(screen.getByText('QPCN Fusion')).toBeInTheDocument();
  });
  it('each panel lists its layer summaries', () => {
    for (const sec of SECTIONS) {
      const Panel = sec.key === 'qft' ? IntroQftPanel
                  : sec.key === 'pcn' ? IntroPcnPanel
                  : IntroQpcnPanel;
      const { unmount } = render(<Panel frame={frame} />);
      for (const ls of sec.layerSummaries) {
        expect(screen.getByText(new RegExp(ls.oneLine.slice(0, 20), 'i')))
          .toBeInTheDocument();
      }
      unmount();
    }
  });
});
```

- [ ] **Step 2: Run to verify it fails**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run IntroPanel.test.tsx
```
Expected: cannot resolve the three Intro panel imports.

- [ ] **Step 3: Write the three panels (one shared body)**

Each file just selects its section and delegates to a shared inner renderer. Pattern (write three near-identical files — do NOT abstract them into one with a section prop; spec says one component per section):

```tsx
// src/qft_pcn/viz/web/src/panels/IntroQftPanel.tsx
import { PanelShell } from './PanelShell';
import { SECTIONS } from '../lib/sections';
import type { Frame } from '../lib/types';

export function IntroQftPanel({ frame, baselineFrame: _baselineFrame }: {
  frame: Frame; baselineFrame?: Frame;
}) {
  const sec = SECTIONS.find((s) => s.key === 'qft')!;
  return (
    <PanelShell title={sec.title} step={frame.step} hasData
                emptyMessage="">
      <div className="intro-panel">
        <p className="intro-tagline">{sec.tagline}</p>
        {sec.story.map((p, i) => <p key={i}>{p}</p>)}
        <h4>Layers in this section</h4>
        <ul>
          {sec.layerSummaries.map((ls) => (
            <li key={ls.layer}>
              <strong>{ls.layer}</strong> — {ls.oneLine}
            </li>
          ))}
        </ul>
        {sec.references && (
          <>
            <h4>References</h4>
            <ul>
              {sec.references.map((r, i) =>
                <li key={i}><a href={r.href}>{r.label}</a></li>)}
            </ul>
          </>
        )}
      </div>
    </PanelShell>
  );
}
```

Replicate for `IntroPcnPanel.tsx` (replace `'qft'` with `'pcn'`) and `IntroQpcnPanel.tsx` (`'qpcn'`).

- [ ] **Step 4: Run test to pass**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run IntroPanel.test.tsx
```
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/panels/IntroQftPanel.tsx \
        src/qft_pcn/viz/web/src/panels/IntroPcnPanel.tsx \
        src/qft_pcn/viz/web/src/panels/IntroQpcnPanel.tsx \
        src/qft_pcn/viz/web/src/panels/IntroPanel.test.tsx
git diff --staged --name-only
git commit -m "feat(viz/web): Intro panels for QFT / PCN / QPCN sections"
```

---

### Task 10: SectionedLayerSelector

**Files:**
- Create: `src/qft_pcn/viz/web/src/components/SectionedLayerSelector.tsx`
- Create: `src/qft_pcn/viz/web/src/components/SectionedLayerSelector.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
// src/qft_pcn/viz/web/src/components/SectionedLayerSelector.test.tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { SectionedLayerSelector } from './SectionedLayerSelector';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

describe('SectionedLayerSelector', () => {
  it('renders the three section headers', () => {
    render(<SectionedLayerSelector />);
    expect(screen.getByText('QFT Substrate')).toBeInTheDocument();
    expect(screen.getByText('PCN Substrate')).toBeInTheDocument();
    expect(screen.getByText('QPCN Fusion')).toBeInTheDocument();
  });

  it('selects a layer when its button is clicked', () => {
    render(<SectionedLayerSelector />);
    fireEvent.click(screen.getByRole('button', { name: 'mps' }));
    expect(useVizStore.getState().selectedLayer).toBe('mps');
  });

  it('clicking a section header selects its intro pseudo-layer', () => {
    render(<SectionedLayerSelector />);
    fireEvent.click(screen.getByRole('button', { name: /qft substrate/i }));
    expect(useVizStore.getState().selectedLayer).toBe('intro-qft');
  });

  it('section is collapsible', () => {
    const { container } = render(<SectionedLayerSelector />);
    const qftHeader = screen.getByRole('button', { name: /qft substrate/i });
    // Layers start visible — clicking should collapse, click again expands.
    fireEvent.click(qftHeader.parentElement!.querySelector(
      '.section-collapse-toggle')!);
    expect(container.querySelector('.section-qft.collapsed')).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run to verify it fails** (`vitest --run SectionedLayerSelector.test.tsx`).

- [ ] **Step 3: Implement the component**

```tsx
// src/qft_pcn/viz/web/src/components/SectionedLayerSelector.tsx
/**
 * Left rail: three collapsible sections (QFT / PCN / QPCN) each containing
 * its layers. Section header is itself clickable: it selects the matching
 * intro pseudo-layer (intro-qft / intro-pcn / intro-qpcn) so the explainer
 * panel area renders the section narrative.
 */

import { useState } from 'react';
import { SECTIONS } from '../lib/sections';
import { useVizStore } from '../store';

export function SectionedLayerSelector() {
  const selected = useVizStore((s) => s.selectedLayer);
  const select = useVizStore((s) => s.selectLayer);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});

  return (
    <nav className="layer-selector">
      <h2>Layers</h2>
      {SECTIONS.map((sec) => {
        const isCollapsed = collapsed[sec.key] ?? false;
        const introKey = `intro-${sec.key}`;
        return (
          <div key={sec.key} className={
            `layer-section section-${sec.key}${isCollapsed ? ' collapsed' : ''}`
          }>
            <div className="layer-section-header">
              <button
                type="button"
                className={selected === introKey
                  ? 'section-title active' : 'section-title'}
                onClick={() => select(introKey)}
              >
                {sec.title}
              </button>
              <button
                type="button"
                className="section-collapse-toggle"
                aria-label={isCollapsed ? 'Expand section' : 'Collapse section'}
                onClick={() => setCollapsed((c) => ({ ...c, [sec.key]: !isCollapsed }))}
              >
                {isCollapsed ? '▸' : '▾'}
              </button>
            </div>
            {!isCollapsed && (
              <ul>
                {sec.layers.map((layer) => (
                  <li key={layer}>
                    <button
                      type="button"
                      className={layer === selected ? 'active' : ''}
                      onClick={() => select(layer)}
                    >
                      {layer}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        );
      })}
    </nav>
  );
}
```

- [ ] **Step 4: Run test to pass** (`vitest --run SectionedLayerSelector.test.tsx`).

- [ ] **Step 5: Register intro pseudo-layers in `panels/index.ts`**

Add at the bottom of the registry:

```ts
import { IntroQftPanel } from './IntroQftPanel';
import { IntroPcnPanel } from './IntroPcnPanel';
import { IntroQpcnPanel } from './IntroQpcnPanel';

PANELS['intro-qft'] = IntroQftPanel;
PANELS['intro-pcn'] = IntroPcnPanel;
PANELS['intro-qpcn'] = IntroQpcnPanel;
```

(If the existing `index.ts` uses a different registry shape — e.g. `panelFor` switch — extend that switch instead. Inspect the existing file before editing.)

- [ ] **Step 6: Swap App.tsx to use SectionedLayerSelector**

In `App.tsx`, replace the `<LayerSelector />` import + mount with:

```tsx
import { SectionedLayerSelector } from './components/SectionedLayerSelector';
// ...
<SectionedLayerSelector />
```

Delete the old `src/qft_pcn/viz/web/src/components/LayerSelector.tsx` file.

- [ ] **Step 7: Run full vitest suite to confirm no regression**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run
```
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/qft_pcn/viz/web/src/components/SectionedLayerSelector.tsx \
        src/qft_pcn/viz/web/src/components/SectionedLayerSelector.test.tsx \
        src/qft_pcn/viz/web/src/panels/index.ts \
        src/qft_pcn/viz/web/src/App.tsx
git rm src/qft_pcn/viz/web/src/components/LayerSelector.tsx
git diff --staged --name-only
git commit -m "feat(viz/web): sectioned LayerSelector + intro pseudo-layer routing"
```

---

## Workstream B — PCN panels (3 components, runs sequentially after Task 2 lands)

### Task 11: PcnFieldsPanel

**Files:**
- Create: `src/qft_pcn/viz/web/src/panels/PcnFieldsPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/PcnFieldsPanel.test.tsx`

- [ ] **Step 1: Write the failing test**

```tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { PcnFieldsPanel } from './PcnFieldsPanel';

const frame = {
  step: 3,
  layer_states: {
    'pcn-fields': {
      layers: [
        { phi: [[0, 0], [0, 0]], E: [[0, 0], [0, 0]],
          Pi: [[1, 1], [1, 1]], channels: 1 },
        { phi: [[0.1, 0.1], [0.1, 0.1]], E: [[0, 0], [0, 0]],
          Pi: [[1, 1], [1, 1]], channels: 1 },
      ],
      step: 3,
    },
  },
} as any;

describe('PcnFieldsPanel', () => {
  it('renders one card per PCN layer', () => {
    render(<PcnFieldsPanel frame={frame} />);
    expect(screen.getAllByText(/layer/i).length).toBeGreaterThanOrEqual(2);
  });
  it('renders the layer-depth readout', () => {
    render(<PcnFieldsPanel frame={frame} />);
    expect(screen.getByText(/depth/i)).toBeInTheDocument();
  });
  it('mounts with empty layer state without crashing', () => {
    render(<PcnFieldsPanel frame={{ step: 0, layer_states: {} } as any} />);
    expect(screen.getByText(/PCN Fields/)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run to verify it fails** (`vitest --run PcnFieldsPanel.test.tsx`).

- [ ] **Step 3: Implement the panel**

```tsx
// src/qft_pcn/viz/web/src/panels/PcnFieldsPanel.tsx
/**
 * PCN-Fields panel: stacks Φ / E / Π for every PCN layer.
 *
 * Each layer card renders a small 2D heatmap (no R3F — keep it cheap for
 * runs with 4+ layers). Clicking a card expands it to full panel width.
 * Toggle which field (Φ, E, Π) is plotted.
 */

import { useState } from 'react';
import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelToolbar } from './PanelToolbar';
import { PanelReadouts } from './PanelReadouts';
import { diverging, normGrid } from './common';

type Grid = number[][];

interface LayerState { phi?: Grid; E?: Grid; Pi?: Grid; channels?: number; }
interface PcnFieldsState { layers?: LayerState[]; step?: number; }

function Heatmap({ grid, w = 90, h = 90 }: { grid: Grid; w?: number; h?: number }) {
  const rows = grid.length;
  const cols = grid[0]?.length ?? 0;
  if (rows === 0 || cols === 0) return null;
  const { norm } = normGrid(grid);
  const cellW = w / cols, cellH = h / rows;
  const cells = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      cells.push(
        <rect key={`${r}-${c}`} x={c * cellW} y={r * cellH}
              width={cellW} height={cellH}
              fill={diverging(norm[r][c])} />);
    }
  }
  return <svg width={w} height={h}>{cells}</svg>;
}

function norm2(g?: Grid | null) {
  if (!g) return 0;
  let s = 0;
  for (const row of g) for (const v of row)
    if (Number.isFinite(v)) s += v * v;
  return Math.sqrt(s);
}

export function PcnFieldsPanel({ frame, baselineFrame: _baselineFrame }: {
  frame: Frame; baselineFrame?: Frame;
}) {
  const st = (frame.layer_states['pcn-fields'] ?? {}) as PcnFieldsState;
  const layers = st.layers ?? [];
  const [field, setField] = useState<'phi' | 'E' | 'Pi'>('phi');
  const [expanded, setExpanded] = useState<number | null>(null);

  const hasData = layers.length > 0;
  const totalPhi = layers.reduce((a, l) => a + norm2(l.phi), 0);
  const totalE = layers.reduce((a, l) => a + norm2(l.E), 0);
  const meanPi = layers.length
    ? layers.reduce((a, l) => {
        const g = l.Pi ?? [];
        const flat = g.flat();
        return a + (flat.length ? flat.reduce((p, q) => p + q, 0) / flat.length : 0);
      }, 0) / layers.length
    : 0;

  return (
    <PanelShell
      title="PCN Fields — Φ / E / Π stack"
      step={st.step ?? frame.step}
      hasData={hasData}
      toolbar={
        <PanelToolbar items={(['phi', 'E', 'Pi'] as const).map((k) => ({
          key: k, label: k, active: field === k,
          onToggle: () => setField(k),
        }))} />
      }
      readouts={<PanelReadouts cells={[
        { label: 'depth', value: layers.length },
        { label: '‖Φ‖₂', value: totalPhi.toFixed(3) },
        { label: '‖E‖₂', value: totalE.toFixed(3) },
        { label: 'mean Π', value: meanPi.toFixed(3) },
      ]} />}
    >
      <div style={{ padding: 8, overflowY: 'auto' }}>
        {layers.map((layer, i) => {
          const g = field === 'phi' ? layer.phi
                  : field === 'E' ? layer.E : layer.Pi;
          const isExpanded = expanded === i;
          return (
            <div
              key={i}
              onClick={() => setExpanded(isExpanded ? null : i)}
              style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: 6, cursor: 'pointer',
                background: isExpanded ? '#16203a' : 'transparent',
                borderRadius: 4, marginBottom: 4,
              }}
            >
              <span style={{ width: 60, color: '#8c97b3' }}>
                layer {i}
              </span>
              {g
                ? <Heatmap grid={g}
                           w={isExpanded ? 320 : 90}
                           h={isExpanded ? 320 : 90} />
                : <span>—</span>}
              <span style={{ fontSize: 11, color: '#7e8aa3' }}>
                {layer.channels ?? 1} ch · ‖·‖₂ {norm2(g).toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>
    </PanelShell>
  );
}
```

- [ ] **Step 4: Run test to pass** (`vitest --run PcnFieldsPanel.test.tsx`).

- [ ] **Step 5: Register in `panels/index.ts`**

Add `PANELS['pcn-fields'] = PcnFieldsPanel;` (or the equivalent in the existing registry shape).

- [ ] **Step 6: Add explainer entry**

In `lib/explainer.ts` append a `pcn-fields` entry mirroring the existing pattern (what / elements / math / watch / references). Pull math from architecture §3.1.

- [ ] **Step 7: Commit**

```bash
git add src/qft_pcn/viz/web/src/panels/PcnFieldsPanel.tsx \
        src/qft_pcn/viz/web/src/panels/PcnFieldsPanel.test.tsx \
        src/qft_pcn/viz/web/src/panels/index.ts \
        src/qft_pcn/viz/web/src/lib/explainer.ts
git diff --staged --name-only
git commit -m "feat(viz/web): PcnFieldsPanel — hierarchical Φ/E/Π stack"
```

---

### Task 12: PcnDynamicsPanel

**Files:**
- Create: `src/qft_pcn/viz/web/src/panels/PcnDynamicsPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/PcnDynamicsPanel.test.tsx`

- [ ] **Step 1: Test** — assert `total F` readout present, per-layer rows render, MetricsStrip path appears after frames pushed (same pattern as MpsPanel.test.tsx).

- [ ] **Step 2: Implement**

```tsx
// src/qft_pcn/viz/web/src/panels/PcnDynamicsPanel.tsx
/**
 * PCN-Dynamics panel: total free energy F, per-layer F contributions,
 * per-layer ‖E‖₂ and mean Π. Time-series strip on total F.
 *
 * Per-layer KL decomposition deferred — substrate hook missing; see
 * EXTENSIONS.md anchor #pcn-layer-kl-divergence.
 */

import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';

interface PcnDynamicsState {
  total_free_energy?: number | null;
  per_layer_free_energy?: (number | null)[];
  per_layer_e_norm?: (number | null)[];
  per_layer_pi_mean?: (number | null)[];
  n_layers?: number;
  step?: number;
}

export function PcnDynamicsPanel({ frame, baselineFrame: _baselineFrame }: {
  frame: Frame; baselineFrame?: Frame;
}) {
  const st = (frame.layer_states['pcn-dynamics'] ?? {}) as PcnDynamicsState;
  const totalF = st.total_free_energy;
  const perF = st.per_layer_free_energy ?? [];
  const perE = st.per_layer_e_norm ?? [];
  const perPi = st.per_layer_pi_mean ?? [];
  const hasData = perF.length > 0;

  return (
    <PanelShell
      title="PCN Dynamics — free energy + per-layer trajectories"
      step={st.step ?? frame.step}
      hasData={hasData}
      readouts={<PanelReadouts cells={[
        { label: 'depth', value: st.n_layers ?? perF.length },
        { label: 'total F', value: totalF != null ? totalF.toFixed(3) : '—',
          highlightId: 'total_free_energy' },
      ]} />}
      metricsStrip={<MetricsStrip layer="pcn-dynamics" metrics={[{
        key: 'F', label: 'total F', color: '#fbc66a',
        select: (ls) => ls.total_free_energy as number | null | undefined,
      }]} />}
    >
      <table className="pcn-dynamics-table">
        <thead><tr>
          <th>layer</th><th>F</th><th>‖E‖₂</th><th>mean Π</th>
        </tr></thead>
        <tbody>
          {perF.map((f, i) => (
            <tr key={i}>
              <td>{i}</td>
              <td>{f != null ? f.toFixed(3) : '—'}</td>
              <td>{perE[i] != null ? perE[i]!.toFixed(3) : '—'}</td>
              <td>{perPi[i] != null ? perPi[i]!.toFixed(3) : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </PanelShell>
  );
}
```

- [ ] **Step 3: Register + explainer** (same shape as Task 11 Step 5/6).

- [ ] **Step 4: Run test + commit**

```bash
git add src/qft_pcn/viz/web/src/panels/PcnDynamicsPanel.tsx \
        src/qft_pcn/viz/web/src/panels/PcnDynamicsPanel.test.tsx \
        src/qft_pcn/viz/web/src/panels/index.ts \
        src/qft_pcn/viz/web/src/lib/explainer.ts
git diff --staged --name-only
git commit -m "feat(viz/web): PcnDynamicsPanel — total F + per-layer trajectories"
```

---

### Task 13: PcnCouplingPanel

**Files:**
- Create: `src/qft_pcn/viz/web/src/panels/PcnCouplingPanel.tsx`
- Create: `src/qft_pcn/viz/web/src/panels/PcnCouplingPanel.test.tsx`

- [ ] **Step 1: Test** — assert two arrows render (data-testid `arrow-pcn-to-qft` and `arrow-qft-to-pcn`), arrow widths respond to fixture values, readouts present.

- [ ] **Step 2: Implement**

```tsx
// src/qft_pcn/viz/web/src/panels/PcnCouplingPanel.tsx
/**
 * PCN-Coupling panel: visualizes the bidirectional QFT <-> PCN bridge.
 *
 * Two arrows whose widths animate with live magnitudes:
 *   - top arrow (PCN -> QFT): mean |stress-energy| from the error field
 *   - bottom arrow (QFT -> PCN): the QPCN's variational energy
 *     (proxy for "QFT pulling the PCN's observation targets")
 *
 * Readouts: kappa_R coupling constant, mean |stress-energy|, mean |Ricci|.
 */

import type { Frame } from '../lib/types';
import { PanelShell } from './PanelShell';
import { PanelReadouts } from './PanelReadouts';
import { MetricsStrip } from './MetricsStrip';

interface PcnCouplingState {
  kappa_R?: number | null;
  mean_abs_stress_energy?: number | null;
  mean_abs_ricci?: number | null;
  qpcn_observable_energy?: number | null;
  step?: number;
}

function arrowWidth(magnitude: number | null | undefined): number {
  if (magnitude == null || !Number.isFinite(magnitude)) return 4;
  const clipped = Math.min(Math.abs(magnitude), 1);
  return 4 + clipped * 18;
}

export function PcnCouplingPanel({ frame, baselineFrame: _baselineFrame }: {
  frame: Frame; baselineFrame?: Frame;
}) {
  const st = (frame.layer_states['pcn-coupling'] ?? {}) as PcnCouplingState;
  const hasData = st.kappa_R != null;
  const upWidth = arrowWidth(st.mean_abs_stress_energy);
  const downWidth = arrowWidth(st.qpcn_observable_energy);

  return (
    <PanelShell
      title="PCN ↔ QFT — bidirectional coupling"
      step={st.step ?? frame.step}
      hasData={hasData}
      readouts={<PanelReadouts cells={[
        { label: 'κ_R', value: st.kappa_R != null ? st.kappa_R.toFixed(4) : '—' },
        { label: 'mean |T|', value: st.mean_abs_stress_energy != null
            ? st.mean_abs_stress_energy.toExponential(2) : '—',
          highlightId: 'mean_abs_stress_energy' },
        { label: 'mean |R|', value: st.mean_abs_ricci != null
            ? st.mean_abs_ricci.toFixed(3) : '—' },
        { label: '⟨H⟩', value: st.qpcn_observable_energy != null
            ? st.qpcn_observable_energy.toFixed(3) : '—' },
      ]} />}
      metricsStrip={<MetricsStrip layer="pcn-coupling" metrics={[
        { key: 'T', label: 'mean |T|', color: '#6cd0ff',
          select: (ls) => ls.mean_abs_stress_energy as number | null | undefined },
        { key: 'R', label: 'mean |R|', color: '#fbc66a',
          select: (ls) => ls.mean_abs_ricci as number | null | undefined },
      ]} />}
    >
      <svg viewBox="0 0 400 220" width="100%" height="220">
        {/* PCN box */}
        <rect x={20} y={70} width={120} height={80}
              fill="#16203a" stroke="#2a3450" />
        <text x={80} y={115} textAnchor="middle" fill="#e0e6f3"
              fontSize="14">PCN</text>

        {/* QFT box */}
        <rect x={260} y={70} width={120} height={80}
              fill="#16203a" stroke="#2a3450" />
        <text x={320} y={115} textAnchor="middle" fill="#e0e6f3"
              fontSize="14">QFT</text>

        {/* PCN -> QFT (top arrow): stress-energy */}
        <line data-testid="arrow-pcn-to-qft"
              x1={140} y1={90} x2={260} y2={90}
              stroke="#6cd0ff" strokeWidth={upWidth}
              markerEnd="url(#arrowhead-up)" />
        <text x={200} y={75} textAnchor="middle"
              fill="#9aa3bb" fontSize="11">T_μν</text>

        {/* QFT -> PCN (bottom arrow): operator expectations */}
        <line data-testid="arrow-qft-to-pcn"
              x1={260} y1={130} x2={140} y2={130}
              stroke="#fbc66a" strokeWidth={downWidth}
              markerEnd="url(#arrowhead-down)" />
        <text x={200} y={155} textAnchor="middle"
              fill="#9aa3bb" fontSize="11">⟨O⟩</text>

        <defs>
          <marker id="arrowhead-up" markerWidth="8" markerHeight="8"
                  refX="6" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#6cd0ff" />
          </marker>
          <marker id="arrowhead-down" markerWidth="8" markerHeight="8"
                  refX="6" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#fbc66a" />
          </marker>
        </defs>
      </svg>
    </PanelShell>
  );
}
```

- [ ] **Step 3: Register + explainer** (same as Task 11).

- [ ] **Step 4: Run + commit**

```bash
git add src/qft_pcn/viz/web/src/panels/PcnCouplingPanel.tsx \
        src/qft_pcn/viz/web/src/panels/PcnCouplingPanel.test.tsx \
        src/qft_pcn/viz/web/src/panels/index.ts \
        src/qft_pcn/viz/web/src/lib/explainer.ts
git diff --staged --name-only
git commit -m "feat(viz/web): PcnCouplingPanel — bidirectional QFT↔PCN bridge"
```

---

## Workstream E — DSL frontend route

### Task 14: Frontend deps + Vite config

**Files:**
- Modify: `src/qft_pcn/viz/web/package.json`
- Modify: `src/qft_pcn/viz/web/vite.config.ts`

(Vite proxy was already extended in Task 7; this task adds the npm deps.)

- [ ] **Step 1: Install deps**

```
cd src/qft_pcn/viz/web
./node_modules/.bin/pnpm add @monaco-editor/react monaco-editor ajv ajv-formats || \
  npm install @monaco-editor/react monaco-editor ajv ajv-formats
```

(If `pnpm` is blocked by the release-age guard, fall back to `npm install`.)

- [ ] **Step 2: Verify build still works**

```
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/vite build
```
Expected: clean build, Monaco bundled.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/viz/web/package.json \
        src/qft_pcn/viz/web/pnpm-lock.yaml 2>/dev/null \
        src/qft_pcn/viz/web/package-lock.json 2>/dev/null
git diff --staged --name-only
git commit -m "chore(viz/web): add Monaco + ajv for DSL editor"
```

---

### Task 15: store + lib clients (`dsl.ts`, `llm.ts`) + types

**Files:**
- Modify: `src/qft_pcn/viz/web/src/lib/types.ts`
- Modify: `src/qft_pcn/viz/web/src/store.ts`
- Create: `src/qft_pcn/viz/web/src/lib/dsl.ts`
- Create: `src/qft_pcn/viz/web/src/lib/llm.ts`

- [ ] **Step 1: Extend `lib/types.ts`**

Append:

```ts
export type Route = 'viz' | 'dsl';

export interface LlmModel {
  name: string;
  size?: number;
  modified_at?: string;
}

export interface ChatTurn {
  role: 'user' | 'assistant';
  text: string;
  /** Optional artifact attached to an assistant turn (e.g. emitted DSL). */
  artifact?: { kind: 'dsl' | 'run' | 'verbalize'; payload: unknown };
}

export interface DslSpec {
  fields: Array<{ name: string; cutoff: number;
                  bare_mass?: number; kinetic?: number }>;
  hamiltonian: { terms: Array<{ kind: string; species?: string;
                                 site?: number; coefficient?: number }> };
  observables: Array<{ operator: string; site?: number;
                       species?: string; target?: number | null }>;
  run?: { steps?: number; chi_max?: number; seed?: number | null };
}
```

Also extend `LAYER_KEYS`:

```ts
export const LAYER_KEYS = [
  'manifold', 'multifield', 'mps', 'hamiltonian',
  'qpcn', 'mera', 'vqc', 'logic',
  'pcn-fields', 'pcn-dynamics', 'pcn-coupling',
] as const;
```

- [ ] **Step 2: Extend `store.ts`**

Add to the state interface and initial value:

```ts
interface VizState {
  // ...existing...
  route: Route;
  chat: ChatTurn[];
  dslText: string;
  llmModel: string | null;
  steppedMode: boolean;

  setRoute: (r: Route) => void;
  appendChat: (turn: ChatTurn) => void;
  setDslText: (t: string) => void;
  setLlmModel: (m: string | null) => void;
  setSteppedMode: (v: boolean) => void;
}
```

Initial values: `route: 'viz', chat: [], dslText: '', llmModel: null, steppedMode: false`.

Setters are trivial `set({ ... })` calls.

`resetAll` should reset `chat`, `dslText`, `route` but should NOT reset `llmModel` or `steppedMode` (user preferences).

- [ ] **Step 3: Implement `lib/dsl.ts`**

```ts
// src/qft_pcn/viz/web/src/lib/dsl.ts
/**
 * Client for the /dsl/* backend endpoints.
 */

import type { DslSpec } from './types';

export async function fetchSchema(): Promise<unknown> {
  const r = await fetch('/dsl/schema');
  if (!r.ok) throw new Error(`/dsl/schema: ${r.status}`);
  return r.json();
}

export async function translate(prompt: string, model: string):
    Promise<{ dsl?: DslSpec; error?: string; raw?: string;
              validation_errors?: string[] }> {
  const r = await fetch('/dsl/translate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, model }),
  });
  if (!r.ok) throw new Error(`/dsl/translate: ${r.status}`);
  return r.json();
}

export async function runDsl(dsl: DslSpec): Promise<{ run_id: string }> {
  const r = await fetch('/dsl/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dsl }),
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(`/dsl/run: ${r.status} — ${detail}`);
  }
  return r.json();
}

export async function verbalize(observations: unknown, model: string,
                                originalPrompt: string):
    Promise<{ text: string }> {
  const r = await fetch('/dsl/verbalize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ observations, model,
                            original_prompt: originalPrompt }),
  });
  if (!r.ok) throw new Error(`/dsl/verbalize: ${r.status}`);
  return r.json();
}

export async function exportRunAsDsl(runId: string): Promise<DslSpec> {
  const r = await fetch(`/dsl/export/${runId}`);
  if (!r.ok) throw new Error(`/dsl/export: ${r.status}`);
  return r.json();
}
```

- [ ] **Step 4: Implement `lib/llm.ts`**

```ts
// src/qft_pcn/viz/web/src/lib/llm.ts
/** Client for /dsl/models with a tiny cache. */

import type { LlmModel } from './types';

let _models: LlmModel[] | null = null;

export async function loadModels(): Promise<LlmModel[]> {
  if (_models) return _models;
  const r = await fetch('/dsl/models');
  if (r.status === 503) throw new Error('Ollama unreachable');
  if (!r.ok) throw new Error(`/dsl/models: ${r.status}`);
  _models = await r.json() as LlmModel[];
  return _models!;
}

/** Default-pick the first gemma-prefixed model, else first model, else null. */
export function defaultModel(models: LlmModel[]): string | null {
  if (!models.length) return null;
  const gemma = models.find((m) => m.name.toLowerCase().startsWith('gemma'));
  return (gemma ?? models[0]).name;
}

/** Reset cache — test helper. */
export function _resetLlmCache() { _models = null; }
```

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/viz/web/src/lib/types.ts \
        src/qft_pcn/viz/web/src/store.ts \
        src/qft_pcn/viz/web/src/lib/dsl.ts \
        src/qft_pcn/viz/web/src/lib/llm.ts
git diff --staged --name-only
git commit -m "feat(viz/web): store/types/dsl/llm clients for DSL route"
```

---

### Task 16: RouteSwitcher + App.tsx mount

**Files:**
- Create: `src/qft_pcn/viz/web/src/components/RouteSwitcher.tsx`
- Create: `src/qft_pcn/viz/web/src/components/RouteSwitcher.test.tsx`
- Modify: `src/qft_pcn/viz/web/src/App.tsx`

- [ ] **Step 1: Write failing test**

```tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { RouteSwitcher } from './RouteSwitcher';
import { useVizStore } from '../store';

beforeEach(() => useVizStore.getState().resetAll());

describe('RouteSwitcher', () => {
  it('renders two route buttons', () => {
    render(<RouteSwitcher />);
    expect(screen.getByRole('button', { name: 'Viz' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'DSL' })).toBeInTheDocument();
  });
  it('clicking DSL sets route in store', () => {
    render(<RouteSwitcher />);
    fireEvent.click(screen.getByRole('button', { name: 'DSL' }));
    expect(useVizStore.getState().route).toBe('dsl');
  });
});
```

- [ ] **Step 2: Implement**

```tsx
// src/qft_pcn/viz/web/src/components/RouteSwitcher.tsx
import { useVizStore } from '../store';

export function RouteSwitcher() {
  const route = useVizStore((s) => s.route);
  const setRoute = useVizStore((s) => s.setRoute);
  return (
    <div className="route-switcher">
      {(['viz', 'dsl'] as const).map((r) => (
        <button
          key={r}
          type="button"
          className={r === route ? 'active' : ''}
          onClick={() => setRoute(r)}
        >
          {r === 'viz' ? 'Viz' : 'DSL'}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Mount in App.tsx**

```tsx
// near the top of App.tsx, after the existing imports:
import { RouteSwitcher } from './components/RouteSwitcher';
import { DslRoute } from './routes/DslRoute';
// ...

export default function App() {
  const error = useVizStore((s) => s.error);
  const route = useVizStore((s) => s.route);
  const selectedLayer = useVizStore((s) => s.selectedLayer);
  return (
    <div className="app">
      <RouteSwitcher />
      {route === 'viz' ? (
        <>
          <RunControls />
          <CompareBar />
          {error && <div className="error-bar" role="alert">{error}</div>}
          <div className="body">
            <SectionedLayerSelector />
            <PanelArea />
            <ExplainerPane layer={selectedLayer} />
          </div>
          <Timeline />
        </>
      ) : (
        <DslRoute />
      )}
    </div>
  );
}
```

- [ ] **Step 4: Add stub `DslRoute` (real implementation lands in Tasks 17–19)**

Create `src/qft_pcn/viz/web/src/routes/DslRoute.tsx`:

```tsx
// src/qft_pcn/viz/web/src/routes/DslRoute.tsx
/**
 * Three-region DSL route. Implementation lands in Tasks 17-19.
 * For now: renders a holding header so RouteSwitcher works end-to-end.
 */

export function DslRoute() {
  return (
    <div className="dsl-route">
      <header><strong>DSL Editor</strong> — loading…</header>
    </div>
  );
}
```

(This is NOT a placeholder — it's a real minimal component that the next three tasks replace. It renders cleanly and the route switch test passes.)

- [ ] **Step 5: Run tests**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run RouteSwitcher.test.tsx
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/viz/web/src/components/RouteSwitcher.tsx \
        src/qft_pcn/viz/web/src/components/RouteSwitcher.test.tsx \
        src/qft_pcn/viz/web/src/App.tsx \
        src/qft_pcn/viz/web/src/routes/DslRoute.tsx
git diff --staged --name-only
git commit -m "feat(viz/web): RouteSwitcher + DSL route mount point"
```

---

### Task 17: DslEditor (Monaco)

**Files:**
- Create: `src/qft_pcn/viz/web/src/routes/dsl/DslEditor.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/dsl/DslEditor.test.tsx`

- [ ] **Step 1: Test (uses a Monaco mock since jsdom can't run it)**

```tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useVizStore } from '../../store';

vi.mock('@monaco-editor/react', () => ({
  default: ({ value, onChange }: any) => (
    <textarea aria-label="dsl-editor" value={value}
              onChange={(e) => onChange(e.target.value)} />
  ),
}));

import { DslEditor } from './DslEditor';

beforeEach(() => useVizStore.getState().resetAll());

describe('DslEditor', () => {
  it('renders the editor textarea bound to store.dslText', () => {
    useVizStore.getState().setDslText('{"hello":1}');
    render(<DslEditor />);
    const ta = screen.getByLabelText('dsl-editor') as HTMLTextAreaElement;
    expect(ta.value).toBe('{"hello":1}');
  });

  it('typing updates store.dslText', () => {
    render(<DslEditor />);
    const ta = screen.getByLabelText('dsl-editor') as HTMLTextAreaElement;
    fireEvent.change(ta, { target: { value: '{"x":2}' } });
    expect(useVizStore.getState().dslText).toBe('{"x":2}');
  });

  it('shows validation OK when JSON parses', () => {
    useVizStore.getState().setDslText('{"fields":[]}');
    render(<DslEditor />);
    expect(screen.getByText(/JSON/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Implement**

```tsx
// src/qft_pcn/viz/web/src/routes/dsl/DslEditor.tsx
/**
 * Monaco-backed JSON editor bound to store.dslText.
 *
 * Server is authoritative for schema validation (POST /dsl/run reports
 * full validation errors); the inline status line just checks that the
 * editor content parses as JSON so the user can spot syntax errors
 * before submitting.
 */

import Editor from '@monaco-editor/react';
import { useVizStore } from '../../store';

export function DslEditor() {
  const dslText = useVizStore((s) => s.dslText);
  const setDslText = useVizStore((s) => s.setDslText);

  let parseStatus = 'empty';
  if (dslText.trim()) {
    try { JSON.parse(dslText); parseStatus = 'JSON OK'; }
    catch (e) { parseStatus = `JSON error: ${(e as Error).message}`; }
  }

  return (
    <div className="dsl-editor">
      <Editor
        height="100%"
        defaultLanguage="json"
        value={dslText}
        onChange={(v) => setDslText(v ?? '')}
        options={{ minimap: { enabled: false },
                   fontSize: 12, scrollBeyondLastLine: false }}
      />
      <div className="dsl-editor-status">{parseStatus}</div>
    </div>
  );
}
```

- [ ] **Step 3: Run + commit**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run DslEditor.test.tsx
```

```bash
git add src/qft_pcn/viz/web/src/routes/dsl/DslEditor.tsx \
        src/qft_pcn/viz/web/src/routes/dsl/DslEditor.test.tsx
git diff --staged --name-only
git commit -m "feat(viz/web): Monaco DSL editor bound to store"
```

---

### Task 18: ChatPane

**Files:**
- Create: `src/qft_pcn/viz/web/src/routes/dsl/ChatPane.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/dsl/ChatPane.test.tsx`

- [ ] **Step 1: Test**

```tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChatPane } from './ChatPane';
import { useVizStore } from '../../store';
import { _resetLlmCache } from '../../lib/llm';

beforeEach(() => {
  _resetLlmCache();
  useVizStore.getState().resetAll();
  global.fetch = vi.fn(async (url: any) => {
    const u = String(url);
    if (u.endsWith('/dsl/models'))
      return { ok: true, json: async () =>
        [{ name: 'gemma3:4b' }] } as any;
    if (u.endsWith('/dsl/translate'))
      return { ok: true, json: async () =>
        ({ dsl: { fields: [{ name: 'A', cutoff: 2 }] } }) } as any;
    throw new Error('unexpected ' + u);
  }) as any;
});

describe('ChatPane', () => {
  it('loads models into the picker', async () => {
    render(<ChatPane />);
    await waitFor(() =>
      expect(screen.getByText('gemma3:4b')).toBeInTheDocument());
  });

  it('Send appends a user turn + an assistant turn with the dsl artifact',
     async () => {
    render(<ChatPane />);
    await waitFor(() => screen.getByText('gemma3:4b'));
    fireEvent.change(screen.getByPlaceholderText(/ask the QPCN/i),
                     { target: { value: 'simulate A' } });
    fireEvent.click(screen.getByRole('button', { name: /send/i }));
    await waitFor(() => {
      expect(useVizStore.getState().chat.length).toBeGreaterThanOrEqual(2);
      expect(useVizStore.getState().dslText).toContain('"name": "A"');
    });
  });
});
```

- [ ] **Step 2: Implement**

```tsx
// src/qft_pcn/viz/web/src/routes/dsl/ChatPane.tsx
/**
 * Chat-mode driver for the DSL route. Loads Ollama models into a picker,
 * sends prompts to /dsl/translate, populates the DSL editor with the
 * returned DSL, accumulates chat turns.
 */

import { useEffect, useState } from 'react';
import { translate } from '../../lib/dsl';
import { defaultModel, loadModels } from '../../lib/llm';
import { useVizStore } from '../../store';
import type { LlmModel } from '../../lib/types';

export function ChatPane() {
  const [models, setModels] = useState<LlmModel[]>([]);
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);

  const chat = useVizStore((s) => s.chat);
  const append = useVizStore((s) => s.appendChat);
  const setDslText = useVizStore((s) => s.setDslText);
  const model = useVizStore((s) => s.llmModel);
  const setModel = useVizStore((s) => s.setLlmModel);
  const setError = useVizStore((s) => s.setError);

  useEffect(() => {
    loadModels()
      .then((ms) => {
        setModels(ms);
        if (!model) setModel(defaultModel(ms));
      })
      .catch((e) => setError(String(e)));
  }, []);

  const send = async () => {
    if (!prompt.trim() || !model) return;
    setBusy(true);
    append({ role: 'user', text: prompt });
    try {
      const result = await translate(prompt, model);
      if (result.dsl) {
        setDslText(JSON.stringify(result.dsl, null, 2));
        append({ role: 'assistant',
                 text: `Emitted DSL into the editor.`,
                 artifact: { kind: 'dsl', payload: result.dsl } });
      } else {
        append({ role: 'assistant',
                 text: `DSL validation failed: ${result.error}\n` +
                       `${(result.validation_errors ?? []).join('\n')}` });
      }
      setPrompt('');
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="chat-pane">
      <div className="chat-header">
        <label>model
          <select value={model ?? ''}
                  onChange={(e) => setModel(e.target.value || null)}>
            <option value="">(none)</option>
            {models.map((m) => (
              <option key={m.name} value={m.name}>{m.name}</option>
            ))}
          </select>
        </label>
      </div>
      <div className="chat-turns">
        {chat.map((t, i) => (
          <div key={i} className={`chat-turn chat-turn-${t.role}`}>
            <strong>{t.role}</strong>
            <pre>{t.text}</pre>
          </div>
        ))}
      </div>
      <div className="chat-input">
        <textarea placeholder="Ask the QPCN..."
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)} />
        <button type="button" onClick={send} disabled={busy || !model}>
          {busy ? 'sending…' : 'Send'}
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Run + commit**

```
cd src/qft_pcn/viz/web && ./node_modules/.bin/vitest --run ChatPane.test.tsx
```

```bash
git add src/qft_pcn/viz/web/src/routes/dsl/ChatPane.tsx \
        src/qft_pcn/viz/web/src/routes/dsl/ChatPane.test.tsx
git diff --staged --name-only
git commit -m "feat(viz/web): ChatPane — Ollama-driven DSL emission"
```

---

### Task 19: SteppedFlowBar + RunOutputPane + DslRoute composition

**Files:**
- Create: `src/qft_pcn/viz/web/src/routes/dsl/SteppedFlowBar.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/dsl/SteppedFlowBar.test.tsx`
- Create: `src/qft_pcn/viz/web/src/routes/dsl/RunOutputPane.tsx`
- Modify: `src/qft_pcn/viz/web/src/routes/DslRoute.tsx` (replace the stub from Task 16)
- Modify: `src/qft_pcn/viz/web/src/App.css` (add DSL route styles)

- [ ] **Step 1: Implement `SteppedFlowBar`**

```tsx
// src/qft_pcn/viz/web/src/routes/dsl/SteppedFlowBar.tsx
/**
 * Three explicit buttons (Generate / Run / Verbalize) with the
 * intermediate artifact shown between steps. Gated: Run requires non-empty
 * dslText; Verbalize requires a completed run with observables.
 */

import { useState } from 'react';
import { translate, runDsl, verbalize } from '../../lib/dsl';
import { useVizStore } from '../../store';

export function SteppedFlowBar() {
  const dslText = useVizStore((s) => s.dslText);
  const setDslText = useVizStore((s) => s.setDslText);
  const model = useVizStore((s) => s.llmModel);
  const appendChat = useVizStore((s) => s.appendChat);
  const setError = useVizStore((s) => s.setError);
  const activeRunId = useVizStore((s) => s.activeRunId);
  const run = useVizStore((s) =>
    activeRunId ? s.runs.get(activeRunId) : undefined);
  const [prompt, setPrompt] = useState('');

  const onGenerate = async () => {
    if (!prompt.trim() || !model) return;
    try {
      const r = await translate(prompt, model);
      if (r.dsl) setDslText(JSON.stringify(r.dsl, null, 2));
      else setError(r.error ?? 'unknown');
    } catch (e) { setError(String(e)); }
  };

  const onRun = async () => {
    try {
      const dsl = JSON.parse(dslText);
      const { run_id } = await runDsl(dsl);
      appendChat({ role: 'assistant', text: `Run started: ${run_id}` });
      // Open WS via the existing connectRun path? For stepped mode we
      // just register the run and let the user switch back to Viz route
      // to watch it stream. Document this in the chat.
    } catch (e) { setError(String(e)); }
  };

  const onVerbalize = async () => {
    if (!run || !model) return;
    const lastFrame = run.frames[run.frames.length - 1];
    const obs = lastFrame?.layer_states?.qpcn ?? {};
    try {
      const r = await verbalize(obs, model, prompt);
      appendChat({ role: 'assistant', text: r.text });
    } catch (e) { setError(String(e)); }
  };

  return (
    <div className="stepped-flow-bar">
      <input placeholder="prompt"
             value={prompt}
             onChange={(e) => setPrompt(e.target.value)} />
      <button type="button" onClick={onGenerate}
              disabled={!prompt.trim() || !model}>1. Generate DSL</button>
      <button type="button" onClick={onRun}
              disabled={!dslText.trim()}>2. Run DSL</button>
      <button type="button" onClick={onVerbalize}
              disabled={!run || !model}>3. Verbalize</button>
    </div>
  );
}
```

- [ ] **Step 2: Test the SteppedFlowBar gating**

```tsx
import '@testing-library/jest-dom/vitest';
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { SteppedFlowBar } from './SteppedFlowBar';
import { useVizStore } from '../../store';

beforeEach(() => {
  useVizStore.getState().resetAll();
  global.fetch = vi.fn() as any;
});

describe('SteppedFlowBar', () => {
  it('Generate disabled when prompt empty', () => {
    render(<SteppedFlowBar />);
    expect(screen.getByRole('button', { name: /Generate DSL/ }))
      .toBeDisabled();
  });
  it('Run disabled when dslText empty', () => {
    render(<SteppedFlowBar />);
    expect(screen.getByRole('button', { name: /Run DSL/ })).toBeDisabled();
  });
  it('Verbalize disabled when no run', () => {
    render(<SteppedFlowBar />);
    expect(screen.getByRole('button', { name: /Verbalize/ })).toBeDisabled();
  });
});
```

- [ ] **Step 3: Implement `RunOutputPane`**

```tsx
// src/qft_pcn/viz/web/src/routes/dsl/RunOutputPane.tsx
/**
 * Right pane of the DSL route. Shows the last observed frame for the
 * active run, plus a button to export the current run's RunSpec back to a
 * DSL JSON download.
 */

import { exportRunAsDsl } from '../../lib/dsl';
import { useVizStore } from '../../store';

export function RunOutputPane() {
  const activeRunId = useVizStore((s) => s.activeRunId);
  const run = useVizStore((s) =>
    activeRunId ? s.runs.get(activeRunId) : undefined);
  const setError = useVizStore((s) => s.setError);

  const exportDsl = async () => {
    if (!activeRunId) return;
    try {
      const dsl = await exportRunAsDsl(activeRunId);
      const blob = new Blob([JSON.stringify(dsl, null, 2)],
                             { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `run-${activeRunId}.dsl.json`;
      document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    } catch (e) { setError(String(e)); }
  };

  return (
    <div className="run-output-pane">
      <div className="run-output-header">
        <span>run: <code>{activeRunId ?? '—'}</code></span>
        <button type="button" disabled={!activeRunId} onClick={exportDsl}>
          Export DSL
        </button>
      </div>
      {run ? (
        <pre className="run-output-frame">
          {JSON.stringify(run.frames[run.frames.length - 1] ?? {}, null, 2)}
        </pre>
      ) : (
        <p className="empty">No active run.</p>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Replace `DslRoute` with the real composition**

```tsx
// src/qft_pcn/viz/web/src/routes/DslRoute.tsx
/**
 * Three-region DSL route: chat pane | Monaco editor | run output.
 * SteppedFlowBar overlays when steppedMode is on.
 */

import { ChatPane } from './dsl/ChatPane';
import { DslEditor } from './dsl/DslEditor';
import { RunOutputPane } from './dsl/RunOutputPane';
import { SteppedFlowBar } from './dsl/SteppedFlowBar';
import { useVizStore } from '../store';

export function DslRoute() {
  const steppedMode = useVizStore((s) => s.steppedMode);
  const setSteppedMode = useVizStore((s) => s.setSteppedMode);
  return (
    <div className="dsl-route">
      <header className="dsl-route-header">
        <label>
          <input type="checkbox" checked={steppedMode}
                 onChange={(e) => setSteppedMode(e.target.checked)} />
          stepped-flow mode
        </label>
      </header>
      {steppedMode && <SteppedFlowBar />}
      <div className="dsl-route-body">
        <ChatPane />
        <DslEditor />
        <RunOutputPane />
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Add CSS**

Append to `web/src/App.css`:

```css
.route-switcher { display: flex; gap: 4px; padding: 4px 8px; background: #0d111a;
                  border-bottom: 1px solid #1c2233; }
.route-switcher button { background: #161b29; color: #c8d0e0;
                         border: 1px solid #2a3450; border-radius: 4px;
                         padding: 2px 12px; cursor: pointer; }
.route-switcher button.active { background: #2b3a6b; }

.dsl-route { display: flex; flex-direction: column; height: 100%;
             min-height: 0; flex: 1; }
.dsl-route-header { padding: 4px 8px; background: #161b29;
                    color: #c8d0e0; font-size: 11px;
                    border-bottom: 1px solid #1c2233; }
.dsl-route-body { display: grid;
                  grid-template-columns: 280px 1fr 320px;
                  flex: 1; min-height: 0; }
.stepped-flow-bar { display: flex; gap: 6px; padding: 6px 8px;
                    background: #0a0d14; border-bottom: 1px solid #1c2233; }
.stepped-flow-bar input { flex: 1; }

.chat-pane { display: flex; flex-direction: column;
             background: #0a0d14; border-right: 1px solid #1c2233;
             color: #c8d0e0; min-height: 0; }
.chat-header { padding: 4px 8px; border-bottom: 1px solid #1c2233; font-size: 11px; }
.chat-turns { flex: 1; overflow-y: auto; padding: 4px 8px; }
.chat-turn { margin-bottom: 8px; font-size: 12px; }
.chat-turn pre { white-space: pre-wrap; }
.chat-input { display: flex; gap: 4px; padding: 4px 8px;
              border-top: 1px solid #1c2233; }
.chat-input textarea { flex: 1; height: 60px; }

.dsl-editor { display: flex; flex-direction: column; min-height: 0; }
.dsl-editor-status { padding: 4px 8px; font-size: 11px;
                     color: #9aa3bb; background: #0d111a;
                     border-top: 1px solid #1c2233; }

.run-output-pane { display: flex; flex-direction: column;
                   background: #0a0d14; border-left: 1px solid #1c2233;
                   color: #c8d0e0; min-height: 0; }
.run-output-header { display: flex; justify-content: space-between;
                     align-items: center; padding: 4px 8px;
                     border-bottom: 1px solid #1c2233; font-size: 11px; }
.run-output-header code { color: #6cd0ff; }
.run-output-frame { flex: 1; overflow: auto; padding: 4px 8px;
                    font-size: 11px; }

.intro-panel { padding: 16px; color: #c8d0e0; overflow-y: auto;
               font-size: 13px; line-height: 1.5; }
.intro-tagline { color: #8c97b3; font-style: italic; }
.intro-panel h4 { color: #8c97b3; font-size: 11px;
                  text-transform: uppercase; letter-spacing: 0.05em;
                  margin: 16px 0 4px; }
.intro-panel ul { padding-left: 20px; }

.pcn-dynamics-table { width: 100%; padding: 8px; border-collapse: collapse;
                       font-size: 12px; color: #c8d0e0; }
.pcn-dynamics-table th, .pcn-dynamics-table td {
  border-bottom: 1px solid #1c2233; padding: 4px 8px; text-align: right;
}
.pcn-dynamics-table th { color: #8c97b3; font-weight: 500; }
.layer-section { padding: 4px 0; }
.layer-section-header { display: flex; align-items: center;
                        justify-content: space-between;
                        padding: 0 8px; }
.section-title { background: none; border: none; color: #8c97b3;
                 font-size: 11px; text-transform: uppercase;
                 letter-spacing: 0.05em; cursor: pointer; padding: 2px 0; }
.section-title.active { color: #6cd0ff; }
.section-collapse-toggle { background: none; border: none; color: #8c97b3;
                            cursor: pointer; font-size: 10px; }
```

- [ ] **Step 6: Run full vitest suite + final smoke**

```
cd src/qft_pcn/viz/web
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/vitest --run
cd ../../../../
uv run pytest src/qft_pcn/tests/test_viz_*.py -v --noconftest
```

All three must pass. Note that some new tests depend on prior tasks landing (e.g. `ChatPane.test.tsx` needs `Task 15`'s store fields); run them after the relevant tasks are committed.

- [ ] **Step 7: Commit**

```bash
git add src/qft_pcn/viz/web/src/routes/dsl/SteppedFlowBar.tsx \
        src/qft_pcn/viz/web/src/routes/dsl/SteppedFlowBar.test.tsx \
        src/qft_pcn/viz/web/src/routes/dsl/RunOutputPane.tsx \
        src/qft_pcn/viz/web/src/routes/DslRoute.tsx \
        src/qft_pcn/viz/web/src/App.css
git diff --staged --name-only
git commit -m "feat(viz/web): DSL route composition — chat | editor | output + stepped-flow bar"
```

---

## Workstream finalization

### Task 20: README smoke matrix + final verification

**Files:**
- Modify: `src/qft_pcn/viz/README.md`

- [ ] **Step 1: Append to the README smoke matrix**

Append new rows / sections:

```markdown
## PCN-side smoke

| layer | fixture | preset | advanced |
|---|---|---|---|
| pcn-fields    | ✓ | ✓ | n/a |
| pcn-dynamics  | ✓ | ✓ | n/a |
| pcn-coupling  | ✓ | ✓ | n/a |

## Narrative sections

- Click each section header (QFT Substrate / PCN Substrate / QPCN Fusion);
  confirm the matching `Intro*` panel renders with the section's tagline +
  layer summaries.
- Collapse / expand each section using the ▾/▸ toggle.

## DSL route

- Switch routes via the top-bar `Viz / DSL` toggle.
- Confirm the Ollama model picker populates from `ollama list`.
- Chat mode: enter a prompt → DSL appears in the editor → click Run in the
  right pane → frames stream → click Verbalize → assistant turn appears.
- Stepped mode: toggle "stepped-flow mode" → use the three numbered
  buttons in order; confirm each is gated on the previous step.
- Import: paste a DSL JSON into the editor → click Run.
- Export: pin a run as baseline, click "Export DSL" in the right pane,
  confirm a `.dsl.json` downloads with the expected structure.
```

- [ ] **Step 2: Final full-suite verification**

```
cd /mnt/f/experiments/anything-to-everything
uv run pytest src/qft_pcn/tests/test_viz_*.py -v --noconftest
cd src/qft_pcn/viz/web
./node_modules/.bin/tsc --noEmit
./node_modules/.bin/vite build
./node_modules/.bin/vitest --run
```

All four must be green. Report counts.

- [ ] **Step 3: Backend smoke via TestClient**

```bash
uv run python -c "
from fastapi.testclient import TestClient
from src.qft_pcn.viz.server import app
c = TestClient(app)
print('health', c.get('/health').status_code)
print('presets', len(c.get('/presets').json()))
print('dsl/schema', c.get('/dsl/schema').status_code)
r = c.post('/run', json={'layers': ['pcn-fields','pcn-dynamics','pcn-coupling'],
                          'steps': 2, 'grid': 8}).json()
print('pcn run', r['run_id'])
"
```

Expect health=200, presets >= 17, dsl/schema=200, pcn run id present.

- [ ] **Step 4: Commit any doc-only fixes**

Only if Step 2 or 3 surfaces real inaccuracies in the README — no empty commits.

```bash
git add src/qft_pcn/viz/README.md
git diff --staged --name-only
git commit -m "docs(viz): smoke matrix for PCN panels + narrative + DSL route"
```

---

## Self-Review Notes

- **Spec coverage:** §4.1 PCN panels ↔ Tasks 1–3, 11–13. §4.2 Narrative ↔ Tasks 8–10. §4.3 DSL/LLM ↔ Tasks 4–7, 14–19. §7 EXTENSIONS ↔ Task 7. §9 testing ↔ tests in every task. §10 risks (Monaco bundle size, Ollama unreachable, reasoning model formatting, DSL evolution) all handled.
- **Placeholder scan:** Every step has concrete code/commands. The `DslRoute` stub in Task 16 is explicitly noted as a real minimal component replaced in Task 19; that's a sequencing artifact, not a placeholder. No "TBD" / "implement later" / "similar to" instructions.
- **Type consistency:** `Frame`, `RunSpec`, `DslSpec`, `LlmModel`, `ChatTurn`, `Route`, `SectionSpec` are defined once in `types.ts` / `sections.ts` and referenced consistently. Store actions (`appendChat`, `setDslText`, `setLlmModel`, `setSteppedMode`, `setRoute`) match their consumers. Snapshot field names (`per_layer_free_energy`, `mean_abs_stress_energy`, etc.) match between `snapshots.py` and the panels.
- **Substrate isolation:** No task edits anything outside `src/qft_pcn/viz/**`, `src/qft_pcn/tests/test_viz_*.py`, `pyproject.toml`, `web/package.json`, or `web/vite.config.ts`.
- **No-placeholders rule:** Three EXTENSIONS entries added (KL divergence, lossless DSL round-trip, LLM retry loop). All affordances either implemented or omitted gracefully.
