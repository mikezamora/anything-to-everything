"""Preset catalog tests: shape, uniqueness, and round-trip via run_simulation."""

import pytest

from src.qft_pcn.viz.presets import PRESETS, PARAM_SCHEMA, Preset
from src.qft_pcn.viz.runs import RunSpec, run_simulation


def test_catalog_is_nonempty_and_ids_unique():
    assert len(PRESETS) > 0
    ids = [p.id for p in PRESETS]
    assert len(ids) == len(set(ids))


def test_every_preset_has_required_fields():
    for p in PRESETS:
        assert isinstance(p, Preset)
        assert p.id and p.layer and p.label and p.description
        assert isinstance(p.spec_overrides, dict)
        # spec_overrides may set layers/steps/grid/seed/params
        for key in p.spec_overrides:
            assert key in {"layers", "steps", "grid", "seed", "params"}


def test_param_schema_has_entry_per_live_layer():
    for layer in ("manifold", "multifield", "mps", "qpcn",
                  "hamiltonian", "mera"):
        assert layer in PARAM_SCHEMA
        entry = PARAM_SCHEMA[layer]
        assert entry["type"] == "object"
        assert "properties" in entry


@pytest.mark.parametrize("preset", PRESETS, ids=lambda p: p.id)
def test_preset_produces_a_nonempty_frame_for_its_layer(preset):
    spec_dict = {"layers": [preset.layer], "steps": 2, "grid": 8,
                 **preset.spec_overrides}
    spec = RunSpec.from_dict(spec_dict)
    frames = list(run_simulation(spec))
    assert frames, f"{preset.id} produced zero frames"
    assert frames[0].layer_states.get(preset.layer), (
        f"{preset.id} produced empty {preset.layer} snapshot")
