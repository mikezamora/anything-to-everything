"""Curated presets and per-layer param schema for the viz UI.

A `Preset` is a frozen record whose `spec_overrides` dict is merged into the
JSON body of `POST /run`. Presets only set values that `RunSpec.from_dict`
already accepts; substrate code is never touched from here.

`PARAM_SCHEMA` is the JSON Schema served by `GET /params/schema` and consumed
by the frontend Advanced expander. It is hand-authored and kept in sync with
`runs.py` substrate builders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Preset:
    """A named starting condition for a single layer."""

    id: str
    layer: str
    label: str
    description: str
    spec_overrides: dict[str, Any] = field(default_factory=dict)


PRESETS: list[Preset] = [
    # ---- manifold ----------------------------------------------------------
    Preset(
        id="manifold.flat",
        layer="manifold",
        label="Flat — zero curvature start",
        description="Uniform metric, no curvature sources. Baseline for "
                    "watching how prediction error alone sources geometry.",
        spec_overrides={"steps": 30, "grid": 12, "seed": 0,
                        "params": {"manifold": {"source": "flat"}}},
    ),
    Preset(
        id="manifold.hot-spot",
        layer="manifold",
        label="Hot spot — central Gaussian source",
        description="A single Gaussian bump in the error field, so curvature "
                    "concentrates at the centre as the run progresses.",
        spec_overrides={"steps": 40, "grid": 14, "seed": 1,
                        "params": {"manifold": {"source": "hot-spot"}}},
    ),
    Preset(
        id="manifold.two-source",
        layer="manifold",
        label="Two-source — competing curvature wells",
        description="Two offset Gaussian sources. Watch the curvature ridge "
                    "between them rise as both wells deepen.",
        spec_overrides={"steps": 40, "grid": 14, "seed": 2,
                        "params": {"manifold": {"source": "two-source"}}},
    ),
    # ---- multifield --------------------------------------------------------
    Preset(
        id="multifield.uncoupled",
        layer="multifield",
        label="Uncoupled — g_ab pinned at 0",
        description="Two fields, coupling frozen at zero. Baseline run.",
        spec_overrides={"steps": 30, "grid": 12, "seed": 0,
                        "params": {"multifield": {"learn_coupling": False,
                                                  "initial_coupling": 0.0}}},
    ),
    Preset(
        id="multifield.symmetric-coupling",
        layer="multifield",
        label="Symmetric coupling — g_ab = 0.5",
        description="Two fields with a fixed symmetric Yukawa coupling.",
        spec_overrides={"steps": 30, "grid": 12, "seed": 0,
                        "params": {"multifield": {"learn_coupling": False,
                                                  "initial_coupling": 0.5}}},
    ),
    Preset(
        id="multifield.learn-coupling",
        layer="multifield",
        label="Learn coupling — g_ab descends",
        description="Coupling is learnable; watch g_ab adapt to the joint "
                    "free energy.",
        spec_overrides={"steps": 50, "grid": 12, "seed": 0,
                        "params": {"multifield": {"learn_coupling": True,
                                                  "initial_coupling": 0.0}}},
    ),
    # ---- qpcn / mps / hamiltonian -----------------------------------------
    Preset(
        id="qpcn.ground-state-relax",
        layer="qpcn",
        label="Ground-state relax — no observation target",
        description="Imaginary-time relaxation with the default Hamiltonian.",
        spec_overrides={"steps": 40, "seed": 0,
                        "params": {"qpcn": {"target_n0": None,
                                            "mass": 1.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="qpcn.quarter-density-target",
        layer="qpcn",
        label="Target ⟨n₀⟩ = 0.25",
        description="Observable-driven update toward a quarter-occupation "
                    "target on site 0.",
        spec_overrides={"steps": 50, "seed": 0,
                        "params": {"qpcn": {"target_n0": 0.25,
                                            "mass": 1.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="qpcn.high-mass",
        layer="qpcn",
        label="High-mass regime (m = 3.0)",
        description="Heavy bare mass; ground state should localise more "
                    "tightly than the default.",
        spec_overrides={"steps": 40, "seed": 0,
                        "params": {"qpcn": {"target_n0": 0.25,
                                            "mass": 3.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="mps.product-state",
        layer="mps",
        label="Product state — bond-dim 1",
        description="MPS warmup starting from a product state; bonds grow as "
                    "entanglement accumulates.",
        spec_overrides={"steps": 30, "seed": 0,
                        "params": {"qpcn": {"chi_max": 8,
                                            "mass": 1.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="mps.entangled-warmup",
        layer="mps",
        label="Entangled warmup — random bond-2",
        description="Random superposition seed (handled by qpcn builder); "
                    "watch entropy plateau as relaxation kicks in.",
        spec_overrides={"steps": 30, "seed": 7,
                        "params": {"qpcn": {"chi_max": 12,
                                            "mass": 1.0, "kinetic": 0.5}}},
    ),
    Preset(
        id="hamiltonian.single-species",
        layer="hamiltonian",
        label="Single species (A only)",
        description="Default single-species Hamiltonian. Watch species_dims "
                    "and the curvature mini-map.",
        spec_overrides={"steps": 10, "seed": 0, "params": {"qpcn": {}}},
    ),
    Preset(
        id="hamiltonian.two-species",
        layer="hamiltonian",
        label="Two species (A + B)",
        description="Two-species Hamiltonian to exercise the species legend "
                    "and cross-species curvature.",
        spec_overrides={"steps": 10, "seed": 0,
                        "params": {"qpcn": {"species": ["A", "B"]}}},
    ),
    # ---- vqc --------------------------------------------------------------
    Preset(
        id="vqc.parameter-shift-train",
        layer="vqc",
        label="Parameter-shift training toward ⟨Z⟩ = 0.5",
        description="Train a small VQC via parameter-shift gradients toward "
                    "a fixed Z-target; watch theta evolve and Bloch spheres "
                    "rotate.",
        spec_overrides={"steps": 20, "seed": 0,
                        "params": {"vqc": {"n_qubits": 3, "n_layers": 2}}},
    ),
    # ---- logic ------------------------------------------------------------
    Preset(
        id="logic.beta-reduce",
        layer="logic",
        label="β-reduce — superposition relaxation",
        description="Imaginary-time relaxation of a logic-encoded MPS under "
                    "EvalHamiltonian; watch per-term residuals decay.",
        spec_overrides={"steps": 30, "seed": 0,
                        "params": {"logic": {"N": 6, "chi_max": 8}}},
    ),
    # ---- mera -------------------------------------------------------------
    Preset(
        id="mera.vacuum-small",
        layer="mera",
        label="Vacuum MERA (4 leaves)",
        description="Static vacuum MERA on 4 leaves. Useful for inspecting "
                    "the tree layout and per-cut entropy.",
        spec_overrides={"steps": 5, "seed": 0,
                        "params": {"mera": {"leaves": 4, "chi_layer": 4}}},
    ),
]


# Hand-authored JSON Schema; each property's `default` mirrors what the
# corresponding `_build_*` function in runs.py uses when the key is absent.
PARAM_SCHEMA: dict[str, dict] = {
    "manifold": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "enum": ["flat", "hot-spot", "two-source"],
                "default": "flat",
                "description": "Initial error-field source pattern.",
            },
        },
    },
    "multifield": {
        "type": "object",
        "properties": {
            "learn_coupling": {"type": "boolean", "default": True,
                               "description": "Adapt g_ab via gradient descent."},
            "initial_coupling": {"type": "number", "default": 0.0,
                                 "minimum": -1.0, "maximum": 1.0,
                                 "description": "Starting value of g_ab."},
        },
    },
    "mps": {
        "type": "object",
        "properties": {},  # mps shares the qpcn substrate; tuned via qpcn
    },
    "qpcn": {
        "type": "object",
        "properties": {
            "mass": {"type": "number", "default": 1.0,
                     "minimum": 0.0, "maximum": 10.0,
                     "description": "Bare mass of species A."},
            "kinetic": {"type": "number", "default": 0.5,
                        "minimum": 0.0, "maximum": 5.0,
                        "description": "Kinetic coefficient."},
            "chi_max": {"type": "integer", "default": 8,
                        "minimum": 1, "maximum": 16,
                        "description": "MPS bond-dimension cap."},
            "target_n0": {"type": ["number", "null"], "default": 0.25,
                          "minimum": 0.0, "maximum": 1.0,
                          "description": "Target ⟨n⟩ on site 0; null disables."},
            "species": {"type": "array", "items": {"type": "string"},
                        "default": ["A"],
                        "description": "List of species names."},
        },
    },
    "hamiltonian": {
        "type": "object",
        "properties": {},  # hamiltonian shares qpcn config
    },
    "mera": {
        "type": "object",
        "properties": {
            "leaves": {"type": "integer", "default": 4,
                       "enum": [2, 4, 8],
                       "description": "Number of leaf sites (power of two)."},
            "chi_layer": {"type": "integer", "default": 4,
                          "minimum": 2, "maximum": 16,
                          "description": "Per-layer bond dimension."},
        },
    },
    "vqc": {
        "type": "object",
        "properties": {
            "n_qubits": {"type": "integer", "default": 3,
                         "minimum": 1, "maximum": 4,
                         "description": "Number of qubits in the VQC."},
            "n_layers": {"type": "integer", "default": 2,
                         "minimum": 1, "maximum": 4,
                         "description": "Number of variational layers."},
        },
    },
    "logic": {
        "type": "object",
        "properties": {
            "N": {"type": "integer", "default": 6,
                  "minimum": 2, "maximum": 16,
                  "description": "Number of MPS sites for the logic chain."},
            "chi_max": {"type": "integer", "default": 8,
                        "minimum": 1, "maximum": 16,
                        "description": "MPS bond-dimension cap for relaxation."},
        },
    },
}
