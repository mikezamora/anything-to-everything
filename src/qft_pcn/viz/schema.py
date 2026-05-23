"""Frame schema for viz instrumentation.

A `Frame` is one captured timestep: a step index plus a mapping from layer
name to a plain dict of JSON-ready diagnostic values. Serialization handles
NumPy arrays/scalars and complex numbers so snapshot extractors can hand back
raw NumPy without manual conversion.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

# Canonical per-layer keys used throughout the viz package. Recorder capture
# methods and the server contract reference these names.
LAYER_KEYS = (
    "manifold",
    "multifield",
    "mps",
    "hamiltonian",
    "qpcn",
    "mera",
    "vqc",
    "logic",
    "mera_relax",
    "bridge",
    "pcn-fields",
    "pcn-dynamics",
    "pcn-coupling",
)


class _VizJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy and complex values."""

    def default(self, obj):  # noqa: D102
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, complex):
            return {"re": obj.real, "im": obj.imag}
        return super().default(obj)


@dataclass
class Frame:
    """A single captured simulation step."""

    step: int
    layer_states: dict[str, dict] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to a JSON string (arrays -> lists, complex -> {re,im})."""
        return json.dumps(
            {"step": self.step, "layer_states": self.layer_states},
            cls=_VizJSONEncoder,
        )

    @classmethod
    def from_json(cls, blob: str) -> "Frame":
        """Deserialize from a JSON string. Arrays stay plain lists."""
        data = json.loads(blob)
        return cls(
            step=data["step"],
            layer_states=data.get("layer_states", {}),
        )

    def to_dict(self) -> dict:
        """Plain-dict form (still possibly containing NumPy values)."""
        return {"step": self.step, "layer_states": self.layer_states}
