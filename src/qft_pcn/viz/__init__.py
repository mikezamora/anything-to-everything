"""Non-invasive layer-visualization instrumentation for QFT-PCN.

This package reads public attributes of existing substrate objects via free
functions and records them as JSON-ready `Frame`s. It modifies nothing in the
simulation: a recorded run behaves identically to an unrecorded one.
"""

from __future__ import annotations

from .schema import Frame, LAYER_KEYS
from .recorder import Recorder

__all__ = ["Frame", "Recorder", "LAYER_KEYS"]
