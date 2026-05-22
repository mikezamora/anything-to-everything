"""Frame recorder.

A `Recorder` accumulates `Frame`s as a simulation runs. Each `capture_*`
method delegates to the matching `snapshot_*` extractor, wraps the result in a
`Frame` under the canonical layer key, and appends it. The recorder owns an
internal step counter that increments once per capture — recording is purely
additive and never touches the substrate objects.
"""

from __future__ import annotations

import json

from .schema import Frame, _VizJSONEncoder
from . import snapshots


class Recorder:
    """Accumulates per-step `Frame`s from live substrate objects."""

    def __init__(self) -> None:
        self.frames: list[Frame] = []
        self._step = 0

    # ---- internal ------------------------------------------------------------

    def _append(self, layer_states: dict[str, dict]) -> Frame:
        frame = Frame(step=self._step, layer_states=layer_states)
        self.frames.append(frame)
        self._step += 1
        return frame

    # ---- per-substrate capture ----------------------------------------------

    def capture_network(self, net) -> Frame:
        """Capture a `QFTPCNNetwork` snapshot (manifold key)."""
        return self._append({"manifold": snapshots.snapshot_network(net)})

    def capture_qpcn(self, q) -> Frame:
        """Capture a `QPCN` snapshot."""
        return self._append({"qpcn": snapshots.snapshot_qpcn(q)})

    def capture_mps(self, mps) -> Frame:
        """Capture an `MPS` snapshot."""
        return self._append({"mps": snapshots.snapshot_mps(mps)})

    def capture_multifield(self, mf) -> Frame:
        """Capture a `MultiFieldNetwork` snapshot."""
        return self._append({"multifield": snapshots.snapshot_multifield(mf)})

    def capture_hamiltonian(self, H) -> Frame:
        """Capture a `Hamiltonian` snapshot."""
        return self._append({"hamiltonian": snapshots.snapshot_hamiltonian(H)})

    def capture_mera(self, m) -> Frame:
        """Capture a `MERA` snapshot."""
        return self._append({"mera": snapshots.snapshot_mera(m)})

    def capture_vqc(self, vqc) -> Frame:
        """Capture a variational-quantum-circuit snapshot."""
        return self._append({"vqc": snapshots.snapshot_vqc(vqc)})

    def capture_logic(self, enc) -> Frame:
        """Capture a logic-encoder snapshot."""
        return self._append({"logic": snapshots.snapshot_logic(enc)})

    # ---- generic -------------------------------------------------------------

    def capture(self, **layer_snaps: dict) -> Frame:
        """Merge several already-built named layer snapshots into one Frame.

        Example: ``rec.capture(manifold=snapshot_network(net),
        qpcn=snapshot_qpcn(q))``.
        """
        return self._append(dict(layer_snaps))

    # ---- management ----------------------------------------------------------

    def clear(self) -> None:
        """Drop all recorded frames and reset the step counter."""
        self.frames = []
        self._step = 0

    def to_json_list(self) -> str:
        """Serialize all frames to a single JSON array string."""
        return json.dumps(
            [f.to_dict() for f in self.frames],
            cls=_VizJSONEncoder,
        )
