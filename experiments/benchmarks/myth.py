"""Myth synthesis benchmark (Osera & Zdancewic 2015).

Spec §14.1 row 4: "Small STLC synthesis from examples. Established
small-scale benchmark." Myth's input format is a typed sketch + a list
of input-output examples. The QPCN's STLC synthesis pipeline (P1..P8 in
``logic/demo_stlc_synthesis.py``) is the natural runner for it.

This loader returns one ProblemSpec per Myth task. Where the upstream
Myth corpus (https://github.com/peterzakin/myth) is not on disk, we
emit the eight built-in P1..P8 problems that the QPCN already exercises
as its STLC milestone (E sub-project) -- the payload references the
``BUILDERS`` map by name so the runner constructs the SynthesisProblem
on the fly.
"""
from __future__ import annotations

from typing import Optional

from ..schema import ProblemSpec


# The eight P-problems live in src/qft_pcn/logic/demo_stlc_synthesis.py.
# We re-expose them here as benchmark problems so the runner can drive
# the SAME pipeline used by test_synthesis_acceptance.py.
_P_DIFFICULTIES = {
    "P1": 0.10,   # identity
    "P2": 0.15,   # identity-via-example
    "P3": 0.40,   # f x (higher-order)
    "P4": 0.55,   # if x<n then x else x+k
    "P5": 0.15,   # K combinator
    "P6": 0.45,   # f (g x)
    "P7": 0.40,   # x < y
    "P8": 0.90,   # negative example (correct refusal)
}


def load(limit: Optional[int] = None) -> list[ProblemSpec]:
    out: list[ProblemSpec] = []
    for name, diff in _P_DIFFICULTIES.items():
        # The spec §14.1 tag "myth" is what the runner branches on to
        # pick the STLC synthesis driver (vs. the proof driver).
        out.append(ProblemSpec(
            domain="synthesis",
            problem_id=f"myth/{name}",
            statement=f"Myth-style STLC synthesis problem {name}",
            payload={"builder_name": name},
            difficulty=diff,
            tags=("myth", "stlc"),
        ))
    if limit is not None:
        out = out[:limit]
    return out
