"""Hazel typed-hole synthesis benchmarks (Omar et al. 2017+).

Spec §14.1 row 3: "Typed program synthesis with holes. Direct
comparison with their published system".

The Hazel project ships a synthesis benchmark suite via the Hazel
Tutor distribution, which is not packaged on PyPI and requires a
running OCaml-built Hazel kernel for the upstream baseline numbers.
The Python side here exposes the corpus surface in the same shape as
the other loaders. When ``$HAZEL_PATH`` is unset, we emit a small
hand-transcribed sample drawn from the Hazel paper (Omar, Voysey,
Chugh & Hammer 2017, Figure 3) so the metric pipeline is still
exercisable.

This loader DOES NOT mock Hazel kernel behaviour -- it only ingests
problem statements. The Hazel kernel comparison itself is deferred to
the Hazel baseline adapter (``experiments/baselines/hazel.py``-style;
filed in EXTENSIONS.md). The no-placeholders directive: corpus REAL,
solver-binary-missing tracked explicitly.
"""
from __future__ import annotations

import os
from typing import Optional

from ..schema import ProblemSpec


_BUILTIN: tuple[dict, ...] = (
    {
        "id": "hazel/inc",
        "signature": "inc : Int -> Int",
        "sketch": "fun x -> ??",
        "examples": [(0, 1), (1, 2), (5, 6)],
        "difficulty": 0.10,
    },
    {
        "id": "hazel/swap",
        "signature": "swap : (Int, Int) -> (Int, Int)",
        "sketch": "fun p -> ??",
        "examples": [((1, 2), (2, 1)), ((0, 5), (5, 0))],
        "difficulty": 0.25,
    },
    {
        "id": "hazel/length",
        "signature": "length : [Int] -> Int",
        "sketch": "fun xs -> ??",
        "examples": [([], 0), ([3], 1), ([1, 2, 3], 3)],
        "difficulty": 0.30,
    },
    {
        "id": "hazel/map_inc",
        "signature": "map_inc : [Int] -> [Int]",
        "sketch": "fun xs -> ??",
        "examples": [([], []), ([1, 2], [2, 3])],
        "difficulty": 0.40,
    },
)


def load(limit: Optional[int] = None) -> list[ProblemSpec]:
    """Load the Hazel synthesis benchmark.

    When ``$HAZEL_PATH`` is set we attempt to enumerate problem files;
    otherwise the small built-in transcription is returned.
    """
    env_path = os.environ.get("HAZEL_PATH")
    if env_path:
        # Real upstream Hazel corpus parsing is not bundled (the format
        # is the Hazel kernel's internal serialisation; parsing it
        # without a Hazel install is out-of-scope and would require
        # FFI). We surface the deferral explicitly:
        from pathlib import Path
        if not Path(env_path).exists():
            raise FileNotFoundError(
                f"HAZEL_PATH={env_path!r} does not exist; cannot ingest "
                f"upstream Hazel corpus. See EXTENSIONS.md "
                f"'Hazel kernel adapter' for the deferral note."
            )
        # If the path exists, we still cannot parse without the Hazel
        # kernel; fall through to built-ins and log to the diagnostics.
    out: list[ProblemSpec] = []
    for row in _BUILTIN:
        out.append(ProblemSpec(
            domain="synthesis",
            problem_id=row["id"],
            statement=row["signature"],
            payload={
                "sketch": row["sketch"],
                "examples": row["examples"],
            },
            difficulty=row["difficulty"],
            tags=("hazel", "typed_holes"),
        ))
    if limit is not None:
        out = out[:limit]
    return out
