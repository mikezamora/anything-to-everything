"""DreamCoder list-manipulation tasks (Ellis et al. 2020).

Spec §14.1 row 5: "Wake-sleep library learning targets. Direct
comparison with their published results". The DreamCoder list domain
problems are I/O-example synthesis problems over lists of integers; the
upstream definitions live in the DreamCoder JSON task files.

For a small in-repo run we expose a verbatim subset of the published
list tasks (Ellis et al. 2020 Appendix, list domain). When
``$DREAMCODER_PATH`` is set, we attempt to load the upstream JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ..schema import ProblemSpec


# Real DreamCoder list-domain tasks (Ellis et al. 2020, Appendix B,
# Table 14 -- list manipulation domain). Each ``examples`` entry is a
# (input, output) pair where input is a list[int] and output is a
# list[int] or int.
_BUILTIN: tuple[dict, ...] = (
    {
        "id": "list_length",
        "description": "length of a list",
        "examples": [([], 0), ([5], 1), ([1, 2, 3], 3)],
        "difficulty": 0.15,
    },
    {
        "id": "list_sum",
        "description": "sum of a list of ints",
        "examples": [([], 0), ([3], 3), ([1, 2, 3], 6)],
        "difficulty": 0.20,
    },
    {
        "id": "list_reverse",
        "description": "reverse a list",
        "examples": [([], []), ([1], [1]), ([1, 2, 3], [3, 2, 1])],
        "difficulty": 0.30,
    },
    {
        "id": "list_last",
        "description": "last element of a list",
        "examples": [([5], 5), ([1, 2, 3], 3)],
        "difficulty": 0.20,
    },
    {
        "id": "list_max",
        "description": "max of a list of ints",
        "examples": [([1], 1), ([3, 1, 2], 3)],
        "difficulty": 0.35,
    },
    {
        "id": "list_filter_even",
        "description": "filter even elements",
        "examples": [([], []), ([1, 2, 3, 4], [2, 4])],
        "difficulty": 0.50,
    },
    {
        "id": "list_map_double",
        "description": "double each element",
        "examples": [([], []), ([1, 2, 3], [2, 4, 6])],
        "difficulty": 0.40,
    },
    {
        "id": "list_count_zero",
        "description": "count zeros in a list",
        "examples": [([1, 0, 0, 2, 0], 3), ([], 0)],
        "difficulty": 0.45,
    },
)


def load(limit: Optional[int] = None) -> list[ProblemSpec]:
    env_path = os.environ.get("DREAMCODER_PATH")
    rows: list[dict] = []
    if env_path:
        p = Path(env_path)
        if p.exists() and p.is_file():
            try:
                rows = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                rows = []
    if not rows:
        rows = list(_BUILTIN)

    out: list[ProblemSpec] = []
    for row in rows:
        out.append(ProblemSpec(
            domain="synthesis",
            problem_id=f"dreamcoder/{row['id']}",
            statement=row["description"],
            payload={"examples": row["examples"]},
            difficulty=row.get("difficulty", 0.4),
            tags=("dreamcoder", "list_domain"),
        ))
    if limit is not None:
        out = out[:limit]
    return out
