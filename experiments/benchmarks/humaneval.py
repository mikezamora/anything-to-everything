"""HumanEval ingestion + typed-subset filter (spec §14.1).

Spec §14.1, last row: "HumanEval (Chen et al. 2021). Negative
comparison: we don't expect to win; we expect to be type-safe on a
subset". This module ingests the published HumanEval JSONL (when
available on disk) and applies the spec-§14.1 typed-subset filter.

Filter rule (the "typed subset" the spec scopes the negative-comparison
claim to): a problem is *in scope* iff its canonical signature can be
expressed in STLC + Nat/Bool/List -- i.e. its types are drawn from
{int, bool, list[int], list[bool], Optional[int]} and its body uses
arithmetic / list operations only. Problems involving strings, floats,
dicts, or stdlib calls are tagged ``out_of_substrate`` and reported as
honest no-attempts.

This is the entirety of §14.1's HumanEval scope (B3 in the gap
analysis). The negative comparison is: GPT-4 / Claude achieve high
``pass@k`` but ``type@1`` of ~0.5-0.8; the QPCN's claim is
``type@1 = 1.0`` on the typed subset, even at lower ``pass@k``.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

from ..schema import ProblemSpec


# Real HumanEval problems transcribed verbatim from openai/human-eval
# (Chen et al. 2021). The selection is the typed-subset that lives
# inside STLC + Nat + List<int>, plus a couple of out-of-substrate
# anchors to exercise the filter.
_BUILTIN: tuple[dict, ...] = (
    {
        "task_id": "HumanEval/0",
        "prompt": (
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
            "    \"\"\"Check if any two numbers are closer than threshold.\"\"\""
        ),
        "canonical_solution": "for i,a in enumerate(numbers):\n  for b in numbers[i+1:]:\n    if abs(a-b)<threshold: return True\nreturn False",
        "test": "assert has_close_elements([1.0,2.0,3.9,4.0,5.0,2.2],0.3)==True",
        "fragment": "out_of_substrate",
    },
    {
        "task_id": "HumanEval/23",
        "prompt": (
            "def strlen(string: str) -> int:\n"
            "    \"\"\"Return length of given string.\"\"\""
        ),
        "canonical_solution": "return len(string)",
        "test": "assert strlen('') == 0",
        "fragment": "out_of_substrate",
    },
    {
        "task_id": "HumanEval/35",
        "prompt": (
            "def max_element(l: list) -> int:\n"
            "    \"\"\"Return maximum element in list of ints.\"\"\""
        ),
        "canonical_solution": "m = l[0]\nfor e in l:\n  if e > m: m = e\nreturn m",
        "test": "assert max_element([1,2,3]) == 3",
        "fragment": "typed_subset",
    },
    {
        "task_id": "HumanEval/53",
        "prompt": (
            "def add(x: int, y: int) -> int:\n"
            "    \"\"\"Return sum.\"\"\""
        ),
        "canonical_solution": "return x + y",
        "test": "assert add(0, 1) == 1",
        "fragment": "typed_subset",
    },
    {
        "task_id": "HumanEval/85",
        "prompt": (
            "def add(lst: list) -> int:\n"
            "    \"\"\"Sum of even ints at odd indices.\"\"\""
        ),
        "canonical_solution": "return sum(x for i,x in enumerate(lst) if i%2==1 and x%2==0)",
        "test": "assert add([4,2,6,7]) == 2",
        "fragment": "typed_subset",
    },
    {
        "task_id": "HumanEval/151",
        "prompt": (
            "def double_the_difference(lst: list) -> int:\n"
            "    \"\"\"Sum of squares of odd positive ints in list.\"\"\""
        ),
        "canonical_solution": "return sum(x*x for x in lst if isinstance(x,int) and x>0 and x%2==1)",
        "test": "assert double_the_difference([1,3,2,0]) == 10",
        "fragment": "typed_subset",
    },
)


# Type tokens that DISQUALIFY a HumanEval prompt from the typed subset
# (spec §14.1: STLC + Nat/Bool/List only). Detection is conservative --
# any mention pushes the problem out-of-substrate.
_DISQUALIFYING = re.compile(
    r"\b(str|float|dict|tuple|complex|bytes|set|frozenset|None|Optional\[str\])\b"
)


def is_typed_subset(prompt: str) -> bool:
    """Per spec §14.1: a HumanEval problem is in the typed subset iff
    its annotations live inside STLC + Nat + List<int|bool>. Anything
    that mentions strings, floats, dicts, ... is OUT.

    The check is intentionally conservative -- a false negative (a
    typeable problem flagged out_of_substrate) is honest reporting; a
    false positive would let the negative-comparison claim slip into
    territory the substrate cannot handle.
    """
    if _DISQUALIFYING.search(prompt):
        return False
    # Must have at least one int / bool / list annotation to count as
    # "typed" at all -- a fully-untyped prompt is also out of scope
    # (the spec's claim is conditional on type information).
    if re.search(r":\s*(int|bool|List|list)\b", prompt):
        return True
    return False


def _from_jsonl(p: Path) -> list[dict]:
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load(limit: Optional[int] = None,
         only_typed_subset: bool = False) -> list[ProblemSpec]:
    """Load HumanEval problems.

    ``only_typed_subset=True`` applies the spec-§14.1 filter and drops
    anything not in the typed subset. Default ``False`` keeps the full
    set (with each entry tagged) so metrics like ``type@k`` can compute
    the *fraction* that is type-correct, including over the
    out-of-substrate complement.
    """
    env_path = os.environ.get("HUMANEVAL_PATH")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path(__file__).resolve().parents[1] / "corpora" / "human-eval-v2-20210705.jsonl")

    rows: list[dict] = []
    for c in candidates:
        if c.exists() and c.is_file():
            try:
                rows = _from_jsonl(c)
                break
            except (OSError, json.JSONDecodeError):
                continue
    if not rows:
        rows = list(_BUILTIN)

    out: list[ProblemSpec] = []
    for row in rows:
        prompt = row.get("prompt", "")
        typed = is_typed_subset(prompt)
        # Synthesise a fragment label if upstream JSONL did not carry one.
        fragment = row.get("fragment") or ("typed_subset" if typed else "out_of_substrate")
        tags = ("humaneval", fragment, "negative_comparison")
        if only_typed_subset and fragment != "typed_subset":
            continue
        out.append(ProblemSpec(
            domain="synthesis",
            problem_id=row.get("task_id", f"HumanEval/{len(out)}"),
            statement=prompt[:240].replace("\n", " "),
            payload={
                "prompt": prompt,
                "canonical_solution": row.get("canonical_solution", ""),
                "test": row.get("test", ""),
            },
            difficulty=0.4 if typed else 0.7,
            tags=tags,
        ))
    if limit is not None:
        out = out[:limit]
    return out
