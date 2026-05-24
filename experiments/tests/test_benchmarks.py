"""Smoke + contract tests for each benchmark loader."""
from __future__ import annotations

import pytest

from experiments.benchmarks import (
    load_dreamcoder, load_hazel, load_humaneval, load_minif2f, load_myth,
    load_qm9,
)
from experiments.benchmarks.humaneval import is_typed_subset


@pytest.mark.parametrize("loader", [
    load_minif2f, load_humaneval, load_myth, load_dreamcoder, load_hazel,
    load_qm9,
])
def test_loader_returns_non_empty(loader):
    out = loader(limit=3)
    assert len(out) >= 1, f"{loader.__name__} returned no problems"
    for p in out:
        assert p.problem_id
        assert p.statement
        assert p.domain in {"proof", "synthesis", "chemistry"}
        assert 0.0 <= p.difficulty <= 1.0


def test_humaneval_typed_subset_filter():
    """Spec §14.1: typed subset must exclude string / float / dict types."""
    assert is_typed_subset("def f(x: int) -> int: ...")
    assert is_typed_subset("def f(l: list) -> bool: ...")
    assert not is_typed_subset("def f(s: str) -> str: ...")
    assert not is_typed_subset("def f(x: float) -> int: ...")
    assert not is_typed_subset("def f(d: dict) -> int: ...")
    # Untyped is also out -- the negative-comparison claim is conditional
    # on type information being available at all.
    assert not is_typed_subset("def f(x): return x")


def test_humaneval_only_typed_subset_excludes_string_problems():
    full = load_humaneval()
    typed = load_humaneval(only_typed_subset=True)
    assert len(typed) <= len(full)
    for p in typed:
        assert "out_of_substrate" not in p.tags


def test_minif2f_built_in_carries_fragment_tag():
    out = load_minif2f()
    fragments = {p.payload.get("fragment") for p in out}
    assert fragments & {"nat_arith", "out_of_substrate"}


def test_qm9_atom_count_filter():
    out = load_qm9(max_atoms=1)
    for p in out:
        assert len(p.payload["atoms"]) <= 1
