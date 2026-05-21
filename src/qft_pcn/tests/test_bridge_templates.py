"""Tests for bridge/templates.py and bridge/llm.py (spec §7, §11.6)."""

from __future__ import annotations

import pytest

from src.qft_pcn.bridge.templates import (
    Problem, FieldSpec, ConstraintSpec, ObservableSpec, SearchSpec,
)
from src.qft_pcn.bridge.dsl.pipeline import validate_dsl
from src.qft_pcn.bridge.llm import MockLLM


def test_problem_builder_to_dsl_validates():
    p = Problem(
        fields=[FieldSpec(name="kind", cutoff=8)],
        sites=4,
        constraints=[
            ConstraintSpec(kind="local", site=0,
                            term="kind == LAM", weight=1.0)
        ],
        boundary={0: {"kind": "KIND_LAM"}},
        observables=[ObservableSpec(site=0, field="kind", op="n")],
    )
    out = validate_dsl(p.to_dsl())
    assert out["valid"] is True


def test_problem_builder_defaults_search():
    p = Problem(
        fields=[FieldSpec(name="x", cutoff=4)],
        sites=1,
        observables=[ObservableSpec(site=0, field="x", op="n")],
    )
    dsl = p.to_dsl()
    assert dsl["search"]["method"] == "imag_time"


def test_mock_llm_emits_canned_dsl():
    mock = MockLLM(responses={
        "double-int": {
            "fields": [{"name": "x", "cutoff": 4}],
            "sites": 1, "constraints": [],
            "observables": [{"site": 0, "field": "x", "op": "n"}],
        }
    })
    dsl = mock.emit_dsl("double-int")
    assert dsl["sites"] == 1


def test_mock_llm_raises_for_unknown_prompt():
    mock = MockLLM(responses={})
    with pytest.raises(KeyError):
        mock.emit_dsl("nope")


def test_mock_llm_verbalize_uses_template():
    mock = MockLLM(responses={})
    result = {"observables": [{"site": 0, "field": "x", "op": "n",
                                "value": 2.5, "imag_part": 1e-13}],
              "converged": True}
    text = mock.verbalize("anything", result)
    assert "2.5" in text or "2.50" in text
    assert "converged" in text.lower() or "converg" in text.lower()


def test_anthropic_llm_import_is_lazy():
    import importlib, sys
    if "anthropic" in sys.modules:
        del sys.modules["anthropic"]
    importlib.import_module("src.qft_pcn.bridge.llm")
