"""Ollama wrapper tests (mocks httpx — no live Ollama needed)."""

import json
from unittest.mock import patch, MagicMock

import pytest

from src.qft_pcn.viz.llm import (
    list_models, generate_dsl, verbalize, OLLAMA_URL,
)


def _mock_response(payload: dict, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    r.raise_for_status = MagicMock()
    return r


@patch("src.qft_pcn.viz.llm.httpx.get")
def test_list_models(mock_get):
    mock_get.return_value = _mock_response(
        {"models": [{"name": "gemma3:4b"}, {"name": "deepseek-r1:32b"}]})
    out = list_models()
    assert [m["name"] for m in out] == ["gemma3:4b", "deepseek-r1:32b"]


@patch("src.qft_pcn.viz.llm.httpx.post")
def test_generate_dsl_returns_validated(mock_post):
    dsl = {"fields": [{"name": "A", "cutoff": 2}],
           "hamiltonian": {"terms": [
               {"kind": "mass", "species": "A", "coefficient": 1.0}]},
           "observables": [
               {"operator": "n", "site": 0, "species": "A", "target": 0.25}]}
    mock_post.return_value = _mock_response(
        {"response": json.dumps(dsl)})
    from src.qft_pcn.viz.dsl import load_schema, EXAMPLES
    result = generate_dsl("simulate A at quarter density",
                          model="gemma3:4b",
                          schema=load_schema(), examples=EXAMPLES)
    assert "dsl" in result
    assert result["dsl"]["fields"][0]["name"] == "A"


@patch("src.qft_pcn.viz.llm.httpx.post")
def test_generate_dsl_surfaces_validation_errors(mock_post):
    mock_post.return_value = _mock_response(
        {"response": json.dumps({"fields": []})})  # missing required keys
    from src.qft_pcn.viz.dsl import load_schema, EXAMPLES
    result = generate_dsl("garbage prompt", model="gemma3:4b",
                          schema=load_schema(), examples=EXAMPLES)
    assert "error" in result
    assert "raw" in result
    assert isinstance(result["validation_errors"], list)


@patch("src.qft_pcn.viz.llm.httpx.post")
def test_verbalize_strips_think_blocks(mock_post):
    mock_post.return_value = _mock_response(
        {"response": "<think>some chain</think>The result is 0.25."})
    txt = verbalize({"obs": {"n_0": 0.25}}, model="deepseek-r1:32b",
                    original_prompt="what is n_0?")
    assert "<think>" not in txt
    assert "result is 0.25" in txt
