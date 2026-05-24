"""Tests for `generate_dsl(max_retries=...)` retry-on-validation-failure loop.

Mocks the single-round-trip helper `_attempt_generate_dsl` so the substrate
under test is the retry loop itself (no live Ollama dependency). The existing
contract is: success returns `{"dsl": <dict>}`, failure returns
`{"error", "raw", "validation_errors"}`. `max_retries=0` must preserve the
historical behavior of returning the failure dict from the first attempt.
"""
from __future__ import annotations

from unittest.mock import patch

from src.qft_pcn.viz import llm as llm_mod


_FAIL = {
    "error": "DSL validation failed",
    "raw": "{}",
    "validation_errors": ["missing field: version"],
}
_FAIL2 = {
    "error": "DSL validation failed",
    "raw": "{}",
    "validation_errors": ["missing field: sites"],
}
_OK = {"dsl": {"version": "1", "fields": [], "sites": 2,
               "constraints": [], "observables": [],
               "search": {"method": "tebd", "runtime": "mps",
                          "steps": 1, "chi_max": 4, "dt": 0.1}}}


def test_no_retry_default_behavior_preserved():
    """Default `max_retries=0` returns first-attempt failure dict unchanged."""
    with patch.object(llm_mod, "_attempt_generate_dsl",
                      return_value=_FAIL) as mock_attempt:
        result = llm_mod.generate_dsl("hello",
                                      model="m", schema={}, examples=[])
    assert result == _FAIL
    assert mock_attempt.call_count == 1


def test_retry_recovers_from_validation_failure():
    """Invalid first emit, valid second emit -> success after one retry."""
    with patch.object(llm_mod, "_attempt_generate_dsl",
                      side_effect=[_FAIL, _OK]) as mock_attempt:
        result = llm_mod.generate_dsl("hello",
                                      model="m", schema={}, examples=[],
                                      max_retries=3)
    assert "dsl" in result
    assert result == _OK
    assert mock_attempt.call_count == 2
    # Second call's prompt must reference the validation errors (feedback turn).
    retry_prompt = mock_attempt.call_args_list[1].args[0]
    assert "missing field: version" in retry_prompt
    assert "failed validation" in retry_prompt.lower()


def test_retry_exhausted_returns_last_failure():
    """All attempts fail -> return final failure dict after max_retries+1 calls."""
    with patch.object(llm_mod, "_attempt_generate_dsl",
                      side_effect=[_FAIL, _FAIL, _FAIL2]) as mock_attempt:
        result = llm_mod.generate_dsl("hello",
                                      model="m", schema={}, examples=[],
                                      max_retries=2)
    assert "dsl" not in result
    assert result == _FAIL2  # The *last* failure surfaces to the caller.
    assert mock_attempt.call_count == 3  # 1 initial + 2 retries.
