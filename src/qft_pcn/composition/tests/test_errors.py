"""Tests for the composition error family (spec §6)."""
from __future__ import annotations
import pytest
from src.qft_pcn.composition.errors import (
    CompositionError, LemmaHashCollision, LemmaNotFound,
    LemmaLeafCountMismatch, LemmaSpeciesMismatch,
    ConditionalLemmaRefused, LemmaValidationError, CompressionError,
    GoalGraphError, DispatchTimeout, IntegrationRefused, RevisionExhausted,
)

ALL = [LemmaHashCollision, LemmaNotFound, LemmaLeafCountMismatch,
       LemmaSpeciesMismatch, ConditionalLemmaRefused,
       LemmaValidationError, CompressionError]


@pytest.mark.parametrize("exc", ALL)
def test_all_subclass_composition_error(exc):
    assert issubclass(exc, CompositionError)


@pytest.mark.parametrize("exc", ALL)
def test_each_raisable_with_message(exc):
    with pytest.raises(CompositionError, match="boom"):
        raise exc("boom")


def test_k_all_inherit_composition_error():
    for cls in (GoalGraphError, DispatchTimeout, IntegrationRefused, RevisionExhausted):
        assert issubclass(cls, CompositionError)


def test_integration_refused_carries_residual_and_gate():
    exc = IntegrationRefused(residual=0.5, gate=0.1)
    assert exc.residual == 0.5
    assert exc.gate == 0.1
    assert "0.5" in str(exc) and "0.1" in str(exc)


def test_revision_exhausted_carries_attempts():
    exc = RevisionExhausted(goal_id="g7", attempts=3)
    assert exc.goal_id == "g7"
    assert exc.attempts == 3
    assert "g7" in str(exc) and "3" in str(exc)


def test_dispatch_timeout_carries_seconds():
    exc = DispatchTimeout(goal_id="g9", timeout_s=2.5)
    assert exc.goal_id == "g9"
    assert exc.timeout_s == 2.5
    assert "g9" in str(exc) and "2.5" in str(exc)
