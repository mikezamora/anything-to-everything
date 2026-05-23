"""Tests for the composition error family (spec §6)."""
from __future__ import annotations
import pytest
from src.qft_pcn.composition.errors import (
    CompositionError, LemmaHashCollision, LemmaNotFound,
    LemmaLeafCountMismatch, LemmaSpeciesMismatch,
    ConditionalLemmaRefused, LemmaValidationError, CompressionError,
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
