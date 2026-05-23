"""Constants and exception family (spec §8.1, §10)."""
from __future__ import annotations
import pytest
from src.qft_pcn.composition._abstraction_const import (
    S_MIN, S_MAX, FP_DECIMALS, FP_RANK,
    DEFAULT_DISTANCE_THRESHOLD, DEFAULT_ALPHA, DEFAULT_CHI_CAP,
    N_QUIESCENT, REPURIFICATION_TAIL_TOL, DENSITY_HERMITICITY_TOL,
    AbstractionError, MiningError, RepurificationWarning, LibraryContractError,
)


def test_constant_values():
    assert (S_MIN, S_MAX) == (3, 8)
    assert (FP_DECIMALS, FP_RANK) == (3, 4)
    assert DEFAULT_DISTANCE_THRESHOLD == 0.15
    assert DEFAULT_ALPHA == 1e-3
    assert DEFAULT_CHI_CAP == 16
    assert N_QUIESCENT == 2
    assert REPURIFICATION_TAIL_TOL == 1e-6
    assert DENSITY_HERMITICITY_TOL == 1e-9


def test_exception_hierarchy():
    assert issubclass(MiningError, AbstractionError)
    assert issubclass(LibraryContractError, AbstractionError)
    assert issubclass(AbstractionError, Exception)
    assert issubclass(RepurificationWarning, Warning)


def test_mining_error_raisable():
    with pytest.raises(MiningError):
        raise MiningError("bad density")
