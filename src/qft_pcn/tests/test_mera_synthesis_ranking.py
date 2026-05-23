"""Tests for synthesis ranking (spec §6.3, §6.4)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import Lam, TInt, Var
from src.qft_pcn.logic.mera_decoder import DecodeResult
from src.qft_pcn.logic.mera_synthesis.ranking import (
    dedupe_by_alpha_eq, classify_failure_mode,
)
from src.qft_pcn.logic.mera_synthesis.problem import Completion


def _id():
    return Lam(param="x", param_ty=TInt(), body=Var(name="x"))


def _id_renamed():
    return Lam(param="y", param_ty=TInt(), body=Var(name="y"))


def test_dedupe_groups_alpha_equivalent_samples():
    samples = [DecodeResult(ast=_id(), residual_norm=0.0),
               DecodeResult(ast=_id_renamed(), residual_norm=0.0),
               DecodeResult(ast=_id(), residual_norm=0.0)]
    groups = dedupe_by_alpha_eq(samples)
    assert len(groups) == 1
    assert groups[0][1] == 3            # multiplicity


def test_dedupe_drops_high_residual_samples():
    samples = [DecodeResult(ast=_id(), residual_norm=0.5)]
    assert dedupe_by_alpha_eq(samples) == []


def test_classify_failure_mode_none_on_low_energy():
    c = Completion(ast=_id(), energy=1e-5, energy_breakdown={},
                   diagnostics={}, multiplicity=1)
    assert classify_failure_mode([c], tolerance_correct=1e-2) is None


def test_classify_failure_mode_no_completion():
    assert classify_failure_mode([], tolerance_correct=1e-2) \
        == "no_valid_completion"


def test_classify_failure_mode_ambiguous():
    a = Completion(ast=_id(), energy=1e-5, energy_breakdown={},
                   diagnostics={}, multiplicity=1)
    b = Completion(ast=_id_renamed(), energy=1e-5 + 1e-7,
                   energy_breakdown={}, diagnostics={}, multiplicity=1)
    assert classify_failure_mode([a, b], tolerance_correct=1e-2,
                                 tolerance_ambiguous=1e-3) == "ambiguous_top1"
