"""Tests for the baseline adapters."""
from __future__ import annotations

import os

import pytest

from experiments.baselines import (
    AlphaProofBaseline, BaselineUnavailable, QPCNBaseline,
    ReProverBaseline, SynquidBaseline,
)
from experiments.benchmarks import load_minif2f, load_myth
from experiments.schema import ProblemSpec


def _strip_env(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


def test_alphaproof_raises_unavailable_without_binary(monkeypatch):
    _strip_env(monkeypatch, "ALPHAPROOF_CMD")
    problems = load_minif2f(limit=1)
    with pytest.raises(BaselineUnavailable) as exc:
        AlphaProofBaseline().solve(problems[0])
    assert exc.value.baseline == "alphaproof"
    assert "EXTENSIONS.md" in str(exc.value)


def test_reprover_raises_unavailable_without_binary(monkeypatch):
    _strip_env(monkeypatch, "REPROVER_CMD")
    problems = load_minif2f(limit=1)
    with pytest.raises(BaselineUnavailable):
        ReProverBaseline().solve(problems[0])


def test_synquid_raises_unavailable_without_binary(monkeypatch):
    _strip_env(monkeypatch, "SYNQUID_CMD")
    problems = load_myth(limit=1)
    with pytest.raises(BaselineUnavailable):
        SynquidBaseline().solve(problems[0])


def test_qpcn_chemistry_honest_no_attempt():
    """QM9 has no QPCN substrate yet; the adapter must surface an
    honest no-attempt rather than fake a result (no-placeholders)."""
    from experiments.benchmarks import load_qm9
    p = load_qm9(limit=1)[0]
    att = QPCNBaseline().solve(p)
    assert att.solved is False
    assert att.error and "chemistry" in att.error.lower()


def test_qpcn_proof_solves_supported_minif2f():
    """The K-8/L acceptance proof family (forall x:Nat. x+0 = x) must
    SOLVE under the QPCN. This is the load-bearing acceptance.
    """
    problems = load_minif2f(limit=4)
    qpcn = QPCNBaseline()
    # Find a problem inside the supported fragment.
    supported = [p for p in problems
                 if p.payload.get("fragment") == "nat_arith"]
    assert supported, "no nat_arith problem in the loaded subset"
    # Take the easiest (mathd_algebra_478: x+0 = x is the K-8 family).
    p = next(p for p in supported if "478" in p.problem_id or "447" in p.problem_id)
    att = qpcn.solve(p)
    assert att.solved, (
        f"K-8 family must solve via QPCN. residual={att.residual_energy}, "
        f"error={att.error}"
    )
    assert att.well_typed
    assert att.residual_energy is not None
    assert att.residual_energy < 0.1


def test_qpcn_out_of_substrate_minif2f_honest_no_attempt():
    """out_of_substrate miniF2F rows must surface an honest no-attempt."""
    problems = load_minif2f()
    out = [p for p in problems
           if p.payload.get("fragment") == "out_of_substrate"]
    assert out, "expected at least one out_of_substrate fixture"
    att = QPCNBaseline().solve(out[0])
    assert att.solved is False
    assert "out_of_substrate" in (att.error or "")
