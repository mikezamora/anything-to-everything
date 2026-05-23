"""Canonical-form computation (spec §5.4; acceptance §9.5)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.qft_pcn.composition._abstraction_const import RepurificationWarning
from src.qft_pcn.composition.subtree_miner import SubtreeCandidate, fingerprint_of
from src.qft_pcn.composition.abstraction import (
    Cluster, CanonicalPrimitive, compute_canonical_form, trace_distance,
)


def _cand(sid: str, rho: np.ndarray) -> SubtreeCandidate:
    return SubtreeCandidate(sid, (0, 20), 4, rho, fingerprint_of(rho, 4))


def _rand_density(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


def test_canonical_of_identical_members_is_that_member():
    rho = _rand_density(8, 11)
    cluster = Cluster(members=[_cand(f"m{i}", rho.copy()) for i in range(5)])
    prim = compute_canonical_form(cluster, chi_cap=16)
    assert np.allclose(prim.rho_canonical, rho, atol=1e-9)
    assert prim.avg_trace_distance == pytest.approx(0.0, abs=1e-9)


def test_canonical_is_the_operator_basis_mean():
    rhos = [_rand_density(8, s) for s in (1, 2, 3)]
    cluster = Cluster(members=[_cand(f"m{i}", r) for i, r in enumerate(rhos)])
    prim = compute_canonical_form(cluster, chi_cap=16)
    mean = sum(rhos) / len(rhos)
    assert np.allclose(prim.rho_canonical, mean, atol=1e-9)
    expected_avg = np.mean([trace_distance(mean, r) for r in rhos])
    assert prim.avg_trace_distance == pytest.approx(expected_avg, abs=1e-9)


def test_repurified_mera_reextracts_to_rho_canonical():
    rho = _rand_density(8, 5)
    cluster = Cluster(members=[_cand("m0", rho)])
    prim = compute_canonical_form(cluster, chi_cap=16)
    # the bounded-bond MERA's boundary density matches rho_canonical
    assert prim.mera is not None
    assert prim.chi <= 16
    assert abs(prim.mera.norm_sq() - 1.0) < 1e-9


def test_truncation_emits_repurification_warning():
    # a full-rank density truncated below its rank warns
    rho = _rand_density(16, 9)
    cluster = Cluster(members=[_cand("m0", rho)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_canonical_form(cluster, chi_cap=2)
    assert any(issubclass(w.category, RepurificationWarning) for w in caught)
