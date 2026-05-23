"""Trace distance + significance test (spec §5.1, §5.3; acceptance §9.3–9.4)."""
from __future__ import annotations
import math
import numpy as np
import pytest
from src.qft_pcn.composition.abstraction import trace_distance, k_min


def _rand_density(d: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


def test_trace_distance_of_equal_is_zero():
    rng = np.random.default_rng(0)
    rho = _rand_density(8, rng)
    assert trace_distance(rho, rho) == pytest.approx(0.0, abs=1e-12)


def test_trace_distance_orthogonal_pure_states_is_one():
    rho0 = np.zeros((4, 4), dtype=complex); rho0[0, 0] = 1.0
    rho1 = np.zeros((4, 4), dtype=complex); rho1[1, 1] = 1.0
    assert trace_distance(rho0, rho1) == pytest.approx(1.0, abs=1e-12)


def test_trace_distance_matches_schatten_one_norm():
    rng = np.random.default_rng(7)
    for _ in range(20):
        r1, r2 = _rand_density(6, rng), _rand_density(6, rng)
        diff = r1 - r2
        # 0.5 * sum singular values of the Hermitian difference
        direct = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff)))
        assert trace_distance(r1, r2) == pytest.approx(direct, abs=1e-10)
        assert 0.0 - 1e-12 <= trace_distance(r1, r2) <= 1.0 + 1e-12


def test_trace_distance_zero_pads_unequal_dims():
    rng = np.random.default_rng(3)
    small = _rand_density(2, rng)
    big = np.zeros((4, 4), dtype=complex)
    big[:2, :2] = small
    # big is `small` embedded with the extra bond empty -> distance 0
    assert trace_distance(small, big) == pytest.approx(0.0, abs=1e-10)


def test_k_min_floored_at_two():
    assert k_min(2) >= 2
    assert k_min(1) >= 2


def test_k_min_rejects_chance_clusters():
    n = 500
    k = k_min(n, alpha=1e-3)
    # a cluster of size k is below the alpha chance bound...
    assert math.exp(-k * math.log(max(n, 2))) <= 1e-3
    # ...and k-1 is not (k is the *smallest* significant size)
    assert math.exp(-(k - 1) * math.log(max(n, 2))) > 1e-3
