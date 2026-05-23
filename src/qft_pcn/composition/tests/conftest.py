"""Shared fixtures: memory guard, FakeLemmaLibrary, stub solvers, corpora."""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.qft.mera import MERA

# --- §9.7 memory ceiling: no dense tensor larger than chi_cap**2 = 256 -------
_MAX_ELEMS = 256


@pytest.fixture(autouse=True)
def _no_large_dense(monkeypatch):
    """Trip if any code under test allocates a dense ndarray above the ceiling.

    Wraps numpy.zeros/empty/ones; subtree leaf spaces (16**s) would blow this.
    """
    real_zeros, real_empty, real_ones = np.zeros, np.empty, np.ones

    def _guard(real):
        def wrapped(shape, *a, **k):
            arr = real(shape, *a, **k)
            if arr.size > _MAX_ELEMS and arr.ndim >= 2:
                raise AssertionError(
                    f"dense tensor of size {arr.size} exceeds ceiling {_MAX_ELEMS}"
                )
            return arr
        return wrapped

    # Only guard 2-D+; MERA internals allocate small 1-D buffers freely.
    monkeypatch.setattr(np, "zeros", _guard(real_zeros))
    monkeypatch.setattr(np, "empty", _guard(real_empty))
    monkeypatch.setattr(np, "ones", _guard(real_ones))
    yield


def make_product_mera(leaf_vectors: list[np.ndarray]) -> MERA:
    """A normalized product MERA over the given 16-dim leaf vectors."""
    return MERA.from_product(leaf_vectors).normalize()


def basis_leaf(idx: int, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, dtype=complex)
    v[idx] = 1.0
    return v


# induction corpus builders: Task 6
