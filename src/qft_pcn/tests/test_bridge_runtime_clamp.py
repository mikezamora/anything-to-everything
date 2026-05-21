"""Tests for runtime boundary-clamp construction and projection."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.bridge.dsl.term import FieldSpec
from src.qft_pcn.bridge.runtime.initial_state import build_initial_state
from src.qft_pcn.bridge.runtime.clamp import project_site, Clamp


def _canon_fields() -> list[FieldSpec]:
    return [
        FieldSpec(name="kind", cutoff=4),
        FieldSpec(name="value", cutoff=4),
    ]   # d_local = 16


def _proj_onto_basis(idx: int, d: int) -> np.ndarray:
    p = np.zeros((d, d), dtype=complex)
    p[idx, idx] = 1.0
    return p


def test_build_initial_state_no_boundary_is_vacuum():
    state = build_initial_state(fields=_canon_fields(), sites=3, boundary={})
    assert state.N == 3
    e0 = state.local_expectation(0, _proj_onto_basis(0, d=16))
    assert abs(e0.real - 1.0) < 1e-10


def test_build_initial_state_with_boundary_clamps_named_site():
    fields = _canon_fields()
    boundary = {1: {"kind": 2}}
    state = build_initial_state(fields=fields, sites=3, boundary=boundary)
    # kind=2, value=0 -> idx = 2*4 + 0 = 8
    e8 = state.local_expectation(1, _proj_onto_basis(8, d=16))
    assert abs(e8.real - 1.0) < 1e-10
    e0_s0 = state.local_expectation(0, _proj_onto_basis(0, d=16))
    assert abs(e0_s0.real - 1.0) < 1e-10


def test_project_site_restores_clamp_after_disturbance():
    fields = _canon_fields()
    boundary = {0: {"kind": 1}}
    state = build_initial_state(fields=fields, sites=2, boundary=boundary)
    rng = np.random.default_rng(0)
    pert = np.eye(16) + 0.01 * (rng.standard_normal((16, 16))
                                + 1j * rng.standard_normal((16, 16)))
    state.apply_local_gate(0, pert)
    state.normalize()
    clamp = Clamp(site=0, field="kind", basis_index=1)
    project_site(state, clamp, fields=fields)
    state.normalize()
    # kind=1 means kind register pinned to 1; site can be any value in [0..3]
    # so projector onto kind=1 subspace should give 1.0 expectation.
    # Sum over value of |kind=1,value=v>: indices 4,5,6,7.
    P = np.zeros((16, 16), dtype=complex)
    for i in range(4, 8):
        P[i, i] = 1.0
    val = state.local_expectation(0, P)
    assert abs(val.real - 1.0) < 1e-10
