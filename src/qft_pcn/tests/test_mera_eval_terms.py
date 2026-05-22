"""Tests for the MERA evaluation redex-penalty builders (spec §7.1)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic._mera_eval_terms import (
    beta_penalty_ops, arith_penalty_ops, cmp_penalty_ops,
    if_penalty_ops, succ_penalty_ops,
    DEFAULT_LAMBDA_BETA, DEFAULT_LAMBDA_ARITH, DEFAULT_LAMBDA_IF,
    DEFAULT_LAMBDA_FIX,
)


def test_default_lambdas_are_positive():
    for lam in (DEFAULT_LAMBDA_BETA, DEFAULT_LAMBDA_ARITH,
                DEFAULT_LAMBDA_IF, DEFAULT_LAMBDA_FIX):
        assert lam > 0


def test_beta_penalty_ops_returns_two_leaf_ops():
    ops = beta_penalty_ops(app_kind_leaf=0, fn_kind_leaf=5, lam=1.0)
    assert set(ops.keys()) == {0, 5}
    for op in ops.values():
        assert op.shape == (16, 16)


def test_arith_penalty_ops_three_leaves():
    ops = arith_penalty_ops(bin_kind_leaf=0, bin_value_leaf=3,
                            lhs_kind_leaf=5, rhs_kind_leaf=10, lam=1.0)
    assert len(ops) == 4


def test_penalty_ops_are_projectors():
    ops = if_penalty_ops(if_kind_leaf=0, cond_kind_leaf=5, lam=1.0)
    for op in ops.values():
        assert np.allclose(op, np.diag(np.diag(op)))
