from __future__ import annotations

import math

import numpy as np
import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.encoding import (
    SPECIES, EncodingTooLarge, TooManyBinders, IntLiteralOutOfRange,
    IllScopedVar, BinderHandle,
)
from src.qft_pcn.qft.mps import MPS


def test_encode_p1_returns_mps_and_meta():
    state, meta = encode(parse(r"\x:Int. x"), N=8, chi_max=16)
    assert isinstance(state, MPS)
    assert state.N == 8
    assert meta.N == 8
    assert meta.chi_max == 16
    assert meta.species == list(SPECIES)


def test_encode_produces_unit_norm():
    state, meta = encode(parse(r"\x:Int. x"), N=8)
    assert abs(state.norm_sq() - 1.0) < 1e-10


def test_encode_p5_unit_norm():
    state, meta = encode(parse(r"(\x:Int. (\y:Int. x + y)(3))(4)"),
                          N=32, chi_max=16)
    assert abs(state.norm_sq() - 1.0) < 1e-10


def test_encode_records_live_binders():
    state, meta = encode(parse(r"\x:Int. \y:Int. x"), N=8)
    # Layout: LAM_x@0, LAM_y@1, VAR_x@2, PAD@3...
    # Bond 0 (LAM_x | LAM_y): [Lam_x]
    # Bond 1 (LAM_y | VAR_x): [Lam_x, Lam_y]
    # Bond 2 (VAR_x | PAD): []
    assert len(meta.live_binders_per_bond) == 7
    assert len(meta.live_binders_per_bond[0]) == 1
    assert meta.live_binders_per_bond[0][0].lam_site == 0
    assert len(meta.live_binders_per_bond[1]) == 2
    assert len(meta.live_binders_per_bond[2]) == 0


def test_encode_too_large_raises():
    with pytest.raises(EncodingTooLarge):
        encode(parse(r"\x:Int. x + x + x"), N=3)


def test_encode_too_many_binders_raises():
    src = (r"\a:Int. \b:Int. \c:Int. \d:Int. \e:Int. \f:Int. \g:Int. "
           r"\h:Int. a")
    with pytest.raises(TooManyBinders):
        encode(parse(src), N=32)


def test_encode_int_literal_out_of_range_raises():
    with pytest.raises(IntLiteralOutOfRange):
        encode(parse(r"\x:Int. x + 100"), N=8)


def test_encode_ill_scoped_var_raises():
    with pytest.raises(IllScopedVar):
        encode(parse("x"), N=4)


def test_encode_pad_sites_have_zero_amplitude_for_non_pad():
    """Quick PAD test: site 3 of \\x.x encoding must have no amplitude for
    KIND_VAR (or anything other than the PAD basis state)."""
    state, meta = encode(parse(r"\x:Int. x"), N=4)
    from src.qft_pcn.qft.fock import embed_op
    from src.qft_pcn.logic.encoding import KIND_PAD, KIND_CUTOFF
    import numpy as np
    p = np.eye(KIND_CUTOFF, dtype=complex)
    p[KIND_PAD, KIND_PAD] = 0.0
    p_local = embed_op(p, 0, (8, 8, 8, 16))
    val = state.local_expectation(3, p_local)
    assert abs(val) < 1e-10
