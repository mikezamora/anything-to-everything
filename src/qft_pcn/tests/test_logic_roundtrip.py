from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.decoder import decode, ast_alpha_eq


PROGRAMS = {
    "P1": r"\x:Int. x",
    "P2": r"(\x:Int. x + 1)(2)",
    "P3": r"\f:Int->Int. \x:Int. f (f x)",
    "P4": r"if 1 < 2 then ((\x:Bool. x)(true)) else false",
    "P5": r"(\x:Int. (\y:Int. x + y)(3))(4)",
}


@pytest.mark.parametrize("name,src", list(PROGRAMS.items()))
def test_roundtrip(name, src):
    expected = parse(src)
    state, meta = encode(expected, N=32, chi_max=16)
    assert abs(state.norm_sq() - 1.0) < 1e-10, f"{name}: not unit-norm"
    result = decode(state, meta)
    assert result.residual_norm < 1e-10, (
        f"{name}: residual_norm={result.residual_norm} too large"
    )
    assert ast_alpha_eq(result.ast, expected), (
        f"{name}: roundtrip mismatch.\n"
        f"  input:  {src}\n"
        f"  output: {result.ast!r}"
    )
