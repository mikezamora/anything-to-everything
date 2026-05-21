from __future__ import annotations

import numpy as np

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic._gate_construction import encode_gate


def test_gate_construction_matches_analytic_p1():
    state_a, meta = encode(parse(r"\x:Int. x"), N=8, chi_max=16)
    state_g, _   = encode_gate(parse(r"\x:Int. x"), N=8, chi_max=16)
    fidelity = abs(state_a.inner(state_g)) ** 2
    assert fidelity > 1.0 - 1e-8, f"fidelity={fidelity}"


def test_gate_construction_matches_analytic_p5():
    src = r"(\x:Int. (\y:Int. x + y)(3))(4)"
    state_a, meta = encode(parse(src), N=32, chi_max=16)
    state_g, _   = encode_gate(parse(src), N=32, chi_max=16)
    fidelity = abs(state_a.inner(state_g)) ** 2
    assert fidelity > 1.0 - 1e-8, f"fidelity={fidelity}"
