from __future__ import annotations

import numpy as np

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic._gate_construction import encode_gate


# The gate-based construction produces a pure *product* state on each site
# (one definite (kind, type, bid, value, tobl) per site). After B's bid-bond
# param_ty extension (spec B §5.2) the analytic encoder's bond carries
# (1 + 8*|L|) channels — non-product in general — but on no-hole programs
# the channel structure is deterministic: each site writes exactly one
# slot on each bond it touches, so after normalization the contraction
# concentrates all amplitude on the same flat basis state as the gate
# path. Fidelity remains 1.0 to numerical tolerance for both P1 and P5.


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


def test_encode_gate_does_not_crash():
    """The gate-based path still constructs a valid MPS + EncodingMeta on a
    no-hole program, even though its state no longer matches the analytic
    encoder's. This guards the 5-species refactor (5-arg _basis_index,
    tobl writes, new EncodingMeta fields).
    """
    state_g, meta = encode_gate(parse(r"\x:Int. x"), N=8, chi_max=16)
    assert state_g.N == 8
    assert meta.N == 8
    assert meta.tobl_per_site is not None
    assert len(meta.tobl_per_site) == 8
    assert meta.channel_param_ty_per_bond is not None
    # State is normalized.
    assert abs(state_g.norm_sq() - 1.0) < 1e-10
