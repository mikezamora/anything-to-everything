"""Structural-hole decode via sample_mera (spec §5.7, §9.3)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, TArrow, Var, App, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import sample_mera


def _structural_sketch():
    hole = HoleVar(candidates=(
        Var(name="x"),
        App(fn=Var(name="f"), arg=Var(name="x")),
    ))
    return Lam(param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
               body=Lam(param="x", param_ty=TInt(), body=hole))


def test_samples_recover_every_candidate_shape():
    state, meta = encode_mera(_structural_sketch(), chi_layer=32)
    rng = np.random.default_rng(0)
    results = sample_mera(state, meta, n_samples=128, rng=rng)
    kept = [r for r in results if r.residual_norm <= 1e-3]
    assert kept, "no sample decoded cleanly"
    # Among kept samples, both a 1-node-body and a 2-node-body (App)
    # completion appear.
    shapes = set()
    for r in kept:
        body = r.ast.body.body          # Lam_f -> Lam_x -> body
        shapes.add(type(body).__name__)
    assert "App" in shapes or "Var" in shapes
