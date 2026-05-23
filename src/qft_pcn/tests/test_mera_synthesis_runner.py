"""Synthesis runner smoke test (P1 only -- full P1-P8 acceptance in a later task)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, Var, HoleVar
from src.qft_pcn.logic.mera_synthesis import synthesize, SynthesisProblem
from src.qft_pcn.logic.decoder import ast_alpha_eq


def _P1():
    """Identity over Int: \\x:Int. ?HOLE  with candidate "x"."""
    return SynthesisProblem(
        name="P1",
        sketch=Lam(param="x", param_ty=TInt(),
                   body=HoleVar(candidates=("x",))),
        # Small anneal budget for the smoke test -- full §7 budget arrives
        # with the P1-P8 acceptance task once perf is profiled.
        anneal_steps=50,
    )


def test_p1_runs_without_raising():
    res = synthesize(_P1(), rng=np.random.default_rng(0))
    assert res is not None
    assert res.problem.name == "P1"


def test_p1_top1_is_identity():
    res = synthesize(_P1(), rng=np.random.default_rng(0))
    assert res.completions, f"no completions, failure_mode={res.failure_mode}"
    top = res.completions[0].ast
    expected = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    assert ast_alpha_eq(top, expected), (
        f"top-1 {top!r} != identity (failure_mode={res.failure_mode}, "
        f"breakdown={res.completions[0].energy_breakdown})")
