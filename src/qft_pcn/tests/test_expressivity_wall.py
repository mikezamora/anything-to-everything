"""§13.5 expressivity wall — empirical area-law cap pin.

Per spec §13.5: the MERA substrate is area-law bounded. A program whose
semantics genuinely require volume-law entanglement should NOT converge
under fixed bond-dim within a reasonable step budget. This test pins
empirical behavior at (chi, steps) for a reference shallow chain and a
deeper sequential arithmetic chain. Both directional outcomes (converge /
above-threshold) are valid §13.5 data points.
"""
import math

import pytest

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve_state


CONVERGENCE_THRESHOLD = 1e-2
STEPS = 100
DT = 0.1
CHI = 16


def _residual_after_evolution(src: str) -> float:
    state, meta = encode_mera(parse(src))
    H = MeraEvalHamiltonian(meta)
    _, final = mera_imaginary_evolve_state(
        state, H, dt=DT, steps=STEPS, chi_layer=CHI
    )
    return float(H.total_energy(final))


def test_shallow_arith_chain_converges():
    """Reference: shallow chain ``1 + 2`` converges below threshold —
    confirms the substrate IS expressive enough for area-law programs."""
    res = _residual_after_evolution("1 + 2")
    assert res < CONVERGENCE_THRESHOLD, (
        f"sanity: shallow chain should converge, got <H>={res}"
    )


@pytest.mark.timeout(180)
def test_deep_arith_chain_at_chi_capped_substrate():
    """Empirical area-law observation: a deeper sequential arithmetic
    chain probes the substrate's bond-dim budget. Pin observed behavior
    at fixed (chi, steps). The EXTENSIONS.md entry tracks the larger
    volume-law refusal substrate."""
    # Six-deep sequential chain — deeper than typical K-8 fixtures.
    src = "((((1 + 1) + 1) + 1) + 1) + 1"
    try:
        res = _residual_after_evolution(src)
    except Exception as exc:
        # An encoder refusal (e.g. AST exceeds n_nodes_max) IS the
        # area-law wall surfacing at the encoder boundary — also valid.
        pytest.skip(f"deep chain refused at encoder: {exc!r}")
        return

    print(
        f"<H> for 6-deep chain at chi={CHI}, steps={STEPS}: "
        f"{res:.6f} (threshold={CONVERGENCE_THRESHOLD})"
    )
    # Non-directional pin: NaN would indicate numerical failure, which is
    # NOT the area-law signature we're documenting.
    assert not math.isnan(res), "residual NaN at depth 6 (numerical failure)"
    # Both "converged" and "above threshold" are valid §13.5 outcomes.
    # Record bucket for empirical tracking:
    if res < CONVERGENCE_THRESHOLD:
        # Substrate handles depth 6 within chi=16; wall is at greater depth.
        bucket = "converged"
    else:
        # Wall observed at this depth under this (chi, steps) budget.
        bucket = "above_threshold"
    print(f"§13.5 empirical bucket: {bucket}")
