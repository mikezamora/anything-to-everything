"""Tests for §12.7 replica-method typical-case complexity.

Spec: ``QFT_PCN_ARCHITECTURE.md`` §12.7. Predicts ``<log Z>`` over a
problem-class ensemble via the replica trick. The S3 substrate
(``qft.replica.compute_log_z_from_replicas``, landed at f34ebda) does
the integer-n -> 0 continuation; this module wires real
``encode_mera(parse(...))`` MERA states into the substrate so the
ensemble Z values are *operator-algebraic* (§1.1 architecture-soul) —
never AST counts, never fingerprints.

All tests use real QPCN theorems via ``logic.ast.parse``. No mocks.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.replica_complexity import (
    ComplexityPrediction,
    DEFAULT_INVERSE_TEMP,
    DEFAULT_REPLICA_N_GRID,
    compute_typical_field_marginal_complexity,
    instance_partition_function,
    predict_proof_difficulty,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


# ---------------------------------------------------------------------------
# Sanity: per-instance Z is real, positive, and reproducible
# ---------------------------------------------------------------------------


def test_instance_partition_function_is_positive_real() -> None:
    state, _ = encode_mera(parse(r"\x:Int. x"))
    z = instance_partition_function(state)
    assert isinstance(z, float)
    assert np.isfinite(z)
    assert z > 0.0


def test_instance_partition_function_alpha_invariant() -> None:
    """Two alpha-equivalent encodings must produce identical Z values.

    Per §1.1 ``encode_mera`` is alpha-invariant (bitwise identical bond
    entanglement under bound-variable rename); our Z is a deterministic
    function of the encoded state, so the substrate-derived equality
    flows through.
    """
    z1 = instance_partition_function(encode_mera(parse(r"\x:Int. x"))[0])
    z2 = instance_partition_function(encode_mera(parse(r"\y:Int. y"))[0])
    assert z1 == pytest.approx(z2, abs=1e-12, rel=1e-12)


# ---------------------------------------------------------------------------
# §12.7 acceptance tests (per task spec)
# ---------------------------------------------------------------------------


def test_homogeneous_ensemble_predicts_low_complexity() -> None:
    """All-identical-theorems ensemble => low typical complexity.

    Each instance has the same Z, so ``<Z^n> = Z^n`` exactly, the
    S3 continuation returns ``log Z`` (exact), and the ensemble has
    zero heterogeneity. Difficulty is dominated by ``max(0, -log Z)``;
    we verify it is bounded by the per-instance value (no extra
    heterogeneity term inflating it).
    """
    src = r"\x:Int. x"
    ensemble = [src] * 8
    pred = compute_typical_field_marginal_complexity(ensemble)
    assert isinstance(pred, ComplexityPrediction)
    assert pred.ensemble_size == 8

    # Heterogeneity is zero on an identical ensemble.
    assert pred.log_z_std == pytest.approx(0.0, abs=1e-12)

    # typical_log_z must match the single-instance log Z exactly
    # (modulo float interpolation noise).
    z_single = instance_partition_function(encode_mera(parse(src))[0])
    assert pred.typical_log_z == pytest.approx(
        float(np.log(z_single)), abs=1e-6
    )

    # Difficulty must equal max(0, -log Z) — no heterogeneity term.
    expected_difficulty = max(0.0, -float(np.log(z_single)))
    assert pred.difficulty == pytest.approx(expected_difficulty, abs=1e-6)


def test_diverse_ensemble_predicts_higher_complexity() -> None:
    """Heterogeneous ensemble => higher difficulty than homogeneous baseline.

    Building two real ensembles:
      * ``homo``: 6 copies of the same theorem (zero heterogeneity).
      * ``hetero``: 6 structurally distinct theorems (real
        encode_mera outputs differ -> different leaf marginals ->
        different Z -> non-zero ``log_z_std``).
    Per the §12.7 spec the diverse ensemble is the harder typical case.
    """
    homo_src = r"\x:Int. x"
    homo = [homo_src] * 6

    hetero = [
        r"\x:Int. x",
        r"\x:Int. \y:Int. x",
        r"\x:Int. \y:Int. y",
        r"\x:Int. \y:Int. x + y",
        r"\x:Int. \y:Int. \z:Int. x + y + z",
        r"\f:Int->Int. \x:Int. f x",
    ]

    pred_homo = compute_typical_field_marginal_complexity(homo)
    pred_hetero = compute_typical_field_marginal_complexity(hetero)

    # Heterogeneity strictly larger.
    assert pred_hetero.log_z_std > pred_homo.log_z_std
    assert pred_hetero.log_z_std > 1e-6

    # Difficulty is strictly larger on the heterogeneous ensemble
    # (the log_z_std term lifts it above the homogeneous baseline).
    assert pred_hetero.difficulty > pred_homo.difficulty


def test_complexity_invariant_under_alpha_renaming() -> None:
    """Alpha-equivalent ensembles must yield identical predictions (§1.1).

    Construct two ensembles that differ only by bound-variable rename;
    the architecture-soul guarantees encode_mera produces bitwise-
    identical states, so every downstream quantity — Z, ``<Z^n>``,
    ``<log Z>``, difficulty — must match exactly.
    """
    ens_x = [
        r"\x:Int. x",
        r"\x:Int. \y:Int. x + y",
        r"\f:Int->Int. \x:Int. f x",
    ]
    ens_renamed = [
        r"\a:Int. a",
        r"\u:Int. \v:Int. u + v",
        r"\g:Int->Int. \w:Int. g w",
    ]

    pred_x = compute_typical_field_marginal_complexity(ens_x)
    pred_r = compute_typical_field_marginal_complexity(ens_renamed)

    assert pred_x.typical_log_z == pytest.approx(
        pred_r.typical_log_z, abs=1e-12, rel=1e-12
    )
    assert pred_x.log_z_std == pytest.approx(
        pred_r.log_z_std, abs=1e-12, rel=1e-12
    )
    assert pred_x.difficulty == pytest.approx(
        pred_r.difficulty, abs=1e-12, rel=1e-12
    )
    assert pred_x.regime == pred_r.regime


def test_complexity_regime_limit_documented() -> None:
    """Pin the S3 well-conditioned regime + assert graceful degradation.

    Well-conditioned regime (per the module docstring):
      * multi-instance ensemble
      * Z bounded so ``max(Z)**max(n_grid)`` is float-representable
      * no Z = 0
    Outside this regime ``regime != "well_conditioned"`` and
    ``notes`` carries a human-readable cause; the call must NOT raise.
    """
    # (a) Healthy multi-instance ensemble lands in well_conditioned.
    healthy = [
        r"\x:Int. x",
        r"\x:Int. \y:Int. x + y",
        r"\f:Int->Int. \x:Int. f x",
        r"\x:Int. \y:Int. \z:Int. x + y + z",
    ]
    pred_healthy = compute_typical_field_marginal_complexity(healthy)
    assert pred_healthy.regime == "well_conditioned"
    assert pred_healthy.ensemble_size == len(healthy)
    assert np.isfinite(pred_healthy.typical_log_z)
    assert np.isfinite(pred_healthy.difficulty)

    # (b) Single-instance ensemble: documented degraded regime
    # (well-defined value, but no replica averaging).
    pred_single = compute_typical_field_marginal_complexity([r"\x:Int. x"])
    assert pred_single.regime == "degraded"
    assert any("single-instance" in n for n in pred_single.notes)
    assert pred_single.ensemble_size == 1
    # The value is still finite and equal to log Z of that instance.
    z = instance_partition_function(encode_mera(parse(r"\x:Int. x"))[0])
    assert pred_single.typical_log_z == pytest.approx(
        float(np.log(z)), abs=1e-6
    )

    # (c) Fully-empty / unparseable ensemble: degrades to "empty"
    # without raising; difficulty is +inf.
    pred_empty = compute_typical_field_marginal_complexity([])
    assert pred_empty.regime == "empty"
    assert pred_empty.ensemble_size == 0
    assert pred_empty.difficulty == float("inf")

    # (d) Default replica n-grid is what the docstring promises.
    assert DEFAULT_REPLICA_N_GRID == (1, 2, 3, 4, 5)
    assert DEFAULT_INVERSE_TEMP == 1.0


# ---------------------------------------------------------------------------
# predict_proof_difficulty wrapper
# ---------------------------------------------------------------------------


def test_predict_proof_difficulty_returns_finite_scalar() -> None:
    """The wrapper composes theorem + corpus and returns a numeric score."""
    corpus = [
        r"\x:Int. x",
        r"\x:Int. \y:Int. x",
        r"\x:Int. \y:Int. x + y",
    ]
    theorem = r"\f:Int->Int. \x:Int. f x"
    score = predict_proof_difficulty(theorem, corpus)
    assert isinstance(score, float)
    assert np.isfinite(score)
    assert score >= 0.0


def test_predict_proof_difficulty_in_distribution_lower_than_out() -> None:
    """A theorem already in the corpus should score no higher than a
    structurally novel one (the in-distribution theorem does not raise
    the ensemble heterogeneity).
    """
    corpus = [
        r"\x:Int. x",
        r"\y:Int. y",
        r"\z:Int. z",
    ]
    in_dist = r"\w:Int. w"  # alpha-equivalent to every corpus entry
    out_dist = r"\x:Int. \y:Int. \z:Int. x + y + z"

    score_in = predict_proof_difficulty(in_dist, corpus)
    score_out = predict_proof_difficulty(out_dist, corpus)

    assert score_out >= score_in
