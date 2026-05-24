"""Acceptance tests for §12.10 holographic compilation.

These tests exercise the real M2/M3 substrate:
  * AST compilation goes through ``encode_mera`` (the real §6 MERA
    encoder) — no AST-walker surrogate.
  * Layer equivalence is verified by §12.2
    ``compute_wilson_loop_signature`` on the encoded state — not by
    AST diffing.
  * Convergence detection runs §12.8 ``detect_dpt_in_wake_sleep_log``
    over real MERA snapshots, using :meth:`MERA.inner` for the
    Loschmidt echo.

No mocks, no stubs, no skips.
"""
from __future__ import annotations

from src.qft_pcn.composition.holographic_compilation import (
    CompilationLayer,
    detect_compilation_convergence,
    record_mera_layer_sequence,
    verify_layer_state_identical,
)
from src.qft_pcn.composition.tests.conftest import (
    basis_leaf,
    make_product_mera,
)
from src.qft_pcn.logic.ast import parse


# ---------------------------------------------------------------------------
# record_mera_layer_sequence
# ---------------------------------------------------------------------------


def test_compile_produces_layer_sequence():
    """Compiling a real AST yields at least one CompilationLayer whose
    per-layer tensors and shared encoded state are REAL MERA objects
    (§1.1: layers carry isometries+disentanglers, not AST passes)."""
    ast = parse(r"\x:Int. x")
    layers = record_mera_layer_sequence(ast)

    assert len(layers) >= 1, "record_mera_layer_sequence must produce >= 1 layer"
    # Per-layer record matches the MERA layer count.
    state = layers[0].state
    assert len(layers) == state.L

    # Layer indices are 0..L-1 in order.
    assert [lyr.layer_index for lyr in layers] == list(range(state.L))

    # Each layer carries the real per-layer tensors from the encoded MERA.
    for ell, lyr in enumerate(layers):
        assert isinstance(lyr, CompilationLayer)
        # Isometries / disentanglers / inter-disentanglers match the
        # encoded state's per-layer slots (reference-shared, §1.1).
        assert len(lyr.isometries) == len(state.isometries[ell])
        assert len(lyr.disentanglers) == len(state.disentanglers[ell])
        assert len(lyr.inter_disentanglers) == len(state.inter_disentanglers[ell])
        for w_layer, w_lyr in zip(state.isometries[ell], lyr.isometries):
            assert w_layer is w_lyr, "layer isometry must be a real MERA tensor"


# ---------------------------------------------------------------------------
# verify_layer_state_identical (via §12.2)
# ---------------------------------------------------------------------------


def test_wilson_signature_verifies_equivalence():
    """Re-encoding the same program yields layers whose Wilson-loop
    signatures (§12.2) agree, so verify_layer_state_identical returns True.
    Re-encoding a structurally different program disagrees."""
    src_a = r"\x:Int. x"
    src_b = r"\x:Int. \y:Int. x"

    layers_a1 = record_mera_layer_sequence(parse(src_a))
    layers_a2 = record_mera_layer_sequence(parse(src_a))
    layers_b = record_mera_layer_sequence(parse(src_b))

    # Same-program layers: §12.2 Wilson signatures must match.
    assert verify_layer_state_identical(layers_a1[0], layers_a2[0]) is True
    # Different-program layers: Wilson signatures differ (the encoded
    # states have different leaf bonds, so the Wilson loops cannot
    # coincide).
    if len(layers_b) > 0 and len(layers_a1) > 0:
        # The signatures must not match — different N and/or different
        # loop dict makes approx_equal short-circuit to False.
        assert verify_layer_state_identical(layers_a1[0], layers_b[0]) is False

    # Two layers from the same compilation trivially share the same
    # underlying state, so their signatures match.
    if len(layers_a1) >= 2:
        assert verify_layer_state_identical(layers_a1[0], layers_a1[1]) is True


# ---------------------------------------------------------------------------
# detect_compilation_convergence (via §12.8)
# ---------------------------------------------------------------------------


def test_dpt_detector_finds_compilation_phase_transition():
    """A synthetic non-convergent compilation snapshot sequence — built
    from orthogonal basis-state MERAs — triggers a §12.8 DPT event
    (Loschmidt echo = 0 between consecutive snapshots), so
    detect_compilation_convergence returns False.

    A constant-state snapshot sequence (identical MERA repeated) has
    unit Loschmidt echoes throughout, no DPT, and the detector returns
    True (converged)."""
    # Non-convergent: orthogonal basis states give |<Psi_i|Psi_{i+1}>|^2 = 0
    # which is below the default 1e-2 absolute DPT threshold.
    state_0 = make_product_mera([basis_leaf(0), basis_leaf(0)])
    state_1 = make_product_mera([basis_leaf(1), basis_leaf(1)])
    state_2 = make_product_mera([basis_leaf(2), basis_leaf(2)])

    non_convergent_snapshots = [state_0, state_1, state_2]
    assert detect_compilation_convergence(non_convergent_snapshots) is False

    # Convergent: identical states give unit echo, no DPT event.
    convergent_snapshots = [state_0, state_0, state_0]
    assert detect_compilation_convergence(convergent_snapshots) is True

    # Vacuously convergent: fewer than 2 snapshots cannot have a
    # consecutive-pair non-analyticity.
    assert detect_compilation_convergence([]) is True
    assert detect_compilation_convergence([state_0]) is True
