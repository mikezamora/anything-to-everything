"""Tests for the tensor-network typechecker (EXTENSIONS.md E11).

The typechecker MUST consult bond structure -- leaf one-hot tags and
``meta.forall_protected_leaves`` -- not the decoded AST. These tests
pin both the accept and reject paths and the dependent-product (TPi)
fiber-bundle check.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.lemma_library import (
    DerivationMetadata, Lemma, LemmaLibrary, bundle_from_mera,
    structural_fingerprint, register_lemma,
)
from src.qft_pcn.composition.tn_typechecker import (
    TypeCheckOk, TypeCheckError,
    tn_typecheck, tn_typecheck_bundle, pi_type_projector_expectation,
    _leaf_index,
)
from src.qft_pcn.logic.ast import (
    parse, TInt, TBool, TArrow, TNat, TProp, TPi,
)
from src.qft_pcn.logic.mera_encoder import encode_mera


def _build_lemma(src: str) -> Lemma:
    state, meta = encode_mera(parse(src))
    bundle = bundle_from_mera(state)
    fp = structural_fingerprint(state)
    deriv = DerivationMetadata(
        hamiltonian_id="tn-tc-test",
        residual_energy=0.0,
        energy_gap=1.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id="tn-tc-test",
    )
    return Lemma(
        lemma_id=f"tn-tc:{src}",
        proposition_type="ignored-for-tn-test",
        mera_tensors=bundle,
        encoding_meta=meta,
        derivation=deriv,
        fingerprint=fp,
    )


# ---- arrow / atomic accept paths -----------------------------------------


def test_well_typed_int_to_int_lemma_accepts():
    """``\\x:Int. x`` carries TYPE_ARR_II in the root type leaf and
    KIND_LAM in the root kind leaf -- the bond signature for Int -> Int."""
    lem = _build_lemma(r"\x:Int. x")
    verdict = tn_typecheck(lem, TArrow(src=TInt(), dst=TInt()))
    assert isinstance(verdict, TypeCheckOk), (
        f"expected accept; got {verdict!r}")
    assert "arrow" in verdict.bond_signature


def test_well_typed_int_literal_accepts():
    """Atomic Int literal -- root type leaf carries TYPE_INT."""
    lem = _build_lemma("7")
    verdict = tn_typecheck(lem, TInt())
    assert isinstance(verdict, TypeCheckOk), (
        f"expected accept; got {verdict!r}")
    assert "atomic" in verdict.bond_signature


def test_well_typed_bool_to_bool_lemma_accepts():
    lem = _build_lemma(r"\x:Bool. x")
    verdict = tn_typecheck(lem, TArrow(src=TBool(), dst=TBool()))
    assert isinstance(verdict, TypeCheckOk), (
        f"expected accept; got {verdict!r}")


# ---- reject paths --------------------------------------------------------


def test_ill_typed_arrow_rejected_on_type_tag():
    """``\\x:Int. x`` claimed as Int -> Bool must be rejected: the
    encoded type leaf is TYPE_ARR_II, not TYPE_ARR_IB."""
    lem = _build_lemma(r"\x:Int. x")
    verdict = tn_typecheck(lem, TArrow(src=TInt(), dst=TBool()))
    assert isinstance(verdict, TypeCheckError), (
        f"expected reject; got {verdict!r}")
    assert verdict.kind == "type_tag_mismatch"
    assert "tag" in verdict.detail


def test_ill_typed_atomic_rejected_kind_or_type_tag():
    """Int literal claimed as Bool must be rejected via the type-leaf
    bond tag mismatch (no kind constraint imposed for atomics)."""
    lem = _build_lemma("7")
    verdict = tn_typecheck(lem, TBool())
    assert isinstance(verdict, TypeCheckError), (
        f"expected reject; got {verdict!r}")
    assert verdict.kind == "type_tag_mismatch"


def test_atomic_claimed_as_arrow_rejected_on_kind():
    """An Int literal claimed as Int -> Int must be rejected because
    the root kind is KIND_INT, not KIND_LAM. The kind-leaf bond check
    is the load-bearing gate here."""
    lem = _build_lemma("7")
    verdict = tn_typecheck(lem, TArrow(src=TInt(), dst=TInt()))
    assert isinstance(verdict, TypeCheckError), (
        f"expected reject; got {verdict!r}")
    assert verdict.kind == "kind_tag_mismatch"


def test_ill_typed_corrupted_type_leaf_rejected():
    """Corrupt the root type leaf to a tag that does not match Int->Int.
    This drives the leaf-vector readback, *not* an AST walk: the AST is
    untouched; only the substrate amplitudes change."""
    lem = _build_lemma(r"\x:Int. x")
    bundle = lem.mera_tensors
    leaf_vectors = [np.array(v, copy=True) for v in bundle.leaf_vectors]
    type_leaf_idx = _leaf_index(0, "type")
    # Overwrite the one-hot tag at position TYPE_ARR_II with one at
    # TYPE_BOOL (clearly the wrong arrow tag). leaf shape (1,16,1).
    v = leaf_vectors[type_leaf_idx]
    v[...] = 0.0
    v[0, 2, 0] = 1.0  # TYPE_BOOL = 2
    from src.qft_pcn.composition.lemma_library import MeraTensorBundle
    corrupted = MeraTensorBundle(
        n_leaves=bundle.n_leaves,
        leaf_dim=bundle.leaf_dim,
        n_layers=bundle.n_layers,
        leaf_vectors=leaf_vectors,
        disentanglers=bundle.disentanglers,
        isometries=bundle.isometries,
        inter_disentanglers=bundle.inter_disentanglers,
        top=bundle.top,
        layer_dims=bundle.layer_dims,
    )
    verdict = tn_typecheck_bundle(
        corrupted, lem.encoding_meta,
        TArrow(src=TInt(), dst=TInt()),
    )
    assert isinstance(verdict, TypeCheckError), (
        f"corrupted type leaf must be rejected; got {verdict!r}")
    assert verdict.kind == "type_tag_mismatch"


# ---- Π_type (dependent product) ------------------------------------------


def test_pi_type_dependent_forall_accepts():
    """``forall x:Nat. Eq x x`` is a genuine dependent product: the
    body mentions the bound variable, so the encoder populates
    ``forall_protected_leaves`` with the binder's bid leaf and the
    bound-Var uses. The TN typechecker accepts ``TPi(src=TNat, ...)``
    against this lemma."""
    lem = _build_lemma("forall x:Nat. Eq x x")
    # Sanity: the encoder must produce a non-empty protected set, else
    # this test is checking the wrong substrate feature.
    assert lem.encoding_meta.forall_protected_leaves, (
        "encoder must record forall_protected_leaves for dependent "
        "forall; precondition for the TPi accept path")
    verdict = tn_typecheck(lem, TPi(src=TNat(), dst=TProp()))
    assert isinstance(verdict, TypeCheckOk), (
        f"expected accept; got {verdict!r}")
    assert "pi" in verdict.bond_signature
    assert "protected_leaves=" in verdict.bond_signature


def test_pi_type_wrong_param_type_rejected():
    """Same dependent forall claimed with TPi(src=TBool, ...) must be
    rejected: the value-leaf bond at the Forall site carries the Nat
    tag, not the Bool tag."""
    lem = _build_lemma("forall x:Nat. Eq x x")
    verdict = tn_typecheck(lem, TPi(src=TBool(), dst=TProp()))
    assert isinstance(verdict, TypeCheckError), (
        f"expected reject; got {verdict!r}")
    assert verdict.kind == "value_tag_mismatch"


def test_pi_type_rejects_arrow_lemma():
    """A plain ``\\x:Int. x`` lemma claimed as a Π type must be
    rejected on the root-kind bond: KIND_LAM, not KIND_FORALL.
    This pins the "Π is NOT just arrow" distinction at the substrate
    level."""
    lem = _build_lemma(r"\x:Int. x")
    verdict = tn_typecheck(lem, TPi(src=TInt(), dst=TInt()))
    assert isinstance(verdict, TypeCheckError), (
        f"expected reject; got {verdict!r}")
    assert verdict.kind == "kind_tag_mismatch"


# ---- projector measurement -----------------------------------------------


def test_pi_type_projector_returns_one_on_accept():
    lem = _build_lemma("forall x:Nat. Eq x x")
    val = pi_type_projector_expectation(lem, TPi(src=TNat(), dst=TProp()))
    assert val == pytest.approx(1.0)


def test_pi_type_projector_returns_zero_on_reject():
    lem = _build_lemma(r"\x:Int. x")
    val = pi_type_projector_expectation(lem, TPi(src=TInt(), dst=TInt()))
    assert val == pytest.approx(0.0)


def test_pi_type_projector_logs_warning_on_malformed_bundle(caplog):
    """E11 reviewer follow-up: the swallowed ``mera_from_bundle`` failure
    must surface as a warning record (so silent bundle corruption is
    diagnosable) while the projector still returns ``0.0`` to keep the
    surface total."""
    import logging
    from dataclasses import replace

    lem = _build_lemma(r"\x:Int. x")
    # Corrupt the bundle so ``mera_from_bundle`` raises during rebuild.
    # Replacing leaf_vectors with an all-zero array of the wrong shape
    # forces a shape/contract error in the rebuild path.
    bad_leaf = np.zeros((1, 1, 1), dtype=np.complex128)
    bad_bundle = replace(lem.mera_tensors, leaf_vectors=[bad_leaf])
    bad_lem = Lemma(
        lemma_id=lem.lemma_id,
        proposition_type=lem.proposition_type,
        mera_tensors=bad_bundle,
        encoding_meta=lem.encoding_meta,
        derivation=lem.derivation,
        fingerprint=lem.fingerprint,
    )
    with caplog.at_level(
        logging.WARNING,
        logger="src.qft_pcn.composition.tn_typechecker",
    ):
        val = pi_type_projector_expectation(
            bad_lem, TPi(src=TInt(), dst=TInt()),
        )
    assert val == pytest.approx(0.0)
    assert any(
        "pi_type_projector" in rec.message and "mera_from_bundle failed" in rec.message
        for rec in caplog.records
    ), f"expected warning record; got {[r.message for r in caplog.records]!r}"


# ---- register_lemma wiring -----------------------------------------------


def test_register_lemma_with_expected_type_gates_admission(tmp_path):
    """When ``expected_type`` is supplied, ``register_lemma`` runs the
    TN typechecker as a second gate. An ill-typed claim must be
    rejected with ``tn_typecheck_failed:*`` even when the residual is
    zero."""
    lib = LemmaLibrary(tmp_path)
    state, meta = encode_mera(parse(r"\x:Int. x"))
    deriv = DerivationMetadata(
        hamiltonian_id="reg-test",
        residual_energy=0.0,
        energy_gap=1.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id="reg-test",
    )
    # Correct expected type -> accepted.
    ok = register_lemma(
        lib, state, meta, hamiltonian=None, derivation=deriv,
        expected_type=TArrow(src=TInt(), dst=TInt()),
    )
    assert ok.accepted, f"correct expected_type must accept; got {ok!r}"
    # Wrong expected type -> rejected on the TN gate, not the residual.
    bad = register_lemma(
        lib, state, meta, hamiltonian=None, derivation=deriv,
        expected_type=TArrow(src=TInt(), dst=TBool()),
    )
    assert not bad.accepted
    assert bad.reason.startswith("tn_typecheck_failed:"), (
        f"expected tn_typecheck_failed prefix; got reason={bad.reason!r}")


def test_register_lemma_default_preserves_prior_behavior(tmp_path):
    """Without an ``expected_type`` arg, the new TN gate is bypassed:
    the residual gate + the soft ``_validate_decoded`` tautology
    admit the lemma exactly as before."""
    lib = LemmaLibrary(tmp_path)
    state, meta = encode_mera(parse(r"\x:Int. x"))
    deriv = DerivationMetadata(
        hamiltonian_id="reg-test",
        residual_energy=0.0,
        energy_gap=1.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id="reg-test",
    )
    res = register_lemma(
        lib, state, meta, hamiltonian=None, derivation=deriv,
    )
    assert res.accepted, f"default path must accept; got {res!r}"
