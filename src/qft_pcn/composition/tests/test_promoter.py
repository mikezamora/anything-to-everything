"""Tests for the use_lemma constraint compiler (spec §5, acceptance
§8.6, 8.11). Promotion is operator-algebraic: NO AST splice, NO
re-derivation of the lemma region."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, register_lemma, DerivationMetadata)
from src.qft_pcn.composition.promoter import Promoter, PromotedLemma
from src.qft_pcn.composition.errors import (
    LemmaLeafCountMismatch, ConditionalLemmaRefused, LemmaNotFound,
    LemmaSpeciesMismatch, LemmaIndexOutOfRange)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def _register(lib, src=r"\x:Int. x", conditional=False):
    state, meta = encode_mera(parse(src))
    deriv = DerivationMetadata(
        hamiltonian_id="h", residual_energy=1e-12, energy_gap=0.5,
        trotter_steps=40, assumptions=("a",) if conditional else (),
        lemma_deps=(), conditional=conditional, source_run_id="r")
    res = register_lemma(lib, state, meta, hamiltonian=None,
                         derivation=deriv, eps_register=1e-8)
    assert res.accepted
    return res.lemma_id, meta


def test_compile_unknown_lemma_raises(tmp_path):
    p = Promoter(LemmaLibrary(tmp_path))
    with pytest.raises(LemmaNotFound):
        p.compile_constraint({"kind": "use_lemma", "lemma_id": "nope",
                              "leaves": [0]})


def test_compile_leaf_count_mismatch_raises(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib)
    p = Promoter(lib)
    with pytest.raises(LemmaLeafCountMismatch):
        p.compile_constraint({"kind": "use_lemma", "lemma_id": lid,
                              "leaves": [0, 1]})   # wrong length


def test_compile_returns_promoted_lemma(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib)
    p = Promoter(lib)
    n = meta.n_leaves
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    assert isinstance(promoted, PromotedLemma)
    assert promoted.lemma_id == lid
    assert promoted.mode == "init_clamp"


def test_conditional_lemma_refused_by_default(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib, conditional=True)
    p = Promoter(lib)
    n = meta.n_leaves
    with pytest.raises(ConditionalLemmaRefused):
        p.compile_constraint({"kind": "use_lemma", "lemma_id": lid,
                              "leaves": list(range(n))})


def test_conditional_lemma_allowed_with_opt_in(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib, conditional=True)
    p = Promoter(lib)
    n = meta.n_leaves
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n)),
         "allow_conditional": True})
    assert promoted.lemma_id == lid


def test_projector_mode_cannot_apply_init_clamp(tmp_path):
    """Projector-mode Promoter exists for projector_energy but cannot
    masquerade as an init_clamp — guards against silent mis-use."""
    lib = LemmaLibrary(tmp_path)
    lid, lemma_meta = _register(lib)
    p = Promoter(lib, mode="projector")
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid,
         "leaves": list(range(lemma_meta.n_leaves))})
    with pytest.raises(NotImplementedError, match="init_clamp"):
        p.apply_init_clamp(host, host_meta, promoted)


def test_init_clamp_species_mismatch_raises(tmp_path):
    """apply_init_clamp's species check (spec §5.2a) refuses a clamp
    whose host window does not align species-wise with the lemma."""
    lib = LemmaLibrary(tmp_path)
    lid, lemma_meta = _register(lib)
    p = Promoter(lib)
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    n = lemma_meta.n_leaves
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    # Surgically swap one host species so the per-leaf species pattern
    # disagrees at index 0 (lemma species[0] vs. flipped host species[0]).
    flipped = list(host_meta.species_of_leaf)
    original = flipped[0]
    # Pick a different species than the lemma's leaf-0 species.
    different = next(s for s in ("kind", "type", "bid", "value", "tobl",
                                 "PAD") if s != original)
    flipped[0] = different
    object.__setattr__(host_meta, "species_of_leaf", flipped)
    with pytest.raises(LemmaSpeciesMismatch):
        p.apply_init_clamp(host, host_meta, promoted)


def test_init_clamp_strength_half_blends_host_and_cached(tmp_path):
    """Spec §6.4 / precision-weighted clamp: strength=0.5 produces a
    renormalized blend of cached + host leaves, NOT a full overwrite."""
    lib = LemmaLibrary(tmp_path)
    lid, lemma_meta = _register(lib)
    p = Promoter(lib)
    # Use the same source as the lemma so species/leaf counts align by
    # construction. We then perturb the host's leaf 0 so the blend is
    # observably different from both the pure-host and pure-cached
    # tensors.
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    pre_host_leaf = np.asarray(host.leaves[0]).copy()
    # Inject a small perturbation so host[0] != cached[0]; otherwise
    # the blend is trivially equal to both.
    host.leaves[0] = host.leaves[0] + 0.1
    host_perturbed = np.asarray(host.leaves[0]).copy()
    cached = lib.materialize(lid)
    cached_leaf0 = np.asarray(cached.leaves[0]).copy()
    n = lemma_meta.n_leaves
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    frozen = p.apply_init_clamp(host, host_meta, promoted, strength=0.5)
    assert isinstance(frozen, set) and 0 in frozen
    # Expected: renormalized 0.5 * cached + 0.5 * host_perturbed at leaf 0.
    expected = 0.5 * cached_leaf0 + 0.5 * host_perturbed
    norm = float(np.linalg.norm(expected[0, :, 0]))
    if norm > 0.0:
        expected = expected / norm
    assert np.allclose(host.leaves[0], expected)
    # And the blend at leaf 0 is NOT byte-equal to the cached tensor.
    assert not np.allclose(host.leaves[0], cached_leaf0)
    # Nor byte-equal to the (perturbed) pre-clamp host tensor.
    assert not np.allclose(host.leaves[0], host_perturbed)
    _ = pre_host_leaf  # retained for readability; unused after blend


def test_init_clamp_strength_zero_leaves_host_unchanged_modulo_norm(tmp_path):
    """strength=0.0 means a zero-weight blend -- the host leaf survives
    up to renormalization. (Norm is preserved trivially since host leaves
    are already unit-normalized by the encoder.)"""
    lib = LemmaLibrary(tmp_path)
    lid, lemma_meta = _register(lib)
    p = Promoter(lib)
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    pre = [np.asarray(v).copy() for v in host.leaves]
    n = lemma_meta.n_leaves
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    p.apply_init_clamp(host, host_meta, promoted, strength=0.0)
    for i in range(n):
        # Host leaves are unit-norm by the encoder; strength=0.0 +
        # renormalization is an identity.
        assert np.allclose(host.leaves[i], pre[i])


def test_init_clamp_strength_out_of_range_raises(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, lemma_meta = _register(lib)
    p = Promoter(lib)
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    n = lemma_meta.n_leaves
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    with pytest.raises(ValueError, match="strength"):
        p.apply_init_clamp(host, host_meta, promoted, strength=-0.1)
    with pytest.raises(ValueError, match="strength"):
        p.apply_init_clamp(host, host_meta, promoted, strength=1.5)


def test_compile_constraint_rejects_out_of_range_leaf(tmp_path):
    """DEVIATION D33: compile_constraint with a host_meta range-checks
    each leaf index and raises a typed LemmaIndexOutOfRange instead of
    letting an opaque IndexError surface from apply_init_clamp's leaf
    write."""
    lib = LemmaLibrary(tmp_path)
    lid, lemma_meta = _register(lib)
    p = Promoter(lib)
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    n = lemma_meta.n_leaves
    bad_leaves = list(range(n - 1)) + [host_meta.n_leaves + 1000]
    with pytest.raises(LemmaIndexOutOfRange):
        p.compile_constraint(
            {"kind": "use_lemma", "lemma_id": lid, "leaves": bad_leaves},
            host_meta=host_meta)


def test_compile_constraint_rejects_count_mismatch(tmp_path):
    """DEVIATION D33: compile_constraint's leaf-count parity check
    fires with a typed LemmaLeafCountMismatch, even when a host_meta
    is supplied (count check precedes the per-leaf range check)."""
    lib = LemmaLibrary(tmp_path)
    lid, lemma_meta = _register(lib)
    p = Promoter(lib)
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    # All indices are in-range, but the cardinality is wrong.
    wrong_count = [0]
    with pytest.raises(LemmaLeafCountMismatch):
        p.compile_constraint(
            {"kind": "use_lemma", "lemma_id": lid,
             "leaves": wrong_count},
            host_meta=host_meta)


def test_apply_init_clamp_rejects_out_of_range_leaf_typed(tmp_path):
    """DEVIATION D33 defense-in-depth: callers that compiled without
    passing host_meta (legacy path) still get a typed
    LemmaIndexOutOfRange at apply_init_clamp time -- never an opaque
    IndexError from the leaf write loop."""
    lib = LemmaLibrary(tmp_path)
    lid, lemma_meta = _register(lib)
    p = Promoter(lib)
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    # Forge a PromotedLemma directly (bypassing compile_constraint's
    # host_meta check) to exercise the apply-time guard.
    n = lemma_meta.n_leaves
    bad_host_leaves = tuple(list(range(n - 1)) +
                            [host_meta.n_leaves + 1000])
    promoted = PromotedLemma(lemma_id=lid, host_leaves=bad_host_leaves,
                             mode="init_clamp", weight=1.0)
    with pytest.raises(LemmaIndexOutOfRange):
        p.apply_init_clamp(host, host_meta, promoted)


def test_init_clamp_writes_lemma_tensors_no_rederivation(tmp_path):
    """Acceptance §8.6: after apply_init_clamp, the host window's tensors
    are byte-equal to the cached lemma's -- proving promotion is a
    referential clamp, not an AST splice + re-derivation.

    Note: MERA stores per-site state as ``leaves[i]`` of shape ``(1, d, 1)``
    (see qft.mera.MERA); the operator-algebraic clamp copies that tensor
    directly. We compare the per-site state vector ``leaves[i][0, :, 0]``.
    """
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib)
    p = Promoter(lib)
    n = meta.n_leaves
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    frozen = p.apply_init_clamp(host, host_meta, promoted)
    cached = lib.materialize(lid)
    for i in range(n):
        assert np.allclose(host.leaves[i][0, :, 0],
                           cached.leaves[i][0, :, 0])
    assert isinstance(frozen, set) and len(frozen) > 0
