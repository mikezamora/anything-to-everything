"""Gap D regression tests: `_meta_to_json` / `_meta_from_json` must
round-trip the I-10-added ``MeraEncodingMeta.forall_protected_leaves:
set[int]`` field. Prior to the fix, ``json.dumps`` raised
``TypeError: Object of type set is not JSON serializable`` and every
``LemmaLibrary.save(lemma)`` against any current encoder's meta
failed before persistence."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.lemma_library import (
    DerivationMetadata,
    Lemma,
    LemmaLibrary,
    _META_SET_FIELDS,
    _meta_from_json,
    _meta_to_json,
    _ty_from_json,
    _ty_to_json,
    bundle_from_mera,
    structural_fingerprint,
)
from src.qft_pcn.logic.ast import (
    Bin, BoolLit, Cons, Eq, Forall, Nil, TArrow, TBool, TInt, TList, TNat,
    TProp, Var, Zero,
)
from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta, encode_mera


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadow the conftest §9.7 ceiling for this file
    """Opt out of the composition/tests/conftest.py dense-tensor cap:
    the real ``encode_mera`` of ``forall x:Nat. Eq (add x Zero) x``
    allocates 16-dim leaves and 16-up isometries (4096-element pair
    matrices) that exceed the conftest cap by construction. The hand-
    built ``MeraEncodingMeta`` tests below do not allocate; this fixture
    only matters for ``test_library_save_load_with_forall_meta``."""
    yield


def _forall_addzero_ast() -> Forall:
    """``forall x:Nat. Eq (add x Zero) x`` -- the §10.10 composite."""
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    return Forall(param="x", param_ty=TNat(), body=body)


def _minimal_meta(forall_protected: set[int]) -> MeraEncodingMeta:
    """Hand-built meta isolating the set-typed field. The other fields
    are populated with the minimum the dataclass requires; we are only
    exercising the JSON serializer."""
    return MeraEncodingMeta(
        n_nodes=1,
        n_leaves=5,
        L=1,
        leaf_dim=8,
        species_of_leaf=["kind", "bid", "tobl", "value", "type"],
        node_of_leaf=[0, 0, 0, 0, 0],
        site_to_ast_path={0: (0,)},
        binder_leaves={},
        use_to_binder={},
        nested_type_index={},
        children_of_node={0: []},
        n_nodes_max=1,
        hole_regions=[],
        witness_node_ranges=[],
        forall_protected_leaves=forall_protected,
        typehole_regions=[],
    )


def test_meta_to_json_handles_forall_protected_leaves_set():
    """Gap D core: serializer must accept ``set[int]`` without raising."""
    meta = _minimal_meta({1, 2, 3})
    s = _meta_to_json(meta)
    # Sanity: the persisted form is JSON-parseable and carries the field
    # as a list. Sorted-list invariant: deterministic on-disk shape.
    import json
    parsed = json.loads(s)
    assert parsed["forall_protected_leaves"] == [1, 2, 3]


def test_meta_from_json_restores_set():
    """Round-trip ``set[int]`` -> JSON -> ``set[int]``. The restored
    field must compare equal AS A SET (not as a list); evolution
    drivers consume it with ``in``-tests and set-algebra."""
    original = {1, 2, 3}
    meta = _minimal_meta(original)
    restored = _meta_from_json(_meta_to_json(meta))
    assert isinstance(restored.forall_protected_leaves, set)
    assert restored.forall_protected_leaves == original
    # Element type also matters -- evolution drivers compare leaf
    # indices as ints, not strings.
    for x in restored.forall_protected_leaves:
        assert isinstance(x, int)


def test_meta_to_json_handles_empty_set():
    """Empty ``set()`` default must also serialize and round-trip --
    every current encoder produces this for non-Forall ASTs."""
    meta = _minimal_meta(set())
    restored = _meta_from_json(_meta_to_json(meta))
    assert restored.forall_protected_leaves == set()
    assert isinstance(restored.forall_protected_leaves, set)


def test_meta_set_fields_autodetected_via_introspection():
    """``_META_SET_FIELDS`` is the single source of truth for which
    ``MeraEncodingMeta`` fields round-trip through a ``set`` restore.
    It is autodetected from ``typing.get_type_hints(MeraEncodingMeta)``
    -- the test pins the contract:

    1. ``forall_protected_leaves: set[int]`` MUST appear (the field that
       motivated Gap D's fix; missing it silently regresses to lists on
       load, breaking ``in``-tests in evolution drivers).
    2. ``list``-typed fields (``species_of_leaf``, ``node_of_leaf``,
       ``hole_regions``, ``witness_node_ranges``, ``typehole_regions``)
       MUST NOT appear -- a false positive would attempt ``int(x) for x``
       over a list of strings / tuples / dicts and crash on load.

    A future maintainer who adds e.g. ``set[tuple[int, int]]`` will see
    this test still pass (the field is autodetected) but the loader
    (``set(int(x) for x in ...)``) will then crash on first load -- by
    design, so the schema-vs-loader mismatch surfaces immediately.
    """
    assert "forall_protected_leaves" in _META_SET_FIELDS, (
        "Gap D regression: forall_protected_leaves dropped from the "
        "autodetected set-field tuple -- introspection over "
        "MeraEncodingMeta type hints failed to spot set[int]"
    )
    # No false positives: every current list-typed field stays out.
    list_typed_fields = (
        "species_of_leaf",
        "node_of_leaf",
        "hole_regions",
        "witness_node_ranges",
        "typehole_regions",
    )
    for name in list_typed_fields:
        assert name not in _META_SET_FIELDS, (
            f"introspection false positive: list-typed field {name!r} "
            "leaked into _META_SET_FIELDS -- the int-coerce loader will "
            "crash on first load"
        )


def test_library_save_load_with_forall_meta(tmp_path):
    """End-to-end: encode a Forall AST through the real encoder, pack
    it as a Lemma, save through LemmaLibrary, load it back, and assert
    the loaded ``forall_protected_leaves`` matches the encoder's
    original. Pre-fix, ``save`` raised ``TypeError`` before persistence."""
    ast = _forall_addzero_ast()
    state, meta = encode_mera(ast)
    # The encoder is required to populate the protected set for a
    # Forall AST; otherwise this test would not be exercising the gap.
    assert isinstance(meta.forall_protected_leaves, set)
    assert meta.forall_protected_leaves, (
        "encoder produced an empty forall_protected_leaves -- the "
        "set serialization gap cannot be exercised on this AST"
    )

    bundle = bundle_from_mera(state)
    fp = structural_fingerprint(state)
    deriv = DerivationMetadata(
        hamiltonian_id="test-gap-d",
        residual_energy=0.0,
        energy_gap=1.0,
        trotter_steps=0,
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id="test-gap-d",
    )
    lemma = Lemma(
        lemma_id="test-gap-d:forall_eq_addzero",
        proposition_type="forall x:Nat. Eq (add x Zero) x",
        mera_tensors=bundle,
        encoding_meta=meta,
        derivation=deriv,
        fingerprint=fp,
    )

    lib = LemmaLibrary(tmp_path)
    lib.save(lemma)  # would raise TypeError pre-fix

    loaded = lib.load(lemma.lemma_id)
    assert isinstance(loaded.encoding_meta.forall_protected_leaves, set)
    assert (
        loaded.encoding_meta.forall_protected_leaves
        == meta.forall_protected_leaves
    ), (
        "Round-tripped forall_protected_leaves diverged from the "
        "encoder's original set"
    )


# ---- Ty (de)serializer unit + regression (silent data-loss) ---------------


def test_ty_serializer_round_trip_unit():
    """``_ty_to_json`` / ``_ty_from_json`` cover the canonical Ty
    subclasses parked in ``nested_type_index``: TNat, TBool, TInt,
    TProp, TList, TArrow (recursive on elem and on src/dst)."""
    cases = [
        TNat(),
        TBool(),
        TInt(),
        TProp(),
        TList(elem=TBool()),
        TList(elem=TList(elem=TArrow(src=TNat(), dst=TBool()))),
        TArrow(src=TNat(), dst=TArrow(src=TBool(), dst=TInt())),
    ]
    for ty in cases:
        assert _ty_from_json(_ty_to_json(ty)) == ty, (
            f"Ty round-trip diverged for {ty!r}"
        )


def test_ty_serializer_rejects_unknown_subclass():
    """Unknown Ty subclass must raise ``TypeError`` (no silent drop).

    Pins the loud-failure contract: a future maintainer adding e.g.
    ``TRefine`` will see a clear error rather than the prior best-effort
    ``json.dumps``-then-skip behavior that silently lost data on load.
    """
    class _BogusTy:
        pass
    with pytest.raises(TypeError, match="unknown Ty subclass"):
        _ty_to_json(_BogusTy())
    with pytest.raises(TypeError, match="unknown Ty kind tag"):
        _ty_from_json({"kind": "TRefine"})


def _forall_list_bool_ast() -> Forall:
    """``forall xs:List Bool. Eq (Cons true Nil) xs`` -- a Forall whose
    param_ty is the non-flat ``TList(TBool())``. The encoder stores the
    full Ty in ``nested_type_index`` at the binder site; the silent-skip
    serializer dropped it, and the decoder fell back to TList(TNat()).
    """
    body = Eq(lhs=Cons(head=BoolLit(val=True), tail=Nil()),
              rhs=Var(name="xs"))
    return Forall(param="xs", param_ty=TList(elem=TBool()), body=body)


def test_meta_to_json_preserves_list_bool_through_nested_type_index(tmp_path):
    """Round-trip ``forall xs:List Bool`` lemma through ``_meta_to_json``
    / ``_meta_from_json``; assert decoded ``param_ty.elem`` is ``TBool``,
    NOT ``TNat``. Pins the silent-data-loss bug surfaced by Forall
    param_ty spec review (`6a455ad`).

    Pre-fix: ``_meta_to_json`` silently dropped every ``Ty`` value from
    ``nested_type_index`` (``json.dumps(TList(...))`` raised; the
    except-pass swallowed it). Cross-session load then defaulted to
    ``TList(elem=TNat())`` via ``_extended_type_from_tag``'s legacy
    fallback.
    """
    from src.qft_pcn.logic.mera_encoder import encode_mera
    from src.qft_pcn.logic.mera_decoder import decode_mera

    ast = _forall_list_bool_ast()
    state, meta = encode_mera(ast)

    # Sanity: encoder did park a TList(TBool) somewhere in the side table.
    list_bool_sites = [
        site for site, ty in meta.nested_type_index.items()
        if isinstance(ty, TList) and isinstance(ty.elem, TBool)
    ]
    assert list_bool_sites, (
        "encoder did not record TList(TBool()) in nested_type_index -- "
        "this test cannot exercise the silent-data-loss path"
    )

    # Round-trip the meta through JSON.
    restored = _meta_from_json(_meta_to_json(meta))

    # The restored side table preserves TList(TBool()) at the same site(s).
    for site in list_bool_sites:
        cached = restored.nested_type_index.get(site)
        assert isinstance(cached, TList), (
            f"site {site}: TList entry dropped from nested_type_index "
            f"(silent-data-loss bug); got {cached!r}"
        )
        assert isinstance(cached.elem, TBool), (
            f"site {site}: TList.elem regressed to {cached.elem!r}; "
            "expected TBool() -- silent fallback to TList(TNat()) means "
            "the Ty (de)serializer is not wired"
        )

    # End-to-end: decode_mera against the restored meta gives back a
    # Forall whose param_ty is TList(TBool()), not TList(TNat()).
    decoded = decode_mera(state, restored)
    decoded_ast = getattr(decoded, "ast", decoded)
    assert isinstance(decoded_ast, Forall)
    assert isinstance(decoded_ast.param_ty, TList)
    assert isinstance(decoded_ast.param_ty.elem, TBool), (
        f"decode_mera produced param_ty={decoded_ast.param_ty!r}; "
        "expected TList(TBool()) -- silent data loss in lemma_library "
        "JSON round-trip regressed param_ty.elem to TNat"
    )
