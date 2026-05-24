"""Gap F regression: parse_kind_stream accepts trailing Forall-protected
KIND_VAR sites as structurally-dead-yet-entanglement-alive (§1.1
binding-as-entanglement, I-Task-10 #5).

After R-Eq-Refl promotes ``Eq(add x Zero, x)`` to ``BoolLit(True)`` the
non-protected lhs descendants collapse to KIND_PAD, but the bound
``Var x`` use sites stay KIND_VAR -- their entanglement IS the
universal-quantification proof. The trailing-PAD scan in
``parse_kind_stream`` must read the protected-leaves set as a
structural-deadness oracle and accept those sites.

Three regression tiers:
  1. Unit-level: a synthetic stream where a downstream site is
     KIND_VAR but its 5 species leaves are all in
     ``forall_protected_leaves`` -- parse must succeed.
  2. Unit-level (negative): same stream WITHOUT the protected-leaves
     argument -- parse must raise DecodeError (Gap E contract: a
     non-protected non-PAD trailing site is a substrate bug).
  3. End-to-end: the §10.10 composite ``forall x:Nat. Eq (add x Zero) x``
     encoded + evolved through R-AddZero + R-Eq-Refl, then decoded via
     ``decode_mera`` -- no DecodeError.
"""
from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import (
    Forall, Eq, Bin, Var, Zero, TNat,
)
from src.qft_pcn.logic.decoder import parse_kind_stream, DecodeError
from src.qft_pcn.logic.encoding import (
    KIND_PAD, KIND_VAR, KIND_BOOL,
    TYPE_NONE, TYPE_BOOL,
    VALUE_TRUE,
)
from src.qft_pcn.logic.mera_encoding import LEAVES_PER_NODE
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera


def _synthetic_stream_with_protected_var_trailing():
    """Stream: site 0 is BoolLit(True) (active AST); sites 1..N carry
    KIND_VAR (the §1.1 leftover, protected). Returns the 5-tuple list
    plus the full protected-leaves set (all 5 species of every
    trailing site)."""
    # Site 0: active AST -- BoolLit(True). 5-tuple: (kind, type, bid, value, tobl).
    site0 = (KIND_BOOL, TYPE_BOOL, 0, VALUE_TRUE, 0)
    # Sites 1 and 2: KIND_VAR survivors (bound-Var-use leftovers from
    # a collapsed Forall body). bid=1 references the (now-implicit)
    # binder; the value/type bits are immaterial to the trailing scan.
    var_site = (KIND_VAR, TYPE_NONE, 1, 0, 0)
    decoded = [site0, var_site, var_site]
    # All 5 species leaves of sites 1 and 2 are protected.
    protected: set[int] = set()
    for site_idx in (1, 2):
        for offset in range(LEAVES_PER_NODE):
            protected.add(LEAVES_PER_NODE * site_idx + offset)
    return decoded, protected


def test_parse_kind_stream_accepts_protected_trailing_var():
    """Trailing KIND_VAR sites whose 5 species leaves are entirely
    contained in ``forall_protected_leaves`` are accepted as
    structurally-dead-yet-entanglement-alive (Gap F fix)."""
    decoded, protected = _synthetic_stream_with_protected_var_trailing()
    ast = parse_kind_stream(
        decoded, nested_type_index={}, forall_protected_leaves=protected,
    )
    # The active AST is site 0 (BoolLit True). The trailing KIND_VAR
    # sites are entanglement-alive but structural-AST-dead.
    from src.qft_pcn.logic.ast import BoolLit
    assert isinstance(ast, BoolLit), f"expected BoolLit, got {type(ast).__name__}"
    assert ast.val is True


def test_parse_kind_stream_rejects_non_protected_trailing_var():
    """Without ``forall_protected_leaves`` (or with sites NOT in the
    protected set), trailing non-PAD sites remain a hard failure --
    Gap E's contract (R-Eq-Refl DFS collapses non-protected
    descendants to PAD) must not be weakened."""
    decoded, _protected = _synthetic_stream_with_protected_var_trailing()
    # Pass an empty protected set: the trailing scan must reject.
    with pytest.raises(DecodeError, match="not PAD after AST parse"):
        parse_kind_stream(
            decoded, nested_type_index={}, forall_protected_leaves=set(),
        )
    # Also: passing None (the MPS / strict default) must reject too.
    with pytest.raises(DecodeError, match="not PAD after AST parse"):
        parse_kind_stream(decoded, nested_type_index={})


def test_parse_kind_stream_rejects_partial_protection():
    """A site whose protection is INCOMPLETE (only some of its 5
    species leaves are in the protected set) must still be rejected
    -- the oracle is "ALL 5 leaves protected", not "any leaf
    protected". This pins the strict-by-default contract per the
    spec: a partially-protected site is a substrate bug, not a
    §1.1 binding."""
    decoded, _full = _synthetic_stream_with_protected_var_trailing()
    # Only protect 4 of the 5 leaves for site 1 (drop one).
    partial: set[int] = set()
    for offset in range(LEAVES_PER_NODE - 1):  # skip last
        partial.add(LEAVES_PER_NODE * 1 + offset)
    for offset in range(LEAVES_PER_NODE):
        partial.add(LEAVES_PER_NODE * 2 + offset)
    with pytest.raises(DecodeError, match="not PAD after AST parse"):
        parse_kind_stream(
            decoded, nested_type_index={}, forall_protected_leaves=partial,
        )


@pytest.mark.timeout(240)
def test_decode_mera_post_eqrefl_succeeds():
    """End-to-end Gap F regression: the §10.10 composite
    ``forall x:Nat. Eq (add x Zero) x`` is encoded + evolved through
    the real MERA imaginary-time loop with frozen-leaves
    (I-Task-10 #5); ``decode_mera`` must NOT trip the trailing-PAD
    check on the Forall-protected Var leftover sites.

    Pre-fix this raised
    ``DecodeError("site N not PAD after AST parse (kind=1)")`` because
    the bound-Var-use sites kept their KIND_VAR bits while the rest of
    the Eq subtree collapsed to PAD. Post-fix the parse succeeds; the
    decoded AST is the surviving active root (the form depends on the
    extent of the in-substrate R-Eq-Refl + co-projection, but the
    parse-success invariant is what Gap F is about).
    """
    # Build forall x:Nat. Eq (add x Zero) x directly (the surface
    # parser may not expose Eq).
    src = Forall(
        param="x", param_ty=TNat(),
        body=Eq(
            lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
            rhs=Var(name="x"),
        ),
    )
    from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
        MeraEvalHamiltonian,
    )
    from src.qft_pcn.logic.mera_evolution_logic import (
        mera_imaginary_evolve_state,
    )

    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    protected = set(meta.forall_protected_leaves)
    # Use the same evolution parameters as the K-8 acceptance test
    # (see test_cross_level_acceptance._real_child_runner).
    _, final = mera_imaginary_evolve_state(
        state, H, dt=0.1, steps=40, chi_layer=32,
        frozen_leaves=protected,
    )
    # The acid test: decode_mera must succeed (no DecodeError) on a
    # state where R-Eq-Refl has promoted the Eq while leaving the
    # bound-Var sites entanglement-alive.
    result = decode_mera(final, meta)
    # The decoded AST must be SOMETHING (not None) -- the parse
    # succeeded. We do not over-specify the exact shape here: that's
    # what test_mera_eqrefl_rule covers. Gap F is about parse-success
    # alone.
    assert result.ast is not None
