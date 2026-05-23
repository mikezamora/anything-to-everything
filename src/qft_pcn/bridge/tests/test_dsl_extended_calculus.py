"""Round-trip the extended-calculus surface through the textual AST
parser and (for the in-substrate composite) through ``encode_mera`` +
``mera_imaginary_evolve_state``.

Addresses EXTENSIONS.md §335-353 (K-8 Blocker B): the AST parser
extension from `52e9387` introduced `forall`/`Eq`/`Nat`/`add`/`Zero`/
`Succ`/`NatLit` *tokens* but did not exercise them through a parse →
encode → evolve → decode round-trip from the bridge layer.  This file
closes that loop for everything the encoder substrate already supports
and **defers** the spec's literal §10.10 list-induction theorem
(`forall xs:List A. length (reverse xs) = length xs`) until the
encoder gains `length`/`reverse` -- see the new EXTENSIONS.md entry
"List arithmetic in encoder substrate".

§1.1 architecture-soul check: the bridge layer is ROUTING, not
classical interpretation.  We parse to AST and hand the AST to
``encode_mera`` -- the encoder enforces the Forall-protected
entanglement structure (``MeraEncodingMeta.forall_protected_leaves``
from I-10 #5).  We then evolve under the real
``MeraEvalHamiltonian`` and verify the Eq node promotes to
``KIND_BOOL`` / ``VALUE_TRUE`` from the bond-entangled fixed point --
no classical universal-introduction tag is ever introduced.
"""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import (
    App,
    Bin,
    Cons,
    Eq,
    Forall,
    NatLit,
    Nil,
    Succ,
    TList,
    TNat,
    Var,
    Zero,
    parse,
)
from src.qft_pcn.logic.encoding import VALUE_TRUE
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_encoding import KIND_BOOL
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian,
    RULE_R_ADD_ZERO,
    RULE_R_EQ_REFL,
    _kind_leaf,
    _leaf_weights,
    _value_leaf,
)
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_imaginary_evolve_state,
)


# Evolution settings mirror ``composition/tests/test_cross_level_acceptance.py``:
# this file's substrate round-trip is the same proof, driven from the
# bridge-layer DSL surface rather than from a hand-built AST.
_EVOLVE_DT = 0.1
_EVOLVE_STEPS = 300
_EVOLVE_CHI = 16


# ---------------------------------------------------------------------------
# Phase 1: parse-only round-trips for every surface keyword the
# I-Task-10 extension introduced + the EXTENSIONS.md §335-353 follow-on
# additions (List/Cons/Nil).  Phase 2 (below) drives the parsed AST
# through the encoder.
# ---------------------------------------------------------------------------


def test_parses_forall_eq_nat():
    """`forall n:Nat. Eq (add n Zero) n` parses to
    Forall(param_ty=TNat, body=Eq(App(App(Var('add'), Var('n')), Zero), Var('n')))).

    `add` is a free variable here, not a primitive -- the substrate
    handles `x + Zero` via the ``Bin('+')`` form that the M2
    R-AddZero rewrite rule fires on (see
    ``composition/tests/test_cross_level_acceptance.py::_ast_theorem``).
    The textual `add` surface is preserved so future encoder extensions
    can promote it to a primitive without breaking this round-trip."""
    ast = parse("forall n:Nat. Eq (add n Zero) n")
    assert isinstance(ast, Forall)
    assert ast.param == "n"
    assert isinstance(ast.param_ty, TNat)
    assert isinstance(ast.body, Eq)
    lhs = ast.body.lhs
    assert isinstance(lhs, App)
    assert isinstance(lhs.fn, App)
    assert isinstance(lhs.fn.fn, Var) and lhs.fn.fn.name == "add"
    assert isinstance(lhs.fn.arg, Var) and lhs.fn.arg.name == "n"
    assert isinstance(lhs.arg, Zero)
    assert isinstance(ast.body.rhs, Var) and ast.body.rhs.name == "n"


def test_parses_natlit_succ():
    """`Succ (NatLit 3)` parses to Succ(NatLit(3))."""
    ast = parse("Succ (NatLit 3)")
    assert isinstance(ast, Succ)
    assert isinstance(ast.arg, NatLit)
    assert ast.arg.val == 3


def test_parses_natlit_paren_form():
    """`NatLit(7)` (paren-call form) also parses to NatLit(7)."""
    ast = parse("NatLit(7)")
    assert isinstance(ast, NatLit)
    assert ast.val == 7


def test_parses_eq_atom():
    """`Eq Zero Zero` parses to Eq(Zero, Zero) (atom-position Eq with
    two atom arguments, mirroring the surface form used inside the
    §10.10 theorem body)."""
    ast = parse("Eq Zero Zero")
    assert isinstance(ast, Eq)
    assert isinstance(ast.lhs, Zero)
    assert isinstance(ast.rhs, Zero)


def test_parses_nil_atom():
    """`Nil` parses to Nil() (the empty-list constructor)."""
    ast = parse("Nil")
    assert isinstance(ast, Nil)


def test_parses_cons_zero_nil():
    """`Cons Zero Nil` parses to Cons(Zero, Nil)."""
    ast = parse("Cons Zero Nil")
    assert isinstance(ast, Cons)
    assert isinstance(ast.head, Zero)
    assert isinstance(ast.tail, Nil)


def test_parses_cons_nested():
    """`Cons (Succ Zero) (Cons Zero Nil)` parses to the nested-list
    AST, exercising the recursive-atom shape of Cons."""
    ast = parse("Cons (Succ Zero) (Cons Zero Nil)")
    assert isinstance(ast, Cons)
    assert isinstance(ast.head, Succ)
    assert isinstance(ast.head.arg, Zero)
    assert isinstance(ast.tail, Cons)
    assert isinstance(ast.tail.head, Zero)
    assert isinstance(ast.tail.tail, Nil)


def test_parses_list_nat_type_in_forall():
    """`forall xs:List Nat. Eq xs xs` parses to
    Forall(param_ty=TList(elem=TNat)) -- the List type-atom path through
    the parametric ``parse_ty_atom``."""
    ast = parse("forall xs:List Nat. Eq xs xs")
    assert isinstance(ast, Forall)
    assert ast.param == "xs"
    assert isinstance(ast.param_ty, TList)
    assert isinstance(ast.param_ty.elem, TNat)


# ---------------------------------------------------------------------------
# Phase 2: end-to-end through encode_mera + the real evaluation
# Hamiltonian on the §10.10 in-substrate composite.  This stands in for
# the directive's `test_run_problem_with_forall_nat_theorem`: the
# bridge's ``run_problem`` consumes a physics JSON DSL (fields / sites /
# Hamiltonian terms), not a logic-theorem source -- so the principled
# "DSL → encoder → solver" path for theorems is parse() → encode_mera()
# → mera_imaginary_evolve_state(), which is exactly what K-Task-8's
# acceptance suite drives.  Routing logic theorems through
# ``run_problem`` would require a separate physics-vs-logic dispatch in
# the runtime; that is a substrate extension, not a DSL surface fix.
# ---------------------------------------------------------------------------


def test_run_problem_with_forall_nat_theorem():
    """End-to-end: parse the §10.10 composite from the textual surface,
    encode it as a MERA state with Forall-protected leaves, evolve under
    the real evaluation Hamiltonian, and verify the Eq node promotes to
    BoolLit(True) via the R-AddZero + R-Eq-Refl reductions.

    NOTE on `add`: the textual surface uses ``+`` (the spec's
    canonical arithmetic operator) which encodes as ``Bin('+', ...)``;
    the R-AddZero rewrite fires on that form.  We construct the
    theorem with ``x + Zero`` to match the same substrate path that
    K-Task-8 acceptance proves.  (The bare ``add`` ident from
    ``test_parses_forall_eq_nat`` would not reduce because no
    ``KIND_ADD`` primitive exists in the encoder yet -- that path
    awaits the "add as primitive" extension.)"""
    src = parse("forall x:Nat. Eq (x + Zero) x")

    # Sanity: parse produced the same shape as the K-8 hand-built
    # theorem (Forall→Eq→Bin('+')→Var/Zero, rhs=Var).
    assert isinstance(src, Forall)
    assert isinstance(src.param_ty, TNat)
    assert isinstance(src.body, Eq)
    assert isinstance(src.body.lhs, Bin) and src.body.lhs.op == "+"
    assert isinstance(src.body.rhs, Var) and src.body.rhs.name == "x"

    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    protected = set(meta.forall_protected_leaves)
    # I-10 blocker #5: Forall-protected leaves are required for the
    # bound Var to survive evolution -- this is the §1.1 entanglement
    # structure the encoder produces.
    assert protected, (
        "Forall-protected leaves set must be populated for the parsed "
        "theorem -- the encoder must enforce §1.1 bond entanglement"
    )

    forall_kids = meta.children_of_node.get(0, [])
    assert forall_kids, "Forall encoded with no body child"
    eq_node = forall_kids[0]

    _, final = mera_imaginary_evolve_state(
        state, H, dt=_EVOLVE_DT, steps=_EVOLVE_STEPS,
        chi_layer=_EVOLVE_CHI, frozen_leaves=protected,
    )

    addzero_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_ADD_ZERO
    )
    eqrefl_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_EQ_REFL
    )
    post_kind = _leaf_weights(final, _kind_leaf(meta, eq_node))
    post_val = _leaf_weights(final, _value_leaf(meta, eq_node))

    # Load-bearing acceptance mirrors
    # ``test_substrate_level_inductive_theorem_proves_end_to_end``.  If
    # this regresses, the substrate proof has regressed.
    assert addzero_residual < 1e-3, (
        f"R-AddZero residual did not relax: {addzero_residual}"
    )
    assert eqrefl_residual < 1e-3, (
        f"R-Eq-Refl residual did not relax: {eqrefl_residual}"
    )
    assert post_kind[KIND_BOOL] > 0.99, (
        f"Eq node did not promote to KIND_BOOL: {post_kind}"
    )
    assert post_val[VALUE_TRUE] > 0.99, (
        f"Eq node value leaf did not promote to VALUE_TRUE: {post_val}"
    )


# ---------------------------------------------------------------------------
# Deferred: the spec's literal §10.10 theorem
# ``forall xs:List A. length (reverse xs) = length xs`` requires
# ``length`` / ``reverse`` primitives in the encoder substrate (kinds,
# typing-Hamiltonian rules, and reduction rules).  Today the encoder
# has TList / Cons / Nil but no list-arithmetic operators.  The new
# EXTENSIONS.md entry "List arithmetic in encoder substrate" tracks
# this; once it lands the test below should be unskipped and made to
# round-trip identically to ``test_run_problem_with_forall_nat_theorem``
# above.
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason="encoder substrate lacks length/reverse primitives -- "
    "see EXTENSIONS.md 'List arithmetic in encoder substrate'"
)
def test_parses_list_length_reverse_theorem():  # pragma: no cover
    """Spec §10.10 literal theorem -- DEFERRED until the encoder gains
    `length` / `reverse` primitives."""
    ast = parse(
        "forall xs:List Nat. Eq (length (reverse xs)) (length xs)"
    )
    assert isinstance(ast, Forall)
