"""Tests for the AST data classes in logic/ast.py."""

from __future__ import annotations

import pytest

from src.qft_pcn.logic.ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin,
    Ty, TInt, TBool, TArrow,
)


def test_var_construction():
    v = Var(name="x")
    assert v.name == "x"


def test_lam_construction_holds_param_and_body():
    lam = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    assert lam.param == "x"
    assert isinstance(lam.param_ty, TInt)
    assert isinstance(lam.body, Var)
    assert lam.body.name == "x"


def test_app_holds_fn_and_arg():
    app = App(fn=Var(name="f"), arg=IntLit(val=3))
    assert isinstance(app.fn, Var)
    assert app.arg.val == 3


def test_intlit_and_boollit():
    assert IntLit(val=5).val == 5
    assert BoolLit(val=True).val is True
    assert BoolLit(val=False).val is False


def test_if_holds_three_branches():
    e = If(cond=BoolLit(val=True), then_b=IntLit(val=1), else_b=IntLit(val=0))
    assert e.cond.val is True
    assert e.then_b.val == 1
    assert e.else_b.val == 0


def test_bin_holds_op_and_operands():
    b = Bin(op="+", lhs=IntLit(val=2), rhs=IntLit(val=3))
    assert b.op == "+"
    assert b.lhs.val == 2 and b.rhs.val == 3


def test_bin_op_must_be_supported():
    # Out-of-grammar op should be a runtime error from __post_init__.
    with pytest.raises(ValueError, match="op must be one of"):
        Bin(op="**", lhs=IntLit(val=1), rhs=IntLit(val=1))


def test_arrow_type_construction():
    t = TArrow(src=TInt(), dst=TBool())
    assert isinstance(t.src, TInt)
    assert isinstance(t.dst, TBool)


def test_ty_equality_by_structure():
    assert TInt() == TInt()
    assert TBool() == TBool()
    assert TArrow(src=TInt(), dst=TInt()) == TArrow(src=TInt(), dst=TInt())
    assert TArrow(src=TInt(), dst=TInt()) != TArrow(src=TBool(), dst=TInt())


def test_node_equality_by_structure():
    # Structural equality on nodes (modulo names, which DO matter for raw AST).
    assert Var(name="x") == Var(name="x")
    assert Var(name="x") != Var(name="y")
    assert IntLit(val=3) == IntLit(val=3)
    assert IntLit(val=3) != IntLit(val=4)


from src.qft_pcn.logic.ast import parse


def test_parse_var():
    assert parse("x") == Var(name="x")


def test_parse_intlit():
    assert parse("42") == IntLit(val=42)
    assert parse("0") == IntLit(val=0)


def test_parse_bool():
    assert parse("true") == BoolLit(val=True)
    assert parse("false") == BoolLit(val=False)


def test_parse_lambda_identity():
    expected = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    assert parse(r"\x:Int. x") == expected


def test_parse_application_left_assoc():
    # f x y = (f x) y
    expected = App(fn=App(fn=Var("f"), arg=Var("x")), arg=Var("y"))
    assert parse("f x y") == expected


def test_parse_addition():
    expected = Bin(op="+", lhs=IntLit(val=2), rhs=IntLit(val=3))
    assert parse("2 + 3") == expected


def test_parse_if():
    expected = If(
        cond=Bin(op="<", lhs=IntLit(val=1), rhs=IntLit(val=2)),
        then_b=IntLit(val=10),
        else_b=IntLit(val=20),
    )
    assert parse("if 1 < 2 then 10 else 20") == expected


def test_parse_arrow_type():
    # \\f:Int->Int. f
    expected = Lam(
        param="f",
        param_ty=TArrow(src=TInt(), dst=TInt()),
        body=Var("f"),
    )
    assert parse(r"\f:Int->Int. f") == expected


def test_parse_arrow_right_assoc():
    # Int -> Int -> Bool  ==  Int -> (Int -> Bool)
    expected = Lam(
        param="g",
        param_ty=TArrow(src=TInt(), dst=TArrow(src=TInt(), dst=TBool())),
        body=Var("g"),
    )
    assert parse(r"\g:Int->Int->Bool. g") == expected


def test_parse_p5_complex_program():
    src = r"(\x:Int. (\y:Int. x + y)(3))(4)"
    p = parse(src)
    # outer App
    assert isinstance(p, App)
    assert isinstance(p.arg, IntLit) and p.arg.val == 4
    # outer Lam
    assert isinstance(p.fn, Lam)
    assert p.fn.param == "x"
    assert isinstance(p.fn.param_ty, TInt)
    # inner App
    inner_app = p.fn.body
    assert isinstance(inner_app, App)
    assert isinstance(inner_app.arg, IntLit) and inner_app.arg.val == 3
    # inner Lam
    assert isinstance(inner_app.fn, Lam)
    assert inner_app.fn.param == "y"
    # x + y
    body = inner_app.fn.body
    assert isinstance(body, Bin) and body.op == "+"
    assert body.lhs == Var(name="x") and body.rhs == Var(name="y")


def test_parse_rejects_unknown_op():
    with pytest.raises(ValueError, match="unexpected token|unexpected character"):
        parse("2 ** 3")


def test_parse_lt_no_spaces():
    """Tokenizer must accept `5<2` (no surrounding whitespace)."""
    expected = Bin(op="<", lhs=IntLit(val=5), rhs=IntLit(val=2))
    assert parse("5<2") == expected


def test_parse_negative_int_literal():
    assert parse("-3") == IntLit(val=-3)
    # And inside an expression:
    assert parse("0 - -1") == Bin(op="-", lhs=IntLit(val=0), rhs=IntLit(val=-1))
    # Inside a lambda body:
    expected = Lam(param="x", param_ty=TInt(), body=IntLit(val=-5))
    assert parse(r"\x:Int. -5") == expected


from src.qft_pcn.logic.ast import pretty


def test_pretty_roundtrip_simple():
    src = r"\x:Int. x"
    assert parse(pretty(parse(src))) == parse(src)


def test_pretty_roundtrip_p2():
    src = r"(\x:Int. x + 1)(2)"
    assert parse(pretty(parse(src))) == parse(src)


def test_pretty_roundtrip_p3():
    src = r"\f:Int->Int. \x:Int. f (f x)"
    assert parse(pretty(parse(src))) == parse(src)


def test_pretty_roundtrip_p4():
    src = r"if 1 < 2 then ((\x:Bool. x)(true)) else false"
    assert parse(pretty(parse(src))) == parse(src)


def test_pretty_roundtrip_p5():
    src = r"(\x:Int. (\y:Int. x + y)(3))(4)"
    assert parse(pretty(parse(src))) == parse(src)


from src.qft_pcn.logic.ast import HoleVar, substitute_hole


def test_holevar_construction():
    h = HoleVar(candidates=["x", "y"])
    assert h.candidates == ["x", "y"]


def test_substitute_hole_finds_and_replaces():
    h = HoleVar(candidates=["x"])
    ast = Lam(param="x", param_ty=TInt(),
              body=Lam(param="y", param_ty=TInt(), body=h))
    replaced = substitute_hole(ast, h, Var(name="x"))
    assert isinstance(replaced, Lam)
    assert isinstance(replaced.body, Lam)
    assert isinstance(replaced.body.body, Var)
    assert replaced.body.body.name == "x"


def test_substitute_hole_returns_unchanged_if_not_found():
    h = HoleVar(candidates=["x"])
    ast = Lam(param="x", param_ty=TInt(), body=Var(name="x"))
    replaced = substitute_hole(ast, h, Var(name="x"))
    assert replaced == ast


# ---- Task 21: Rec node (EXPERIMENTAL) -----------------------------------


def test_experimental_rec_flag_set():
    from src.qft_pcn.logic.ast import EXPERIMENTAL_REC
    assert EXPERIMENTAL_REC is True


def test_rec_node_construction():
    from src.qft_pcn.logic.ast import Rec, TInt, TArrow, Lam, Var
    body = Lam(param="n", param_ty=TInt(), body=Var(name="f"))
    rec = Rec(name="f", name_ty=TArrow(src=TInt(), dst=TInt()), body=body)
    assert rec.name == "f"
    assert isinstance(rec.name_ty, TArrow)
    assert isinstance(rec.body, Lam)
