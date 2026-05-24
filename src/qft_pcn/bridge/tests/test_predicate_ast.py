"""§2.7 Pythonic-restricted predicate AST visitor."""
import pytest
from src.qft_pcn.bridge.dsl.predicate_ast import parse_predicate, PredicateRejected, evaluate_predicate


@pytest.mark.parametrize("term", [
    "expr == 'Lambda'",
    "type(expr) == 'Nat'",
    "type(arg1) == type(arg2)",
    "binder(expr) == 1",
    "node_kind == 'App' and type(expr) != 'unknown'",
    "value(expr) in [0, 1, 2]",
    "not (expr == 'Nil')",
])
def test_allowed_terms_parse(term):
    # Pass user_field_names for the one term that references a bare field
    if "node_kind" in term:
        parse_predicate(term, user_field_names=frozenset({"node_kind"}))
    elif "arg1" in term or "arg2" in term:
        parse_predicate(term, two_site=True)
    else:
        parse_predicate(term)


@pytest.mark.parametrize("term", [
    "lambda x: x",
    "[x for x in xs]",
    "__import__('os')",
    "expr.__class__",
    "f'{expr}'",
    "x if y else z",
])
def test_disallowed_terms_reject(term):
    with pytest.raises(PredicateRejected):
        parse_predicate(term)


def test_evaluate_simple_equality():
    p = parse_predicate("expr == 'Lambda'")
    assert evaluate_predicate(p, {"expr": "Lambda"}) is True
    assert evaluate_predicate(p, {"expr": "App"}) is False


def test_evaluate_helper_call():
    p = parse_predicate("type(expr) == 'Nat'")
    env = {"expr": "Zero", "_helpers": {"type": lambda x: "Nat" if x == "Zero" else "?"}}
    assert evaluate_predicate(p, env) is True


def test_evaluate_two_site_helper():
    p = parse_predicate("type(arg1) == type(arg2)", two_site=True)
    env = {"arg1": "Zero", "arg2": "Succ",
           "_helpers": {"type": lambda x: "Nat"}}
    assert evaluate_predicate(p, env) is True


def test_evaluate_boolop_and_compare():
    p = parse_predicate("node_kind == 'App' and value(expr) in [0, 1, 2]",
                        user_field_names=frozenset({"node_kind"}))
    env = {"expr": "x", "node_kind": "App",
           "_helpers": {"value": lambda _: 1}}
    assert evaluate_predicate(p, env) is True
    env2 = {**env, "node_kind": "Var"}
    assert evaluate_predicate(p, env2) is False
