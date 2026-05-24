"""§2.7 Pythonic-restricted predicate AST visitor."""
import pytest
from src.qft_pcn.bridge.dsl.predicate_ast import parse_predicate, PredicateRejected


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
