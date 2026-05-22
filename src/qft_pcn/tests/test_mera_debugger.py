"""Tests for the MERA constraint debugger (spec §3, §9.1)."""
from __future__ import annotations
import json
from dataclasses import dataclass
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_debugger import (
    NamedMeraTerm, DiagnosticReport, RuleViolation, TermEvaluationError,
    diagnose, format_report, register_explanation, clear_explanations,
)


@dataclass
class _MockTerm:
    name: str
    rule_class: str
    node: int
    leaves: tuple
    value: float = 0.0
    def expectation(self, state) -> float:
        return float(self.value)


@dataclass
class _MockRaises:
    name: str
    rule_class: str
    node: int
    leaves: tuple
    def expectation(self, state) -> float:
        raise ValueError("intentional")


def _mock_terms(violated):
    """One term per (rule_class, node); only `violated` names are nonzero."""
    base = [("T-App", 0), ("T-Var", 1), ("T-Abs", 0)]
    out = []
    for rc, nd in base:
        nm = f"{rc}@node_{nd}"
        out.append(_MockTerm(nm, rc, nd, (5 * nd,), violated.get(nm, 0.0)))
    return out


def test_diagnose_emits_structured_report():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    report = diagnose(state, meta, _mock_terms({"T-App@node_0": 1.0}))
    assert isinstance(report, DiagnosticReport)
    assert abs(report.total_energy - 1.0) < 1e-10
    assert report.n_violations == 1
    v = report.rule_violations[0]
    assert v.rule_class == "T-App"
    assert v.node == 0
    assert v.lookup_failed is False
    assert "T-App" in v.explanation


def test_report_json_roundtrip():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    report = diagnose(state, meta, _mock_terms({"T-App@node_0": 1.0}))
    rt = DiagnosticReport.from_dict(json.loads(report.to_json()))
    assert rt == report


def test_threshold_filters_and_sorts():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    terms = _mock_terms({"T-App@node_0": 0.5, "T-Abs@node_0": 1.5,
                         "T-Var@node_1": 1e-9})
    desc = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    assert desc.n_violations == 2
    assert desc.rule_violations[0].rule_class == "T-Abs"
    asc = diagnose(state, meta, terms, sort="ascending")
    assert asc.rule_violations[0].rule_class == "T-App"


def test_term_that_raises_is_captured():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    good = _MockTerm("T-Var@node_1", "T-Var", 1, (5,), value=0.5)
    bad = _MockRaises("T-App@node_0", "T-App", 0, (0,))
    report = diagnose(state, meta, [good, bad], threshold=0.0)
    assert report.n_violations == 1
    assert len(report.errors) == 1
    assert report.errors[0].exception_type == "ValueError"


def test_non_conforming_term_raises_typeerror():
    class NotATerm: pass
    state, meta = encode_mera(parse(r"\x:Int. x"))
    with pytest.raises(TypeError) as ei:
        diagnose(state, meta, [NotATerm()])
    assert "index 0" in str(ei.value)
    assert "NamedMeraTerm" in str(ei.value)


def test_empty_terms_zero_report():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    report = diagnose(state, meta, [], threshold=1e-6)
    assert report.total_energy == 0.0
    assert report.n_violations == 0
    assert report.by_rule_class == {}


def test_bad_threshold_and_sort_raise():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    with pytest.raises(ValueError):
        diagnose(state, meta, [], threshold=-1.0)
    with pytest.raises(ValueError):
        diagnose(state, meta, [], sort="sideways")


def test_explanation_registry():
    clear_explanations()
    register_explanation("T-X", "rule at node {node}")
    register_explanation("T-X", "rule at node {node}")     # idempotent
    with pytest.raises(ValueError):
        register_explanation("T-X", "different")
    state, meta = encode_mera(parse(r"\x:Int. x"))
    term = _MockTerm("T-X@node_0", "T-X", 0, (0,), value=0.5)
    report = diagnose(state, meta, [term], threshold=0.0)
    assert report.rule_violations[0].explanation == "rule at node 0"


def test_unregistered_rule_uses_fallback():
    clear_explanations()
    state, meta = encode_mera(parse(r"\x:Int. x"))
    term = _MockTerm("X-Z@node_0", "X-Z", 0, (0,), value=0.5)
    report = diagnose(state, meta, [term], threshold=0.0)
    assert "X-Z" in report.rule_violations[0].explanation


def test_format_report_human_readable():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    report = diagnose(state, meta, _mock_terms({"T-App@node_0": 1.0}))
    text = format_report(report)
    assert "T-App" in text
    assert "node 0" in text
