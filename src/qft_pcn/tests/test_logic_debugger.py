"""Tests for the constraint debugger (sub-project D)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.logic import encode, parse


# ===========================================================================
# Task 2: Protocol
# ===========================================================================


def test_named_hamiltonian_term_protocol_importable():
    """The Protocol exists at the expected import path so that B and C
    can import it without circular dependencies."""
    from src.qft_pcn.logic.debugger import NamedHamiltonianTerm
    assert NamedHamiltonianTerm.__module__ == "src.qft_pcn.logic.debugger"


def test_named_hamiltonian_term_is_runtime_checkable():
    """Protocol is runtime_checkable so diagnose() can validate inputs."""
    from src.qft_pcn.logic.debugger import NamedHamiltonianTerm

    @dataclass
    class Conforming:
        name: str
        rule_class: str
        site: int
        sites: tuple[int, ...]
        def expectation(self, state) -> float:
            return 0.0

    c = Conforming("X@0", "X", 0, (0,))
    assert isinstance(c, NamedHamiltonianTerm)


def test_named_hamiltonian_term_rejects_missing_attribute():
    """A class missing `expectation` does not satisfy the Protocol."""
    from src.qft_pcn.logic.debugger import NamedHamiltonianTerm

    class Missing:
        name = "X@0"
        rule_class = "X"
        site = 0
        sites = (0,)
        # no expectation

    assert not isinstance(Missing(), NamedHamiltonianTerm)


# ===========================================================================
# Task 3: Dataclasses
# ===========================================================================


def test_rule_violation_dataclass_shape():
    from src.qft_pcn.logic.debugger import RuleViolation
    v = RuleViolation(
        name="T-App@site_5",
        rule_class="T-App",
        site=5,
        sites=[5, 6, 8],
        energy_contribution=1.0,
        ast_path=[0, 1],
        lookup_failed=False,
        explanation="T-App violated",
        context={"foo": "bar"},
    )
    assert v.name == "T-App@site_5"
    assert v.rule_class == "T-App"
    assert v.site == 5
    assert v.sites == [5, 6, 8]
    assert v.energy_contribution == 1.0
    assert v.ast_path == [0, 1]
    assert v.lookup_failed is False
    assert v.explanation == "T-App violated"
    assert v.context == {"foo": "bar"}


def test_rule_violation_is_frozen():
    from src.qft_pcn.logic.debugger import RuleViolation
    import dataclasses
    v = RuleViolation(
        name="X@0", rule_class="X", site=0, sites=[0],
        energy_contribution=0.5, ast_path=None, lookup_failed=True,
        explanation="x", context={},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.name = "Y@0"  # type: ignore[misc]


def test_rule_violation_to_dict_is_json_safe():
    from src.qft_pcn.logic.debugger import RuleViolation
    v = RuleViolation(
        name="T-App@site_5",
        rule_class="T-App",
        site=5,
        sites=[5, 6, 8],
        energy_contribution=1.0,
        ast_path=[0, 1],
        lookup_failed=False,
        explanation="T-App violated",
        context={"a": 1, "b": "x"},
    )
    d = v.to_dict()
    s = json.dumps(d)
    rt = json.loads(s)
    assert rt == d
    assert isinstance(d["sites"], list)
    assert isinstance(d["site"], int)
    assert isinstance(d["energy_contribution"], float)


def test_rule_violation_to_dict_handles_none_ast_path():
    from src.qft_pcn.logic.debugger import RuleViolation
    v = RuleViolation(
        name="X@0", rule_class="X", site=0, sites=[0],
        energy_contribution=0.0, ast_path=None, lookup_failed=True,
        explanation="x", context={},
    )
    d = v.to_dict()
    assert d["ast_path"] is None
    assert d["lookup_failed"] is True


def test_term_evaluation_error_dataclass_shape():
    from src.qft_pcn.logic.debugger import TermEvaluationError
    e = TermEvaluationError(
        term_name="T-App@site_0",
        rule_class="T-App",
        site=0,
        exception_type="ValueError",
        exception_message="bad",
    )
    d = e.to_dict()
    assert json.dumps(d)
    assert d["term_name"] == "T-App@site_0"
    assert d["exception_type"] == "ValueError"


def test_diagnostic_report_dataclass_shape():
    from src.qft_pcn.logic.debugger import DiagnosticReport
    r = DiagnosticReport(
        total_energy=1.0,
        threshold=1e-6,
        n_terms_evaluated=3,
        n_violations=1,
        rule_violations=[],
        by_rule_class={"T-App": 1.0},
        errors=[],
    )
    d = r.to_dict()
    assert json.dumps(d, indent=2)
    assert d["total_energy"] == 1.0
    assert d["by_rule_class"] == {"T-App": 1.0}


# ===========================================================================
# Task 4: JSON round-trip
# ===========================================================================


def test_diagnostic_report_json_roundtrip():
    from src.qft_pcn.logic.debugger import (
        DiagnosticReport, RuleViolation, TermEvaluationError,
    )
    v = RuleViolation(
        name="T-App@site_5", rule_class="T-App", site=5,
        sites=[5, 6, 8], energy_contribution=1.0,
        ast_path=[0, 1], lookup_failed=False,
        explanation="T-App violated", context={"foo": "bar"},
    )
    e = TermEvaluationError(
        term_name="bad", rule_class="X", site=0,
        exception_type="ValueError", exception_message="oops",
    )
    r = DiagnosticReport(
        total_energy=1.0, threshold=1e-6,
        n_terms_evaluated=3, n_violations=1,
        rule_violations=[v],
        by_rule_class={"T-App": 1.0, "T-Var": 0.0},
        errors=[e],
    )
    s = r.to_json()
    assert isinstance(s, str)
    rt = DiagnosticReport.from_dict(json.loads(s))
    assert rt == r


def test_diagnostic_report_to_json_default_indent():
    from src.qft_pcn.logic.debugger import DiagnosticReport
    r = DiagnosticReport(
        total_energy=0.0, threshold=1e-6,
        n_terms_evaluated=0, n_violations=0,
        rule_violations=[], by_rule_class={}, errors=[],
    )
    s = r.to_json()
    assert "\n" in s
    s_compact = r.to_json(indent=None)
    assert "\n" not in s_compact


# ===========================================================================
# Task 5: Explanation registry
# ===========================================================================


def test_register_explanation_basic():
    from src.qft_pcn.logic.debugger import (
        register_explanation, get_explanation, clear_explanations,
    )
    clear_explanations()
    register_explanation("T-App", "T-App at site {site}")
    tpl, ext = get_explanation("T-App")
    assert tpl == "T-App at site {site}"
    assert ext is None


def test_register_explanation_idempotent_same_template():
    from src.qft_pcn.logic.debugger import (
        register_explanation, clear_explanations,
    )
    clear_explanations()
    register_explanation("T-App", "tpl")
    register_explanation("T-App", "tpl")  # no error


def test_register_explanation_conflict_raises():
    from src.qft_pcn.logic.debugger import (
        register_explanation, clear_explanations,
    )
    clear_explanations()
    register_explanation("T-App", "first")
    with pytest.raises(ValueError) as exc:
        register_explanation("T-App", "different")
    assert "T-App" in str(exc.value)


def test_register_explanation_idempotent_same_extractor():
    from src.qft_pcn.logic.debugger import (
        register_explanation, clear_explanations,
    )
    clear_explanations()
    def ext(term, state, meta):
        return {}
    register_explanation("T-X", "tpl", ext)
    register_explanation("T-X", "tpl", ext)


def test_register_explanation_extractor_conflict_raises():
    from src.qft_pcn.logic.debugger import (
        register_explanation, clear_explanations,
    )
    clear_explanations()
    def ext1(term, state, meta):
        return {}
    def ext2(term, state, meta):
        return {}
    register_explanation("T-X", "tpl", ext1)
    with pytest.raises(ValueError):
        register_explanation("T-X", "tpl", ext2)


def test_get_explanation_missing_returns_none():
    from src.qft_pcn.logic.debugger import (
        get_explanation, clear_explanations,
    )
    clear_explanations()
    assert get_explanation("UNREGISTERED") is None


def test_clear_explanations_empties_registry():
    from src.qft_pcn.logic.debugger import (
        register_explanation, get_explanation, clear_explanations,
    )
    register_explanation("Z", "z")
    clear_explanations()
    assert get_explanation("Z") is None


# ===========================================================================
# Task 6: Mock fixtures
# ===========================================================================


@dataclass
class _MockTerm:
    """A NamedHamiltonianTerm satisfying the Protocol structurally."""
    name: str
    rule_class: str
    site: int
    sites: tuple[int, ...]
    value: float = 0.0

    def expectation(self, state) -> float:
        return float(self.value)


@dataclass
class _MockTermThatRaises:
    """Mock term whose expectation() raises a configured exception."""
    name: str
    rule_class: str
    site: int
    sites: tuple[int, ...]
    exc: BaseException

    def expectation(self, state) -> float:
        raise self.exc


def _mock_stlc_terms_for(p, state, meta, *, violated: dict[str, float]):
    """Build a list of mock STLC typing terms.

    For each (rule_class, site) we emit one mock returning 0.0, except
    where the term's name appears in `violated`, in which case the mock
    returns the specified energy.
    """
    rule_classes = ["T-Var", "T-Abs", "T-App", "T-If", "T-Bin",
                    "T-IntLit", "T-BoolLit"]
    terms = []
    for site in sorted(meta.site_to_ast_path):
        for rc in rule_classes:
            name = f"{rc}@site_{site}"
            terms.append(_MockTerm(
                name=name, rule_class=rc, site=site, sites=(site,),
                value=violated.get(name, 0.0),
            ))
    return terms


def test_mock_term_satisfies_protocol():
    """Smoke test: the mock conforms to the Protocol used by diagnose()."""
    from src.qft_pcn.logic.debugger import NamedHamiltonianTerm
    m = _MockTerm("T-App@site_0", "T-App", 0, (0,))
    assert isinstance(m, NamedHamiltonianTerm)
    p = parse(r"\x:Int. x")
    state, _ = encode(p, N=32, chi_max=16)
    e = m.expectation(state)
    assert type(e) is float


# ===========================================================================
# Task 7: diagnose() core — validation, totals, error capture
# ===========================================================================


def test_diagnose_rejects_negative_threshold():
    from src.qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    with pytest.raises(ValueError) as exc:
        diagnose(state, meta, [], threshold=-0.1)
    assert "threshold" in str(exc.value)


def test_diagnose_rejects_bad_sort():
    from src.qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    with pytest.raises(ValueError) as exc:
        diagnose(state, meta, [], sort="banana")
    assert "sort" in str(exc.value)


def test_diagnose_rejects_non_conforming_term():
    from src.qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)

    class NotATerm:
        pass

    with pytest.raises(TypeError) as exc:
        diagnose(state, meta, [NotATerm()])
    msg = str(exc.value)
    assert "index 0" in msg
    assert "NamedHamiltonianTerm" in msg


def test_diagnose_empty_terms_returns_zero():
    from src.qft_pcn.logic.debugger import diagnose, DiagnosticReport
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    r = diagnose(state, meta, [], threshold=1e-6)
    assert isinstance(r, DiagnosticReport)
    assert r.total_energy == 0.0
    assert r.threshold == 1e-6
    assert r.n_terms_evaluated == 0
    assert r.n_violations == 0
    assert r.rule_violations == []
    assert r.by_rule_class == {}
    assert r.errors == []


def test_diagnose_total_energy_is_sum_of_all_terms():
    """total_energy is independent of threshold."""
    from src.qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    terms = [
        _MockTerm("A@0", "A", 0, (0,), value=0.3),
        _MockTerm("B@1", "B", 1, (1,), value=0.7),
        _MockTerm("C@2", "C", 2, (2,), value=1e-9),
    ]
    r = diagnose(state, meta, terms, threshold=1.0)
    assert abs(r.total_energy - (0.3 + 0.7 + 1e-9)) < 1e-12
    assert r.n_terms_evaluated == 3
    assert r.n_violations == 0


def test_diagnose_captures_term_evaluation_errors():
    """Terms that raise are NOT propagated; they go into report.errors."""
    from src.qft_pcn.logic.debugger import diagnose
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    good = _MockTerm("T-Var@1", "T-Var", 1, (1,), value=0.5)
    bad = _MockTermThatRaises(
        "T-App@0", "T-App", 0, (0,), exc=ValueError("intentional"),
    )
    r = diagnose(state, meta, [good, bad], threshold=0.0)
    assert abs(r.total_energy - 0.5) < 1e-12
    assert r.n_terms_evaluated == 2
    assert r.n_violations == 1
    assert r.rule_violations[0].rule_class == "T-Var"
    assert len(r.errors) == 1
    e = r.errors[0]
    assert e.term_name == "T-App@0"
    assert e.exception_type == "ValueError"
    assert "intentional" in e.exception_message


# ===========================================================================
# Task 8: _build_violations — threshold, sort, AST path
# ===========================================================================


def test_diagnose_threshold_filters_low_energy_terms():
    """|<H>| < threshold means the term doesn't make rule_violations."""
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 0.5,
        "T-Abs@site_0": 1.5,
        "T-Var@site_1": 1e-9,
    })
    r = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    assert r.n_violations == 2
    assert r.rule_violations[0].rule_class == "T-Abs"
    assert r.rule_violations[1].rule_class == "T-App"
    assert abs(r.by_rule_class["T-Var"] - 1e-9) < 1e-15


def test_diagnose_threshold_zero_includes_every_violator():
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-Var@site_1": 1e-12,
    })
    r = diagnose(state, meta, terms, threshold=0.0)
    names = {v.name for v in r.rule_violations}
    assert "T-Var@site_1" in names


def test_diagnose_sort_descending():
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 0.3, "T-Abs@site_1": 0.7, "T-Var@site_2": 0.5,
    })
    r = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    assert [v.energy_contribution
            for v in r.rule_violations] == [0.7, 0.5, 0.3]


def test_diagnose_sort_ascending():
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 0.3, "T-Abs@site_1": 0.7, "T-Var@site_2": 0.5,
    })
    r = diagnose(state, meta, terms, threshold=1e-6, sort="ascending")
    assert [v.energy_contribution
            for v in r.rule_violations] == [0.3, 0.5, 0.7]


def test_diagnose_sort_by_site():
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 0.3, "T-Abs@site_1": 0.7, "T-Var@site_2": 0.5,
    })
    r = diagnose(state, meta, terms, threshold=1e-6, sort="site")
    assert [v.site for v in r.rule_violations] == [0, 1, 2]


def test_diagnose_ast_path_is_looked_up():
    """For sites in meta.site_to_ast_path, ast_path is the looked-up tuple."""
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    term = _MockTerm("T-Abs@site_0", "T-Abs", 0, (0,), value=1.0)
    r = diagnose(state, meta, [term], threshold=1e-6)
    assert r.n_violations == 1
    v = r.rule_violations[0]
    assert v.ast_path == list(meta.site_to_ast_path[0])
    assert v.lookup_failed is False


def test_diagnose_ast_path_lookup_failure_does_not_raise():
    """A term at a site not in meta.site_to_ast_path: ast_path=None,
    lookup_failed=True (NOT an exception)."""
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    pad_site = max(meta.site_to_ast_path) + 5
    bogus = _MockTerm("Global@99", "Global", pad_site, (pad_site,), value=0.7)
    r = diagnose(state, meta, [bogus], threshold=0.0)
    assert r.n_violations >= 1
    matches = [v for v in r.rule_violations if v.name == "Global@99"]
    assert len(matches) == 1
    v = matches[0]
    assert v.ast_path is None
    assert v.lookup_failed is True


def test_diagnose_default_tie_breaking_is_deterministic():
    """When two violations have equal energy, site asc then name lex."""
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = [
        _MockTerm("Z@1", "Z", 1, (1,), value=1.0),
        _MockTerm("A@1", "A", 1, (1,), value=1.0),
        _MockTerm("A@0", "A", 0, (0,), value=1.0),
    ]
    r = diagnose(state, meta, terms, threshold=1e-6, sort="descending")
    names = [v.name for v in r.rule_violations]
    assert names == ["A@0", "A@1", "Z@1"]


# ===========================================================================
# Task 9: Explanation registry wiring
# ===========================================================================


def test_unregistered_rule_class_uses_fallback_explanation():
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations
    clear_explanations()
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    t = _MockTerm("X-Custom@site_0", "X-Custom", 0, (0,), value=0.5)
    r = diagnose(state, meta, [t], threshold=0.0)
    v = r.rule_violations[0]
    assert "X-Custom" in v.explanation
    assert "0.5" in v.explanation


def test_registered_template_default_slots():
    """A template using only default slots needs no extractor."""
    from src.qft_pcn.logic.debugger import (
        diagnose, register_explanation, clear_explanations,
    )
    clear_explanations()
    register_explanation(
        "T-Foo",
        "{rule_class} fired at site {site} (path {ast_path}, e={energy})",
    )
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    t = _MockTerm("T-Foo@site_0", "T-Foo", 0, (0,), value=0.5)
    r = diagnose(state, meta, [t], threshold=0.0)
    v = r.rule_violations[0]
    assert v.explanation.startswith("T-Foo fired at site 0")


def test_registered_template_with_extractor():
    from src.qft_pcn.logic.debugger import (
        diagnose, register_explanation, clear_explanations,
    )
    clear_explanations()
    def ext(term, state, meta):
        return {"foo": "bar", "baz": 42}
    register_explanation("T-X", "rule {foo}/{baz} at site {site}", ext)
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    t = _MockTerm("T-X@site_0", "T-X", 0, (0,), value=0.5)
    r = diagnose(state, meta, [t], threshold=0.0)
    v = r.rule_violations[0]
    assert v.explanation == "rule bar/42 at site 0"
    assert v.context == {"foo": "bar", "baz": 42}


def test_extractor_failure_falls_back_gracefully():
    """If a registered extractor raises, the violation gets the fallback
    explanation; the diagnose() call still succeeds."""
    from src.qft_pcn.logic.debugger import (
        diagnose, register_explanation, clear_explanations,
    )
    clear_explanations()
    def bad_ext(term, state, meta):
        raise RuntimeError("extractor blew up")
    register_explanation("T-Bad", "rule {missing_slot}", bad_ext)
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    t = _MockTerm("T-Bad@site_0", "T-Bad", 0, (0,), value=0.5)
    r = diagnose(state, meta, [t], threshold=0.0)
    assert r.n_violations == 1
    v = r.rule_violations[0]
    assert "T-Bad" in v.explanation
    assert any(e.exception_type == "RuntimeError" for e in r.errors)


# ===========================================================================
# Task 10: Pretty-printer
# ===========================================================================


def test_format_report_basic():
    from src.qft_pcn.logic.debugger import (
        diagnose, format_report, clear_explanations,
    )
    clear_explanations()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 1.0,
    })
    r = diagnose(state, meta, terms, threshold=1e-6)
    text = format_report(r)
    assert isinstance(text, str)
    assert "Total energy" in text
    assert "T-App" in text
    assert "site 0" in text


def test_format_report_empty():
    from src.qft_pcn.logic.debugger import format_report, DiagnosticReport
    r = DiagnosticReport(
        total_energy=0.0, threshold=1e-6,
        n_terms_evaluated=0, n_violations=0,
        rule_violations=[], by_rule_class={}, errors=[],
    )
    text = format_report(r)
    assert "Total energy: 0" in text
    assert "No violations" in text


def test_format_report_includes_errors_section():
    from src.qft_pcn.logic.debugger import (
        diagnose, format_report, clear_explanations,
    )
    clear_explanations()
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)
    bad = _MockTermThatRaises(
        "T-App@0", "T-App", 0, (0,), exc=ValueError("oops"),
    )
    r = diagnose(state, meta, [bad], threshold=0.0)
    text = format_report(r)
    assert "Errors" in text or "errors" in text
    assert "oops" in text


# ===========================================================================
# Task 11: Seed STLC templates
# ===========================================================================


def test_register_stlc_seed_templates_populates_registry():
    from src.qft_pcn.logic.debugger import (
        register_stlc_seed_templates, get_explanation, clear_explanations,
    )
    clear_explanations()
    register_stlc_seed_templates()
    for rc in ("T-Var", "T-Abs", "T-App", "T-If", "T-Bin",
               "T-IntLit", "T-BoolLit"):
        assert get_explanation(rc) is not None, (
            f"seed template for {rc} not registered"
        )


def test_register_stlc_seed_templates_is_idempotent():
    from src.qft_pcn.logic.debugger import (
        register_stlc_seed_templates, clear_explanations,
    )
    clear_explanations()
    register_stlc_seed_templates()
    register_stlc_seed_templates()


def test_seed_t_abs_explanation_renders_default_slots():
    from src.qft_pcn.logic.debugger import (
        register_stlc_seed_templates, diagnose, clear_explanations,
    )
    clear_explanations()
    register_stlc_seed_templates()
    p = parse(r"\x:Int. x")
    state, meta = encode(p, N=32, chi_max=16)
    t = _MockTerm("T-Abs@site_0", "T-Abs", 0, (0,), value=1.0)
    r = diagnose(state, meta, [t], threshold=1e-6)
    v = r.rule_violations[0]
    assert "T-Abs" in v.explanation
    assert "site 0" in v.explanation


def test_seed_t_app_arrow_template_for_b_rule_id():
    """B's actual rule_id is 'T-App-Arrow' — the seed set covers it."""
    from src.qft_pcn.logic.debugger import (
        register_stlc_seed_templates, get_explanation, clear_explanations,
    )
    clear_explanations()
    register_stlc_seed_templates()
    entry = get_explanation("T-App-Arrow")
    assert entry is not None
    tpl, _ = entry
    assert "T-App-Arrow" in tpl


# ===========================================================================
# Task 12: Headline acceptance test
# ===========================================================================


def test_diagnose_emits_structured_report_for_ill_typed_program():
    """End-to-end with mock H_typing: ((\\x:Int. x + 1)(true)) produces
    a report identifying the T-App violation at the application site."""
    from src.qft_pcn.logic.debugger import (
        diagnose, register_stlc_seed_templates, clear_explanations,
        DiagnosticReport, format_report,
    )
    clear_explanations()
    register_stlc_seed_templates()

    p = parse(r"(\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)

    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 1.0,
    })

    report = diagnose(state, meta, terms, threshold=1e-6)

    assert isinstance(report, DiagnosticReport)
    assert abs(report.total_energy - 1.0) < 1e-10
    assert report.n_violations == 1

    v = report.rule_violations[0]
    assert v.rule_class == "T-App"
    assert v.site == 0
    assert v.ast_path == list(meta.site_to_ast_path[0])
    assert v.lookup_failed is False
    assert "T-App" in v.explanation

    text = format_report(report)
    assert "T-App" in text


# ===========================================================================
# Task 13: Gated end-to-end test for sub-project B
# ===========================================================================


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def _has_b_adapter() -> bool:
    """B/E ship a `build_typing_terms(meta)` helper that wraps TypingTerm
    instances into NamedHamiltonianTerm-conforming objects. Until that
    adapter exists, the integration test stays skipped — the mock test
    above proves D's structural correctness."""
    try:
        from src.qft_pcn.logic.typing_hamiltonian import (  # noqa: F401
            build_typing_terms,
        )
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_b_adapter(),
    reason="sub-project B's NamedHamiltonianTerm adapter "
           "(build_typing_terms) not yet implemented",
)
def test_end_to_end_ill_typed_program_with_real_typing_hamiltonian():
    """Once B (or E's B-adapter) ships, this is the test that proves the
    whole stack works: ill-typed program → real H_typing → diagnose →
    report with the right rule_class and site."""
    from src.qft_pcn.logic.typing_hamiltonian import build_typing_terms
    from src.qft_pcn.logic.debugger import diagnose

    p = parse(r"(\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)
    terms = build_typing_terms(meta)
    report = diagnose(state, meta, terms, threshold=1e-6)

    rule_classes = {v.rule_class for v in report.rule_violations}
    # B uses "T-App-Arrow" as the rule_id for the application rule.
    assert "T-App-Arrow" in rule_classes or "T-App" in rule_classes


def test_full_report_json_roundtrip_for_ill_typed_program():
    """Combine §7.1 and §7.2: produce a real-looking report, serialize,
    parse back, and assert equality plus structural correctness."""
    from src.qft_pcn.logic.debugger import (
        diagnose, register_stlc_seed_templates, clear_explanations,
        DiagnosticReport,
    )
    clear_explanations()
    register_stlc_seed_templates()
    p = parse(r"(\x:Int. x + 1)(true)")
    state, meta = encode(p, N=32, chi_max=16)
    terms = _mock_stlc_terms_for(p, state, meta, violated={
        "T-App@site_0": 1.0,
    })
    r = diagnose(state, meta, terms, threshold=1e-6)
    s = r.to_json()
    rt = DiagnosticReport.from_dict(json.loads(s))
    assert rt == r
    parsed = json.loads(s)
    assert isinstance(parsed["rule_violations"], list)
    assert parsed["rule_violations"][0]["rule_class"] == "T-App"


# ===========================================================================
# Task 14: Public re-exports
# ===========================================================================


def test_debugger_public_reexports_from_logic_init():
    """logic.__init__ exposes the debugger's public surface."""
    from src.qft_pcn.logic import (
        NamedHamiltonianTerm,
        DiagnosticReport,
        RuleViolation,
        TermEvaluationError,
        diagnose,
        format_report,
        register_explanation,
        get_explanation,
        clear_explanations,
        register_stlc_seed_templates,
    )
    from src.qft_pcn.logic.debugger import (
        NamedHamiltonianTerm as NHT_direct,
    )
    assert NamedHamiltonianTerm is NHT_direct
    assert callable(diagnose)
    assert callable(format_report)
    assert callable(register_explanation)
    assert callable(get_explanation)
    assert callable(clear_explanations)
    assert callable(register_stlc_seed_templates)
    assert DiagnosticReport is not None
    assert RuleViolation is not None
    assert TermEvaluationError is not None


def test_top_level_qft_pcn_reexports_diagnose():
    """qft_pcn top-level re-exports the most commonly-used names."""
    from src.qft_pcn import diagnose, DiagnosticReport
    from src.qft_pcn.logic.debugger import (
        diagnose as diagnose_direct,
        DiagnosticReport as DR_direct,
    )
    assert diagnose is diagnose_direct
    assert DiagnosticReport is DR_direct


# ===========================================================================
# Bonus: Protocol-generic contract (per user mission acceptance #3)
# ===========================================================================


def test_diagnose_works_with_non_stlc_protocol_conformant_term():
    """D's Protocol contract: ANY object with name/rule_class/site/sites
    /expectation works — not just STLC. This proves the genericity claim."""
    from src.qft_pcn.logic.debugger import diagnose, clear_explanations

    clear_explanations()
    state, meta = encode(parse(r"\x:Int. x"), N=32, chi_max=16)

    @dataclass
    class CircuitConstraint:
        """An entirely non-STLC, non-typing term. Just satisfies Protocol."""
        name: str
        rule_class: str
        site: int
        sites: tuple[int, ...]
        def expectation(self, state) -> float:
            return 0.42

    t = CircuitConstraint("XOR@gate_2", "XOR-Constraint", 2, (2, 3))
    r = diagnose(state, meta, [t], threshold=1e-6)
    assert r.n_violations == 1
    v = r.rule_violations[0]
    assert v.rule_class == "XOR-Constraint"
    assert v.sites == [2, 3]
    assert abs(v.energy_contribution - 0.42) < 1e-12
