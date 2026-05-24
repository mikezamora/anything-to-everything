"""§9.2 v1 schema extensions: version, mera runtime, argmax, decomposition."""
import pytest
from src.qft_pcn.bridge.dsl.schema import parse_and_validate
from src.qft_pcn.bridge.errors import BadSchemaError


def _minimal_v1() -> str:
    return """
    {
      "version": "1",
      "fields": [{"name": "n", "cutoff": 4}],
      "sites": 2,
      "constraints": [{"kind": "local", "site": 0, "term": "n == 1", "weight": 1.0}],
      "observables": [{"site": 1, "field": "n", "op": "n"}],
      "search": {"method": "imag_time", "steps": 10, "chi_max": 8, "dt": 0.05}
    }
    """


def test_v1_minimal_parses():
    dsl = parse_and_validate(_minimal_v1())
    assert dsl["version"] == "1"


def test_missing_version_rejects():
    spec = _minimal_v1().replace('"version": "1",', "")
    with pytest.raises(BadSchemaError, match="version"):
        parse_and_validate(spec)


def test_search_runtime_mera_accepted():
    spec = _minimal_v1().replace('"method": "imag_time"', '"method": "imag_time", "runtime": "mera"')
    dsl = parse_and_validate(spec)
    assert dsl["search"]["runtime"] == "mera"


def test_search_runtime_invalid_rejects():
    spec = _minimal_v1().replace('"method": "imag_time"', '"method": "imag_time", "runtime": "tebd"')
    with pytest.raises(BadSchemaError, match=r"not one of"):
        parse_and_validate(spec)


def test_search_runtime_default_mps_when_absent():
    # _minimal_v1 has no runtime; default fill should set runtime to 'mps'
    dsl = parse_and_validate(_minimal_v1())
    assert dsl["search"]["runtime"] == "mps"


def test_observable_argmax_accepted():
    spec = _minimal_v1().replace('"op": "n"', '"op": "argmax"')
    dsl = parse_and_validate(spec)
    assert dsl["observables"][0]["op"] == "argmax"


def _v1_with_constraint(c: str) -> str:
    return _minimal_v1().replace(
        '"constraints": [{"kind": "local", "site": 0, "term": "n == 1", "weight": 1.0}]',
        f'"constraints": [{c}]'
    )


def test_well_typed_subtree_accepted():
    dsl = parse_and_validate(_v1_with_constraint(
        '{"kind": "well_typed_subtree", "root": 0, "weight": 10.0}'
    ))
    assert dsl["constraints"][0]["kind"] == "well_typed_subtree"


def test_example_constraint_accepted():
    dsl = parse_and_validate(_v1_with_constraint(
        '{"kind": "example", "input": "[]", "output": "0", "weight": 5.0}'
    ))
    assert dsl["constraints"][0]["kind"] == "example"


def test_vocabulary_constraint_accepted():
    dsl = parse_and_validate(_v1_with_constraint(
        '{"kind": "vocabulary", "primitives": ["Match", "Cons", "Nil"]}'
    ))
    assert dsl["constraints"][0]["primitives"] == ["Match", "Cons", "Nil"]


def test_use_lemma_constraint_accepted():
    dsl = parse_and_validate(_v1_with_constraint(
        '{"kind": "use_lemma", "lemma_id": "length-base", "sites": [0, 1], "weight": 8.0}'
    ))
    assert dsl["constraints"][0]["lemma_id"] == "length-base"


def test_use_lemma_requires_nonempty_sites():
    with pytest.raises(BadSchemaError):
        parse_and_validate(_v1_with_constraint(
            '{"kind": "use_lemma", "lemma_id": "x", "sites": [], "weight": 1.0}'
        ))
