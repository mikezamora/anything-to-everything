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
