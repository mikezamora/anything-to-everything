"""Few-shot DSL examples must validate against the §9.2 v1 schema."""
import json
import pathlib
import pytest
from src.qft_pcn.bridge.dsl.schema import parse_and_validate


EXAMPLES_DIR = pathlib.Path(__file__).resolve().parent.parent / "llm_examples"


@pytest.mark.parametrize("path", sorted(EXAMPLES_DIR.glob("*.json")),
                         ids=lambda p: p.stem)
def test_example_validates(path: pathlib.Path):
    parse_and_validate(path.read_text())
