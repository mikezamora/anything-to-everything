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


def test_system_prompt_contains_v1_schema_keywords():
    from src.qft_pcn.viz.llm import build_system_prompt
    prompt = build_system_prompt()
    for kw in ["version", "fields", "sites", "constraints", "observables",
               "search", "runtime", "well_typed_subtree", "example",
               "vocabulary", "use_lemma", "argmax"]:
        assert kw in prompt, f"system prompt missing keyword: {kw}"


def test_system_prompt_includes_length_synthesis_few_shot():
    from src.qft_pcn.viz.llm import build_system_prompt
    prompt = build_system_prompt()
    assert '"runtime": "mera"' in prompt
    assert 'List_a_to_Nat' in prompt
