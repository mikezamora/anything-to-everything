"""Tests for §15 length-synthesis demo + §10.5 real-LLM bridge wiring.

These tests exercise the FULL pipeline (NL -> Ollama -> DSL ->
encode_mera -> MeraEvalHamiltonian -> mera_imaginary_evolve_state ->
decode_mera -> pretty -> example-verification) end-to-end. The LLM
call is real (Ollama, gemma4:31b by default); when the host is
unreachable the tests SKIP loudly rather than mock — the §10.5 / §15
acceptance is meaningful only against a real LLM.

Anti-shortcut (§1.1): no MockLLM canned response is wired here; the
LLM is the load-bearing producer of the structured spec the substrate
consumes.
"""
from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import urlopen

import pytest

from src.qft_pcn.bridge.errors import LlmUnavailableError
from src.qft_pcn.bridge.llm_emitter import LLMToDslEmitter
from src.qft_pcn.composition.demo_length_synthesis import (
    NL_PROMPT,
    DemoResult,
    run_demo,
)


# Opt this file out of the composition/tests/conftest.py §9.7 dense-tensor
# memory ceiling (same rationale as test_hierarchical_proof_demo.py:
# the §10.10 composite's encode_mera saturates the 16-dim leaf substrate).
@pytest.fixture(autouse=True)
def _no_large_dense():
    yield


def _ollama_host_port() -> tuple[str, int]:
    host = os.environ.get("QFT_PCN_OLLAMA_HOST", "localhost")
    port = int(os.environ.get("QFT_PCN_OLLAMA_PORT", "11434"))
    return host, port


def _ollama_reachable() -> bool:
    host, port = _ollama_host_port()
    try:
        with urlopen(f"http://{host}:{port}/api/tags", timeout=3) as r:
            r.read()
        return True
    except (URLError, OSError):
        return False


_OLLAMA_AVAILABLE = _ollama_reachable()
_SKIP_REASON = (
    "Ollama unreachable on QFT_PCN_OLLAMA_HOST:PORT — the §15 acceptance "
    "requires a real LLM; no mock fallback per §1.1 anti-shortcut. Set "
    "QFT_PCN_OLLAMA_HOST to a host running `ollama serve` with the model "
    "preloaded."
)


@pytest.mark.skipif(not _OLLAMA_AVAILABLE, reason=_SKIP_REASON)
def test_llm_emits_parseable_dsl():
    """B1 acceptance: the real LLM emits a JSON spec whose ``theorem``
    string round-trips through the host's surface parser (the validator
    inside LLMToDslEmitter.emit_theorem_spec).
    """
    emitter = LLMToDslEmitter(timeout=180.0)
    spec = emitter.emit_theorem_spec(NL_PROMPT)
    assert isinstance(spec.theorem, str) and spec.theorem.strip(), (
        f"LLM emitted empty / non-string theorem: {spec.theorem!r}"
    )
    # The emitter has already invoked parse_ast on the theorem; if we
    # got here without an exception, the surface parse succeeded.
    assert "forall" in spec.theorem.lower(), (
        f"theorem is not universally quantified: {spec.theorem!r}"
    )
    assert spec.examples, "LLM emitted zero examples"
    for e in spec.examples:
        assert "input" in e and "expected" in e


@pytest.mark.skipif(not _OLLAMA_AVAILABLE, reason=_SKIP_REASON)
def test_length_demo_end_to_end_runs():
    """A2 acceptance: the §15 pipeline runs end-to-end against a real
    LLM. The main theorem relaxes to <H> < 1e-3 (substrate convergence)
    and ALL canonical Peano example-witnesses relax to <H> < 1e-3.

    The §15.11 worked-example numerics: the demo evaluates
    ``forall x:Nat. (x + 0) == x`` at x = 0, 1, 3 (the Peano analogs of
    ``length [], length [x], length [x,y,z]`` -- see module docstring
    for the EXTENSIONS S4 List-substrate adaptation). All three must
    pass.
    """
    result = run_demo()
    assert isinstance(result, DemoResult)
    # Main theorem: substrate converged to ground state.
    assert result.main_converged, (
        f"main theorem failed to converge: <H>={result.main_residual_energy}"
    )
    assert result.main_residual_energy < 1e-3
    # All examples pass.
    n_examples = len(result.examples)
    assert n_examples >= 3, f"expected >= 3 examples, got {n_examples}"
    for ex in result.examples:
        assert ex["passed"], (
            f"example x={ex['x']} did not pass: "
            f"residual={ex['residual_energy']}, instance={ex['instance']!r}"
        )
    # Narrative carries the §15 step banners (basic shape check).
    for marker in ("STEP 1", "STEP 4-5", "STEP 6", "STEP 11", "SUMMARY"):
        assert marker in result.narrative, (
            f"narrative missing §15 banner {marker!r}"
        )
