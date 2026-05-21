"""Tests for the OllamaLLM shim — uses a mock HTTP server / urlopen patch.

We re-import the llm module inside each test fixture rather than relying on
the import at the top of the file. The neighbouring
``test_bridge_dependency_graph.test_bridge_does_not_import_anthropic_eagerly``
test pops ``src.qft_pcn.bridge.*`` out of ``sys.modules`` and reimports the
package, which means any class object imported at module-collection time is
bound to a stale module dict. Patching ``urlopen`` on the *current* module
object is the only reliable way to inject a mock that the live ``OllamaLLM``
class will actually see.
"""
from __future__ import annotations

import importlib
import json

import pytest
from unittest.mock import MagicMock, patch


def _llm_module():
    """Return the *current* llm module from sys.modules, importing if needed."""
    return importlib.import_module("src.qft_pcn.bridge.llm")


def _mock_urlopen(response_body: dict):
    """Returns a context-manager mock mimicking urllib.request.urlopen()."""
    body = json.dumps(response_body).encode("utf-8")
    cm = MagicMock()
    cm.read.return_value = body
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def test_ollama_emit_dsl_parses_json_response():
    """Given a chat response containing a DSL JSON string, emit_dsl returns the parsed dict."""
    mod = _llm_module()
    spec = {
        "fields": [],
        "sites": 4,
        "constraints": [],
        "boundary": {},
        "observables": [],
        "search": {"method": "imag_time", "steps": 1, "chi_max": 4},
    }
    ollama_response = {"message": {"role": "assistant", "content": json.dumps(spec)}, "done": True}
    llm = mod.OllamaLLM(model="llama3.2:latest")
    with patch.object(mod, "urlopen", return_value=_mock_urlopen(ollama_response)):
        out = llm.emit_dsl("test prompt")
    assert out == spec


def test_ollama_verbalize_returns_assistant_content():
    mod = _llm_module()
    ollama_response = {"message": {"role": "assistant", "content": "The result is 3."}, "done": True}
    llm = mod.OllamaLLM(model="llama3.2:latest")
    with patch.object(mod, "urlopen", return_value=_mock_urlopen(ollama_response)):
        text = llm.verbalize("what is the answer?", {"observables": [{"site": 0, "value": 3.0}]})
    assert "3" in text


def test_ollama_default_model_is_gemma4():
    mod = _llm_module()
    llm = mod.OllamaLLM()
    assert llm.model == "gemma4:31b"


def test_ollama_custom_host_and_port():
    mod = _llm_module()
    llm = mod.OllamaLLM(model="llama3.2:latest", host="example.com", port=12345)
    assert "example.com" in llm.base_url
    assert "12345" in llm.base_url


def test_ollama_unavailable_raises_bridge_error():
    """If Ollama can't be reached, emit_dsl raises a typed BridgeError, not a bare URLError."""
    mod = _llm_module()
    errs = importlib.import_module("src.qft_pcn.bridge.errors")
    llm = mod.OllamaLLM(model="llama3.2:latest", host="localhost", port=11111)
    # No mock — should genuinely fail to connect.
    with pytest.raises(errs.BridgeError):
        llm.emit_dsl("test prompt")
