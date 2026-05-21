"""Enforce the §1.3 dependency rule: bridge depends on logic/qft, not the
other way around. Also verify the public surface exists and that importing
bridge does not transitively load `anthropic`."""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
QFT_PCN = REPO_ROOT / "src" / "qft_pcn"


def _python_files_under(d: pathlib.Path) -> list[pathlib.Path]:
    return [p for p in d.rglob("*.py") if "__pycache__" not in str(p)]


def _imports_in(path: pathlib.Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[str] = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for alias in n.names:
                out.append(alias.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                out.append(n.module)
    return out


def test_logic_does_not_import_bridge():
    bad: list[str] = []
    for p in _python_files_under(QFT_PCN / "logic"):
        for imp in _imports_in(p):
            if "bridge" in imp.split("."):
                bad.append(f"{p}: {imp}")
    assert not bad, "logic/ files importing bridge:\n" + "\n".join(bad)


def test_qft_does_not_import_bridge():
    bad: list[str] = []
    for p in _python_files_under(QFT_PCN / "qft"):
        for imp in _imports_in(p):
            if "bridge" in imp.split("."):
                bad.append(f"{p}: {imp}")
    assert not bad, "qft/ files importing bridge:\n" + "\n".join(bad)


def test_bridge_does_not_import_anthropic_eagerly():
    sys.modules.pop("anthropic", None)
    for k in list(sys.modules):
        if k.startswith("src.qft_pcn.bridge") or k == "src.qft_pcn.bridge":
            sys.modules.pop(k, None)
    importlib.import_module("src.qft_pcn.bridge")
    assert "anthropic" not in sys.modules, \
        "importing qft_pcn.bridge eagerly loaded anthropic"


def test_public_bridge_surface_present():
    import src.qft_pcn.bridge as br
    for name in ("run_problem", "diagnose_problem", "validate_dsl",
                 "Problem", "MockLLM", "BridgeError"):
        assert hasattr(br, name), f"bridge.{name} missing"
