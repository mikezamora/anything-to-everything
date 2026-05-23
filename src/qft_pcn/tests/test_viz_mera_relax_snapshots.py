"""End-to-end test for the mera_relax viz layer (§10.10 induction-theorem
demo wire-up).

Runs a real `mera_relax` simulation for two steps and asserts:
  - total energy is finite and monotonically decays under imag time;
  - forall_protected_leaves is populated (non-empty list) for the default
    `forall x:Nat. ...` preset — the load-bearing §1.1 / §10.10 invariant;
  - ast_text round-trips to a string containing "forall".
"""

from __future__ import annotations

from src.qft_pcn.viz.runs import RunSpec, run_simulation


def _two_frames():
    spec = RunSpec(
        layers=["mera_relax"],
        steps=2,
        seed=0,
        params={"mera_relax": {
            "expr": "forall x:Nat. Eq (x + Zero) x",
            "dt": 0.05,
            "chi_layer": 16,
            "n_nodes_max": 32,
        }},
    )
    return list(run_simulation(spec))


def test_mera_relax_emits_finite_decaying_energy() -> None:
    frames = _two_frames()
    assert len(frames) == 2
    snap0 = frames[0].layer_states["mera_relax"]
    snap1 = frames[1].layer_states["mera_relax"]
    e0 = snap0["total_energy"]
    e1 = snap1["total_energy"]
    assert e0 is not None and e1 is not None
    assert isinstance(e0, float) and isinstance(e1, float)
    # Imag-time relaxation: energy must be finite and (weakly) decrease.
    import math
    assert math.isfinite(e0) and math.isfinite(e1)
    assert e1 <= e0 + 1e-9


def test_mera_relax_forall_protected_leaves_are_populated() -> None:
    frames = _two_frames()
    snap = frames[0].layer_states["mera_relax"]
    leaves = snap["forall_protected_leaves"]
    assert isinstance(leaves, list)
    assert len(leaves) > 0
    for i in leaves:
        assert isinstance(i, int)
        assert i >= 0


def test_mera_relax_ast_text_round_trip_contains_forall() -> None:
    frames = _two_frames()
    snap = frames[0].layer_states["mera_relax"]
    text = snap["ast_text"]
    assert text is not None and isinstance(text, str)
    assert "forall" in text


def test_mera_relax_residual_entries_have_rule_id_site_value() -> None:
    frames = _two_frames()
    snap = frames[0].layer_states["mera_relax"]
    residuals = snap["residuals"]
    assert isinstance(residuals, list) and len(residuals) > 0
    for r in residuals:
        assert "rule_id" in r and isinstance(r["rule_id"], str)
        assert "site" in r and isinstance(r["site"], int)
        assert "value" in r and isinstance(r["value"], float)


def test_mera_relax_layer_bond_dims_emitted() -> None:
    frames = _two_frames()
    snap = frames[0].layer_states["mera_relax"]
    assert snap["n_leaves"] is not None and snap["n_leaves"] > 0
    dims = snap["layer_bond_dims"]
    assert isinstance(dims, list)
    for d in dims:
        assert isinstance(d, int) and d >= 1
