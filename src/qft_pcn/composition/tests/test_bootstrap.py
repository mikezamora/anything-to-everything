"""§12.4 conformal-bootstrap acceptance tests.

These tests pin the operator-algebraic bootstrap contract on the real
:class:`MeraTypingHamiltonian` plus the real S2 :func:`solve_sdp` (no
mock optimizer, no AST symbol parsing). The three spec §12.4 acceptance
properties are:

  * well-typed program → bootstrap SDP feasible, dimension bound ~ 0;
  * ill-typed program → bootstrap SDP infeasible (impossibility
    certificate, distinct from §12.1's anomaly trace);
  * engineered substrate-level type perturbation → dimension bound
    matches the obstruction magnitude (the bootstrap is *quantitative*,
    not just a yes/no decision).

All states come from :func:`encode_mera` on real ASTs. Ill-typed states
are produced by surgical substrate-level leaf mutation via
:func:`src.qft_pcn.logic._mera_typing_rules.mutate_leaf` — the same
substrate-level mutation the §12.1 anomaly tests use. The bootstrap
solver never touches the AST; the SDP encoding is derived entirely
from the typing Hamiltonian's leaf-projector structure (spec §1.1).
"""
from __future__ import annotations

import pytest

from src.qft_pcn.composition.bootstrap import (
    DEFAULT_TRUNCATION_GAP,
    BootstrapResult,
    BootstrapVerification,
    build_bootstrap_problem,
    solve_bootstrap,
    verify_typing_via_anomaly_sdp,
)
from src.qft_pcn.composition.bootstrap import _TypingConstraints
from src.qft_pcn.logic._mera_typing_rules import mutate_leaf
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_encoding import TYPE_INT, TYPE_BOOL
from src.qft_pcn.logic.mera_typing_hamiltonian import (
    MeraTypingHamiltonian, RULE_T_LIT_INT,
)
from src.qft_pcn.qft.sdp_solver import SDPStatus


# ---------------------------------------------------------------------------
# §9.7 dense-tensor ceiling opt-out: real encode_mera fixtures allocate
# isometries above the per-test cap. See test_anomaly.py for the same
# pattern — physics tests on real MERAs need the cap lifted.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadow the conftest fixture
    yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lit_int_program():
    """``\\x:Int. x`` — identity on Int, the canonical well-typed lambda.

    Node 1 is the VAR node bound to x:Int; both leaves of the encoding
    align consistently with the typing rules, so every one-node
    generator's obstruction expectation vanishes.
    """
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    return state, meta, H


def _make_lit_int_3_program():
    """``\\x:Int. 3`` — integer literal at node 1; the cleanest one-node
    typing-rule activation for surgical mutation tests.
    """
    state, meta = encode_mera(parse(r"\x:Int. 3"))
    H = MeraTypingHamiltonian(meta)
    return state, meta, H


# ---------------------------------------------------------------------------
# Spec §12.4 acceptance: well-typed program → feasible bootstrap SDP
# ---------------------------------------------------------------------------


def test_well_typed_lambda_has_feasible_bootstrap():
    """``\\x:Int. x`` is well-typed: every one-node typing-rule
    obstruction expectation vanishes, so the bootstrap SDP

        minimize trace(X) s.t. X >> 0, X[i,i] == 0, trace(X) <= tau

    has the trivial feasible solution ``X = 0`` with optimum 0. Spec
    §12.4: well-typed → feasible → dimension bound ~ 0.

    The substrate-level sanity check ``H.total_energy(state) ~ 0`` is
    the same gauge the bootstrap operates in: the obstruction
    expectations are the operator-algebraic source of the typing
    Hamiltonian's per-term residuals.
    """
    state, _meta, H = _make_lit_int_program()
    # Substrate sanity: well-typed state has zero typing energy.
    assert abs(H.total_energy(state)) < 1e-9

    verification = verify_typing_via_anomaly_sdp(state, H)

    assert isinstance(verification, BootstrapVerification)
    assert verification.well_typed, (
        f"well-typed program not classified well-typed; "
        f"status={verification.result.status}, "
        f"obstructions={verification.obstructions}"
    )
    assert verification.result.feasible
    assert verification.result.status in (
        SDPStatus.OPTIMAL, SDPStatus.INACCURATE)
    # Optimum trace ~ 0 (within the truncation gap).
    assert verification.result.optimal_value < DEFAULT_TRUNCATION_GAP * 10
    # Dimension bound: every diagonal is 0, so X = 0 is optimal and
    # max eigenvalue is 0.
    assert verification.result.dimension_bound < DEFAULT_TRUNCATION_GAP * 10
    # No active obstruction.
    nonzero = [
        (rid, n, o) for (rid, n, o) in verification.obstructions
        if o > DEFAULT_TRUNCATION_GAP
    ]
    assert nonzero == [], (
        f"well-typed program surfaced obstructions: {nonzero}")


# ---------------------------------------------------------------------------
# Spec §12.4 acceptance: ill-typed program → infeasible bootstrap SDP
# ---------------------------------------------------------------------------


def test_ill_typed_program_bootstrap_infeasible():
    """Surgical typing violation: integer literal whose type leaf has
    been swapped to BOOL at the substrate level. The T-Lit-Int(node=1)
    generator's obstruction expectation
    ``<P_kind=INT . (I - P_type=INT)> = 1`` (full violation), so

        trace(X) >= X[i*, i*] = 1 > tau = 1e-6

    forces the bootstrap SDP **infeasible**. Spec §12.4: ill-typed →
    infeasible bootstrap → §12.4 impossibility certificate.

    The encoding ``\\x:Int. x + true`` from the task spec would require
    a parser that accepts mixed-type ``+`` — the parser rejects that
    surface syntax pre-encoding. Substrate-level mutation produces an
    operator-algebraically equivalent ill-typed state (same projector-
    expectation footprint) without needing a typo-tolerant parser, and
    keeps the test pinned on the operator structure (§1.1) rather than
    on AST parsing quirks. The §12.1 anomaly tests use the same
    substrate-mutation pattern for the same reason (test_anomaly.py:125).
    """
    state, meta, H = _make_lit_int_3_program()
    # Surgical violation: kind says INT, swap type leaf from INT -> BOOL.
    type_leaf = meta.layout.leaf_of(1, "type")
    mutated = mutate_leaf(state, type_leaf, TYPE_INT, TYPE_BOOL)
    # Substrate sanity: the T-Lit-Int(node=1) residual fires.
    residuals = H.residuals(mutated)
    assert residuals[(RULE_T_LIT_INT, 1)] > 0.5, (
        f"substrate mutation failed to trigger T-Lit-Int(node=1) "
        f"residual: {residuals[(RULE_T_LIT_INT, 1)]}"
    )

    verification = verify_typing_via_anomaly_sdp(mutated, H)

    assert not verification.well_typed, (
        f"ill-typed program classified well-typed; "
        f"status={verification.result.status}"
    )
    assert verification.result.status == SDPStatus.INFEASIBLE, (
        f"expected INFEASIBLE, got {verification.result.status} "
        f"({verification.result.dimension_bound})"
    )
    assert not verification.result.feasible
    # The obstructions list MUST localize the violation on the
    # (T-Lit-Int, 1) generator — same algebraic signature the §12.1
    # anomaly certificate uses (the two diagnostics agree on the
    # substrate signature; they differ in how they surface it).
    fired = [
        (rid, n, o) for (rid, n, o) in verification.obstructions
        if o > DEFAULT_TRUNCATION_GAP
    ]
    pairs = {(rid, n) for (rid, n, _o) in fired}
    assert (RULE_T_LIT_INT, 1) in pairs, (
        f"§12.4 bootstrap failed to surface the (T-Lit-Int, 1) "
        f"obstruction; got {pairs}"
    )


# ---------------------------------------------------------------------------
# Spec §12.4 acceptance: dimension bound quantitatively matches
# substrate obstruction magnitude (engineered simple type → expected
# bound).
# ---------------------------------------------------------------------------


def test_bootstrap_dimension_bound_consistent_with_naive():
    """Engineered simple type → expected dimension bound.

    Setup: ``\\x:Int. 3``, well-typed at the substrate. We then build a
    bootstrap problem manually (via :func:`build_bootstrap_problem`)
    with a *single engineered obstruction value* on a one-element
    diagonal — bypassing the typing-Hamiltonian extraction so we can
    pin the bound to a known input.

    Expected: with diagonal ``[d]`` and truncation gap ``tau``:
      * If ``d <= tau``: feasible, optimal trace = d, dimension bound = d.
      * If ``d > tau``: infeasible.

    Tests the SDP-side acceptance contract (bound matches input)
    independently of the substrate-side obstruction-extraction
    pipeline. Together with the two preceding tests this pins the full
    well-typed→feasible, ill-typed→infeasible, quantitative-bound
    chain.
    """
    # Feasible regime: d well below tau. With tau = 1.0, d = 0.3:
    # feasible, optimal trace = d = 0.3, max eigenvalue = 0.3.
    tau = 1.0
    d_feasible = 0.3
    tc = _TypingConstraints(
        obstruction_diag=(d_feasible,),
        generator_refs=(("T-Lit-Int", 0),),
        truncation_gap=tau,
    )
    problem = build_bootstrap_problem(tc)
    result = solve_bootstrap(problem, truncation_gap=tau)
    assert isinstance(result, BootstrapResult)
    assert result.feasible, f"feasible regime not feasible: {result.status}"
    assert result.optimal_value == pytest.approx(d_feasible, abs=1e-4)
    assert result.dimension_bound == pytest.approx(d_feasible, abs=1e-4)

    # Infeasible regime: d above tau. With tau = 0.1, d = 0.5: the
    # diagonal constraint X[0,0] = 0.5 forces trace(X) >= 0.5 > 0.1,
    # so the SDP is infeasible.
    tau_tight = 0.1
    d_infeasible = 0.5
    tc2 = _TypingConstraints(
        obstruction_diag=(d_infeasible,),
        generator_refs=(("T-Lit-Int", 0),),
        truncation_gap=tau_tight,
    )
    problem2 = build_bootstrap_problem(tc2)
    result2 = solve_bootstrap(problem2, truncation_gap=tau_tight)
    assert not result2.feasible, (
        f"infeasible regime classified feasible: {result2}")
    assert result2.status == SDPStatus.INFEASIBLE
    # Dimension bound is +inf when infeasible.
    assert result2.dimension_bound == float("inf")
