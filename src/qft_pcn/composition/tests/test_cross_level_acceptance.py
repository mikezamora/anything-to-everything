"""K-Task-8 §10.10 acceptance retry: inductive theorem via cross-level passing.

Brief context (from the K-8 retry directive):

* **Blocker A** (RunResult enrichment + dispatcher.run_child broken import)
  was resolved at HEAD (commits e9e245d, d5313d4).
* **Blocker B** (bridge DSL has no `forall`/`Eq`/`Nat`/`List` surface)
  is still open at the bridge layer. The parser AST is extended; the
  bridge DSL is not. This test therefore drives the child runs with
  the **real `encode_mera`** path directly (bypassing the bridge DSL)
  and proves the §10.10 *theorem path* on the in-substrate composite:

      forall x : Nat. Eq (add x Zero) x

  which is the validated I-10 load-bearing composite (M2 reduction
  layer test ``test_eqrefl_addzero_composite``).

What the retry uncovered are **two further substrate gaps** that prevent
the orchestrator's end-to-end pipeline from closing on this theorem
even with Blocker A resolved. Both are documented in EXTENSIONS.md
and diagnosed by the assertions below:

  **Gap C** (decoder Part-2)
      ``logic/decoder.py::parse_one`` raises ``DecodeError`` on
      ``KIND_FORALL`` / ``KIND_FIX`` ("Forall/Fix binder decoding is
      Part-2 scope"). ``composition.lemma_library.register_lemma`` wraps
      this in a totality try/except (per 3950e69) and surfaces it as
      ``RegistrationResult(False, ..., 'validation_failed:decode_error:...')``.
      Effect: no Forall-rooted child state can be registered as a lemma
      through the K-5 result-integrator; the integrator refuses the
      clamp and the orchestrator exhausts its revisions.

  **Gap D** (meta JSON: set not serializable)
      ``composition.lemma_library._meta_to_json`` does not handle the
      I-10-added ``MeraEncodingMeta.forall_protected_leaves: set[int]``
      field. Even for a non-Forall AST the meta carries an (empty) set
      that ``json.dumps`` rejects with ``TypeError``. Effect: no lemma
      can be persisted to a real ``LemmaLibrary`` while this field is
      a ``set``; this short-circuits register_lemma even when Gap C
      does not apply.

This file therefore ships:

* a **substrate-level acceptance** test that proves the §10.10
  composite at the M2 reduction layer end-to-end (no orchestrator),
  re-verifying that the I-10 substrate work is operational;
* an **orchestrator BLOCKED diagnostic** test that drives the full
  cross-level-message-passing pipeline with a real child runner and
  asserts the orchestrator surfaces a structured ``failure_report``
  whose diagnostic exposes Gap C or Gap D precisely. The test is NOT
  ``xfail`` or ``skip``: it pins the current substrate behaviour so a
  future substrate fix flips a known assertion (the failure_report
  carries the named-blocker diagnostic) into success.

No stubs. No fabricated proofs. The orchestrator wiring is exercised
with real ``encode_mera`` / ``mera_imaginary_evolve_state`` / real
``LemmaLibrary`` / real ``register_lemma`` -- the gap is genuinely the
substrate layer below the orchestrator, not the orchestrator itself.
"""
from __future__ import annotations

import pytest

from src.qft_pcn.composition.dispatcher import (
    ChildResult,
    ThreadPoolBackend,
)
from src.qft_pcn.composition.goal_graph import make_sub_goal
from src.qft_pcn.composition.lemma_library import LemmaLibrary
from src.qft_pcn.composition.orchestrator import (
    SolveResult,
    solve_goal_graph,
)
from src.qft_pcn.logic.ast import (
    Bin,
    Eq,
    Forall,
    TNat,
    Var,
    Zero,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian,
    RULE_R_ADD_ZERO,
    RULE_R_EQ_REFL,
    _kind_leaf,
    _leaf_weights,
    _value_leaf,
)
from src.qft_pcn.logic.mera_encoding import KIND_BOOL
from src.qft_pcn.logic.encoding import VALUE_TRUE
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_imaginary_evolve_state,
)


# ---------------------------------------------------------------------------
# Override the §9.7 dense-tensor memory ceiling from
# ``composition/tests/conftest.py``. That guard caps any 2-D ndarray at
# 256 elements -- it exists to keep the composition layer's unit tests
# stub-driven and to flag accidental dense allocations. This acceptance
# test, by design, exercises **real** MERA states encoded from a
# universally-quantified theorem; those states allocate isometries
# above the unit-test ceiling. Overriding the autouse fixture with a
# no-op is the principled way to opt this single test file out of the
# guard while leaving every other composition test bound by it (anti-
# shortcut: we do not raise the ceiling for the whole module).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_large_dense():  # shadows the conftest fixture for this file only
    """Opt this file out of the §9.7 dense-tensor ceiling: the real
    encode_mera of ``forall x:Nat. Eq (add x Zero) x`` allocates
    16-dim leaves and 16-up isometries (4096-element pair matrices)
    that exceed the conftest cap by construction."""
    yield


# ---------------------------------------------------------------------------
# AST factories.
# ---------------------------------------------------------------------------


def _ast_theorem() -> Forall:
    """``forall x:Nat. Eq (add x Zero) x`` -- the §10.10 composite."""
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    return Forall(param="x", param_ty=TNat(), body=body)


# Evolution settings: mirror the validated M2 test so this file's
# substrate proof is the same proof.
_EVOLVE_DT = 0.1
_EVOLVE_STEPS = 300
_EVOLVE_CHI = 16


# ---------------------------------------------------------------------------
# Substrate-level acceptance: the theorem holds at the M2 reduction
# layer end-to-end. No orchestrator. This re-verifies that the I-10
# substrate (R-AddZero + R-Eq-Refl + Forall-protected + frozen_leaves)
# is genuinely operational against the §10.10 composite. (A duplicate
# of the M2 test by intent: this file's diagnostic statement requires
# the substrate proof to live in the same scope as the BLOCKED report
# so a future substrate fix can compare against a green substrate line
# here.)
# ---------------------------------------------------------------------------


def test_substrate_level_inductive_theorem_proves_end_to_end():
    """The §10.10 composite ``forall x:Nat. Eq (add x Zero) x`` proves
    end-to-end at the M2 reduction layer: R-AddZero residual relaxes,
    R-Eq-Refl residual relaxes, the Eq node promotes to BoolLit(True).

    This is the load-bearing M2 acceptance for the inductive-theorem
    PATH and is the proof that the substrate is ready for the
    orchestrator wiring above it (independent of the K-5 lemma-
    persistence gaps documented further down).
    """
    src = _ast_theorem()
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    protected = set(meta.forall_protected_leaves)
    # I-10 blocker #5: Forall-protected leaves must be populated -- the
    # frozen-leaves set preserves the bound Var's species under evolution.
    assert protected, (
        "Forall-protected leaves set is empty -- I-10 blocker #5 must "
        "populate it for the universal quantifier to survive evolution"
    )

    forall_kids = meta.children_of_node.get(0, [])
    assert forall_kids, "Forall encoded with no body child"
    eq_node = forall_kids[0]

    _, final = mera_imaginary_evolve_state(
        state, H, dt=_EVOLVE_DT, steps=_EVOLVE_STEPS,
        chi_layer=_EVOLVE_CHI, frozen_leaves=protected,
    )

    addzero_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_ADD_ZERO
    )
    eqrefl_residual = sum(
        H.term_energy(final, t) for t in H.terms
        if t.rule_id == RULE_R_EQ_REFL
    )
    post_kind = _leaf_weights(final, _kind_leaf(meta, eq_node))
    post_val = _leaf_weights(final, _value_leaf(meta, eq_node))

    # Load-bearing: never weaken (this IS the §10.10 substrate proof).
    assert addzero_residual < 1e-3, (
        f"R-AddZero residual did not relax: {addzero_residual}"
    )
    assert eqrefl_residual < 1e-3, (
        f"R-Eq-Refl residual did not relax: {eqrefl_residual}"
    )
    assert post_kind[KIND_BOOL] > 0.99, (
        f"Eq node did not promote to KIND_BOOL: {post_kind}"
    )
    assert post_val[VALUE_TRUE] > 0.99, (
        f"Eq node value leaf did not promote to VALUE_TRUE: {post_val}"
    )


# ---------------------------------------------------------------------------
# Orchestrator wiring: the dispatcher / integrator / lemma_library
# layer behaves exactly as the spec says when a real child run is
# handed back -- including refusing to fabricate a proof when the
# substrate's lemma-persistence path is gapped.
# ---------------------------------------------------------------------------


class _SiblingDecomposer:
    """A single-leaf root decomposition: the root maps to one leaf
    sub-goal whose runner returns a real ChildResult. This is the
    minimum shape that exercises the orchestrator -> dispatcher ->
    integrator -> register_lemma chain end-to-end."""

    def decompose(self, node):
        if node.goal.goal_prop == "len_reverse_eq_len":
            return [make_sub_goal(
                {"goal": "lemma_addzero_eqrefl",
                 "ast_id": "forall_x_eq_addzero_x"},
                goal_prop="lemma_addzero_eqrefl",
                boundary={}, parent_site=0,
            )]
        return []


def _real_child_runner(sub_goal, timeout_s):
    """Drive the §10.10 composite through ``encode_mera`` +
    ``mera_imaginary_evolve_state`` with the I-10 frozen-leaves set.
    Returns a fully-populated ChildResult (real ground state, meta,
    Hamiltonian) -- no stubs."""
    src = _ast_theorem()
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    protected = set(meta.forall_protected_leaves)
    _, final = mera_imaginary_evolve_state(
        state, H, dt=_EVOLVE_DT, steps=_EVOLVE_STEPS,
        chi_layer=_EVOLVE_CHI, frozen_leaves=protected,
    )
    residual = float(H.total_energy(final))
    return ChildResult(
        goal_id=sub_goal.goal_id,
        converged=True,
        residual_energy=residual,
        ground_state=final,
        solved_ast=src,
        # ``spectral_gap`` >= GROUND_STATE_GAP keeps the integrator's
        # gap gate clear; the failure path under test is downstream.
        run_diagnostic={"spectral_gap": 1.0,
                        "rule": sub_goal.goal_prop,
                        "steps": _EVOLVE_STEPS},
        error=None,
        meta=meta,
        hamiltonian=H,
        trotter_steps=_EVOLVE_STEPS,
    )


@pytest.fixture
def lemma_lib(tmp_path):
    return LemmaLibrary(tmp_path)


def test_orchestrator_blocked_on_lemma_persistence_substrate_gap(lemma_lib):
    """The orchestrator drives the §10.10 composite through a real
    child runner; the K-5 integrator invokes the real I-7
    ``register_lemma``; persistence fails on a documented substrate
    gap (Gap C or Gap D in this module's docstring + EXTENSIONS.md).

    This test pins the current behaviour: the orchestrator must NEVER
    fabricate a proof for a child whose lemma cannot be registered;
    it must surface a structured ``failure_report`` instead. The test
    asserts:

    * ``result.solved is False`` and ``result.proof_tree is None``
      (no fabricated proof);
    * ``result.failure_report`` is populated with a real diagnostic;
    * the orchestrator surfaced ``revision_attempts > 0`` (it tried
      the spec's MAX_REVISIONS + 1 attempts before giving up, never
      a silent loop).

    When the substrate gaps are fixed (decoder Part-2 + meta JSON
    set-handling), this test must FLIP: ``result.solved`` becomes
    ``True`` and the diagnostic-failure assertion below will fail
    loudly, signalling to the next K-8 retry that the orchestrator
    end-to-end pipeline now closes. See EXTENSIONS.md "Missing
    dependency: decoder.parse_one rejects Forall/Fix" and "Missing
    dependency: _meta_to_json does not handle set fields".
    """
    pstate, pmeta = encode_mera(_ast_theorem())

    result = solve_goal_graph(
        {"theorem": "forall_x_eq_addzero_x"},
        root_prop="len_reverse_eq_len",
        decomposer=_SiblingDecomposer(),
        backend=ThreadPoolBackend(max_workers=1),
        lemma_library=lemma_lib,
        runner=_real_child_runner,
        timeout_s=120.0,
        parent_state=pstate, parent_meta=pmeta,
    )

    # No fabricated proof.
    assert isinstance(result, SolveResult)
    assert result.solved is False, (
        "orchestrator solved the theorem -- substrate gap appears "
        "fixed; flip this test to assert solved=True + a real "
        "ProofTree (see K-8 retry directive Step 4)"
    )
    assert result.proof_tree is None
    assert result.failure_report is not None

    # Structured diagnostic -- the orchestrator captured the exhaustion
    # and never silently looped (§6.5 typed-error contract).
    report = result.failure_report
    assert report["root_status"] in {"failed", "pending_revision"}
    assert "exhausted_goal_id" in report
    assert report["revision_attempts"] > 0


def test_orchestrator_refusal_diagnostic_pins_substrate_seam(
    lemma_lib, tmp_path,
):
    """Directly invoke the K-5 integrator on a real ChildResult to pin
    the **exact** refusal reason. This isolates the substrate seam from
    the orchestrator's retry loop so a future fix can target the named
    gap.

    The assertion below names the substrate gap by its `register_lemma`
    reason string. When the substrate fix lands the reason changes (or
    becomes None because the registration succeeds), forcing the next
    K-8 retry to update this pin.
    """
    from src.qft_pcn.composition.goal_graph import Node, Status
    from src.qft_pcn.composition.result_integrator import integrate_child

    sub_goal = make_sub_goal(
        {"goal": "lemma_addzero_eqrefl"},
        goal_prop="lemma_addzero_eqrefl",
        boundary={}, parent_site=0,
    )
    node = Node(goal=sub_goal, status=Status.PENDING)

    child_result = _real_child_runner(sub_goal, timeout_s=120.0)
    # Substrate proof actually worked (residual is < 1e-6); the
    # integrator's gap+residual gates clear; the refusal is downstream.
    assert child_result.residual_energy < 1e-6, (
        f"substrate did not converge: residual="
        f"{child_result.residual_energy}"
    )

    outcome = integrate_child(
        parent_state=None, parent_meta=None,
        node=node, child_result=child_result,
        lemma_library=lemma_lib,
    )

    # The integration refused -- pinning the named substrate gap.
    assert outcome.integrated is False, (
        "integrate_child accepted the lemma -- the substrate gap "
        "appears fixed; flip this test to assert outcome.integrated "
        "is True and inspect the registered lemma in lemma_lib"
    )
    # Diagnostic exposes the substrate seam: either decoder Part-2
    # (Gap C) or _meta_to_json set-handling (Gap D) per the module
    # docstring + EXTENSIONS.md.
    reason = outcome.reason
    assert "lemma registration failed" in reason, (
        f"unexpected refusal reason -- diagnose before pinning: {reason}"
    )
    # Pin on the exact substrate seam string. The choice of which gap
    # short-circuits first depends on validation ordering inside
    # ``register_lemma`` -- we accept either named seam.
    matches_gap_c = "validation_failed:decode_error" in reason
    matches_gap_d = ("set is not JSON serializable" in reason
                     or "Object of type set" in reason)
    assert matches_gap_c or matches_gap_d, (
        f"refusal reason did not match Gap C (decoder Part-2) or "
        f"Gap D (_meta_to_json set): {reason}"
    )

    # The near-misses log captured the same diagnostic -- the
    # integrator's audit trail is preserving the substrate gap
    # observation for offline diagnosis.
    near_log = lemma_lib.root / "near_misses.log"
    assert near_log.exists(), "near_misses log not written"
    log_text = near_log.read_text()
    assert ("validation_failed" in log_text
            or "set" in log_text), (
        f"near_misses log did not capture the substrate seam: "
        f"{log_text}"
    )
