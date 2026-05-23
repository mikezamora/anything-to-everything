"""SubGoal.parent_leaves tuple semantics (spec §5.2a, K-5 result-integrator).

Pins the explicit host-leaf footprint contract: ``SubGoal.parent_leaves``
is a ``tuple[int, ...]`` carried verbatim by the goal graph, and the
result integrator's ``_resolve_host_leaves`` returns it unchanged. The
field is canonical, not derived: callers with non-contiguous /
species-permuted layouts publish the explicit tuple; callers with a
contiguous base+count use :func:`make_contiguous_sub_goal`.
"""
from __future__ import annotations

from src.qft_pcn.composition.goal_graph import (
    SubGoal, Node, Status, make_sub_goal, make_contiguous_sub_goal,
)
from src.qft_pcn.composition.result_integrator import _resolve_host_leaves


class _DummyMeta:
    """Stand-in for MeraEncodingMeta.n_leaves.

    The integrator's ``_resolve_host_leaves`` accepts a ``child_meta``
    argument for caller-signature parity but no longer consults it:
    ``parent_leaves`` is the canonical source. Asserting that the
    returned tuple ignores ``n_leaves`` is part of the contract.
    """
    def __init__(self, n_leaves: int) -> None:
        self.n_leaves = n_leaves


def test_subgoal_holds_arbitrary_tuple():
    """``parent_leaves`` round-trips a non-contiguous tuple verbatim --
    no rebuild from a base int, no contiguous-window expansion."""
    leaves = (2, 5, 9)
    g = make_sub_goal({"g": "non_contiguous"}, goal_prop="P",
                      boundary={}, parent_leaves=leaves)
    assert g.parent_leaves == (2, 5, 9)
    assert isinstance(g.parent_leaves, tuple)


def test_subgoal_default_parent_leaves_is_empty_tuple():
    """Root-goal sentinel: an unspecified footprint is ``()``, not
    ``None``. The empty-tuple convention keeps ``parent_leaves``'s type
    monomorphic (always a tuple of ints) and removes a ``None``
    branch from every reader."""
    g = make_sub_goal({"g": "root"}, goal_prop="P", boundary={})
    assert g.parent_leaves == ()


def test_subgoal_normalizes_iterable_to_int_tuple():
    """``make_sub_goal`` accepts any iterable of ints and freezes it
    into a tuple of ints -- callers cannot smuggle in a mutable list."""
    g = make_sub_goal({"g": "x"}, goal_prop="P", boundary={},
                      parent_leaves=[1, 2, 3])  # list -> tuple
    assert g.parent_leaves == (1, 2, 3)
    assert isinstance(g.parent_leaves, tuple)


def test_make_contiguous_sub_goal_builds_dense_window():
    """The contiguous helper builds ``(base, base+1, ..., base+n-1)`` --
    the principled isomorphic-decomposer shape."""
    g = make_contiguous_sub_goal({"g": "c"}, goal_prop="P", boundary={},
                                 base=4, n_leaves=3)
    assert g.parent_leaves == (4, 5, 6)


def test_make_contiguous_sub_goal_zero_leaves_is_empty():
    """``n_leaves=0`` produces an empty tuple -- not a guard, just the
    natural ``range(base, base+0)``."""
    g = make_contiguous_sub_goal({"g": "c"}, goal_prop="P", boundary={},
                                 base=4, n_leaves=0)
    assert g.parent_leaves == ()


def test_resolve_host_leaves_returns_parent_leaves_verbatim():
    """``_resolve_host_leaves`` reads ``parent_leaves`` directly: no
    extension from a single int, no use of ``child_meta.n_leaves``.
    A non-contiguous tuple round-trips through the integrator's
    resolver unchanged."""
    leaves = (2, 5, 9)
    g = make_sub_goal({"g": "x"}, goal_prop="P", boundary={},
                      parent_leaves=leaves)
    node = Node(goal=g, status=Status.PENDING)
    # n_leaves on the meta deliberately disagrees with len(leaves);
    # the resolver must ignore it (parent_leaves is canonical).
    out = _resolve_host_leaves(node, _DummyMeta(n_leaves=42))
    assert out == (2, 5, 9)


def test_resolve_host_leaves_on_contiguous_window_is_unchanged():
    """Contiguous case: ``_resolve_host_leaves`` returns the same tuple
    the helper built -- the integrator does not re-expand it."""
    g = make_contiguous_sub_goal({"g": "c"}, goal_prop="P", boundary={},
                                 base=0, n_leaves=16)
    node = Node(goal=g, status=Status.PENDING)
    out = _resolve_host_leaves(node, _DummyMeta(n_leaves=16))
    assert out == tuple(range(0, 16))


def test_resolve_host_leaves_empty_tuple_is_returned_as_is():
    """An empty ``parent_leaves`` (root goal) yields an empty
    resolver output -- the integrator's caller (``integrate_child``)
    only invokes the resolver in the non-root clamp path, but the
    resolver itself does not raise on the sentinel."""
    g = make_sub_goal({"g": "root"}, goal_prop="P", boundary={})
    node = Node(goal=g, status=Status.PENDING)
    out = _resolve_host_leaves(node, _DummyMeta(n_leaves=8))
    assert out == ()


def test_make_sub_goal_rejects_duplicate_leaves():
    """Duplicate entries in ``parent_leaves`` are a caller-discipline
    failure: the same host leaf cannot carry two distinct child lemmas.
    ``make_sub_goal`` raises ``ValueError`` loudly (no silent dedup --
    silent coerce hides the decomposer bug)."""
    import pytest
    with pytest.raises(ValueError, match=r"unique"):
        make_sub_goal({"g": "dup"}, goal_prop="P", boundary={},
                      parent_leaves=(2, 2, 5))


def test_make_sub_goal_rejects_negative_leaf():
    """A negative leaf index is meaningless on the parent MERA -- the
    decomposer published garbage. ``make_sub_goal`` raises ``ValueError``
    citing the offending tuple."""
    import pytest
    with pytest.raises(ValueError, match=r"non-negative"):
        make_sub_goal({"g": "neg"}, goal_prop="P", boundary={},
                      parent_leaves=(-1, 0, 1))


def test_make_sub_goal_accepts_unsorted_unique_non_negative():
    """Tuple ORDER is significant (it is the entanglement footprint
    order on the parent MERA, §1.1). ``make_sub_goal`` preserves the
    caller's order verbatim -- no sort, no canonicalisation."""
    g = make_sub_goal({"g": "ord"}, goal_prop="P", boundary={},
                     parent_leaves=(9, 2, 5))
    assert g.parent_leaves == (9, 2, 5)


def test_integrator_passes_non_contiguous_window_to_promoter():
    """End-to-end contract assertion: a SubGoal with a non-contiguous
    ``parent_leaves`` tuple drives the integrator to invoke
    ``Promoter.compile_constraint`` with that exact ``leaves`` list.

    Uses stubs for the lemma library + promoter so the assertion is
    specifically about the resolver -> compile_constraint plumbing
    (the I-Task surfaces are exercised end-to-end in
    ``test_result_integrator.py``).
    """
    import math
    from src.qft_pcn.composition import result_integrator as ri
    from src.qft_pcn.composition.dispatcher import ChildResult

    captured: dict = {}

    class _StubLemmaLibrary:
        """Records the registered lemma; .all_ids/.load not used here."""
        pass

    class _StubPromoter:
        def __init__(self, lib, mode):
            captured["promoter_mode"] = mode

        def compile_constraint(self, spec):
            # Pin the exact ``leaves`` argument the integrator built.
            captured["compile_spec"] = dict(spec)
            return {"compiled": True}

        def apply_init_clamp(self, parent_state, parent_meta, promoted,
                             strength):
            captured["clamp_strength"] = strength

    class _StubRegResult:
        accepted = True
        lemma_id = "lemma-xyz"
        reason = ""

    def _stub_register_lemma(library, ground_state, meta, **kwargs):
        captured["register_called"] = True
        return _StubRegResult

    # Stub-patch the integrator's name-bound dependencies. Direct
    # monkeypatching keeps the assertion local; no global state escapes.
    orig_promoter = ri.Promoter
    orig_register = ri.register_lemma
    ri.Promoter = _StubPromoter
    ri.register_lemma = _stub_register_lemma
    try:
        leaves = (2, 5, 9)
        sub_goal = make_sub_goal({"g": "x"}, goal_prop="P", boundary={},
                                 parent_leaves=leaves)
        node = Node(goal=sub_goal, status=Status.ACTIVE)
        node.parent = Node(
            goal=make_sub_goal({"g": "p"}, goal_prop="Par", boundary={}),
            status=Status.ACTIVE,
        )
        child_meta = _DummyMeta(n_leaves=42)  # deliberately wrong
        child_result = ChildResult(
            goal_id=sub_goal.goal_id, converged=True,
            residual_energy=1e-9,
            ground_state=object(), solved_ast="ast",
            run_diagnostic={"spectral_gap": 1.0},
            error=None, meta=child_meta, hamiltonian=None,
            trotter_steps=0,
        )
        outcome = ri.integrate_child(
            parent_state=object(), parent_meta=object(),
            node=node, child_result=child_result,
            lemma_library=_StubLemmaLibrary(),
        )
    finally:
        ri.Promoter = orig_promoter
        ri.register_lemma = orig_register

    # The integrator clamped: clamp_strength was set by the stub promoter.
    assert outcome.integrated is True
    assert math.isfinite(captured["clamp_strength"])
    # The promoter saw the canonical (non-contiguous) tuple as a list --
    # ``parent_leaves`` is the wire format, NOT a derived window.
    spec = captured["compile_spec"]
    assert spec["kind"] == "use_lemma"
    assert spec["lemma_id"] == "lemma-xyz"
    assert spec["leaves"] == [2, 5, 9], (
        f"integrator did not pass parent_leaves verbatim to "
        f"compile_constraint: {spec['leaves']!r}"
    )
