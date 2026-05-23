"""Wake-sleep orchestration + induction discovery (spec §6; acceptance §9.5-9.6, §9.8)."""
from __future__ import annotations

import pytest

from src.qft_pcn.composition.wake_sleep import (
    Problem, WakeSleepConfig, CycleReport, wake_sleep_cycle, wake_sleep_loop,
)
from .conftest import (
    FakeLemmaLibrary, make_stub_solver, make_step_counting_solver,
    build_induction_corpus,
)


def _problems(corpus):
    return [Problem(id=sid, hamiltonian_or_state=(state, meta))
            for state, meta, sid in corpus]


def test_wake_phase_registers_only_solved_problems():
    corpus = build_induction_corpus()
    lib = FakeLemmaLibrary()
    probs = _problems(corpus)
    probs.append(Problem(id="bad", hamiltonian_or_state="UNSOLVABLE"))
    prebuilt = {"UNSOLVABLE": (corpus[0][0], corpus[0][1])}
    report = wake_sleep_cycle(lib, probs, make_stub_solver(prebuilt), cycle_index=0)
    assert report.n_solved == 5                       # the unsolvable one excluded
    assert "bad" not in lib.registered_solutions


def test_induction_primitive_is_discovered():
    """Architecture §10.9 acceptance test, parts 1-4."""
    corpus = build_induction_corpus()
    lib = FakeLemmaLibrary()
    report = wake_sleep_cycle(lib, _problems(corpus),
                              make_stub_solver({}), cycle_index=0)
    # one cluster of the five induction skeletons -> one promoted primitive
    assert report.n_significant >= 1
    assert len(report.promoted) == 1
    assert len(lib.registered_primitives) == 1
    prim = lib.registered_primitives[0]
    assert set(prim.provenance.source_ids) == {f"ind{i}" for i in range(5)}


def test_subsequent_inductive_proof_converges_in_fewer_steps():
    """Architecture §10.9 acceptance test, part 5."""
    corpus = build_induction_corpus()
    sixth_state, sixth_meta, _ = corpus[0]            # a fresh inductive problem

    # without the induction primitive
    lib_empty = FakeLemmaLibrary()
    solve_empty = make_step_counting_solver(sixth_state, sixth_meta)
    wake_sleep_cycle(lib_empty, [], solve_empty, cycle_index=0)  # no discovery
    solve_empty(Problem("p6", (sixth_state, sixth_meta)), lib_empty)
    steps_without = solve_empty.last_steps

    # with the induction primitive present
    lib_grown = FakeLemmaLibrary()
    wake_sleep_cycle(lib_grown, _problems(corpus),
                     make_stub_solver({}), cycle_index=0)
    assert lib_grown.has_induction_primitive()
    solve_grown = make_step_counting_solver(sixth_state, sixth_meta)
    solve_grown(Problem("p6", (sixth_state, sixth_meta)), lib_grown)
    steps_with = solve_grown.last_steps

    assert steps_with < steps_without          # converges in fewer steps


def test_validation_hook_gates_promotion():
    corpus = build_induction_corpus()
    lib = FakeLemmaLibrary()
    report = wake_sleep_cycle(lib, _problems(corpus), make_stub_solver({}),
                              cycle_index=0, validate=lambda prim: False)
    assert report.promoted == []
    assert lib.registered_primitives == []


def test_cycle_is_deterministic():
    corpus = build_induction_corpus()
    r1 = wake_sleep_cycle(FakeLemmaLibrary(), _problems(corpus),
                          make_stub_solver({}), cycle_index=0)
    r2 = wake_sleep_cycle(FakeLemmaLibrary(), _problems(corpus),
                          make_stub_solver({}), cycle_index=0)
    assert (r1.n_solved, r1.n_candidates, r1.n_clusters,
            r1.n_significant, len(r1.promoted)) == \
           (r2.n_solved, r2.n_candidates, r2.n_clusters,
            r2.n_significant, len(r2.promoted))


def test_loop_stops_after_quiescent_cycles():
    corpus = build_induction_corpus()
    batches = [_problems(corpus), [], [], []]
    reports = wake_sleep_loop(FakeLemmaLibrary(), batches,
                              make_stub_solver({}),
                              WakeSleepConfig(n_quiescent=2))
    # discovers in cycle 0, two empty quiescent cycles, then stops
    assert reports[0].promoted
    assert len(reports) <= 3
