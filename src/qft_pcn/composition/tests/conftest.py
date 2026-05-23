"""Shared fixtures: memory guard, FakeLemmaLibrary, stub solvers, corpora."""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.qft.mera import MERA

# --- §9.7 memory ceiling: no dense tensor larger than chi_cap**4 = 65_536 ----
# The spec §9.7 cap is chi_cap=16 on bond/leg dimensions. The largest LEGITIMATE
# dense allocation in the substrate is a 2-leg reduced density matrix:
# (chi_cap**2) x (chi_cap**2) = 256x256 = 65_536 complex entries. The guard
# catches accidental full-Hilbert allocations (e.g. 16**N for N>=4 leaves
# would already exceed this ceiling and trigger). MERA isometry slabs
# (256x16 = 4_096) and density matrices (256x256 = 65_536) fit; anything
# larger is a substrate violation.
_MAX_ELEMS = 65_536


@pytest.fixture(autouse=True)
def _no_large_dense(monkeypatch):
    """Trip if any code under test allocates a dense ndarray above the ceiling.

    Wraps numpy.zeros/empty/ones; subtree leaf spaces (16**s) for s>=4 would
    blow this. Legitimate MERA isometries and 2-leg RDMs fit comfortably.
    """
    real_zeros, real_empty, real_ones = np.zeros, np.empty, np.ones

    def _guard(real):
        def wrapped(shape, *a, **k):
            arr = real(shape, *a, **k)
            if arr.size > _MAX_ELEMS and arr.ndim >= 2:
                raise AssertionError(
                    f"dense tensor of size {arr.size} exceeds ceiling {_MAX_ELEMS}"
                )
            return arr
        return wrapped

    # Only guard 2-D+; MERA internals allocate small 1-D buffers freely.
    monkeypatch.setattr(np, "zeros", _guard(real_zeros))
    monkeypatch.setattr(np, "empty", _guard(real_empty))
    monkeypatch.setattr(np, "ones", _guard(real_ones))
    yield


def make_product_mera(leaf_vectors: list[np.ndarray]) -> MERA:
    """A normalized product MERA over the given 16-dim leaf vectors."""
    return MERA.from_product(leaf_vectors).normalize()


def basis_leaf(idx: int, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, dtype=complex)
    v[idx] = 1.0
    return v


# --- §8.4 test doubles ------------------------------------------------------
from src.qft_pcn.composition.abstraction import CanonicalPrimitive


class FakeLemmaLibrary:
    """In-memory LemmaLibrary satisfying the spec §7 contract.

    Mirrors :class:`LemmaLibraryAdapter`: ``register`` accepts either a
    ``(state, meta, source_id)`` triple (wake-phase solved problem) or a
    :class:`CanonicalPrimitive` (sleep-phase abstraction). ``tier_of`` /
    ``prune`` / ``replace`` operate on source_id keys (the orchestrator
    consults them with the source_ids returned by :meth:`cached_solutions`,
    NOT the synthetic ``e1`` counter ids).
    """

    def __init__(self):
        self._solved: list[tuple[object, object, str]] = []
        self._tiers: dict[str, str] = {}
        self.registered_primitives: list[CanonicalPrimitive] = []
        self.registered_solutions: list[str] = []
        self.replacements: list[tuple[str, CanonicalPrimitive]] = []
        self.pruned: set[str] = set()
        self._counter = 0

    def register(self, entry):
        self._counter += 1
        eid = f"e{self._counter}"
        if isinstance(entry, CanonicalPrimitive):
            self.registered_primitives.append(entry)
            self._tiers[eid] = "dynamic"
        else:                                   # (MERA, meta, id) triple
            state, meta, sid = entry
            self._solved.append((state, meta, sid))
            self.registered_solutions.append(sid)
            self._tiers.setdefault(sid, "dynamic")
            self._tiers[eid] = "dynamic"
        return eid

    def cached_solutions(self):
        return [(s, m, sid) for s, m, sid in self._solved
                if sid not in self.pruned]

    def tier_of(self, entry_id):
        return self._tiers.get(entry_id, "dynamic")

    def replace(self, entry_id, new_primitive):
        """Append-only: record the replacement, stash the new primitive in
        :attr:`registered_primitives`, and mark the old entry pruned so
        :meth:`cached_solutions` no longer surfaces it. Returns a synthetic
        replacement id."""
        self.replacements.append((entry_id, new_primitive))
        if isinstance(new_primitive, CanonicalPrimitive):
            self.registered_primitives.append(new_primitive)
        self.pruned.add(entry_id)
        return f"r{len(self.replacements)}"

    def prune(self, entry_ids):
        """Mark the given source-ids as pruned. Core-tier ids are skipped
        per spec §3.3. Returns the number of ids actually pruned (excluding
        core and already-pruned)."""
        if isinstance(entry_ids, str):
            entry_ids = (entry_ids,)
        n = 0
        for eid in entry_ids:
            if self._tiers.get(eid) == "core":
                continue
            if eid in self.pruned:
                continue
            self.pruned.add(eid)
            n += 1
        return n

    def has_induction_primitive(self):
        return len(self.registered_primitives) > 0


def make_stub_solver(prebuilt):
    """A deterministic `solve`: payload is a (MERA, meta) pair or 'UNSOLVABLE'."""
    def solve(problem, library):
        payload = problem.hamiltonian_or_state
        if payload == "UNSOLVABLE":
            state, meta = prebuilt["UNSOLVABLE"]
            return state, meta, 1.0
        state, meta = payload
        return state, meta, 0.0
    return solve


def make_step_counting_solver(state, meta):
    """A `solve` whose step count drops when the induction primitive exists."""
    def solve(problem, library):
        steps = 4 if getattr(library, "has_induction_primitive", lambda: False)() \
            else 12
        solve.last_steps = steps
        return state, meta, 0.0
    solve.last_steps = None
    return solve


# --- induction corpus (spec §9.6) -------------------------------------------
def build_induction_corpus():
    """Five inductive-proof MERAs sharing a Forall/Fix induction skeleton with
    different Eq predicates. Built directly as ASTs (M1's surface parser does
    not yet support Forall/Fix/Nat/Eq tokens; the plan-specified contract is
    preserved: five programs whose proof MERAs *share* an induction-shaped
    Forall/Fix subtree)."""
    from src.qft_pcn.logic.ast import (
        Forall, Fix, Eq, Succ, NatLit, Var, Bin, TNat, TProp,
    )
    from src.qft_pcn.logic.mera_encoder import encode_mera
    n = Var(name="n")
    # Predicates vary; all wrapped in the same Forall(n:Nat)->Fix(ind:Prop)->...
    # skeleton so the induction-shaped sub-MERA is shared across the corpus.
    # All five predicates have identical AST node count (5), so the wrapped
    # programs share both AST size and the induction skeleton -- the
    # fingerprint (purity, top-eigs, ast_size) pre-filter (spec §4.3) keeps
    # them in the same comparison bucket.
    predicates = [
        Eq(lhs=Bin(op="+", lhs=n, rhs=NatLit(val=0)), rhs=n),     # n + 0 = n
        Eq(lhs=Bin(op="+", lhs=NatLit(val=0), rhs=n), rhs=n),     # 0 + n = n
        Eq(lhs=Bin(op="*", lhs=n, rhs=NatLit(val=1)), rhs=n),     # n * 1 = n
        Eq(lhs=Succ(arg=n), rhs=Succ(arg=n)),                     # S n = S n
        Eq(lhs=Bin(op="-", lhs=n, rhs=NatLit(val=0)), rhs=n),     # n - 0 = n
    ]
    corpus = []
    for i, pred in enumerate(predicates):
        src = Forall(
            param="n", param_ty=TNat(),
            body=Fix(param="ind", param_ty=TProp(), body=pred),
        )
        state, meta = encode_mera(src)
        corpus.append((state, meta, f"ind{i}"))
    return corpus
