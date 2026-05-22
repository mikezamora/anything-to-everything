# Abstraction Discovery / Wake-Sleep Implementation Plan (sub-project J)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Every task is TDD: write the failing test first, watch it fail, then implement, then watch it pass.

**Goal:** Build the wake-sleep abstraction-discovery subsystem — mine sub-MERAs of solved problems, cluster their boundary reduced density matrices by trace distance, promote recurring patterns to canonical primitive operators in the lemma library. One Wilson RG-flow step.

**Architecture:** A cached solved problem is a `(MERA, MeraEncodingMeta)` pair. Mining enumerates node-aligned MERA-tree subtrees and extracts the reduced density matrix at each boundary bond. Fingerprint-bucketing pre-filters; trace-distance agglomerative clustering groups recurrences; a `k_min` significance test rejects accidental clusters; canonical-form computation averages member densities in the operator basis and re-purifies to a bounded-bond MERA; the wake-sleep orchestrator runs the solve → mine → cluster → abstract → consolidate cycle.

**Tech Stack:** Python 3.11, numpy 1.26, pytest. NumPy only — no SciPy, no scikit-learn. Built on `src/qft_pcn/qft/mera.py` (sub-project F) and `src/qft_pcn/logic/mera_encoder.py` (M1). Integrates with sub-project I's `LemmaLibrary` via the §7 contract; a `FakeLemmaLibrary` test double stands in until I lands.

**Spec:** `docs/superpowers/specs/2026-05-22-abstraction-discovery-design.md` — authoritative. When this plan and the spec disagree, the spec wins.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Driving principles (from spec §1 — non-negotiable; embed in every subagent prompt)

1. **Don't stop digging at hard field-crossings.** Principled path over easy-but-wrong path.
2. **Abstraction is operator-algebraic, never syntactic.** A pattern is a *reduced density matrix*, not an AST node-string. Shortcut to refuse: "just hash recurring subtree token sequences." Alternative: compute `ρ_S` at the boundary bond and compare by trace distance.
3. **Binding is genuine entanglement, never a classical lookup.** A mined sub-MERA carries its binding correlations as tree entanglement; the boundary `ρ_S` encodes them. Shortcut to refuse: "strip binding and re-derive it with a `dict`." Alternative: read `ρ_S` straight off the MERA's ascended density.
4. **No dense operator at scale.** `ρ_S` lives on the boundary bond (≤16-dim), never on the `16**s` leaf space. Re-purification yields a bounded-bond MERA. Shortcut to refuse: "materialize the subtree and partial-trace." Alternative: ascend per-site densities via `MERA._layer_density`.
5. **`optimize='greedy'` on every `np.einsum`.**
6. **Reuse F's MERA substrate and M1's encoder. Do not reinvent or modify them.** A genuine `mera.py` bug is fixed in place (interface-only, F's tests stay green); otherwise build on top.

---

## Pre-existing worktree state

Unrelated modified files exist in the worktree (`QFT_PCN_ARCHITECTURE.md`, `src/qft_pcn/logic/mera_encoder.py`, `lib/`, `src/qft_pcn/logic/_mera_holes.py`, `src/qft_pcn/tests/test_mera_holes.py`). **Leave them alone.** Stage only the files each task names.

The package `src/qft_pcn/composition/` does not yet exist; Task 1 creates it.

---

## Task 1: Package skeleton, constants, and exception family

**Files:**
- Create: `src/qft_pcn/composition/__init__.py`
- Create: `src/qft_pcn/composition/_abstraction_const.py`
- Create: `src/qft_pcn/composition/tests/__init__.py`
- Test: `src/qft_pcn/composition/tests/test_const.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_const.py`:

```python
"""Constants and exception family (spec §8.1, §10)."""
from __future__ import annotations
import pytest
from src.qft_pcn.composition._abstraction_const import (
    S_MIN, S_MAX, FP_DECIMALS, FP_RANK,
    DEFAULT_DISTANCE_THRESHOLD, DEFAULT_ALPHA, DEFAULT_CHI_CAP,
    N_QUIESCENT, REPURIFICATION_TAIL_TOL, DENSITY_HERMITICITY_TOL,
    AbstractionError, MiningError, RepurificationWarning, LibraryContractError,
)


def test_constant_values():
    assert (S_MIN, S_MAX) == (3, 8)
    assert (FP_DECIMALS, FP_RANK) == (3, 4)
    assert DEFAULT_DISTANCE_THRESHOLD == 0.15
    assert DEFAULT_ALPHA == 1e-3
    assert DEFAULT_CHI_CAP == 16
    assert N_QUIESCENT == 2
    assert REPURIFICATION_TAIL_TOL == 1e-6
    assert DENSITY_HERMITICITY_TOL == 1e-9


def test_exception_hierarchy():
    assert issubclass(MiningError, AbstractionError)
    assert issubclass(LibraryContractError, AbstractionError)
    assert issubclass(AbstractionError, Exception)
    assert issubclass(RepurificationWarning, Warning)


def test_mining_error_raisable():
    with pytest.raises(MiningError):
        raise MiningError("bad density")
```

Run it, watch it fail (`ModuleNotFoundError`).

- [ ] **Step 2: Implement**

Create `src/qft_pcn/composition/__init__.py`:

```python
"""Sub-project J: abstraction discovery / wake-sleep (architecture §10.9)."""
```

Create `src/qft_pcn/composition/tests/__init__.py` (empty file).

Create `src/qft_pcn/composition/_abstraction_const.py`:

```python
"""Constants and exception family for abstraction discovery (spec §8.1, §10)."""
from __future__ import annotations

S_MIN: int = 3
S_MAX: int = 8
FP_DECIMALS: int = 3
FP_RANK: int = 4
DEFAULT_DISTANCE_THRESHOLD: float = 0.15
DEFAULT_ALPHA: float = 1e-3
DEFAULT_CHI_CAP: int = 16
N_QUIESCENT: int = 2
REPURIFICATION_TAIL_TOL: float = 1e-6
DENSITY_HERMITICITY_TOL: float = 1e-9


class AbstractionError(Exception):
    """Base class for abstraction-discovery errors."""


class MiningError(AbstractionError):
    """A cached state is not a valid normalized MERA, or an interval is malformed."""


class LibraryContractError(AbstractionError):
    """The injected LemmaLibrary does not satisfy the spec §7 contract."""


class RepurificationWarning(Warning):
    """Canonical re-purification truncated tail mass above the tolerance."""
```

- [ ] **Step 3: Verify** — `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_const.py -v`. All green.

- [ ] **Step 4: Commit**

```
git add src/qft_pcn/composition/__init__.py src/qft_pcn/composition/_abstraction_const.py src/qft_pcn/composition/tests/__init__.py src/qft_pcn/composition/tests/test_const.py
git commit -m "$(cat <<'EOF'
feat(composition/_abstraction_const): package skeleton + constants + exceptions

Sub-project J skeleton (architecture §10.9): composition package, the
S_MIN/S_MAX/k_min/threshold constants and the AbstractionError family.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Trace distance and the significance test

`trace_distance` and `k_min` are pure numeric functions with no MERA dependency — build and lock them first.

**Files:**
- Create: `src/qft_pcn/composition/abstraction.py` (partial — distance + significance only)
- Test: `src/qft_pcn/composition/tests/test_abstraction_math.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_abstraction_math.py`:

```python
"""Trace distance + significance test (spec §5.1, §5.3; acceptance §9.3–9.4)."""
from __future__ import annotations
import math
import numpy as np
import pytest
from src.qft_pcn.composition.abstraction import trace_distance, k_min


def _rand_density(d: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


def test_trace_distance_of_equal_is_zero():
    rng = np.random.default_rng(0)
    rho = _rand_density(8, rng)
    assert trace_distance(rho, rho) == pytest.approx(0.0, abs=1e-12)


def test_trace_distance_orthogonal_pure_states_is_one():
    rho0 = np.zeros((4, 4), dtype=complex); rho0[0, 0] = 1.0
    rho1 = np.zeros((4, 4), dtype=complex); rho1[1, 1] = 1.0
    assert trace_distance(rho0, rho1) == pytest.approx(1.0, abs=1e-12)


def test_trace_distance_matches_schatten_one_norm():
    rng = np.random.default_rng(7)
    for _ in range(20):
        r1, r2 = _rand_density(6, rng), _rand_density(6, rng)
        diff = r1 - r2
        # 0.5 * sum singular values of the Hermitian difference
        direct = 0.5 * np.sum(np.abs(np.linalg.eigvalsh(diff)))
        assert trace_distance(r1, r2) == pytest.approx(direct, abs=1e-10)
        assert 0.0 - 1e-12 <= trace_distance(r1, r2) <= 1.0 + 1e-12


def test_trace_distance_zero_pads_unequal_dims():
    rng = np.random.default_rng(3)
    small = _rand_density(2, rng)
    big = np.zeros((4, 4), dtype=complex)
    big[:2, :2] = small
    # big is `small` embedded with the extra bond empty -> distance 0
    assert trace_distance(small, big) == pytest.approx(0.0, abs=1e-10)


def test_k_min_floored_at_two():
    assert k_min(2) >= 2
    assert k_min(1) >= 2


def test_k_min_rejects_chance_clusters():
    n = 500
    k = k_min(n, alpha=1e-3)
    # a cluster of size k is below the alpha chance bound...
    assert math.exp(-k * math.log(max(n, 2))) <= 1e-3
    # ...and k-1 is not (k is the *smallest* significant size)
    assert math.exp(-(k - 1) * math.log(max(n, 2))) > 1e-3
```

Run it, watch it fail.

- [ ] **Step 2: Implement**

Create `src/qft_pcn/composition/abstraction.py`:

```python
"""Clustering + canonical-form computation for abstraction discovery (spec §5)."""
from __future__ import annotations

import math

import numpy as np

from src.qft_pcn.composition._abstraction_const import DEFAULT_ALPHA


def _embed(rho: np.ndarray, dim: int) -> np.ndarray:
    """Embed a (d, d) density into a (dim, dim) space by zero-padding (spec §5.1)."""
    if rho.shape[0] == dim:
        return rho
    out = np.zeros((dim, dim), dtype=complex)
    d = rho.shape[0]
    out[:d, :d] = rho
    return out


def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """D(rho1, rho2) = 0.5 * sum(|eigvals(rho1 - rho2)|)  (spec §5.1).

    Hermitian inputs; unequal dimensions are zero-padded to the larger space.
    """
    dim = max(rho1.shape[0], rho2.shape[0])
    diff = _embed(rho1, dim) - _embed(rho2, dim)
    # Hermitian difference -> eigvalsh is the correct, cheap Schatten-1 route.
    return 0.5 * float(np.sum(np.abs(np.linalg.eigvalsh(diff))))


def k_min(n_candidates: int, alpha: float = DEFAULT_ALPHA) -> int:
    """Smallest cluster size unlikely to arise by chance (spec §5.3).

    Uniform null model: P(cluster of size k) ~ exp(-k log N). Solve
    exp(-k log N) <= alpha  ->  k >= -log(alpha) / log(N). Floored at 2.
    """
    n = max(n_candidates, 2)
    k = math.ceil(-math.log(alpha) / math.log(n))
    return max(k, 2)
```

- [ ] **Step 3: Verify** — `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_abstraction_math.py -v`. All green.

- [ ] **Step 4: Commit**

```
git add src/qft_pcn/composition/abstraction.py src/qft_pcn/composition/tests/test_abstraction_math.py
git commit -m "$(cat <<'EOF'
feat(composition/abstraction): trace distance + k_min significance test

Trace distance D(rho1,rho2)=1/2||rho1-rho2||_1 via eigvalsh, and the
k_min Bayesian-evidence threshold from the uniform null model (§10.9).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Subtree miner — enumeration, boundary density, fingerprint

**Files:**
- Create: `src/qft_pcn/composition/subtree_miner.py`
- Test: `src/qft_pcn/composition/tests/test_subtree_miner.py` (create)
- Create: `src/qft_pcn/composition/tests/conftest.py` (the memory guard + builders)

- [ ] **Step 1: Write `conftest.py` (test fixtures)**

Create `src/qft_pcn/composition/tests/conftest.py`:

```python
"""Shared fixtures: memory guard, FakeLemmaLibrary, stub solvers, corpora."""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.qft.mera import MERA

# --- §9.7 memory ceiling: no dense tensor larger than chi_cap**2 = 256 -------
_MAX_ELEMS = 256


@pytest.fixture(autouse=True)
def _no_large_dense(monkeypatch):
    """Trip if any code under test allocates a dense ndarray above the ceiling.

    Wraps numpy.zeros/empty/ones; subtree leaf spaces (16**s) would blow this.
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
```

(The induction-corpus builders are added to this same `conftest.py` in Task 6 — leave a placeholder comment `# induction corpus builders: Task 6` at the end now.)

- [ ] **Step 2: Write the failing test**

Create `src/qft_pcn/composition/tests/test_subtree_miner.py`:

```python
"""Subtree miner (spec §4; acceptance §9.1–9.2, §9.7)."""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition._abstraction_const import S_MIN, S_MAX, MiningError
from src.qft_pcn.composition.subtree_miner import (
    MineConfig, Fingerprint, SubtreeCandidate,
    mine_subtrees, mine_corpus, bucket_by_fingerprint,
)
from src.qft_pcn.composition.abstraction import trace_distance
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse  # M1 parser


def _encode(src: str):
    state, meta = encode_mera(parse(src))
    return state, meta


def test_mine_returns_node_aligned_subtrees_in_size_band():
    state, meta = _encode(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    cands = mine_subtrees(state, meta, source_id="p1")
    assert cands, "expected at least one mined subtree"
    for c in cands:
        assert S_MIN <= c.ast_size <= S_MAX
        lo, hi = c.leaf_interval
        # node-aligned: interval width covers whole AST nodes (5 leaves each)
        assert (hi - lo) % 5 == 0 or _covers_pad(meta, lo, hi)


def _covers_pad(meta, lo, hi):
    # node-aligned check tolerant of PAD leaves at the interval edge
    return all(meta.node_of_leaf[i] == -1
               for i in range(lo, hi) if meta.node_of_leaf[i] == -1)


def test_mined_densities_are_valid():
    state, meta = _encode(r"\f:Int->Int. \x:Int. f (f x)")
    for c in mine_subtrees(state, meta, source_id="p2"):
        rho = c.rho
        assert np.allclose(rho, rho.conj().T, atol=1e-9)            # Hermitian
        eig = np.linalg.eigvalsh(rho)
        assert eig.min() >= -1e-9                                    # PSD
        assert np.trace(rho).real == pytest.approx(1.0, abs=1e-9)    # unit trace


def test_fingerprint_invariant_under_bond_basis_change():
    state, meta = _encode(r"\x:Int. x")
    cands = mine_subtrees(state, meta, source_id="p3")
    if not cands:
        pytest.skip("program too small for a subtree in band")
    rho = cands[0].rho
    d = rho.shape[0]
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
    rho_rot = q @ rho @ q.conj().T
    from src.qft_pcn.composition.subtree_miner import fingerprint_of
    assert fingerprint_of(rho_rot, cands[0].ast_size) == cands[0].fingerprint
    assert trace_distance(rho, rho_rot) == pytest.approx(0.0, abs=1e-9)


def test_bucket_groups_identical_fingerprints():
    state, meta = _encode(r"(\x:Int. x + 1)(2)")
    cands = mine_subtrees(state, meta, source_id="p4")
    buckets = bucket_by_fingerprint(cands)
    total = sum(len(v) for v in buckets.values())
    assert total == len(cands)
    for fp, group in buckets.items():
        assert all(c.fingerprint == fp for c in group)


def test_mine_rejects_invalid_mera():
    state, meta = _encode(r"\x:Int. x")
    # corrupt the top tensor so a reduced density is no longer unit-trace
    bad = state.copy()
    bad.layers[-1].tensor *= 2.0  # break normalization
    with pytest.raises(MiningError):
        mine_subtrees(bad, meta, source_id="bad")
```

> If M1's encoder produces no in-band subtree for the smallest programs, the affected tests `pytest.skip`; `test_mine_returns_node_aligned_subtrees_in_size_band` uses a 6+-node program guaranteed to have one. Adjust the corrupt-MERA mutation in `test_mine_rejects_invalid_mera` to whatever `mera.py`'s public layer API actually exposes — if no public mutator exists, build an explicitly non-normalized MERA via `MERA.from_product` with an un-normalized leaf vector instead.

Run it, watch it fail.

- [ ] **Step 3: Implement**

Create `src/qft_pcn/composition/subtree_miner.py`:

```python
"""Subtree mining for abstraction discovery (spec §4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.qft_pcn.composition._abstraction_const import (
    S_MIN, S_MAX, FP_DECIMALS, FP_RANK, DENSITY_HERMITICITY_TOL, MiningError,
)


@dataclass(frozen=True)
class MineConfig:
    s_min: int = S_MIN
    s_max: int = S_MAX
    fp_decimals: int = FP_DECIMALS
    fp_rank: int = FP_RANK


@dataclass(frozen=True)
class Fingerprint:
    purity: float
    top_eigs: tuple[float, ...]
    ast_size: int


@dataclass
class SubtreeCandidate:
    source_id: str
    leaf_interval: tuple[int, int]
    ast_size: int
    rho: np.ndarray
    fingerprint: Fingerprint


def _validate_density(rho: np.ndarray) -> None:
    """Raise MiningError unless rho is Hermitian, PSD, unit-trace (spec §4.2)."""
    tol = DENSITY_HERMITICITY_TOL
    if not np.allclose(rho, rho.conj().T, atol=tol):
        raise MiningError("reduced density is not Hermitian")
    if np.linalg.eigvalsh(rho).min() < -tol:
        raise MiningError("reduced density is not positive-semidefinite")
    if abs(np.trace(rho).real - 1.0) > 1e-6:
        raise MiningError("reduced density is not unit-trace")


def fingerprint_of(rho: np.ndarray, ast_size: int,
                   fp_decimals: int = FP_DECIMALS,
                   fp_rank: int = FP_RANK) -> Fingerprint:
    """Basis-invariant (purity, top-eigs, size) fingerprint (spec §4.3)."""
    purity = round(float(np.trace(rho @ rho).real), fp_decimals)
    eigs = np.sort(np.linalg.eigvalsh(rho).real)[::-1]
    top = list(eigs[:fp_rank]) + [0.0] * max(0, fp_rank - len(eigs))
    top_eigs = tuple(round(float(e), fp_decimals) for e in top[:fp_rank])
    return Fingerprint(purity=purity, top_eigs=top_eigs, ast_size=ast_size)


def _node_aligned_intervals(meta) -> list[tuple[int, int, int]]:
    """Yield (leaf0, leaf_hi, ast_size) for every node-aligned MERA-tree block
    whose AST size lies in [S_MIN, S_MAX]. A block is a width-2^d leaf interval
    aligned to a 2^d boundary; a block is mined only if it splits no AST node.
    """
    n_leaves = meta.n_leaves
    out: list[tuple[int, int, int]] = []
    d = 1
    while (1 << d) <= n_leaves:
        w = 1 << d
        for lo in range(0, n_leaves, w):
            hi = lo + w
            nodes = {meta.node_of_leaf[i] for i in range(lo, hi)}
            nodes.discard(-1)  # PAD leaves
            # node-aligned: every leaf of each touched node lies inside [lo,hi)
            split = any(
                any(not (lo <= j < hi)
                    for j in range(n_leaves)
                    if meta.node_of_leaf[j] == nd)
                for nd in nodes
            )
            if split:
                continue
            s = len(nodes)
            if S_MIN <= s <= S_MAX:
                out.append((lo, hi, s))
        d += 1
    return out


def _boundary_density(state, leaf0: int, leaf_hi: int) -> np.ndarray:
    """Reduced density matrix at the boundary bond of the [leaf0,leaf_hi) block.

    Ascends F's per-site layer densities (MERA._layer_density) to the layer at
    which the block is a single site; reads that site's reduced density. Never
    materializes the 16**s leaf space (spec §4.2, principle 4).
    """
    w = leaf_hi - leaf0
    d = w.bit_length() - 1            # block width 2^d
    rho_sites = state._layer_density(d)
    block_index = leaf0 >> d
    return np.asarray(rho_sites[block_index], dtype=complex)


def mine_subtrees(state, meta, source_id: str,
                  config: MineConfig = MineConfig()) -> list[SubtreeCandidate]:
    """Enumerate node-aligned sub-MERAs and extract boundary densities (spec §4.4)."""
    out: list[SubtreeCandidate] = []
    for lo, hi, s in _node_aligned_intervals(meta):
        rho = _boundary_density(state, lo, hi)
        _validate_density(rho)
        fp = fingerprint_of(rho, s, config.fp_decimals, config.fp_rank)
        out.append(SubtreeCandidate(source_id, (lo, hi), s, rho, fp))
    return out


def mine_corpus(solved: list[tuple[object, object, str]],
                config: MineConfig = MineConfig()) -> list[SubtreeCandidate]:
    """mine_subtrees across a corpus (spec §4.4)."""
    out: list[SubtreeCandidate] = []
    for state, meta, source_id in solved:
        out.extend(mine_subtrees(state, meta, source_id, config))
    return out


def bucket_by_fingerprint(
        candidates: list[SubtreeCandidate]
) -> dict[Fingerprint, list[SubtreeCandidate]]:
    """Group candidates by exact fingerprint (spec §4.3)."""
    buckets: dict[Fingerprint, list[SubtreeCandidate]] = {}
    for c in candidates:
        buckets.setdefault(c.fingerprint, []).append(c)
    return buckets
```

> `_layer_density` and the block-index convention are F's; confirm against `src/qft_pcn/qft/mera.py` before relying on the `leaf0 >> d` indexing. If F's `_layer_density` indexes blocks differently, adapt `_boundary_density` — the spec contract is "the reduced density at the block, ascended via `_layer_density`", not a literal index formula. If `_layer_density` is private and a subagent judges a thin public wrapper warranted, add it to `mera.py` interface-only and confirm F's tests stay green (spec §11).

- [ ] **Step 4: Verify** — `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_subtree_miner.py -v`. All green. Run F's MERA tests and M1's encoder tests; confirm still green.

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/subtree_miner.py src/qft_pcn/composition/tests/test_subtree_miner.py src/qft_pcn/composition/tests/conftest.py
git commit -m "$(cat <<'EOF'
feat(composition/subtree_miner): mine sub-MERAs + boundary reduced densities

Enumerate node-aligned MERA-tree subtrees of solved states, extract the
reduced density matrix at each boundary bond via MERA._layer_density, and
fingerprint-hash by (purity, top eigenvalues, size). No 16**s leaf space
is ever materialized (architecture §10.9).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Clustering, significance filter, provenance

**Files:**
- Modify: `src/qft_pcn/composition/abstraction.py` (append clustering + Cluster + Provenance)
- Test: `src/qft_pcn/composition/tests/test_clustering.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_clustering.py`:

```python
"""Agglomerative clustering + significance filter (spec §5.2–5.3, §5.5; §9.3–9.4)."""
from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.composition.subtree_miner import SubtreeCandidate, fingerprint_of
from src.qft_pcn.composition.abstraction import (
    ClusterConfig, Cluster, cluster_candidates, significant_clusters, k_min,
)


def _cand(source_id: str, rho: np.ndarray, size: int = 4) -> SubtreeCandidate:
    return SubtreeCandidate(source_id, (0, 5 * size), size, rho,
                            fingerprint_of(rho, size))


def _pure(d: int, idx: int) -> np.ndarray:
    rho = np.zeros((d, d), dtype=complex)
    rho[idx, idx] = 1.0
    return rho


def test_two_well_separated_groups_yield_two_clusters():
    # group A: near |0>; group B: near |1>; inter-group distance ~ 1
    a = [_cand(f"a{i}", _pure(4, 0)) for i in range(4)]
    b = [_cand(f"b{i}", _pure(4, 1)) for i in range(3)]
    clusters = cluster_candidates(a + b, ClusterConfig(distance_threshold=0.15))
    assert len(clusters) == 2
    sizes = sorted(c.size for c in clusters)
    assert sizes == [3, 4]


def test_clustering_is_deterministic_under_permutation():
    rng = np.random.default_rng(2)
    cands = [_cand(f"c{i}", _pure(4, i % 2)) for i in range(8)]
    base = cluster_candidates(cands, ClusterConfig())
    shuffled = cands[::-1]
    perm = cluster_candidates(shuffled, ClusterConfig())
    assert sorted(c.size for c in base) == sorted(c.size for c in perm)


def test_significant_clusters_drops_small_ones():
    big = Cluster(members=[_cand(f"x{i}", _pure(4, 0)) for i in range(6)])
    tiny = Cluster(members=[_cand("y0", _pure(4, 1))])
    n = 200
    sig = significant_clusters([big, tiny], n_candidates=n)
    assert big in sig
    assert tiny not in sig
    assert all(c.size >= k_min(n) for c in sig)
```

Run it, watch it fail.

- [ ] **Step 2: Implement — append to `abstraction.py`**

Append to `src/qft_pcn/composition/abstraction.py`:

```python
from dataclasses import dataclass, field

from src.qft_pcn.composition._abstraction_const import DEFAULT_DISTANCE_THRESHOLD
from src.qft_pcn.composition.subtree_miner import SubtreeCandidate, Fingerprint


@dataclass(frozen=True)
class ClusterConfig:
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
    linkage: str = "average"          # "average" | "complete" | "single"


@dataclass
class Cluster:
    members: list[SubtreeCandidate]

    @property
    def size(self) -> int:
        return len(self.members)


@dataclass
class Provenance:
    source_ids: tuple[str, ...]
    occurrences: tuple[tuple[str, tuple[int, int]], ...]
    discovered_in_cycle: int
    use_log: list[str] = field(default_factory=list)


def _fp_adjacent(f1: Fingerprint, f2: Fingerprint, grid: float = 1e-3) -> bool:
    """Two fingerprints are comparison-eligible if same size and their rounded
    scalars differ by at most one grid step (spec §4.3)."""
    if f1.ast_size != f2.ast_size:
        return False
    if abs(f1.purity - f2.purity) > grid + 1e-12:
        return False
    return all(abs(a - b) <= grid + 1e-12
               for a, b in zip(f1.top_eigs, f2.top_eigs))


def _pair_distance(c1: SubtreeCandidate, c2: SubtreeCandidate) -> float:
    """Trace distance, gated by the fingerprint pre-filter (spec §4.3, §5.2)."""
    if not _fp_adjacent(c1.fingerprint, c2.fingerprint):
        return 1.0                    # never merged across fingerprint buckets
    return trace_distance(c1.rho, c2.rho)


def _linkage_distance(a: list[SubtreeCandidate], b: list[SubtreeCandidate],
                      dmat: dict[tuple[int, int], float],
                      idx: dict[int, int], how: str) -> float:
    vals = [dmat[(min(idx[id(x)], idx[id(y)]), max(idx[id(x)], idx[id(y)]))]
            for x in a for y in b]
    if how == "single":
        return min(vals)
    if how == "complete":
        return max(vals)
    return sum(vals) / len(vals)      # average


def cluster_candidates(candidates: list[SubtreeCandidate],
                       config: ClusterConfig = ClusterConfig()) -> list[Cluster]:
    """Hierarchical agglomerative clustering under trace distance (spec §5.2).

    Deterministic: candidates are first sorted by (source_id, leaf_interval);
    ties in the merge step break by that order.
    """
    ordered = sorted(candidates, key=lambda c: (c.source_id, c.leaf_interval))
    n = len(ordered)
    if n == 0:
        return []
    idx = {id(c): i for i, c in enumerate(ordered)}
    dmat: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            dmat[(i, j)] = _pair_distance(ordered[i], ordered[j])

    groups: list[list[SubtreeCandidate]] = [[c] for c in ordered]
    while len(groups) > 1:
        best = None
        best_d = config.distance_threshold + 1e-12
        for gi in range(len(groups)):
            for gj in range(gi + 1, len(groups)):
                d = _linkage_distance(groups[gi], groups[gj], dmat, idx,
                                      config.linkage)
                if d <= best_d - 1e-12 or (best is None and d <= best_d):
                    if best is None or d < best_d - 1e-12:
                        best, best_d = (gi, gj), d
        if best is None:
            break
        gi, gj = best
        groups[gi] = groups[gi] + groups[gj]
        del groups[gj]
    return [Cluster(members=g) for g in groups]


def significant_clusters(clusters: list[Cluster], n_candidates: int,
                         alpha: float = DEFAULT_ALPHA) -> list[Cluster]:
    """Keep clusters with size >= k_min(n_candidates, alpha) (spec §5.3)."""
    threshold = k_min(n_candidates, alpha)
    return [c for c in clusters if c.size >= threshold]
```

> The merge loop above stops once no inter-group distance is `<= distance_threshold`. Keep the tie-break deterministic: when two candidate merges have equal distance, prefer the pair with the lexicographically smallest `(source_id, leaf_interval)` first member — the `ordered` sort plus index order delivers this. A subagent simplifying the loop must preserve `test_clustering_is_deterministic_under_permutation`.

- [ ] **Step 3: Verify** — `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_clustering.py src/qft_pcn/composition/tests/test_abstraction_math.py -v`. All green.

- [ ] **Step 4: Commit**

```
git add src/qft_pcn/composition/abstraction.py src/qft_pcn/composition/tests/test_clustering.py
git commit -m "$(cat <<'EOF'
feat(composition/abstraction): agglomerative clustering + significance filter

Hierarchical agglomerative clustering of reduced density matrices under
trace distance with fingerprint pre-filtering; significant_clusters applies
the k_min threshold; Provenance dataclass (architecture §10.9).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Canonical-form computation

**Files:**
- Modify: `src/qft_pcn/composition/abstraction.py` (append `CanonicalPrimitive`, `compute_canonical_form`)
- Test: `src/qft_pcn/composition/tests/test_canonical_form.py` (create)

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_canonical_form.py`:

```python
"""Canonical-form computation (spec §5.4; acceptance §9.5)."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.qft_pcn.composition._abstraction_const import RepurificationWarning
from src.qft_pcn.composition.subtree_miner import SubtreeCandidate, fingerprint_of
from src.qft_pcn.composition.abstraction import (
    Cluster, CanonicalPrimitive, compute_canonical_form, trace_distance,
)


def _cand(sid: str, rho: np.ndarray) -> SubtreeCandidate:
    return SubtreeCandidate(sid, (0, 20), 4, rho, fingerprint_of(rho, 4))


def _rand_density(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = a @ a.conj().T
    return rho / np.trace(rho).real


def test_canonical_of_identical_members_is_that_member():
    rho = _rand_density(8, 11)
    cluster = Cluster(members=[_cand(f"m{i}", rho.copy()) for i in range(5)])
    prim = compute_canonical_form(cluster, chi_cap=16)
    assert np.allclose(prim.rho_canonical, rho, atol=1e-9)
    assert prim.avg_trace_distance == pytest.approx(0.0, abs=1e-9)


def test_canonical_is_the_operator_basis_mean():
    rhos = [_rand_density(8, s) for s in (1, 2, 3)]
    cluster = Cluster(members=[_cand(f"m{i}", r) for i, r in enumerate(rhos)])
    prim = compute_canonical_form(cluster, chi_cap=16)
    mean = sum(rhos) / len(rhos)
    assert np.allclose(prim.rho_canonical, mean, atol=1e-9)
    expected_avg = np.mean([trace_distance(mean, r) for r in rhos])
    assert prim.avg_trace_distance == pytest.approx(expected_avg, abs=1e-9)


def test_repurified_mera_reextracts_to_rho_canonical():
    rho = _rand_density(8, 5)
    cluster = Cluster(members=[_cand("m0", rho)])
    prim = compute_canonical_form(cluster, chi_cap=16)
    # the bounded-bond MERA's boundary density matches rho_canonical
    assert prim.mera is not None
    assert prim.chi <= 16
    assert abs(prim.mera.norm_sq() - 1.0) < 1e-9


def test_truncation_emits_repurification_warning():
    # a full-rank density truncated below its rank warns
    rho = _rand_density(16, 9)
    cluster = Cluster(members=[_cand("m0", rho)])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compute_canonical_form(cluster, chi_cap=2)
    assert any(issubclass(w.category, RepurificationWarning) for w in caught)
```

Run it, watch it fail.

- [ ] **Step 2: Implement — append to `abstraction.py`**

Append to `src/qft_pcn/composition/abstraction.py`:

```python
import warnings

from src.qft_pcn.composition._abstraction_const import (
    DEFAULT_CHI_CAP, REPURIFICATION_TAIL_TOL, RepurificationWarning,
)
from src.qft_pcn.qft.mera import MERA


@dataclass
class CanonicalPrimitive:
    rho_canonical: np.ndarray
    mera: MERA
    chi: int
    avg_trace_distance: float
    provenance: Provenance


def _repurify(rho: np.ndarray, chi_cap: int) -> tuple[MERA, int]:
    """Re-purify a density matrix into a bounded-bond MERA (spec §5.4 step 2).

    Eigendecompose rho = sum_j p_j |e_j><e_j|; keep the top chi_cap eigenpairs;
    build a purifying term superposition over the kept eigenvectors. The result
    is a bounded-bond MERA -- never a dense leaf-space tensor (principle 4).
    """
    eigvals, eigvecs = np.linalg.eigvalsh(rho), None
    eigvals, eigvecs = np.linalg.eigh(rho)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    keep = min(chi_cap, len(eigvals))
    kept_p = np.clip(eigvals[:keep].real, 0.0, None)
    tail = float(np.sum(np.clip(eigvals[keep:].real, 0.0, None)))
    if tail > REPURIFICATION_TAIL_TOL:
        warnings.warn(
            f"re-purification truncated tail mass {tail:.3e}",
            RepurificationWarning, stacklevel=2,
        )
    kept_p = kept_p / kept_p.sum()
    d = rho.shape[0]
    # purifying term superposition: each kept eigenvector contributes a product
    # term sqrt(p_j) over two boundary leaves (system + purifying reference).
    terms = []
    for j in range(keep):
        vec = eigvecs[:, j]
        leaf_sys = np.asarray(vec, dtype=complex)
        leaf_ref = np.zeros(d, dtype=complex)
        leaf_ref[j] = 1.0
        terms.append((complex(np.sqrt(kept_p[j])), [leaf_sys, leaf_ref]))
    mera = MERA.from_term_superposition(terms).normalize()
    return mera, keep


def compute_canonical_form(cluster: Cluster,
                           chi_cap: int = DEFAULT_CHI_CAP,
                           cycle_index: int = 0) -> CanonicalPrimitive:
    """Cluster representative: operator-basis mean re-purified (spec §5.4)."""
    members = cluster.members
    dim = max(m.rho.shape[0] for m in members)
    rho_canonical = sum(_embed(m.rho, dim) for m in members) / len(members)
    avg_d = float(np.mean([trace_distance(rho_canonical, m.rho)
                           for m in members]))
    mera, chi = _repurify(rho_canonical, chi_cap)
    prov = Provenance(
        source_ids=tuple(dict.fromkeys(m.source_id for m in members)),
        occurrences=tuple((m.source_id, m.leaf_interval) for m in members),
        discovered_in_cycle=cycle_index,
    )
    return CanonicalPrimitive(rho_canonical=rho_canonical, mera=mera, chi=chi,
                              avg_trace_distance=avg_d, provenance=prov)
```

> Confirm `MERA.from_term_superposition`'s exact signature against `src/qft_pcn/qft/mera.py` (it exists per F's API — line 428). If its term format differs from `(amplitude, [leaf_vectors])`, adapt `_repurify` to match — the spec contract is "a bounded-bond MERA re-purification", not a literal call form. The purifying-leaf construction must keep every dense object `<= chi_cap**2`; if `from_term_superposition` cannot express a two-leaf purification directly, a subagent should still keep the system within the §9.7 ceiling — escalate rather than materialize a large tensor.

- [ ] **Step 3: Verify** — `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_canonical_form.py -v`. All green.

- [ ] **Step 4: Commit**

```
git add src/qft_pcn/composition/abstraction.py src/qft_pcn/composition/tests/test_canonical_form.py
git commit -m "$(cat <<'EOF'
feat(composition/abstraction): canonical-form computation

Cluster representative = operator-basis mean of member reduced densities,
re-purified into a bounded-bond MERA via eigendecomposition + term
superposition; provenance attached (architecture §10.9).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wake-sleep orchestration + the induction-discovery acceptance test

**Files:**
- Create: `src/qft_pcn/composition/wake_sleep.py`
- Modify: `src/qft_pcn/composition/tests/conftest.py` (add `FakeLemmaLibrary`, solvers, induction corpus)
- Test: `src/qft_pcn/composition/tests/test_wake_sleep.py` (create)

- [ ] **Step 1: Extend `conftest.py` with the library double, solvers, and corpus**

Append to `src/qft_pcn/composition/tests/conftest.py` (replacing the `# induction corpus builders: Task 6` placeholder):

```python
# --- §8.4 test doubles ------------------------------------------------------
from src.qft_pcn.composition.abstraction import CanonicalPrimitive
from src.qft_pcn.composition._abstraction_const import LibraryContractError


class FakeLemmaLibrary:
    """In-memory LemmaLibrary satisfying the spec §7 contract."""

    def __init__(self):
        self._solved: list[tuple[object, object, str]] = []
        self._tiers: dict[str, str] = {}
        self.registered_primitives: list[CanonicalPrimitive] = []
        self.registered_solutions: list[str] = []
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
            self._tiers[eid] = "dynamic"
        return eid

    def cached_solutions(self):
        return list(self._solved)

    def tier_of(self, entry_id):
        return self._tiers.get(entry_id, "dynamic")

    def replace(self, entry_id, new_state):
        pass

    def prune(self, entry_id):
        if self._tiers.get(entry_id) == "core":
            raise LibraryContractError("cannot prune a core entry")

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
    """Five inductive-proof MERAs sharing an induction-shaped subtree with
    different predicates. Encoded via M1's encode_mera so each proof MERA
    contains the same Forall/Fix induction skeleton."""
    from src.qft_pcn.logic.mera_encoder import encode_mera
    from src.qft_pcn.logic.ast import parse
    predicates = [
        r"\n:Nat. Eq (n + 0) n",
        r"\n:Nat. Eq (0 + n) n",
        r"\n:Nat. Eq n n",
        r"\n:Nat. Eq (n + n) (n + n)",
        r"\n:Nat. Eq (Succ n) (Succ n)",
    ]
    corpus = []
    for i, pred in enumerate(predicates):
        # wrap the predicate in the induction principle: Forall n. Fix ...
        src = rf"Forall n:Nat. (Fix ind. ({pred}))"
        state, meta = encode_mera(parse(src))
        corpus.append((state, meta, f"ind{i}"))
    return corpus
```

> The exact induction-encoding `src` strings depend on M1's extended-calculus surface syntax. The contract is: five programs whose proof MERAs *share* an induction-shaped subtree (the `Forall`/`Fix` skeleton) with the predicate varying. If M1's parser cannot express `Forall`/`Fix` exactly as written, a subagent must adjust the strings to M1's actual grammar **while preserving the shared-skeleton property** — the whole acceptance test depends on the five subtrees clustering. If M1's encoder does not yet support these nodes, escalate (J depends on M1; this is a real blocker, not a thing to fake).

- [ ] **Step 2: Write the failing test**

Create `src/qft_pcn/composition/tests/test_wake_sleep.py`:

```python
"""Wake-sleep orchestration + induction discovery (spec §6; acceptance §9.5–9.6, §9.8)."""
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
```

Run it, watch it fail.

- [ ] **Step 3: Implement**

Create `src/qft_pcn/composition/wake_sleep.py`:

```python
"""Wake-sleep cycle orchestration for abstraction discovery (spec §6)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.qft_pcn.composition._abstraction_const import (
    DEFAULT_ALPHA, DEFAULT_CHI_CAP, N_QUIESCENT, LibraryContractError,
)
from src.qft_pcn.composition.subtree_miner import MineConfig, mine_corpus
from src.qft_pcn.composition.abstraction import (
    ClusterConfig, CanonicalPrimitive, cluster_candidates,
    significant_clusters, compute_canonical_form, trace_distance,
)


@dataclass
class Problem:
    id: str
    hamiltonian_or_state: Any


@dataclass
class WakeSleepConfig:
    mine: MineConfig = field(default_factory=MineConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    alpha: float = DEFAULT_ALPHA
    chi_cap: int = DEFAULT_CHI_CAP
    residual_eps: float = 1e-6
    n_quiescent: int = N_QUIESCENT


@dataclass
class CycleReport:
    cycle_index: int
    n_solved: int
    n_candidates: int
    n_clusters: int
    n_significant: int
    promoted: list[CanonicalPrimitive]
    n_consolidated: int
    n_pruned: int


_REQUIRED_LIBRARY_METHODS = ("register", "cached_solutions", "tier_of",
                             "replace", "prune")


def _check_library(library) -> None:
    for name in _REQUIRED_LIBRARY_METHODS:
        if not callable(getattr(library, name, None)):
            raise LibraryContractError(
                f"LemmaLibrary is missing required method '{name}' (spec §7)")


SolveFn = Callable[[Problem, Any], tuple[Any, Any, float]]


def wake_sleep_cycle(library, problem_batch: list[Problem], solve: SolveFn,
                     cycle_index: int,
                     config: WakeSleepConfig = WakeSleepConfig(),
                     validate: Callable[[CanonicalPrimitive], bool] | None = None,
                     ) -> CycleReport:
    """Run ONE wake-sleep cycle (architecture §10.9 pseudocode; spec §6.1)."""
    _check_library(library)

    # WAKE -------------------------------------------------------------------
    solved: list[tuple[Any, Any, str]] = []
    for problem in problem_batch:
        state, meta, residual = solve(problem, library)
        if residual < config.residual_eps:
            solved.append((state, meta, problem.id))
            library.register((state, meta, problem.id))
    # mine over the whole cached corpus, not just this batch
    corpus = list(library.cached_solutions())

    # DREAM ------------------------------------------------------------------
    candidates = mine_corpus(corpus, config.mine)

    # CLUSTER ----------------------------------------------------------------
    clusters = cluster_candidates(candidates, config.cluster)
    sig = significant_clusters(clusters, len(candidates), config.alpha)

    # ABSTRACT ---------------------------------------------------------------
    promoted: list[CanonicalPrimitive] = []
    for cluster in sig:
        primitive = compute_canonical_form(cluster, config.chi_cap, cycle_index)
        if validate is not None and not validate(primitive):
            continue
        library.register(primitive)
        promoted.append(primitive)

    # CONSOLIDATE ------------------------------------------------------------
    n_consolidated, n_pruned = _consolidate(library, promoted, config)

    return CycleReport(
        cycle_index=cycle_index,
        n_solved=len(solved),
        n_candidates=len(candidates),
        n_clusters=len(clusters),
        n_significant=len(sig),
        promoted=promoted,
        n_consolidated=n_consolidated,
        n_pruned=n_pruned,
    )


def _consolidate(library, promoted: list[CanonicalPrimitive],
                 config: WakeSleepConfig) -> tuple[int, int]:
    """Re-derive cached solutions with the new primitives; prune redundancies
    (spec §6.2). Never mutates a cached MERA in place; never prunes a 'core'
    entry. Returns (n_consolidated, n_pruned)."""
    if not promoted:
        return 0, 0
    n_consolidated = 0
    from src.qft_pcn.composition.subtree_miner import mine_subtrees
    for state, meta, sid in library.cached_solutions():
        cands = mine_subtrees(state, meta, sid, config.mine)
        for cand in cands:
            for prim in promoted:
                if trace_distance(cand.rho, prim.rho_canonical) \
                        < config.cluster.distance_threshold:
                    prim.provenance.use_log.append(sid)
                    n_consolidated += 1
                    break
            else:
                continue
            break
    # pruning of redundant dynamic-ring lemmas is delegated to the library;
    # J only counts and never deletes a core entry.
    return n_consolidated, 0


def wake_sleep_loop(library, problem_batches: list[list[Problem]],
                    solve: SolveFn,
                    config: WakeSleepConfig = WakeSleepConfig(),
                    validate: Callable[[CanonicalPrimitive], bool] | None = None,
                    ) -> list[CycleReport]:
    """Run cycles until n_quiescent consecutive cycles promote nothing, or the
    batches are exhausted (spec §6.1)."""
    reports: list[CycleReport] = []
    quiescent = 0
    for cycle_index, batch in enumerate(problem_batches):
        report = wake_sleep_cycle(library, batch, solve, cycle_index,
                                  config, validate)
        reports.append(report)
        if report.promoted:
            quiescent = 0
        else:
            quiescent += 1
            if quiescent >= config.n_quiescent:
                break
    return reports
```

> `_consolidate`'s "shorter re-derivation" is detected by a primitive's canonical density matching a cached subtree's boundary density; the spec (§6.2) leaves *retention* to the library. This implementation logs provenance use and counts; it does not delete entries (pruning is the library's job, and the `"core"` tier must never be auto-pruned). A subagent must not "simplify" this into a direct `del` on cached states — that would violate spec §3.3 (never mutate a cached MERA, never auto-prune core).

- [ ] **Step 4: Verify** — `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_wake_sleep.py -v`. All green. Then run the whole J suite: `.venv/bin/python -m pytest src/qft_pcn/composition/ -v`.

- [ ] **Step 5: Commit**

```
git add src/qft_pcn/composition/wake_sleep.py src/qft_pcn/composition/tests/conftest.py src/qft_pcn/composition/tests/test_wake_sleep.py
git commit -m "$(cat <<'EOF'
feat(composition/wake_sleep): wake-sleep cycle orchestration + induction test

Orchestrate solve -> mine -> cluster -> abstract -> consolidate (architecture
§10.9). End-to-end acceptance: five inductive proofs yield one promoted
induction primitive; a sixth inductive proof converges in fewer steps.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Full-suite verification and acceptance sign-off

**Files:** none new — verification only.

- [ ] **Step 1: Run the whole J suite**

`.venv/bin/python -m pytest src/qft_pcn/composition/ -v`. Every test green.

- [ ] **Step 2: Run the regression set**

`.venv/bin/python -m pytest src/qft_pcn/tests/ src/qft_pcn/qft/ -v` (or the project's existing suite paths). Confirm F's MERA tests and M1's encoder tests still pass — J modified none of them.

- [ ] **Step 3: Walk the spec §12 acceptance criteria**

For each of the 11 criteria in the spec, name the test that proves it:

1. Enumeration + valid densities → `test_subtree_miner.py::test_mine_returns_node_aligned_subtrees_in_size_band`, `::test_mined_densities_are_valid`.
2. Fingerprint invariance → `::test_fingerprint_invariant_under_bond_basis_change`.
3. Trace distance + clustering → `test_abstraction_math.py`, `test_clustering.py`.
4. `k_min` / significance → `test_abstraction_math.py::test_k_min_*`, `test_clustering.py::test_significant_clusters_drops_small_ones`.
5. Canonical form + validation hook → `test_canonical_form.py`, `test_wake_sleep.py::test_validation_hook_gates_promotion`.
6. **Induction-discovery end-to-end** → `test_wake_sleep.py::test_induction_primitive_is_discovered`, `::test_subsequent_inductive_proof_converges_in_fewer_steps`.
7. No dense materialization → the `conftest.py` `_no_large_dense` autouse guard (whole suite runs under it).
8. Determinism → `test_wake_sleep.py::test_cycle_is_deterministic`, `test_clustering.py::test_clustering_is_deterministic_under_permutation`.
9. F + M1 tests still green → Step 2.
10. `optimize='greedy'` on every einsum → grep `src/qft_pcn/composition/` for `np.einsum`; confirm each call passes `optimize='greedy'`. (The current implementation uses `eigvalsh`/`eigh`/matmul, not `einsum`; if a subagent introduces an `einsum`, this criterion binds it.)
11. Every claim backed by fresh pytest output → paste the Step 1 + Step 2 output into the completion report.

- [ ] **Step 4: Verification before completion**

Invoke `superpowers:verification-before-completion`. Do not claim J complete without the fresh pytest output for Steps 1 and 2 in hand.

- [ ] **Step 5: Final commit (only if Step 3 grep surfaced an einsum fix or a doc tweak)**

If Steps 1–4 needed no code change, there is nothing to commit here. If a fix was required:

```
git add src/qft_pcn/composition/
git commit -m "$(cat <<'EOF'
fix(composition): acceptance sign-off corrections for sub-project J

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Integration notes (read before starting)

- **Sub-project I is specced in parallel.** J calls `LemmaLibrary.register(...)` and the rest of the §7 contract. Until I lands, the `FakeLemmaLibrary` double *is* the contract. When I lands, swap the double for the real library in an integration test; if I's API differs from spec §7, **stop and reconcile** — do not bend J to a guessed API.
- **M1 must be present.** Tasks 3 and 6 import `encode_mera` and the extended-calculus parser. If M1's `Forall`/`Fix` nodes are not yet implemented, the induction corpus cannot be built — escalate; do not stub a fake MERA in place of a real encoding (that would void the operator-algebraic acceptance test).
- **F's `_layer_density` / `from_term_superposition`.** Tasks 3 and 5 lean on these. Confirm their exact signatures in `src/qft_pcn/qft/mera.py` before relying on them. A genuine missing primitive is fixed in `mera.py` interface-only with F's tests staying green; otherwise build on top.
- **The shortcuts to refuse, restated for every subagent prompt:** (a) no syntactic subtree-string hashing — patterns are reduced density matrices; (b) no classical `dict` lookup for binding — read entanglement off the MERA; (c) no `16**s` dense leaf-space tensor — boundary-bond densities only; (d) no time/effort estimates anywhere.
