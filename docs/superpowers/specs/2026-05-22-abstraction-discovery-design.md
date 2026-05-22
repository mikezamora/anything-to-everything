# Spec: Abstraction Discovery / Wake-Sleep (migration sub-project J)

**Document type**: Implementation specification.
**Date**: 2026-05-22.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Implements**: `QFT_PCN_ARCHITECTURE.md` §10.9 (abstraction discovery — hierarchical
composition mechanism 2). Related: §11.3 (Wilson RG — wake-sleep is one RG-flow step),
§11.7 (compounding-capability argument).
**Depends on**: sub-project I (lemma library — exposes `LemmaLibrary` with a
`register(...)` method), M1 (MERA-native encoder — `MERA`, `MeraEncodingMeta`),
F (MERA substrate — `src/qft_pcn/qft/mera.py`).
**Status**: Approved design; ready for implementation plan.

---

## 0. How to read this spec

This specifies the **wake-sleep abstraction-discovery subsystem**: the mechanism by
which, when a structural pattern recurs across many solved problems, the system
*automatically abstracts it into a new primitive operator* registered in the lemma
library. It is `QFT_PCN_ARCHITECTURE.md` §10.9 turned into concrete APIs, dataclasses,
and acceptance tests.

Everything here is the contract. If something needed during implementation is missing,
**stop and ask** — do not guess.

No time, duration, or effort estimates appear in this document. Sequencing and
dependency order are specified; calendar and effort are not. Standing project directive.

---

## 1. Driving principles (non-negotiable)

This subsystem is the operational realization of one step of Wilson RG flow on the
operator algebra of the problem domain (§11.3). The temptation to take an easier,
syntactically simpler shortcut is strong here; each shortcut below is **forbidden**,
and the principled alternative is the contract.

1. **No time / duration / effort estimates.** Critical standing directive. This
   document specifies *what* and *in what order*, never *how long*.

2. **Abstraction is operator-algebraic, never syntactic.**
   - *Shortcut (forbidden):* induce a grammar over AST node sequences — find recurring
     subtree *strings* and promote those, the way classical DreamCoder compresses
     program text.
   - *Principled (the contract):* a candidate pattern is the **reduced density matrix
     `ρ_S`** of a solved sub-MERA at its boundary bond. Patterns "recur" in the
     operator-algebra sense — two occurrences are *the same pattern* iff their reduced
     density matrices are close in trace distance, even if the surface AST differs.
     A discovered primitive is a **canonical operator** (a bounded-bond MERA /
     operator), composable with other primitives **by tensor product**, not by code
     substitution (§11.5: this is exactly where the QPCN differs from DreamCoder).

3. **Binding is genuine entanglement, never a classical lookup** (architecture §8.1,
   M1 spec §1.2). A mined sub-MERA carries its binding correlations *as tree
   entanglement*. The reduced density matrix at the boundary bond therefore *encodes*
   whether a binder inside the subtree is referenced from outside it — that nonzero
   boundary entanglement is a load-bearing feature of the fingerprint, not noise to be
   normalized away. The miner must never replace a sub-MERA's binding structure with a
   Python-side `dict[use, binder]`.

4. **No dense operator at scale.** Reduced density matrices live on a **boundary bond**
   whose dimension is the MERA bond dimension `χ` (≤ a few tens), *not* on the
   `16**s`-dimensional leaf space of the subtree. `ρ_S` is at most `(χ², χ²)` for a
   two-bond boundary and is the only object materialized. No sub-MERA's leaf space is
   ever densified. Re-purification (§5.3) produces a **bounded-bond MERA**, never a
   dense tensor.

5. **`optimize='greedy'` on every `np.einsum`.** Every contraction in this subsystem —
   boundary-bond density extraction, trace-distance eigenvalue computation, canonical
   averaging, re-purification — passes `optimize='greedy'`.

6. **Reuse, do not reinvent.** The MERA substrate (F, `mera.py`) provides
   `_layer_density`, `entanglement_entropy`, the causal-cone helpers, `from_product`,
   `from_term_superposition`, `inner`, `normalize`. The encoder (M1) provides `MERA`
   and `MeraEncodingMeta`. The lemma library (I) provides `LemmaLibrary.register(...)`.
   This subsystem *consumes* all of them. If a genuine bug or missing primitive is
   found in `mera.py`, fix it there; otherwise build on top.

---

## 2. Where J sits in the migration

| Sub-project | Scope | Depends on |
|---|---|---|
| F | MERA substrate | — |
| M1 | MERA-native logic encoder | F |
| I | Lemma library + promotion (§10.8) | M1 |
| **J** (this spec) | **Abstraction discovery / wake-sleep (§10.9)** | **I, M1, F** |
| §10.10 | Cross-level message passing | I, J |

J is the **second** hierarchical-composition mechanism. I caches solved sub-solutions
as addressable lemmas; J *mines* those cached solutions for recurring structure and
promotes the recurring structure to new **primitive operators** in the same library.
I makes reuse possible; J makes the vocabulary *grow*.

In the Wilson RG picture (§11.3): I is the library at scale `H_k`; one full J
wake-sleep cycle is the block-spin step that produces `H_{k+1}`, the effective
Hamiltonian whose primitives are composites of the scale below.

---

## 3. Scope

### 3.1 In scope

- **Subtree mining** (`subtree_miner.py`): enumerate sub-MERAs of every solved cached
  state for subtree sizes `s ∈ [S_MIN, S_MAX]`; compute the **reduced density matrix
  at each subtree's boundary bond**; assign each a **fingerprint hash** (trace + sorted
  top-`r` eigenvalues, rounded) so that obviously-different candidates are never
  pairwise-compared.
- **Abstraction** (`abstraction.py`): hierarchical agglomerative clustering of mined
  reduced density matrices under the **trace distance** `D(ρ₁,ρ₂)=½‖ρ₁−ρ₂‖₁`; a
  **significance test** `k_min(N)` rejecting clusters too small to be non-accidental;
  **canonical-form computation** (the cluster representative = operator minimizing
  average trace distance, computed by operator-basis averaging then re-purification to
  a bounded-bond MERA); **provenance** tracking.
- **Wake-sleep orchestration** (`wake_sleep.py`): one cycle = wake (solve a batch with
  the current library) → dream (mine substructures) → cluster → abstract (promote
  significant clusters to library primitives) → consolidate (re-derive cached solutions
  with the new primitives; prune redundant lemmas). A loop wrapper running cycles
  until no new primitive is discovered for `N_QUIESCENT` consecutive cycles.
- An acceptance test suite (§9): subtree-miner correctness, clustering / significance
  correctness, canonical-form correctness, and the **end-to-end induction-discovery
  acceptance test** (§9.6 — architecture §10.9's acceptance test).

### 3.2 Out of scope (other sub-projects / own specs)

- The lemma library itself, its storage, its promotion of *whole* solved problems
  (§10.8) — that is sub-project I. J calls `LemmaLibrary.register(...)`; it does not
  implement the library.
- The QPCN solver / `relax_to_ground` invoked in the wake phase — J calls an injected
  `solve` callable; it does not implement relaxation. The acceptance suite supplies a
  deterministic stub solver (§8.4).
- The LLM frontend (`spec = LLM(P)`) — the wake phase receives already-compiled
  Hamiltonians or already-encoded MERA states; natural-language → DSL is §10.5.
- Cross-level message passing / goal graphs (§10.10) — a later sub-project that
  *consumes* J's grown library.
- Cross-domain held-out validation of promoted primitives (§10.9 risk table row 1):
  the validation *hook* is specified (§6.3 `validate` callable) but a full held-out
  benchmark corpus is out of scope; the hook defaults to accept-all and the acceptance
  suite exercises a reject path with a stub.

### 3.3 Will not do

- Materialize any sub-MERA's `16**s` leaf space densely (principle 4).
- Grammar / syntactic-string subtree induction (principle 2).
- Replace a sub-MERA's binding entanglement with a classical lookup (principle 3).
- Modify the lemma library's internals, the encoder, or the MERA substrate (except a
  genuine bug fix in `mera.py`, interface-only, with F's tests still green).
- Mutate cached solved states in place — consolidation produces *new* states and the
  library decides what to keep; J never overwrites a cached `MERA`.

---

## 4. Subtree mining

### 4.1 What a "subtree" is on the MERA substrate

A cached solved problem is a `(MERA, MeraEncodingMeta)` pair from M1. The MERA is a
single binary tree over `N_leaves = 2^L` species-leaves (M1 §4). A **subtree** is an
**internal node of the MERA tree** — equivalently, a contiguous leaf interval
`[leaf0, leaf0+w)` with `w = 2^d` aligned to a `d`-th-layer block. The subtree's
**size in AST nodes** is `s = (count of non-PAD leaves in the interval) / 5` (five
species-leaves per AST node, M1 §4.3); a subtree is *mined* only when its leaf interval
covers a whole number of AST nodes (no node is split across the boundary) and
`S_MIN ≤ s ≤ S_MAX`. Splitting a node would produce a meaningless mid-node boundary
bond; the miner rejects such intervals (it does not crash — see §10).

`S_MIN = 3` and `S_MAX = 8` are the defaults (architecture §10.9: "very small or very
large subtrees are usually not useful"); both are parameters of `MineConfig`.

### 4.2 The boundary reduced density matrix

For a subtree occupying leaf interval `I`, the **boundary** is the MERA tree edge(s)
connecting `I`'s root tensor to the rest of the tree. The **reduced density matrix
`ρ_S`** is the density matrix of the subtree's degrees of freedom traced against the
complement — obtained as the reduced density of the cached state on the *bond* between
`I`'s subtree-root and its parent isometry.

Concretely `ρ_S` is computed by ascending the cached MERA's per-site density matrices
(F's `MERA._layer_density`) up to the layer `d` at which `I` is a single block, and
reading the reduced density at that block's index. `ρ_S` has shape `(χ_d, χ_d)` where
`χ_d` is the MERA bond dimension entering layer `d` — bounded by `χ_layer` (M1
default 16), so `ρ_S` is at most `(16, 16)`. It is Hermitian, positive-semidefinite,
unit-trace; the miner asserts these to tolerance `1e-9` and raises `MiningError` on
violation (a violation means the cached state was not a valid normalized MERA).

**Why the boundary bond and not the leaf space.** The reduced density on the full
`16**s`-dimensional leaf space of the subtree carries the same spectral information
(the subtree's Schmidt spectrum across its boundary) but would be astronomically large
for `s = 8`. The boundary-bond density `ρ_S` is the *coarse-grained* representation the
MERA already computes — and it is exactly the object whose recurrence signals a shared
substructure (architecture §10.9: "that substructure has a recognizable signature in
the reduced density matrix at the bond where the substructure ends"). Using the bond
density is not a shortcut; it is the principled, RG-correct object.

### 4.3 The fingerprint hash

Pairwise trace distance over all mined candidates is `O(M²)` in the candidate count
`M`. To avoid comparing obviously-different candidates, each `ρ_S` gets a
**fingerprint**: a tuple of

- `round(tr(ρ_S²), FP_DECIMALS)` — the purity (a basis-independent scalar),
- the sorted top-`FP_RANK` eigenvalues of `ρ_S`, each `round(·, FP_DECIMALS)`,
- the AST-node size `s`.

`FP_DECIMALS = 3`, `FP_RANK = 4` defaults (both in `MineConfig`). Candidates are
**bucketed by fingerprint**; trace distance is computed only within a bucket *and* its
fingerprint-adjacent neighbours (buckets whose rounded scalars differ by ≤ 1 ulp of
the rounding grid — see §5.2). The fingerprint is **not** the cluster key — it is a
*pre-filter*; final cluster membership is decided by trace distance (§5). Two reduced
density matrices that are unitarily equivalent have identical eigenvalues and purity,
so the fingerprint is invariant under the basis ambiguity of the boundary bond — which
is exactly the invariance abstraction discovery needs.

### 4.4 API

```python
# src/qft_pcn/composition/subtree_miner.py

@dataclass(frozen=True)
class MineConfig:
    s_min: int = 3
    s_max: int = 8
    fp_decimals: int = 3
    fp_rank: int = 4

@dataclass(frozen=True)
class Fingerprint:
    purity: float                  # round(tr(rho^2), fp_decimals)
    top_eigs: tuple[float, ...]    # sorted desc, len == fp_rank, rounded
    ast_size: int                  # subtree size in AST nodes

@dataclass
class SubtreeCandidate:
    source_id: str                 # cached-problem identifier (provenance)
    leaf_interval: tuple[int, int] # [leaf0, leaf0 + w)
    ast_size: int                  # s
    rho: np.ndarray                # boundary reduced density matrix, (chi, chi)
    fingerprint: Fingerprint

def mine_subtrees(state: "MERA", meta: "MeraEncodingMeta", source_id: str,
                  config: MineConfig = MineConfig()) -> list[SubtreeCandidate]:
    """Enumerate all node-aligned sub-MERAs of `state` with AST size in
    [s_min, s_max]; for each compute the boundary reduced density matrix and
    its fingerprint. Pure: does not mutate `state`.

    Raises MiningError if `state` is not a valid normalized MERA (a reduced
    density fails the Hermitian / PSD / unit-trace check)."""

def mine_corpus(solved: list[tuple["MERA", "MeraEncodingMeta", str]],
                config: MineConfig = MineConfig()) -> list[SubtreeCandidate]:
    """mine_subtrees over a whole corpus, concatenating results."""

def bucket_by_fingerprint(
        candidates: list[SubtreeCandidate]
) -> dict[Fingerprint, list[SubtreeCandidate]]:
    """Group candidates by exact fingerprint. The clustering layer compares
    within a bucket and its fingerprint-adjacent neighbours only."""
```

---

## 5. Abstraction

### 5.1 Trace distance

```python
# src/qft_pcn/composition/abstraction.py

def trace_distance(rho1: np.ndarray, rho2: np.ndarray) -> float:
    """D(rho1, rho2) = 0.5 * sum(|eigvals(rho1 - rho2)|).

    rho1, rho2 are Hermitian; if their dimensions differ they are zero-padded
    (embedded in the larger boundary-bond space) before subtraction — a
    smaller-boundary subtree is the larger one with the extra bond in a pure
    reference state, so zero-padding is the correct embedding. Returns a float
    in [0, 1]."""
```

`D` is computed from the eigenvalues of the Hermitian difference `ρ₁−ρ₂` (`np.linalg.
eigvalsh`), **not** from a dense `‖·‖₁` SVD path — `eigvalsh` is the correct and cheap
route for Hermitian operators. `D = 0` iff the matrices are equal; `D = 1` iff
orthogonal supports.

### 5.2 Hierarchical agglomerative clustering

```python
@dataclass(frozen=True)
class ClusterConfig:
    distance_threshold: float = 0.15   # merge-stop trace distance
    linkage: str = "average"           # "average" | "complete" | "single"

@dataclass
class Cluster:
    members: list[SubtreeCandidate]
    @property
    def size(self) -> int: ...

def cluster_candidates(candidates: list[SubtreeCandidate],
                       config: ClusterConfig = ClusterConfig()) -> list[Cluster]:
    """Hierarchical agglomerative clustering of candidates under trace_distance.
    Bottom-up: start with singletons, repeatedly merge the two clusters whose
    inter-cluster distance (per `linkage`) is minimal, stop when the minimal
    distance exceeds `distance_threshold`. Distances are computed only between
    candidates sharing a fingerprint bucket or fingerprint-adjacent buckets
    (§4.3); cross-bucket pairs are treated as distance 1.0 (never merged).
    Deterministic: ties broken by (source_id, leaf_interval) lexicographic
    order."""
```

Agglomerative clustering is implemented directly (it is a few dozen lines over a
distance matrix); no SciPy dependency is introduced. The distance matrix is the only
`O(M²)`-shaped object and `M` is bounded by the fingerprint pre-filter — within a
bucket, candidate counts are small.

### 5.3 Significance test

```python
def k_min(n_candidates: int, alpha: float = 1e-3) -> int:
    """Minimum cluster size for significance. Architecture §10.9: a cluster of
    size k from N candidates arises by chance with probability ~ exp(-k log N)
    under a uniform null model. Solve exp(-k log N) <= alpha for the smallest
    integer k:  k_min = ceil( -log(alpha) / log(max(N, 2)) ).
    Floored at 2 (a 'cluster' of one is not a recurrence)."""

def significant_clusters(clusters: list[Cluster], n_candidates: int,
                         alpha: float = 1e-3) -> list[Cluster]:
    """Return clusters with size >= k_min(n_candidates, alpha)."""
```

This is the Bayesian-evidence threshold of architecture §10.9 ("Promotion threshold
`k_min` is the Bayesian evidence threshold"). A cluster smaller than `k_min` is
discarded: a pattern seen too few times is not yet evidence of genuine domain
structure (§10.9 risk table: "Bad abstractions").

### 5.4 Canonical-form computation

The cluster representative is the operator minimizing average trace distance to all
members. Per architecture §10.9 it is computed by **operator-basis averaging then
re-purification**:

```python
@dataclass
class CanonicalPrimitive:
    rho_canonical: np.ndarray        # the averaged reduced density matrix
    mera: "MERA"                     # bounded-bond MERA re-purification of rho_canonical
    chi: int                         # the re-purification bond dimension
    avg_trace_distance: float        # mean D(rho_canonical, member.rho)
    provenance: "Provenance"

def compute_canonical_form(cluster: Cluster,
                           chi_cap: int = 16) -> CanonicalPrimitive:
    """1. rho_canonical = (1/|C|) * sum(member.rho for member in cluster),
       members embedded into the common (largest) boundary-bond dimension by
       zero-padding (§5.1). The arithmetic mean in the operator basis is the
       unique minimizer of average *squared* Hilbert-Schmidt distance and is
       the architecture-specified representative; it is Hermitian, PSD,
       unit-trace by linearity.
    2. Re-purify rho_canonical into a bounded-bond MERA: eigendecompose
       rho_canonical = sum_j p_j |e_j><e_j|; keep the top `chi_cap` eigenpairs;
       build a MERA whose boundary bond carries the mixed state sum_j p_j ...
       via MERA.from_term_superposition over the kept eigenvectors (a purifying
       term superposition). The result is a bounded-bond MERA, never a dense
       tensor (principle 4).
    3. avg_trace_distance = mean(trace_distance(rho_canonical, m.rho)).
    Returns the CanonicalPrimitive."""
```

The re-purification keeps at most `chi_cap` eigenpairs; truncated tail mass is recorded
and, if it exceeds `1e-6`, a `RepurificationWarning` is emitted (the canonical operator
is then an approximation — still registrable, but the warning is surfaced to the
consolidation phase).

### 5.5 Provenance

```python
@dataclass
class Provenance:
    source_ids: tuple[str, ...]            # distinct cached problems in the cluster
    occurrences: tuple[tuple[str, tuple[int, int]], ...]  # (source_id, leaf_interval)
    discovered_in_cycle: int               # wake-sleep cycle index
    use_log: list[str] = field(default_factory=list)      # source_ids that later use it
```

Each promoted primitive remembers the problems it was abstracted from
(`source_ids`), the exact occurrences (`occurrences`), and the cycle index. When a
primitive is later *used* (consolidation re-derives a cached solution with it), the
using problem's id is appended to `use_log` — giving each original problem "credit"
(architecture §10.9 "Provenance"). Provenance is what makes pruning safe (drop
primitives with an empty `use_log` after `N_QUIESCENT` cycles) and what makes a primitive
non-opaque (its decomposition into the subtrees it came from is always recoverable).

---

## 6. Wake-sleep orchestration

### 6.1 The cycle

```python
# src/qft_pcn/composition/wake_sleep.py

@dataclass
class WakeSleepConfig:
    mine: MineConfig = MineConfig()
    cluster: ClusterConfig = ClusterConfig()
    alpha: float = 1e-3
    chi_cap: int = 16
    residual_eps: float = 1e-6      # wake-phase acceptance threshold
    n_quiescent: int = 2            # cycles with no discovery before stop

@dataclass
class CycleReport:
    cycle_index: int
    n_solved: int                   # problems solved in the wake phase
    n_candidates: int               # mined subtree candidates
    n_clusters: int
    n_significant: int
    promoted: list[CanonicalPrimitive]
    n_consolidated: int             # cached solutions shortened
    n_pruned: int                   # lemmas dropped
```

```python
def wake_sleep_cycle(library: "LemmaLibrary",
                     problem_batch: list["Problem"],
                     solve: Callable[["Problem", "LemmaLibrary"],
                                     tuple["MERA", "MeraEncodingMeta", float]],
                     cycle_index: int,
                     config: WakeSleepConfig = WakeSleepConfig(),
                     validate: Callable[["CanonicalPrimitive"], bool] | None = None,
                     ) -> CycleReport:
    """Run ONE wake-sleep cycle (architecture §10.9 pseudocode):

    WAKE       — for each problem in problem_batch: (state, meta, residual) =
                 solve(problem, library); if residual < config.residual_eps,
                 the (state, meta, problem.id) triple joins the solved set
                 (and is registered into `library` via library.register(...)).
    DREAM      — mine_corpus over the solved set -> candidates.
    CLUSTER    — cluster_candidates -> clusters; significant_clusters filters.
    ABSTRACT   — for each significant cluster: compute_canonical_form; if
                 `validate` is not None and validate(primitive) is False, skip
                 it (held-out-validation hook, §10.9 risk row 1); else
                 library.register(primitive) and record it in `promoted`.
    CONSOLIDATE— for each cached solution, attempt re-derivation with the new
                 primitives (see §6.2); if a re-derivation is a strictly
                 smaller-bond MERA, replace the cached entry; prune lemmas that
                 are now exact consequences of more primitive ones.
    Returns the CycleReport."""

def wake_sleep_loop(library: "LemmaLibrary",
                    problem_batches: list[list["Problem"]],
                    solve: Callable[..., tuple["MERA", "MeraEncodingMeta", float]],
                    config: WakeSleepConfig = WakeSleepConfig(),
                    validate: Callable[["CanonicalPrimitive"], bool] | None = None,
                    ) -> list[CycleReport]:
    """Run cycles over successive batches until `config.n_quiescent`
    consecutive cycles promote zero primitives, or batches are exhausted.
    Returns the list of per-cycle CycleReports."""
```

`Problem` is a lightweight dataclass `Problem(id: str, hamiltonian_or_state: Any)` —
the wake phase passes it opaquely to the injected `solve`. J does not define how a
problem is solved; that is the QPCN solver's job.

### 6.2 Consolidation

Consolidation re-derives each cached solution against the freshly-grown library:

- For a cached solved state, check whether one of the new primitives' canonical
  reduced density matrix matches (trace distance < `config.cluster.distance_threshold`)
  the boundary density at one of the cached state's subtrees. If so, the subtree can be
  *replaced by a reference to the primitive*: the re-derived solution clamps that
  subtree to the primitive's canonical MERA and is structurally shorter.
- "Shorter" is measured by `MERA.bond_dimensions()` — a re-derivation is kept only if
  its maximum bond dimension is strictly smaller (or equal with fewer non-trivial
  tensors). The library decides retention; J reports the count in `CycleReport`.
- A cached lemma that becomes an *exact* consequence of more primitive lemmas
  (re-derivable with residual below `residual_eps` using only primitives) is pruned —
  but only from the dynamic outer ring of the two-tier library (architecture §10.9
  risk table "Catastrophic compression": a stable core is never auto-pruned). J asks
  the library which tier an entry is in via the contract in §7; it never deletes a
  core entry.

Consolidation never mutates a cached `MERA` in place — it produces a new state and
hands it to `library.register(...)` / the library's replace API; the library owns
retention.

### 6.3 The validation hook

`validate` defaults to `None` (accept every significant cluster). When supplied, it is
the cross-domain held-out check of architecture §10.9's first risk row: a candidate
primitive is registered only if it also helps on problems it was not abstracted from.
The full held-out corpus is out of scope (§3.2); the hook and a reject path are
specified and tested with a stub (§9.5).

---

## 7. Contract J requires from sub-project I

J depends on `LemmaLibrary` exposing at least:

- `register(entry)` — accepts either a solved `(MERA, MeraEncodingMeta, id)` triple
  (wake-phase caching) **or** a `CanonicalPrimitive` (abstract-phase promotion). The
  library distinguishes them by type. Returns a stable entry id.
- `cached_solutions() -> list[tuple[MERA, MeraEncodingMeta, str]]` — the solved states
  available for mining and consolidation.
- `tier_of(entry_id) -> str` — `"core"` or `"dynamic"`; J prunes only `"dynamic"`.
- `replace(entry_id, new_state)` / `prune(entry_id)` — consolidation operations; J
  calls these, the library enforces the two-tier policy.

If sub-project I's final API differs, this section is the integration point to
reconcile — **stop and ask** rather than guessing the library's shape. The acceptance
suite uses an in-memory `FakeLemmaLibrary` implementing exactly this contract (§8.4),
so J's tests do not block on I landing.

---

## 8. Constants, configs, dependencies, test doubles

### 8.1 Module-level constants

`src/qft_pcn/composition/_abstraction_const.py` (new) defines: `S_MIN = 3`,
`S_MAX = 8`, `FP_DECIMALS = 3`, `FP_RANK = 4`, `DEFAULT_DISTANCE_THRESHOLD = 0.15`,
`DEFAULT_ALPHA = 1e-3`, `DEFAULT_CHI_CAP = 16`, `N_QUIESCENT = 2`,
`REPURIFICATION_TAIL_TOL = 1e-6`, `DENSITY_HERMITICITY_TOL = 1e-9`. The dataclass
defaults in §4–§6 reference these.

### 8.2 Numerical tolerances

PSD / Hermitian / unit-trace checks on mined `ρ_S` use `DENSITY_HERMITICITY_TOL`
(`1e-9`). Trace distance and clustering compare to `distance_threshold` directly.
Re-purification tail mass is checked against `REPURIFICATION_TAIL_TOL` (`1e-6`).

### 8.3 External dependencies

NumPy only. No SciPy, no scikit-learn — agglomerative clustering, `eigvalsh`-based
trace distance, and the significance arithmetic are implemented in-module. This keeps
the subsystem inside the project's existing dependency surface.

### 8.4 Test doubles (in the test tree, not shipped)

- `FakeLemmaLibrary` — in-memory dict implementing the §7 contract; records every
  `register` call so tests can assert promotions.
- `stub_solver` — a deterministic `solve` callable: given a `Problem` whose payload is
  a pre-built `(MERA, MeraEncodingMeta)`, returns it with residual `0.0`; given a
  payload marked unsolvable, returns residual `1.0`. No actual relaxation.
- The induction corpus (§9.6) — five hand-built MERA states (via M1's `encode_mera`)
  that all share an induction-shaped subtree with different inductive predicates.

---

## 9. Acceptance tests

`src/qft_pcn/composition/tests/test_subtree_miner.py` and
`src/qft_pcn/composition/tests/test_wake_sleep.py`.

### 9.1 Subtree mining — enumeration and density validity

For a hand-built solved MERA, `mine_subtrees` returns exactly the node-aligned
sub-MERAs with AST size in `[S_MIN, S_MAX]`; every returned `ρ_S` is Hermitian, PSD,
and unit-trace to `1e-9`; node-splitting intervals are absent from the result.

### 9.2 Fingerprint invariance

Two reduced density matrices related by a boundary-bond unitary (same physical
subtree, different bond basis) produce **identical** fingerprints; their trace distance
is `0` to `1e-9`. A subtree with a genuinely different boundary spectrum produces a
different fingerprint.

### 9.3 Trace distance and clustering correctness

- `trace_distance(ρ, ρ) == 0`; `trace_distance` of two orthogonal-support pure states
  `== 1`; agrees with a direct `½‖ρ₁−ρ₂‖₁` SVD computation to `1e-10` on random
  Hermitian unit-trace inputs.
- `cluster_candidates` on a set with two well-separated groups (intra-group distance
  `< 0.05`, inter-group `> 0.5`) returns exactly two clusters; the result is
  deterministic across runs and across input permutations.

### 9.4 Significance test

`k_min(N)` is non-decreasing-then-saturating in the spec'd direction: `k_min` is
floored at 2, and for `N` large enough `k_min` rejects a cluster of size 2 while
`exp(-k_min·log N) ≤ alpha` holds. `significant_clusters` drops sub-`k_min` clusters
and keeps the rest.

### 9.5 Canonical form and validation hook

- `compute_canonical_form` on a cluster of `m` copies of the *same* `ρ` returns
  `rho_canonical == ρ` (to `1e-9`) and `avg_trace_distance == 0`; its `mera` is a valid
  normalized bounded-bond MERA whose boundary density re-extracts to `ρ` within
  `1e-6`.
- On a cluster of distinct members, `rho_canonical` equals the arithmetic mean and
  `avg_trace_distance` equals the mean trace distance to that mean.
- With a `validate` stub returning `False`, the abstract phase registers **no**
  primitive even when a significant cluster exists; with `True`, it registers it.

### 9.6 End-to-end induction-discovery acceptance test (architecture §10.9)

A corpus of **five** problems that all use induction on naturals with *different*
inductive predicates (e.g. `P(n) = n+0 = n`, `P(n) = 0+n = n`, `P(n) = n ≤ n`,
`P(n) = double(n) = n+n`, `P(n) = Eq (Succ n) (Succ n)`). Each is encoded with M1 so
its proof MERA contains the **same induction-shaped subtree** (the `Forall`/`Fix`
binder structure of the induction principle) with the predicate varying.

The test runs `wake_sleep_loop` (or one `wake_sleep_cycle`) with `stub_solver` and a
`FakeLemmaLibrary`, and asserts:

1. The dream phase mines the induction-shaped subtree from **all five** problems.
2. Those five reduced density matrices land in **one** cluster (intra-cluster trace
   distance below `distance_threshold` — the predicate differs but the boundary
   density of the induction skeleton is shared structure).
3. The cluster passes the significance test (`5 ≥ k_min(N)` for the corpus's `N`).
4. The abstract phase promotes exactly **one** new primitive — the "induction
   primitive" — and `FakeLemmaLibrary` records exactly one `CanonicalPrimitive`
   `register` call; its `provenance.source_ids` is the five problem ids.
5. **Convergence in fewer steps:** a sixth inductive problem, solved by a
   `step_counting_solver` that returns a step count (number of relaxation iterations),
   takes **strictly fewer steps when the library contains the promoted induction
   primitive** than the same problem solved against the library without it. The
   `step_counting_solver` is a deterministic stub whose step count is a fixed function
   of "is the induction primitive available" — the test asserts the inequality, which
   is the architecture §10.9 acceptance criterion ("subsequent inductive proofs should
   converge in substantially fewer steps").

### 9.7 No dense materialization

A `conftest.py` memory-ceiling guard (mirroring M1 §12 criterion 10) confirms no test
in this suite materializes a tensor larger than the boundary-bond scale
(`χ_cap² = 256`). The `16**s` leaf space of a mined subtree is never densified.

### 9.8 Determinism

`wake_sleep_cycle` on a fixed corpus with fixed configs produces an identical
`CycleReport` (same promoted primitives, same counts) across repeated runs and across
input permutations — abstraction discovery is a pure function of the solved corpus.

---

## 10. Error model

New exception family in `src/qft_pcn/composition/_abstraction_const.py`
(`AbstractionError` base):

- `MiningError` — a cached state is not a valid normalized MERA (reduced density fails
  the Hermitian / PSD / unit-trace check), or a configured interval is malformed.
- `RepurificationWarning` (a `Warning`, not an error) — canonical re-purification
  truncated tail mass above `REPURIFICATION_TAIL_TOL`.
- `LibraryContractError` — the injected `LemmaLibrary` does not satisfy the §7 contract
  (missing a required method) — raised early, with a message naming the missing method.

Node-splitting leaf intervals are **not** errors — the miner silently skips them
(they are simply not valid subtrees). The miner is total on valid normalized MERA
inputs; it raises rather than returning a malformed candidate.

---

## 11. File layout

```
src/qft_pcn/composition/
├── __init__.py                 # new package
├── _abstraction_const.py       # constants + exception family
├── subtree_miner.py            # MineConfig, Fingerprint, SubtreeCandidate,
│                               #   mine_subtrees, mine_corpus, bucket_by_fingerprint
├── abstraction.py              # trace_distance, ClusterConfig, Cluster,
│                               #   cluster_candidates, k_min, significant_clusters,
│                               #   CanonicalPrimitive, compute_canonical_form,
│                               #   Provenance
├── wake_sleep.py               # Problem, WakeSleepConfig, CycleReport,
│                               #   wake_sleep_cycle, wake_sleep_loop
└── tests/
    ├── __init__.py
    ├── conftest.py             # memory-ceiling guard; FakeLemmaLibrary;
    │                           #   stub_solver; induction-corpus builders
    ├── test_subtree_miner.py   # §9.1–9.2, §9.7
    └── test_wake_sleep.py      # §9.3–9.6, §9.8
```

All files are new. `mera.py`, `mps.py`, the encoder, and the lemma library are not
modified — except `mera.py` may gain a thin public wrapper if `mine_subtrees` needs an
internal density helper exposed (interface-only, no behavior change, F's MERA tests
must still pass).

---

## 12. Acceptance criteria

J is complete when:

1. `mine_subtrees` enumerates exactly the node-aligned sub-MERAs in `[S_MIN, S_MAX]`
   and every mined `ρ_S` is a valid density matrix (§9.1).
2. Fingerprints are invariant under boundary-bond basis change and discriminate
   genuinely different subtrees (§9.2).
3. `trace_distance` matches the direct `½‖·‖₁` definition; `cluster_candidates` is
   correct and deterministic (§9.3).
4. `k_min` / `significant_clusters` implement the §5.3 Bayesian-evidence threshold
   (§9.4).
5. `compute_canonical_form` returns the operator-basis mean re-purified to a
   bounded-bond MERA, with correct `avg_trace_distance`; the validation hook gates
   promotion (§9.5).
6. **The end-to-end induction-discovery acceptance test passes** — five inductive
   problems yield one cluster, one promoted induction primitive with correct
   provenance, and a sixth inductive problem converges in strictly fewer steps with the
   primitive available (§9.6). This is architecture §10.9's acceptance test.
7. No test materializes a tensor larger than `χ_cap²` (§9.7); `conftest.py` memory
   guard confirms it.
8. `wake_sleep_cycle` is deterministic (§9.8).
9. F's MERA tests and M1's encoder tests still pass.
10. Every `np.einsum` uses `optimize='greedy'`.
11. Every completion claim is backed by fresh pytest output (verification-before-
    completion).

---

## 13. Open questions

**None blocking.** The mining granularity (node-aligned MERA-tree blocks), the
recurrence object (boundary-bond reduced density matrix), the distance metric (trace
distance), the significance threshold (`k_min` from the uniform null model), and the
canonical-form recipe (operator-basis mean + bounded-bond re-purification) are all
fixed above. Two items are flagged as integration points, not ambiguities:

- The exact `LemmaLibrary` API (§7) — reconciled when sub-project I lands; the
  `FakeLemmaLibrary` test double lets J proceed and is the precise contract I must meet.
- The held-out cross-domain validation corpus (§6.3, §3.2) — the `validate` hook is
  specified and tested; the corpus itself is a later sub-project.

---

## 14. Contract J exposes downstream

- `wake_sleep_loop` is the entry point §10.10 (cross-level message passing) and the
  §10.11 hierarchical-proof demo call to grow the library between problem batches.
- `CanonicalPrimitive` (with `mera`, `rho_canonical`, `provenance`) is the object the
  lemma library stores and the compiler later instantiates by tensor product into a
  Hamiltonian — the operator-algebraic composability of §11.5.
- `CycleReport` is the observable for §13.8's capability-scaling measurements: the
  per-cycle `promoted` / `n_consolidated` / `n_pruned` counts are the empirical
  `α(t)` / `β` of the wake-sleep dynamics.

---

## 15. Glossary (J-local)

- **Subtree / sub-MERA** — an internal node of the cached state's MERA tree; a
  node-aligned contiguous leaf interval of width `2^d`.
- **Boundary bond** — the MERA tree edge connecting a subtree's root tensor to its
  parent; the reduced density matrix lives here.
- **Reduced density matrix `ρ_S`** — the subtree's density on its boundary bond; the
  operator-algebraic signature of the substructure.
- **Fingerprint** — a rounded (purity, top-eigenvalues, size) tuple; a basis-invariant
  pre-filter that avoids `O(M²)` trace-distance comparisons.
- **Trace distance** — `D(ρ₁,ρ₂)=½‖ρ₁−ρ₂‖₁`; the clustering metric.
- **`k_min`** — the significance threshold; smallest cluster size unlikely to arise by
  chance under the uniform null model.
- **Canonical form** — a cluster's representative operator: the operator-basis mean of
  its members' reduced densities, re-purified to a bounded-bond MERA.
- **Provenance** — the record of which problems a primitive was abstracted from and
  which later problems used it.
- **Wake-sleep cycle** — one RG-flow step: solve (wake) → mine + cluster + abstract
  (dream) → consolidate.
- **Two-tier library** — a stable manually-curated core plus a dynamic outer ring; J
  prunes only the dynamic ring.
