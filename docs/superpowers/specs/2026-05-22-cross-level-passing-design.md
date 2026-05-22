# Spec: Cross-Level Message Passing (migration sub-project K)

**Document type**: Implementation specification.
**Date**: 2026-05-22.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Implements**: `QFT_PCN_ARCHITECTURE.md` §10.10 — *Cross-level message passing (hierarchical composition, mechanism 3)*. See also §11.2 (the five composition mechanisms in one frame).
**Status**: Approved design; ready for implementation plan.

---

## 0. How to read this spec

This specifies **cross-level message passing**: the runtime mechanism by which a parent QPCN run decomposes its target goal into sub-goals, dispatches each sub-goal as a child QPCN run, and integrates the children's solved ground states back into the parent's MERA via lemma-clamping. It is "sub-project K, the third hierarchical-composition mechanism, built on the MERA substrate (F), the LLM bridge (G), and the lemma library (I)."

Everything here is the contract. If something needed during implementation is missing, **stop and ask** — do not guess.

No time, duration, or effort estimates appear in this document. Sequencing and dependency order are specified; calendar/effort are not. Standing project directive.

---

## 1. Driving principles (non-negotiable)

These are the architecture's soul. Embed them verbatim in every subagent prompt that touches this sub-project. A subagent will reach for the easiest path; it must instead be made aware of these and refuse the shortcut.

1. **The architecture is a multi-field synthesis. Do not stop digging at hard field-crossings.** When a fork offers an easy-but-wrong path and a harder principled path, take the principled path.
   - *Shortcut to refuse*: "implement cross-level passing as an ad-hoc RPC — parent calls a child function, gets a return value, splices it in."
   - *Principled alternative*: cross-level passing **is variational free-energy minimization at the scale of whole QPCN runs**. §10.10 states this is a structural claim, not an analogy. The top-down message is a *prediction* (the parent's expected proposition type at a sub-tree), the bottom-up message is a *precision-weighted prediction error* (the child's residual energy gates how strongly its result clamps the parent). Same equation as a single predictive-coding layer update; only the scale changes. The goal/proof exchange must be modelled on that equation, not on a function call.

2. **Binding is genuine entanglement, never a classical lookup** (architecture §8.1, §1.1 of the migration spec).
   - *Shortcut to refuse*: when integrating a child, store `parent_dict[site] = child_ast` and have the decoder read it back.
   - *Principled alternative*: integration is **clamping a sub-MERA into the parent's MERA via I's lemma-promotion machinery** (`result_integrator.py`). The child's solved ground state becomes a clamped region of the parent's tensor network; the parent's subsequent imaginary-time evolution sees it as a fixed boundary condition, not a Python value.

3. **`optimize='greedy'` on every einsum / `numpy` contraction**, including any contraction inside the integrator or the residual checks.

4. **A child is integrated only if it is in a true ground state.** Premature integration (§10.10 risk table) is forbidden: a low-energy *excited* state that happens to be below a loose threshold must not be clamped. The residual-energy gate is strict (§6.3) and is the precision term in the free-energy message — a high residual means low precision means a weak (or refused) clamp, exactly as a noisy sensory channel down-weights its prediction error.

5. **OOM / resource pressure is a signal to optimize or to parallelize better, not to shrink the problem.** Reducing `χ_max`, the goal-graph depth, or the acceptance theorem to dodge a resource limit is forbidden.

6. **Reuse, do not reinvent.** F's MERA substrate (`src/qft_pcn/qft/mera.py`), G's LLM bridge and DSL pipeline (`src/qft_pcn/bridge/`), and I's lemma library/promotion machinery are consumed. K orchestrates them; it does not duplicate a QPCN runner, a DSL compiler, or a clamp primitive.

---

## 2. Where K sits in the migration

| Sub-project | Role | K's relation |
|---|---|---|
| F (built) | MERA substrate: `MERA`, clamping, expectations. | A child QPCN's ground state and the parent's MERA are both `MERA` states. |
| G (built) | LLM bridge: NL → DSL spec, DSL compile + run pipeline (`bridge/dsl/pipeline.py`, `bridge/api.py`, `bridge/runtime/`). | K packages each sub-goal as a **DSL spec** and runs it through G's pipeline. The LLM frontend suggests decompositions and revisions. |
| I (lemma library) | Lemma storage + promotion: clamp a solved sub-MERA into a parent. | K's `result_integrator` calls I's promotion machinery to clamp a solved child. K **does not** re-implement clamping. |
| **K (this spec)** | The runtime that *navigates* the goal graph, dispatches children, integrates results. | Mechanism 3 of §11.2 — "the runtime mechanism by which the composition actually happens." |

§10.8 (lemma library, I) and §10.9 (abstraction discovery) are *what* gets composed; §10.10 (K) is *how* the composition runs.

K is the **runtime**. It owns no proof logic of its own beyond goal-graph bookkeeping and the free-energy message-passing schedule. The proving happens inside child QPCN runs (G's pipeline); the lemma mechanics happen inside I.

---

## 3. Scope

### 3.1 In scope

- The **goal graph**: a DAG of sub-goals. `Node(goal, status, children)`, `status ∈ {pending, active, solved, failed, cycle}`. Construction (top-down from a root goal), traversal, and **cycle detection** via a per-search visited set.
- The **dispatcher**: spawn child QPCN runs (each a DSL spec routed through G's pipeline), collect their results, run **siblings in parallel** (thread pool / asyncio), enforce a **per-child timeout**.
- The **result integrator**: bottom-up clamping. If a child's residual energy is below threshold, clamp its solved sub-MERA into the parent via I's lemma machinery; otherwise mark the parent `pending_revision`.
- **Revision**: LLM-guided alternative decomposition when a sub-goal fails (`revision.py`), routed through G's LLM frontend.
- The **whole-hierarchy free-energy objective**: K reports a single scalar `F_hierarchy` (§7) that the schedule minimizes; this is the structural-claim accounting from §10.10.
- An acceptance test suite (§9): a small inductive proof requiring 3 levels of decomposition.

### 3.2 Out of scope (own specs / already built)

- The lemma library itself and its promotion/clamp internals — sub-project I. K calls I; K does not implement it.
- Abstraction discovery (§10.9) — sub-project J.
- The LLM bridge, DSL schema, DSL→Hamiltonian compiler, and the single-QPCN runtime — sub-project G (built). K consumes `src/qft_pcn/bridge/`.
- The §10.11 hierarchical-proof *demo* (`demo_hierarchical_proof.py`) — a follow-on; K delivers the acceptance test, not the full demo.
- Distributed (Ray / cluster) execution — §10.10 lists it as "heavyweight"; K ships the "lightweight" thread-pool / asyncio path. The dispatcher's concurrency boundary is an interface (§5.2) so a Ray backend can be added later without touching the goal graph or integrator.

### 3.3 Will not do

- Re-implement a QPCN runner, a DSL compiler, or a clamp primitive (principle 6).
- Integrate a child on a loose / non-ground-state residual (principle 4).
- Model the goal/proof exchange as an ad-hoc RPC (principle 1).
- Allow a cyclic goal graph to deadlock (cycle detection is mandatory, §4.4).
- Let one bad sub-goal starve all parallel work (per-child timeout + fail-fast, §5.4).

---

## 4. The goal graph

`src/qft_pcn/composition/goal_graph.py`.

### 4.1 The sub-goal and the node

A **sub-goal** is a DSL spec plus the proposition type expected at that sub-tree's root. It is exactly what G's `bridge/dsl/pipeline.py` consumes as input.

```python
@dataclass(frozen=True)
class SubGoal:
    goal_id: str                     # stable identity; used by the visited set
    dsl_spec: dict                   # a G-compatible DSL spec (bridge/dsl/schema.py shape)
    goal_prop: str                   # the expected proposition type at this sub-tree root
    boundary: dict                   # parent-context constraints -> child boundary conditions
    parent_site: int | None          # the leaf/site in the parent MERA this sub-goal occupies
```

`goal_id` is a content hash of `(goal_prop, dsl_spec)` (canonicalized). Two sub-goals with the same `goal_id` are the same goal — this is what makes cycle detection sound (§4.4) and lemma sharing across siblings real (§8).

```python
class Status(enum.Enum):
    PENDING = "pending"           # created, not yet dispatched
    ACTIVE  = "active"            # a child QPCN run is in flight
    SOLVED  = "solved"            # child returned a true ground state; integrated
    FAILED  = "failed"            # child did not converge; revision exhausted
    CYCLE   = "cycle"             # this goal is already an ancestor under proof
    PENDING_REVISION = "pending_revision"  # child failed; awaiting an alternative decomposition

@dataclass
class Node:
    goal: SubGoal
    status: Status
    children: list["Node"] = field(default_factory=list)
    result: "ChildResult | None" = None      # set when SOLVED or FAILED
    revision_attempts: int = 0
    parent: "Node | None" = None             # back-edge; not part of the DAG semantics
```

### 4.2 Construction (top-down)

```python
def build_goal_graph(root_spec: dict, root_prop: str,
                     decomposer: "Decomposer") -> Node
```

- The **root** is the user's target goal (a DSL spec + proposition).
- A node's children are the sub-goals identified by the **decomposer** (§4.3). Construction is lazy: a node's children are populated when the node is first selected for expansion, not eagerly to full depth (the graph can be large; §10.10 risk "combinatorial explosion").
- Each child carries `parent_site` — the leaf index in the *parent's* MERA encoding (M1's `MeraEncodingMeta.node_of_leaf` / `site_to_ast_path`) that the sub-tree occupies. This is the site the integrator clamps into (§6).

### 4.3 The decomposer

```python
class Decomposer(typing.Protocol):
    def decompose(self, node: Node) -> list[SubGoal]: ...
```

Two implementations:

- `CompilerDecomposer` — inspects the parent's Hamiltonian compilation: sub-trees with high `?`-density (hole density) or marked `requires_lemma` become sub-goals. This is the deterministic path (§10.10 "Goal graph": *children are sub-goals identified by the parent's Hamiltonian compiler*).
- `LLMDecomposer` — calls G's LLM frontend (`bridge/llm.py`) when the compiler is unsure how to break a problem down, or for the root NL→goal-graph conversion. §10.10 "LLM involvement": *always at the root; at decomposition points; at failure points.*

The decomposer used is composable: `LLMDecomposer` is the fallback when `CompilerDecomposer` returns no sub-goals for a still-unsolved node. A node with **no** sub-goals from either decomposer is a **leaf goal** — dispatched directly as a single QPCN run with no further decomposition (the §10.10 acceptance "Level 0 axioms" sit here).

### 4.4 Cycle detection

§10.10 mandate: *maintain a per-search visited set; refuse to dispatch a child whose goal is already being proved as an ancestor.*

- A search carries a `visited: set[str]` of `goal_id`s currently on the active path from the root.
- Before a node is dispatched, `goal_id` is checked against `visited`. On a hit, the node's status is set to `Status.CYCLE` and it is **not** dispatched.
- A `CYCLE` node triggers revision (§4.5): the parent must find an alternative decomposition. If revision is exhausted, the node becomes `FAILED` and the failure propagates up (§6.4).
- The visited set is *per active path*, not global: the same `goal_id` appearing in two independent sibling sub-trees is **not** a cycle — it is a shared lemma (§8), and the second occurrence resolves to the first's cached result.

```python
def detect_cycle(node: Node, visited: frozenset[str]) -> bool:
    return node.goal.goal_id in visited
```

The graph is asserted acyclic after every `solved` integration by `assert_acyclic(root)` (a DFS); a detected back-edge that slipped past `detect_cycle` is a bug, not a recoverable state, and raises `GoalGraphError`.

### 4.5 Goal-graph operations (the §10.10 "Detailed implementation" API)

```python
def dispatch(node: Node) -> "Future[ChildResult]"      # delegates to dispatcher.py
def integrate(parent: Node, child_result: "ChildResult") -> None   # delegates to result_integrator.py
def revise(node: Node) -> list[SubGoal]                # delegates to revision.py
def status_of(node: Node) -> Status
def is_solved(root: Node) -> bool                      # root.status == SOLVED
```

`dispatch`, `integrate`, `revise` are thin façades on the dedicated modules so the goal graph stays a pure data structure + traversal; the policy lives in the dispatcher / integrator / revision modules.

### 4.6 The proof tree

When `is_solved(root)` holds, `extract_proof_tree(root) -> ProofTree` walks the `SOLVED` nodes and returns a verified proof tree: each node carries the child QPCN's solved AST, its residual energy, and the clamp that integrated it. This is the §10.10 acceptance deliverable — "surfacing the final proof."

```python
@dataclass(frozen=True)
class ProofTreeNode:
    goal_prop: str
    solved_ast: object              # the child's decoded AST (M1's decoder output)
    residual_energy: float
    children: tuple["ProofTreeNode", ...]

@dataclass(frozen=True)
class ProofTree:
    root: ProofTreeNode
    total_residual: float           # sum over the tree; the residual term of F_hierarchy
```

---

## 5. The dispatcher

`src/qft_pcn/composition/dispatcher.py`.

### 5.1 Spawning a child QPCN run

```python
@dataclass(frozen=True)
class ChildResult:
    goal_id: str
    converged: bool
    residual_energy: float          # the child run's final residual (RunResult.energy)
    ground_state: object | None     # the child's MERA ground state, or None on failure
    solved_ast: object | None       # decoded AST, or None on failure
    run_diagnostic: dict            # G's RunResult.to_dict() for provenance
    error: str | None               # failure / timeout description
```

```python
def run_child(sub_goal: SubGoal, *, chi_max: int = 32,
              timeout_s: float) -> ChildResult
```

`run_child` packages `sub_goal.dsl_spec` (with `sub_goal.boundary` merged in as boundary conditions) and routes it through G's existing pipeline: `bridge.dsl.pipeline.compile_dsl` → `bridge.runtime` evolution → `RunResult`. K does **not** implement the run; it adapts `RunResult` into `ChildResult`. `RunResult.energy` is the residual; `RunResult.converged` is the convergence flag.

`chi_max = 32` per sub-QPCN is the §10.10 acceptance setting.

### 5.2 Concurrency — parallel siblings

§10.10: *children at the same level in the goal graph are independent; they can be searched in parallel.*

```python
class DispatchBackend(typing.Protocol):
    def submit(self, sub_goal: SubGoal, timeout_s: float) -> "Future[ChildResult]": ...
    def shutdown(self) -> None: ...

class ThreadPoolBackend(DispatchBackend):    # the shipped lightweight path
    def __init__(self, max_workers: int = 4): ...
```

- `dispatch_siblings(nodes: list[Node], backend: DispatchBackend) -> list[ChildResult]` submits all sibling nodes at once and collects results as they complete.
- The lemma library is **shared** across the backend's workers (§8): a lemma proved in one sibling is visible to the others through I's library, which is process-shared and guarded by a lock.
- `DispatchBackend` is the seam where a future Ray backend plugs in (§3.2). The goal graph and integrator never see the backend type.

### 5.3 The free-energy schedule

The dispatch order is **not** an arbitrary BFS/DFS. It is the schedule that minimizes `F_hierarchy` (§7): the dispatcher expands the frontier node with the **largest expected free-energy reduction** — the node whose `goal_prop` has the highest hole-density / precision-weighted error in the parent. This is the higher-scale analog of a predictive-coding layer choosing which prediction error to resolve first. A plain BFS is permitted as a fallback only when the precision estimates are unavailable, and the code must say so explicitly (it is the easy path; the principled path is precision-ordered).

### 5.4 Per-child timeout and fail-fast

§10.10 risk *resource starvation*: *a single bad sub-goal blocks all parallel work.*

- Every `run_child` has a hard `timeout_s`. On timeout the future resolves to a `ChildResult` with `converged=False`, `error="timeout"`, and `residual_energy=inf`.
- A timed-out or non-converged child does **not** block siblings: `dispatch_siblings` collects per-node and never joins on a single straggler beyond its own timeout.
- Fail-fast: a node whose residual stays above threshold after `MAX_REVISIONS` (§6.5) is quarantined (§6.6); the dispatcher does not keep retrying it.

---

## 6. The result integrator

`src/qft_pcn/composition/result_integrator.py`.

### 6.1 Bottom-up integration — the principled mechanism

§10.10: *child QPCN's ground state `|Ψ_child⟩` is returned with its residual energy. If residual below threshold: integrate as a clamped sub-MPS in the parent (via the §10.8 lemma-promotion machinery).*

```python
def integrate_child(parent_state, parent_meta, node: Node,
                     child_result: ChildResult,
                     lemma_library) -> "IntegrationOutcome"
```

The integrator does **not** splice a Python AST into the parent. It:

1. Checks the residual gate (§6.3). If the gate fails → `pending_revision` (§6.4).
2. If the gate passes, promotes the child's ground state to a **lemma** via I's lemma-promotion machinery (`lemma_library.promote(child_result.ground_state, child_result.solved_ast, child_result.goal_id)`).
3. **Clamps** the promoted lemma's sub-MERA into the parent's MERA at `node.goal.parent_site`. The clamp is I's primitive (the MERA analog of `bridge/runtime/clamp.py`'s `project_site`, generalized to a multi-leaf sub-tree window). The parent's subsequent imaginary-time evolution then sees the clamped region as a fixed boundary condition.
4. Sets `node.status = SOLVED`, records `node.result`.

The clamped sub-MERA is genuine network state, not a dictionary entry — principle 2.

### 6.2 The precision-weighted message

The free-energy reading (principle 1): the child's bottom-up message is its ground state; the **precision** of that message is `1 / (residual_energy + ε)`. A low-residual child is a high-precision message and clamps strongly; a higher-residual child (still below the gate) clamps with a proportionally weaker constraint strength. The clamp strength passed to I is:

```python
clamp_strength = STRENGTH_MAX * precision_weight(child_result.residual_energy)
precision_weight(r) = 1.0 / (1.0 + r / RESIDUAL_SCALE)
```

This is the §10.10 fallback option *"lower the child's strength of constraints to allow approximate matching ('treat this as a conjecture')"* — realized as the precision term of the message, not as an ad-hoc knob.

### 6.3 The residual-energy gate (strict)

```python
RESIDUAL_GATE = 1e-6        # absolute residual energy ceiling for a true ground state
GROUND_STATE_GAP = 1e-4     # min gap to the first excited state
```

A child is integrated as `SOLVED` **only if**:

- `child_result.converged is True`, **and**
- `child_result.residual_energy <= RESIDUAL_GATE`, **and**
- the child run reports a spectral gap `>= GROUND_STATE_GAP` to the first excited state (so the returned state is the *ground* state, not a near-degenerate excited state — §10.10 risk *premature integration*).

If `converged` but `RESIDUAL_GATE < residual_energy <= CONJECTURE_CEILING`, the child may be integrated as a **conjecture**: clamped with a reduced `clamp_strength` (§6.2) and the node marked `SOLVED` but flagged `provisional=True` in `node.result`. A provisional integration must be re-evaluated if any sibling's assumptions change (§8 lazy re-evaluation). Above `CONJECTURE_CEILING` → `pending_revision`.

### 6.4 Failure path

If the gate fails outright (`residual_energy > CONJECTURE_CEILING` or `converged is False`):

- `node.status = FAILED` locally; `node.parent.status = PENDING_REVISION`.
- The failure propagates up the goal graph (§6.5) until a node can be revised or the root is reached.

### 6.5 Revision trigger

A `PENDING_REVISION` node is handed to `revision.py` (§10). `revision.revise(node)` returns an alternative `list[SubGoal]`; the node's `children` are replaced and the sub-graph re-dispatched. `node.revision_attempts` is incremented; after `MAX_REVISIONS = 3` the node becomes `FAILED` and the failure propagates to *its* parent. If the *root* becomes `FAILED`, the whole composition fails — the parent's plan was flawed (§10.10) — and K returns a structured failure report, never a fabricated proof.

### 6.6 Quarantine

§10.10 risk *cascade failures*: a `FAILED` sub-graph is **quarantined** — marked and excluded from further dispatch — while alternative decompositions of its parent are explored in parallel. Quarantine is a flag on the `Node`; the dispatcher skips quarantined nodes.

---

## 7. The whole-hierarchy free-energy objective

K reports a single scalar that the schedule minimizes — the §10.10 structural claim made concrete:

```
F_hierarchy = Σ_{nodes n}  residual_energy(n)              # accuracy / prediction-error term
            + Σ_{nodes n}  complexity_penalty(n)           # complexity term (decomposition size)
```

- The **accuracy term** is the sum of child residual energies — the prediction error of every sub-QPCN against its goal proposition.
- The **complexity term** penalizes goal-graph breadth/depth (KL-to-prior in free-energy language), so the schedule prefers the smallest decomposition that drives accuracy to zero.
- `F_hierarchy` decreases monotonically as solved children are integrated (each integration removes a residual term). K asserts this monotone decrease in the acceptance suite (§9.5): a non-monotone step is a bug in the integrator (a premature or wrong clamp).

This is the same `F = accuracy + complexity` decomposition a single predictive-coding layer minimizes. K does not invent a new objective; it instantiates the existing one at the whole-QPCN scale.

`compute_free_energy(root: Node) -> float` lives in `goal_graph.py`.

---

## 8. Shared lemma library across siblings

§10.10: *the lemma library is shared across siblings, so discoveries in one branch immediately benefit others.*

- The dispatcher holds one `lemma_library` (I's object), passed to every `run_child` and every `integrate_child`.
- Access is guarded by a lock (siblings run on a thread pool, §5.2).
- Before dispatching a node, the dispatcher queries the library by `goal_id`: if a lemma for that exact goal already exists (proved by an earlier sibling), the node resolves to the cached lemma with **no** child run — the second occurrence of a shared `goal_id` (§4.4) is a cache hit, not a cycle and not a re-proof.
- **Lazy re-evaluation** (§10.10 risk *stale lemmas*): if a node's `boundary` (assumptions) changes after a lemma was cached under the old assumptions, the cache entry is invalidated and the node re-dispatched. Provisional integrations (§6.3) are always re-evaluated when a sibling completes.

---

## 9. Acceptance tests

`src/qft_pcn/composition/tests/test_goal_graph.py`, `test_dispatcher.py`, `test_result_integrator.py`, `test_revision.py`, and the end-to-end `test_cross_level_acceptance.py`.

### 9.1 Goal-graph construction and cycle detection

- `build_goal_graph` from a root spec produces a DAG; `assert_acyclic` passes.
- A decomposer that emits a child with an ancestor's `goal_id` causes that node's status to become `CYCLE` and it is not dispatched.
- A `goal_id` appearing in two independent sibling sub-trees is **not** flagged a cycle; the second resolves to the first's cached result (§8).
- `extract_proof_tree` on a fully-solved graph returns a `ProofTree` whose nodes mirror the `SOLVED` goal-graph nodes.

### 9.2 Dispatcher concurrency and timeout

- `dispatch_siblings` over N independent sub-goals runs them concurrently on `ThreadPoolBackend` (verified: wall time well under the serial sum when each child sleeps a fixed stub interval).
- A child exceeding `timeout_s` resolves to `ChildResult(converged=False, error="timeout", residual_energy=inf)` and does **not** delay its siblings.
- The shared `lemma_library` is observed by a sibling after another sibling promotes a lemma (cache-hit path, §8).

### 9.3 Integrator residual gate

- A child with `residual_energy <= RESIDUAL_GATE`, `converged`, and sufficient spectral gap is integrated `SOLVED`; the parent MERA gains a clamped region at `parent_site` (verified by an expectation on the clamped leaves matching the child's ground state).
- A child with `RESIDUAL_GATE < residual <= CONJECTURE_CEILING` is integrated `SOLVED, provisional=True` with reduced `clamp_strength`.
- A child with `residual > CONJECTURE_CEILING` or `converged is False` leaves the node `FAILED` and the parent `PENDING_REVISION`; no clamp is applied.
- A near-degenerate child (gap `< GROUND_STATE_GAP`) is **not** integrated even if its residual is tiny — the premature-integration guard (§6.3).

### 9.4 Revision and failure propagation

- A `PENDING_REVISION` node calls `revision.revise`, replaces its children with the alternative decomposition, and re-dispatches.
- After `MAX_REVISIONS` failed revisions a node becomes `FAILED` and its parent becomes `PENDING_REVISION`.
- A root that becomes `FAILED` makes K return a structured failure report — never a fabricated proof.

### 9.5 Free-energy monotonicity

- `compute_free_energy(root)` is recorded after every integration; the sequence is monotone non-increasing across the whole acceptance run.

### 9.6 End-to-end acceptance — the §10.10 theorem

Prove `∀ xs : List A. length (reverse xs) = length xs` by 3 levels of decomposition:

- **Level 0 (axioms)** — definitions of `length`, `reverse`, the constructor `Cons`: leaf goals, dispatched directly.
- **Level 1 (lemmas)** — `length (xs ++ [y]) = length xs + 1` and `reverse (Cons x xs) = reverse xs ++ [x]`: each a sub-goal dispatched as a child QPCN run; the two are **siblings** and run in parallel.
- **Level 2 (induction case)** — the inductive step combining both Level-1 lemmas: a node whose children are the Level-1 lemmas; integrated only after both children are `SOLVED`.
- **Level 3 (theorem)** — the induction principle applied to base + inductive case: the root.

The test asserts: the goal graph reaches depth 3; siblings at Level 1 dispatch in parallel; every integration passes the residual gate; `extract_proof_tree(root)` returns a verified proof tree with `total_residual <= RESIDUAL_GATE * n_nodes`; `is_solved(root)` is `True`. `χ_max = 32` per sub-QPCN (§10.10 setting). No wall-time assertion is made — time estimates are a directive violation (principle, §1); the test asserts *correctness and structure*, not duration.

### 9.7 No-shortcut structural markers

- The integrator never writes the child AST into a parent-side `dict`: a test inspects that integration mutates the parent **MERA state** (a clamped region with the expected expectation values), and that decoding the parent reads the clamped sub-tree out of the network, not out of a side table.
- `optimize='greedy'` appears on every contraction in `composition/` (a grep-style test, mirroring the M1 suite's einsum check).

---

## 10. Revision module

`src/qft_pcn/composition/revision.py`.

```python
def revise(node: Node, llm=None) -> list[SubGoal]
```

- Asks G's LLM frontend (`bridge/llm.py`): *"sub-goal X could not be proved with the current decomposition; suggest a different one"* — §10.10 *"I'm stuck on sub-goal X; can you suggest a different proof strategy?"*.
- A deterministic `HeuristicReviser` fallback exists for offline / no-LLM runs: it permutes the decomposition (e.g. swap induction variable, split a conjunction differently) from a fixed catalogue.
- `revise` caches failed decompositions by `goal_id` (§10.10 risk *combinatorial explosion* — *cache failed approaches*) so a revision never re-proposes a known-dead decomposition.

---

## 11. Error model

`src/qft_pcn/composition/errors.py`:

- `CompositionError` — base.
- `GoalGraphError` — a back-edge slipped past cycle detection (a bug, not recoverable).
- `DispatchTimeout` — surfaced inside `ChildResult.error`, not raised, so it never aborts a sibling batch.
- `IntegrationRefused` — the residual gate refused a clamp; carries the residual and the gate value.
- `RevisionExhausted` — `MAX_REVISIONS` reached; the node is `FAILED`.

K is total on a well-formed root spec: it returns either a `ProofTree` or a structured failure report. It never fabricates a proof and never silently lowers the residual gate.

---

## 12. File layout

```
src/qft_pcn/composition/
├── __init__.py
├── goal_graph.py          # SubGoal, Node, Status, build_goal_graph, cycle detection,
│                          # extract_proof_tree, compute_free_energy, the §4.5 façade ops
├── dispatcher.py          # run_child, ChildResult, DispatchBackend, ThreadPoolBackend,
│                          # dispatch_siblings, the free-energy schedule, per-child timeout
├── result_integrator.py   # integrate_child, the residual gate, precision-weighted clamp,
│                          # quarantine
├── revision.py            # revise, HeuristicReviser, failed-decomposition cache
├── errors.py              # the §11 exception family
└── tests/
    ├── __init__.py
    ├── test_goal_graph.py
    ├── test_dispatcher.py
    ├── test_result_integrator.py
    ├── test_revision.py
    └── test_cross_level_acceptance.py
```

All files are new. `bridge/`, `qft/mera.py`, the lemma library (I), and the MPS/MERA logic stacks are **not** modified — K consumes them. If a genuine missing primitive is found in I's clamp machinery (e.g. multi-leaf sub-tree clamp not exposed), it is fixed in I, not duplicated in K, and I's tests must still pass.

---

## 13. Acceptance criteria

K is complete when:

1. Goal-graph construction, traversal, and cycle detection (§9.1) pass.
2. Dispatcher parallel-sibling execution and per-child timeout (§9.2) pass.
3. The integrator residual gate — strict ground-state, conjecture band, refusal — (§9.3) passes.
4. Revision and failure propagation (§9.4) pass, including a root-failure structured report.
5. `F_hierarchy` is monotone non-increasing across the acceptance run (§9.5).
6. The §10.10 end-to-end theorem `∀ xs. length (reverse xs) = length xs` is proved by 3-level decomposition with parallel Level-1 siblings, `χ_max = 32`, and `extract_proof_tree` returns a verified tree (§9.6).
7. The no-shortcut structural markers (§9.7) pass — integration mutates the parent MERA, not a dict; every contraction uses `optimize='greedy'`.
8. G's bridge tests, F's MERA tests, and I's lemma tests still pass.
9. Every completion claim is backed by fresh pytest output (verification-before-completion).

---

## 14. Contract K exposes upward (to §10.11 demo and beyond)

- `build_goal_graph` + `dispatch_siblings` + `integrate_child` + `extract_proof_tree` are the four entry points the §10.11 hierarchical-proof demo composes.
- `ProofTree` is the verified-proof-tree artefact the demo surfaces.
- `DispatchBackend` is the seam for a future Ray / cluster backend (§3.2) — additive, no change to goal graph or integrator.
- `compute_free_energy` exposes the whole-hierarchy objective for any later monitoring / scheduling work.

---

## 15. Open questions

**None.** The goal-graph node shape, the status set, the per-search visited-set cycle detection, the residual-gate three-band policy (ground / conjecture / refuse), the precision-weighted clamp strength, the thread-pool concurrency path with the Ray seam left as an interface, and the free-energy objective are all resolved above. The choice of `CompilerDecomposer` vs `LLMDecomposer` ordering at a given node is a plan-level policy (the spec fixes the fallback rule §4.3); the implementation plan pins the catalogue of `HeuristicReviser` permutations.

---

## 16. Glossary (K-local)

- **Sub-goal** — a DSL spec plus its expected proposition type; the unit a child QPCN run proves.
- **Goal graph** — the DAG of sub-goals, built top-down, consumed bottom-up.
- **Cross-level message** — top-down: a parent's prediction (goal proposition); bottom-up: a child's precision-weighted prediction error (ground state gated by residual energy).
- **Residual gate** — the strict ground-state test (§6.3) a child must pass before its sub-MERA is clamped into the parent.
- **Conjecture integration** — a child below the conjecture ceiling but above the ground-state gate, clamped with reduced strength and flagged provisional.
- **F_hierarchy** — the whole-hierarchy variational free energy; `accuracy + complexity`; monotone non-increasing as children integrate.
- **Quarantine** — a `FAILED` sub-graph excluded from further dispatch while alternatives are explored.
- **Shared lemma library** — I's library, one instance across all sibling workers; a proved goal is a cache hit, not a re-proof.
