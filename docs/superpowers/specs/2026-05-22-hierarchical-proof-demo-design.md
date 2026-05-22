# Spec: Hierarchical Proof Composition Demo (migration sub-project L)

**Document type**: Implementation specification.
**Date**: 2026-05-22.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Implements**: `QFT_PCN_ARCHITECTURE.md` §10.11 — the second publishable milestone.
**Status**: Approved design; ready for implementation plan.

---

## 0. How to read this spec

This specifies the **hierarchical proof composition demo**: the demo script that takes a
mathematical theorem, builds a goal graph, dispatches sub-QPCNs to find a proof for each
node of a proof tree, integrates the results via the lemma machinery, and surfaces a
*verified* proof tree. It is sub-project L of the migration roadmap — the integration
milestone that composes M1/M2/M3 (the MERA logic substrate, typing/evaluation Hamiltonians,
synthesis) with sub-projects I (lemma library), J (wake-sleep abstraction discovery), and
K (cross-level message passing).

L builds *no new substrate*. It is an orchestration layer over already-specced components.
Everything here is the contract. If something needed during implementation is missing,
**stop and ask** — do not guess, and do not paper over a missing dependency with a stub.

No time, duration, or effort estimates appear in this document. Sequencing and dependency
order are specified; calendar/effort are not. Standing project directive.

---

## 1. Driving principles (non-negotiable)

Inherited project-wide (see `2026-05-22-mera-migration-design.md` §1) and specialized for L.
**Every subagent prompt that touches L must restate §1.1–§1.5 verbatim.** Subagents
default to the easiest path; the easiest path here is fabricated success. Refuse it.

### 1.1 No time or effort estimates

Never emit a wall-clock estimate, a duration, a "this should take", or an effort sizing.
The §10.11 acceptance text says "under 5 minutes on a single machine" — that is an
*acceptance bound on the finished demo's measured runtime*, not a planning estimate, and it
is the only number of its kind permitted. The demo *reports* its measured runtime; the spec
and plan never *predict* one.

### 1.2 Binding is genuine entanglement, never a classical lookup

A proof tree's nodes share variables (the theorem's universally-quantified `xs`, the
inductive variable, the lemma parameters). Where two proof-tree nodes share a variable,
that sharing is realized as **bond entanglement carried through the MERA tree** on the
substrate M1 built (M1 spec §5), and as the shared-variable coupling Hamiltonian
`H_coupling` of architecture §13.3.1. It is **never** a Python-side `dict[var, value]`
threaded between sub-QPCN runs. The cross-level message passing of K (§10.10) passes
*goal propositions and ground-state sub-MPSes*, not variable bindings as classical data.

### 1.3 `optimize='greedy'` on every einsum

Every `numpy.einsum` and every contraction in any code L adds uses `optimize='greedy'`.
No exception. L calls into M1/M2/M3 which already obey this; any contraction L writes
itself (e.g. assembling a composite proof-tree state) obeys it too.

### 1.4 A proof is VERIFIED only at zero residual energy — correctness by construction

This is the load-bearing criterion of the whole demo. Architecture §13.2 (Theorem 13.2):
a QPCN Hamiltonian is `H = Σ_i w_i P_i` with each `P_i` a positive-semidefinite penalty
operator that annihilates exactly the states satisfying constraint `i`. A state with
`⟨Ψ|H|Ψ⟩ = 0` satisfies **every** constraint.

A proof-tree node is **verified** iff the sub-QPCN that produced it reached a ground state
with **residual energy `E_node ≤ ε_verify` where `ε_verify = 1e-8`**. The whole proof tree
is **verified** iff *every* leaf and *every* internal node is verified, and the
cross-level integration (K, §10.10) reports a composition residual `E_compose ≤ ε_verify`
at every shared-variable boundary (architecture §13.3.1: equality of the joint
ground-state energy with the sum of child energies holds iff the shared-variable structure
is consistent — a positive composition residual is exactly "the lemmas clash").

There is **no other notion of "proved" in this demo.** A node is not proved because the
decoded AST "looks like a proof", because a classical type-checker is happy, or because a
search heuristic terminated. The classical type-check (architecture §10.8 registration
validation) is a *necessary cross-check that must also pass*, but it is never *sufficient*
and never a substitute for the energy criterion. If `E_node > ε_verify`, the node is
**not proved**, full stop — report it as unproved.

### 1.5 Honest reporting — partial results are publishable, fabricated results are not

The demo attempts four target theorems (§3) in increasing difficulty. It reports an
honest tally: "solved X of 4 target theorems", where X counts only theorems whose entire
proof tree met §1.4. If theorem 1 verifies and theorems 2–4 do not, the demo prints
`solved 1 of 4` and shows the partial proof trees with the unverified nodes marked.

A subagent that makes the tally read higher than the verified count — by loosening
`ε_verify`, by stubbing a sub-QPCN to return a hand-built ground state, by marking a node
verified on a classical check alone, by skipping a node, or by catching and swallowing a
non-convergence — has violated the project's foundational directive. Negative and partial
results are first-class deliverables here. A truthful `solved 1 of 4` is a publishable
result (§10.11, §11.8); a fabricated `solved 4 of 4` is research misconduct.

---

## 2. Background: what L composes

L is pure orchestration. It depends on the following, each specced/built in parallel.
L assumes their APIs exactly as the architecture doc describes; **§7 pins the exact
import surface L is written against**. If a dependency's shipped API differs, that is a
plan-level reconciliation task (§7.6), not a license for L to reimplement the dependency.

| Dep | Provides | Architecture ref |
|---|---|---|
| **M1** | MERA-native logic encoder: extended-calculus AST (`Forall`, `Fix`, `Eq`, `Nat`, `List`, `Cons`, `Nil`), `encode_mera`/`decode_mera`, `MeraEncodingMeta`, `mera_window_expectation_factored`. | §10 / M1 spec |
| **M2** | MERA-native typing + evaluation Hamiltonians; recursion unfolding; **induction-as-ground-state**. | §10 / M2 spec |
| **M3** | MERA-native synthesis + constraint debugger; imaginary-time relaxation to a ground state (`relax_to_ground`). | §10 / M3 spec |
| **I** | Lemma library: `LemmaLibrary` storage/indexing/registration, `use_lemma` promotion (clamp a sub-MPS / add `−W|Ψ_L⟩⟨Ψ_L|`). | §10.8 |
| **J** | Wake-sleep abstraction discovery: subtree mining, clustering, primitive promotion. **Used by L only in the optional capability-growth measurement (§5.4); not on theorem 1's critical path.** | §10.9 |
| **K** | Cross-level message passing: `GoalGraph` (DAG of sub-goals), `Dispatcher` (spawn + collect sub-QPCN runs), `ResultIntegrator` (bottom-up clamping), `Revision` (alternative decomposition). | §10.10 |

The §10.11 demo is "§10.7 (first synthesis milestone) extended to multi-level proof
trees, once §10.8–10.10 exist." L is the script that wires §10.8–10.10 into a runnable
end-to-end theorem-proving demo.

---

## 3. Scope

### 3.1 In scope

- `src/qft_pcn/composition/demo_hierarchical_proof.py` — the demo: takes a theorem
  identifier, builds the goal graph, dispatches sub-QPCNs via K, integrates via I,
  optionally measures capability growth via J, and surfaces the verified proof tree.
- The four **target-theorem definitions** (§3.3): each a frozen, version-pinned spec —
  axioms, the goal proposition, and the intended decomposition into a proof tree.
- The **proof-tree data structure** (`ProofTree`, `ProofNode`) and the
  **verification predicate** `verify_proof_tree` implementing §1.4 exactly.
- The **honest tally** reporter: `solved X of 4`, per-theorem pass/fail, per-node
  residual energies, total measured runtime.
- Acceptance tests (§6): theorem 1 (list-induction) must pass end-to-end; theorems 2–4
  are attempted with honest pass/fail reporting and are *not* required to pass.

### 3.2 Out of scope

- Building or modifying M1/M2/M3/I/J/K. L consumes them. A genuine bug found in a
  dependency is fixed *there* (with that sub-project's tests still green), not worked
  around in L.
- Distributed/Ray execution (architecture §10.10 "heavyweight"). L runs on a single
  machine with a thread pool at most; the §10.11 acceptance is explicitly single-machine.
- The quantum-walk dispatcher (architecture §12, the `O(log N)` goal-graph navigation).
  L uses K's classical BFS/DFS dispatcher. The quantum-walk dispatcher is a §12 extension.
- LLM-driven decomposition. The four target theorems ship with **fixed, hand-authored
  decompositions** (§3.3). K's `Revision` hook *may* call an LLM for alternative
  decompositions on a failed sub-goal, but theorem 1's critical path must succeed with
  the fixed decomposition and **no LLM call** — the acceptance run is reproducible offline.
- Any target theorem beyond the four in §3.3.

### 3.3 The four target theorems

Attempted in this order. Each is a frozen module under
`src/qft_pcn/composition/theorems/`. **Theorem 1 is the firm acceptance target;
theorems 2–4 are stretch and honestly reported.**

**Theorem 1 — list-induction (FIRM TARGET).**
`∀ xs ys : List Nat. length (xs ++ ys) = length xs + length ys`.
(`length (reverse xs) = length xs` is the §10.10 acceptance variant and is an accepted
substitute — whichever the implementation finds tractable first is theorem 1; the plan
picks `length (xs ++ ys) = length xs + length ys` as primary, the reverse identity as
the documented fallback.) Three-level decomposition (architecture §10.10 acceptance):
- **Level 0 — axioms**: definitions of `length` (`length Nil = 0`,
  `length (Cons x xs) = Succ (length xs)`), `++` (`Nil ++ ys = ys`,
  `(Cons x xs) ++ ys = Cons x (xs ++ ys)`), `+` (`0 + n = n`, `Succ m + n = Succ (m + n)`).
  Each axiom is a proof-tree leaf: a sub-QPCN whose Hamiltonian's ground state *is* the
  axiom's defining equation, residual zero by construction.
- **Level 1 — base case**: `length (Nil ++ ys) = length Nil + length ys`. A sub-QPCN
  ground state; verified by §1.4.
- **Level 1 — inductive step**: assuming `length (xs ++ ys) = length xs + length ys`,
  prove `length ((Cons x xs) ++ ys) = length (Cons x xs) + length ys`. A sub-QPCN ground
  state with the inductive hypothesis clamped as a lemma (I's `use_lemma`).
- **Level 2 — theorem**: the list-induction principle (M2's induction-as-ground-state)
  applied to the base and inductive-step lemmas. A sub-QPCN whose ground state composes
  the two children; `E_compose ≤ ε_verify` at the shared-`ys` boundary.

This is the "axioms → lemmas → inductive step → theorem" structure of §10.11.

**Theorem 2 — algebraic identity (STRETCH).**
`(a + b)^2 = a^2 + 2·a·b + b^2` from commutative-ring axioms. Decomposition: ring axioms
as level-0 leaves; distributivity expansion as level-1 lemmas; the final regrouping as
the level-2 theorem. Tests rewriting.

**Theorem 3 — small group theory (STRETCH).**
Every group of prime order `p` is cyclic. Decomposition: group axioms + Lagrange's
theorem (itself decomposed, or supplied as a registered library lemma) as level-0/1;
the order-of-element argument as level-2. Tests categorical structure.

**Theorem 4 — rank-nullity (STRETCH).**
`rank(T) + nullity(T) = dim(domain)` for a linear map between finite-dimensional spaces.
Decomposition: basis-extension lemma + dimension axioms as level-0/1; the
direct-sum / quotient argument as level-2. Tests dimensional reasoning.

Theorems 2–4 ship with hand-authored decompositions but **carry no obligation to verify**.
If a level cannot even be *encoded* on the M1 substrate within `χ_max = 32` (e.g. the
group-theory or linear-algebra encodings exceed the extended calculus's expressivity),
the demo records `theorem N: not encodable on current substrate` and counts it unsolved —
honestly. This is a real, publishable finding about the substrate's reach.

### 3.4 Will not do

- Loosen `ε_verify` to make a theorem "pass".
- Substitute a hand-constructed ground state for a sub-QPCN that failed to converge.
- Mark a node verified on a classical type-check alone (§1.4).
- Catch a sub-QPCN non-convergence and continue as if it converged.
- Reorder or drop a target theorem to inflate the tally.
- Exceed `χ_max = 32` per sub-QPCN (the §10.11 single-machine acceptance bound) — if a
  sub-QPCN needs more bond dimension, that is a reported failure, not a license to raise χ.

---

## 4. The proof tree

### 4.1 Data structures

```python
# src/qft_pcn/composition/proof_tree.py

@dataclass(frozen=True)
class ProofNode:
    node_id: str                       # stable identifier within the tree
    level: int                         # 0 = axiom leaf, increasing toward the root
    goal: GoalSpec                     # the proposition this node proves (K's goal type)
    children: tuple[str, ...]          # node_ids of sub-proofs this node composes
    state_ref: SubMPSRef | None        # the sub-QPCN ground state (M3 output); None until solved
    residual_energy: float | None      # E_node; None until the sub-QPCN has run
    compose_residual: float | None     # E_compose at this node's shared-var boundary; None for leaves
    classical_check: bool | None       # the §10.8 cross-check type-check result
    status: str                        # "pending" | "solved" | "unverified" | "unencodable"

@dataclass
class ProofTree:
    theorem_id: str
    root_id: str
    nodes: dict[str, ProofNode]

    def is_verified(self) -> bool: ...        # delegates to verify_proof_tree
```

`GoalSpec` and `SubMPSRef` are K's and I's types respectively (§7); L does not redefine
them.

### 4.2 The verification predicate (§1.4 made executable)

```python
# src/qft_pcn/composition/proof_tree.py

EPS_VERIFY = 1e-8

def verify_proof_tree(tree: ProofTree) -> VerificationReport:
    """A ProofTree is verified iff EVERY node satisfies, simultaneously:

      1. node.status == "solved"
      2. node.residual_energy is not None and node.residual_energy <= EPS_VERIFY
      3. node.classical_check is True            (necessary cross-check, never sufficient)
      4. for every internal node: node.compose_residual is not None
         and node.compose_residual <= EPS_VERIFY
      5. every child_id in node.children resolves to a node that is itself verified
         (recursive; the recursion is well-founded because the tree is a DAG)

    Returns a VerificationReport with the overall verdict AND the per-node table
    (node_id, level, residual_energy, compose_residual, classical_check, verdict)
    so the demo can print exactly which nodes carried the proof and which did not.

    This function NEVER returns True on a tree with a missing, None, or above-threshold
    energy anywhere. There is no 'close enough' branch.
    """
```

`VerificationReport.verdict` is `True` only if all five conditions hold for every node.
A theorem counts toward the §1.5 tally iff `verify_proof_tree(tree).verdict is True`.

### 4.3 Why energy zero is genuine verification (architecture grounding)

- A leaf is an *axiom* sub-QPCN: M2 compiles the axiom's defining equation into a
  Hamiltonian whose ground state encodes exactly that equation; residual zero means the
  decoded AST *is* the axiom (Theorem 13.2).
- An internal node is a sub-QPCN whose Hamiltonian includes the inductive-step or
  composition constraints plus the children clamped as lemmas via I (`−W|Ψ_L⟩⟨Ψ_L|`,
  §10.8). Residual zero means the composed structure satisfies every typing, evaluation,
  and induction constraint simultaneously.
- The composition residual `E_compose` at a shared-variable boundary is architecture
  §13.3.1's `H_coupling` energy: zero iff the shared-variable (binding-as-entanglement)
  structure of the two children is *consistent*. A positive `E_compose` is the
  architecture's built-in detection of "lemma A and lemma B clash" (§10.8 false-composition
  risk). The demo surfaces it rather than hiding it.

The proof is therefore *correct by construction* (§13.2) — not "checked after the fact"
and not "asserted by a heuristic."

---

## 5. The demo flow

`demo_hierarchical_proof.py` exposes `run_demo(theorem_ids=(1,2,3,4)) -> DemoReport` and a
`__main__` entry point.

### 5.1 Per-theorem pipeline

For each theorem id, in order:

1. **Load** the frozen theorem module from `theorems/`: its axioms, goal proposition, and
   intended decomposition.
2. **Build the goal graph** (K's `GoalGraph`): one node per intended proof-tree node, edges
   = "child solved before parent." Cycle-checked by K.
3. **Encodability gate**: attempt to encode every node's goal on the M1 substrate with
   `χ_max = 32`. If any node raises an M1 encoding exception or needs `χ > 32`, mark the
   theorem `unencodable`, record which node failed, skip to the next theorem. (Honest
   negative result, §3.3.)
4. **Dispatch bottom-up** (K's `Dispatcher`): solve level-0 leaves first (axiom
   sub-QPCNs), then each higher level, clamping already-solved children as lemmas via I.
   Each dispatch is one sub-QPCN run: M3's `relax_to_ground` on the node's Hamiltonian,
   `χ_max = 32`. Siblings at one level may run on a thread pool (K supports it); the
   shared `LemmaLibrary` is the only shared state and I guarantees it is append-safe.
5. **Record** for each node: `residual_energy` (from the relaxation), `compose_residual`
   (from K's `ResultIntegrator` at the shared-variable boundary), `classical_check` (I's
   registration type-check on the decoded AST), and `status`.
6. **Revision on failure**: if a node's `residual_energy > ε_verify`, K's `Revision` may
   try one alternative decomposition (a fixed, hand-authored fallback decomposition shipped
   with the theorem module — no LLM on theorem 1's path). If the alternative also fails,
   the node stays `unverified` and the failure propagates up the goal graph. The demo does
   **not** abort; it records the partial tree.
7. **Verify** the assembled `ProofTree` with `verify_proof_tree` (§4.2).

### 5.2 Surfacing the proof tree

The demo prints, per theorem:
- The proof tree, indented by level, each node showing
  `node_id  level  goal  residual_energy  compose_residual  classical_check  verdict`.
- For a verified theorem: `THEOREM N: VERIFIED` and the decoded proof term at the root.
- For an unverified theorem: `THEOREM N: NOT VERIFIED` and the specific nodes that failed,
  with their residual energies, so the reader sees exactly where the proof broke.

### 5.3 The honest tally

After all four: `DemoReport` prints

```
=== Hierarchical Proof Composition Demo — results ===
theorem 1 (list-induction)      : VERIFIED        (4 levels, max residual 3.1e-12)
theorem 2 (binomial identity)   : NOT VERIFIED    (level-2 node 'regroup' residual 0.07)
theorem 3 (prime-order cyclic)  : UNENCODABLE     (node 'lagrange' exceeds chi_max=32)
theorem 4 (rank-nullity)        : NOT ATTEMPTED   (substrate gap, see report)
-----------------------------------------------------
solved 1 of 4 target theorems
total measured runtime: 142.7 s on this machine
```

The `solved X of 4` line counts only `verify_proof_tree(...).verdict is True` theorems.
The runtime is *measured*, never predicted. Exact wording of the per-theorem lines reflects
the actual run; the example above is illustrative.

### 5.4 Optional capability-growth measurement (J)

If J is available, the demo *may* run an optional second pass: after theorem 1 verifies,
its inductive substructure is offered to J's wake-sleep cycle; J discovers an "induction"
primitive; the demo re-runs theorem 1 (or a sibling inductive theorem) and reports whether
the sub-QPCN converged in fewer Trotter steps (architecture §10.9 acceptance,
§11.8 point 3 — capability compounding). This pass is **clearly labelled optional**, is
**off by default**, and **its outcome never changes the §5.3 tally**. It is a measurement,
not a pass/fail gate.

---

## 6. Acceptance tests

`tests/test_hierarchical_proof_demo.py` (and per-theorem test modules under `tests/`).

### 6.1 Theorem 1 end-to-end (REQUIRED — the milestone)

`test_theorem1_list_induction_verified`: `run_demo(theorem_ids=(1,))` returns a
`DemoReport` with theorem 1 `VERIFIED`. Asserts:
- The proof tree has exactly the §3.3 structure: level-0 axiom leaves, level-1 base case
  and inductive step, level-2 theorem root — **3 levels of decomposition** (counting
  level-0 axioms, level-1 lemmas, level-2 theorem; the inductive step is itself a level-1
  node, matching the §10.10 "4-level" description of axioms→lemmas→inductive step→theorem).
- Every node's `residual_energy ≤ 1e-8`.
- Every internal node's `compose_residual ≤ 1e-8`.
- Every node's `classical_check is True`.
- `verify_proof_tree(tree).verdict is True`.
- Every sub-QPCN ran with `χ_max ≤ 32`.
- The whole run completes on a single machine. (The §10.11 "under 5 minutes" bound is
  asserted as a generous CI ceiling — `runtime < 300 s` — not as a tight measurement;
  if it is exceeded the test fails and the cause is investigated, not the bound relaxed.)

### 6.2 The verification predicate is honest (REQUIRED)

`test_verify_rejects_missing_energy`: a `ProofTree` with one node's `residual_energy = None`
→ `verify_proof_tree` verdict `False`.
`test_verify_rejects_above_threshold`: a node with `residual_energy = 1e-6` (> `ε_verify`)
→ verdict `False`.
`test_verify_rejects_classical_only`: a node with `residual_energy = None` but
`classical_check = True` → verdict `False` (classical check is never sufficient, §1.4).
`test_verify_rejects_bad_compose`: an internal node with `compose_residual = 0.05` →
verdict `False`.
`test_verify_rejects_unverified_child`: a verified-looking parent whose child is
`unverified` → verdict `False` (condition 5, recursive).
`test_verify_accepts_all_zero`: a tree with every node solved, every energy `≤ 1e-12`,
every classical check `True` → verdict `True`.

### 6.3 Theorems 2–4 attempted, honestly reported (REQUIRED that the *reporting* is honest)

`test_theorems_2_3_4_honest_tally`: `run_demo(theorem_ids=(1,2,3,4))` returns a report
where the `solved X of 4` count **equals** the number of theorems with
`verify_proof_tree(...).verdict is True`. The test does **not** assert any particular X —
it asserts the tally is *consistent with the per-theorem verdicts*. A theorem reported
`VERIFIED` must have a verified tree; a theorem reported `NOT VERIFIED` / `UNENCODABLE`
must not be counted. This is the test that catches a fabricated tally.

### 6.4 No-shortcut structural markers (REQUIRED)

`test_shared_var_is_entanglement`: for theorem 1's level-2 composition, the assembled
proof-tree state has strictly positive entanglement entropy across the cut separating the
two children's shared `ys` leaves — proving the shared variable is bond entanglement, not
a classical lookup (§1.2; the M1-style structural marker, M1 spec §9.4).
`test_no_dense_blowup`: the run materializes no tensor larger than `16**2` densely
(`conftest.py` memory ceiling, inherited from M1 §12.10).
`test_einsum_greedy`: a lint-style check that every `einsum` in
`src/qft_pcn/composition/demo_hierarchical_proof.py` and `proof_tree.py` passes
`optimize='greedy'`.

### 6.5 Determinism / reproducibility (REQUIRED)

`test_theorem1_reproducible`: theorem 1 runs offline (no network, no LLM) and produces the
same proof-tree structure and the same verdict on two consecutive runs with a fixed RNG
seed. Residual energies may differ in the last few digits; the verdict and structure must
not.

---

## 7. The contract L is written against (dependency import surface)

L is written against these exact signatures. If a shipped dependency differs, §7.6 applies.

### 7.1 From M1 (`src/qft_pcn/logic/`)

`encode_mera(ast, n_nodes_max=32, chi_layer=16) -> (MERA, MeraEncodingMeta)`;
`decode_mera(state, meta) -> DecodeResult`; the extended-calculus AST nodes
(`Forall`, `Fix`, `Eq`, `Nat`/`Zero`/`Succ`, `List`/`Nil`/`Cons`).

### 7.2 From M2

A typing/evaluation Hamiltonian compiler and **induction-as-ground-state**: given an AST
and a set of axioms/inductive predicates, produce a Hamiltonian whose ground state encodes
a proof. L calls M2 to compile each proof-tree node's Hamiltonian.

### 7.3 From M3

`relax_to_ground(H, chi_max=32, ...) -> RelaxResult` with `RelaxResult.residual_energy`
and `RelaxResult.state` (the ground-state sub-MPS/MERA). This is the single sub-QPCN run.

### 7.4 From I (`src/qft_pcn/composition/lemma_library.py`, `promoter.py`)

`LemmaLibrary` with `register(state, proposition, metadata)` and a type-directed `lookup`;
`use_lemma` clamping (clamp a sub-MPS to a cached ground state, or add `−W|Ψ_L⟩⟨Ψ_L|`).
I's `register` runs the classical type-check that becomes `ProofNode.classical_check`.

### 7.5 From K (`src/qft_pcn/composition/goal_graph.py`, `dispatcher.py`, `result_integrator.py`, `revision.py`)

`GoalGraph` (`Node(goal, status, children)`, cycle detection); `Dispatcher.dispatch(node)`
(spawn a sub-QPCN run, returns a result/future) and parallel sibling dispatch;
`ResultIntegrator.integrate(parent, child_result)` returning the `compose_residual`;
`Revision.revise(node)` for an alternative decomposition. L *uses* K's goal graph; it does
not reimplement DAG navigation.

### 7.6 Reconciliation rule

If a shipped dependency's API differs from §7.1–§7.5, the implementation plan adds an
explicit reconciliation task: a *thin adapter* in L that maps the shipped API to the
surface above, **or** a fix in the dependency (with that dependency's tests still green).
L never silently reimplements a dependency's responsibility. A missing dependency is a
**stop-and-ask**, not a stub.

---

## 8. File layout

```
src/qft_pcn/composition/
├── __init__.py                       # MODIFY/CREATE: export the demo entry point
├── proof_tree.py                     # NEW: ProofNode, ProofTree, verify_proof_tree, EPS_VERIFY
├── demo_hierarchical_proof.py        # NEW: run_demo, the pipeline, the tally, __main__
└── theorems/
    ├── __init__.py                   # NEW
    ├── theorem1_list_induction.py    # NEW: axioms, goal, fixed decomposition (+ fallback)
    ├── theorem2_binomial.py          # NEW
    ├── theorem3_prime_order_cyclic.py# NEW
    └── theorem4_rank_nullity.py      # NEW
tests/
├── conftest.py                       # NEW or MODIFY: memory ceiling fixture (from M1)
├── test_hierarchical_proof_demo.py   # NEW: §6.1, §6.3, §6.4, §6.5
├── test_proof_tree.py                # NEW: §6.2 verification-predicate tests
└── test_theorem1_list_induction.py   # NEW: theorem-1-specific end-to-end assertions
```

L creates only `proof_tree.py`, `demo_hierarchical_proof.py`, the `theorems/` package, and
the `tests/` modules. It does not modify M1/M2/M3/I/J/K source files (§3.2); the only
permitted edit outside L's own files is adding the demo export to
`src/qft_pcn/composition/__init__.py` and the shared `conftest.py` memory ceiling.

---

## 9. Error model

- A sub-QPCN that fails to converge below `ε_verify` is **not** an exception — it is a
  recorded `unverified` node. The demo continues and reports it.
- An M1 encoding exception (`EncodingTooLarge`, `MeraLeafBudgetExceeded`, …) during the
  §5.1 step-3 encodability gate marks the theorem `unencodable` and is recorded, not raised
  out of `run_demo`.
- A missing or API-incompatible dependency (M1–K) raises `ImportError`/`AttributeError`
  loudly at demo startup — L does **not** degrade to a stub. This is the stop-and-ask
  boundary (§7.6).
- A cyclic goal graph is K's `CycleError`; L surfaces it as a theorem-level failure.
- `verify_proof_tree` never raises on a malformed tree — it returns verdict `False` with
  the offending node identified.

---

## 10. Acceptance criteria

L is complete when:

1. `test_theorem1_list_induction_verified` (§6.1) passes: theorem 1 verified end-to-end,
   3-level decomposition, every node a sub-QPCN ground state at `residual ≤ 1e-8`,
   `χ_max ≤ 32`, single machine.
2. All six verification-predicate tests (§6.2) pass — `verify_proof_tree` provably rejects
   missing/above-threshold/classical-only/bad-compose/unverified-child trees.
3. `test_theorems_2_3_4_honest_tally` (§6.3) passes: the `solved X of 4` count is
   consistent with the per-theorem verdicts (no fabricated tally).
4. The three no-shortcut markers (§6.4) pass: shared variable is entanglement, no dense
   blowup, every einsum greedy.
5. `test_theorem1_reproducible` (§6.5) passes: offline, deterministic structure and verdict.
6. `python -m src.qft_pcn.composition.demo_hierarchical_proof` runs end-to-end, prints the
   proof tree(s) and the honest `solved X of 4` tally with measured runtime.
7. M1/M2/M3/I/J/K test suites still pass (L modified none of their source).
8. Every completion claim is backed by fresh pytest output (verification-before-completion).

The honest expectation, stated up front (§1.5, §3.3): **theorem 1 is the firm milestone
and must verify. Theorems 2–4 are stretch; the demo attempts them and reports the true
tally — `solved 1 of 4` is an acceptable, publishable outcome and `solved 4 of 4` is only
acceptable if every one of the twelve-plus sub-QPCN ground states genuinely reached
`residual ≤ 1e-8`.**

---

## 11. Open questions

**None blocking.** The verification criterion (§1.4, `ε_verify = 1e-8`), the proof-tree
structure (§4), the four frozen theorems (§3.3), the firm-vs-stretch split, and the
single-machine `χ_max = 32` bound are all resolved above. Two delegated plan-level
decisions (not spec ambiguities):

1. Whether theorem 1's primary form is `length (xs ++ ys) = …` or `length (reverse xs) = …`
   — the plan picks the one that encodes within budget first; both are §10.10-sanctioned.
2. The exact adapter shapes if §7.6 reconciliation is needed — discovered when the
   dependencies' shipped APIs are inspected at implementation time.

---

## 12. Glossary (L-local)

- **Proof-tree node** — one sub-problem; a sub-QPCN run produces its ground state.
- **Verified node** — a node meeting all five §4.2 conditions (`residual ≤ ε_verify`,
  classical cross-check passes, compose residual `≤ ε_verify`, children verified).
- **`ε_verify`** — the residual-energy verification threshold, `1e-8`. The single number
  that defines "proved" in this demo.
- **`E_compose`** — composition residual at a shared-variable boundary; architecture
  §13.3.1's `H_coupling` energy. Zero iff the children's shared-variable structure is
  consistent.
- **Honest tally** — `solved X of 4`, where X counts only theorems with a fully verified
  proof tree. Fabricating X is a foundational-directive violation.
- **Encodability gate** — the §5.1 step-3 check that a theorem's nodes fit the M1 substrate
  within `χ_max = 32`; failing it yields an honest `unencodable` result.
