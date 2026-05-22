# Spec: Migration to the §10+§11+§12 Architecture End State

**Document type**: Migration roadmap + full specification of sub-project M1.
**Date**: 2026-05-22.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` (root, 3416 lines — the end-state vision).
**Status**: Approved roadmap; M1 ready for implementation plan.

---

## 0. How to read this spec

This is the migration spec from the current implemented state to the architecture doc's end state. It has two parts:

- **§1–§4: the migration roadmap** — current state, end state, decomposition into phases and sub-projects, dependency order. Roadmap-level: each sub-project named, scoped, and dependency-ordered, but not detailed.
- **§5–§13: sub-project M1 in full** — the first sub-project to build (factored MERA layer-0 + factored MERA expectation). Detailed to the level the implementation plan needs.

Sub-projects M2–L and the eighteen §12 extensions each get their own spec → plan → implement cycle when reached. This document does not detail them.

**No time, duration, or effort estimates appear anywhere in this document.** Sequencing and dependency order are specified; calendar/effort are not. This is a standing project directive.

---

## 1. Driving principles (non-negotiable, inherited project-wide)

These bind every sub-project in the migration. Any subagent prompt derived from this spec must embed them with explicit shortcut+alternative pairs.

1. **The architecture is a multi-field synthesis academia would call impossible. The value is in not stopping when a field-crossing looks too hard.** When a design fork offers an easier-but-architecturally-wrong path and a harder principled path, take the principled path.

2. **Variable binding is genuine entanglement on the bond — never a classical lookup** (architecture doc §8.1). Every encoder, every Hamiltonian, every retarget preserves this.

3. **Never materialize a dense operator at the logic layer's local dimension.** D_LOCAL = 65536. A dense (65536, 65536) operator is ~64 GiB. All operators are factored over the five species (`kind`, `type`, `bid`, `value`, `tobl`) and applied species-by-species.

4. **OOM is a signal to optimize the computation, not a reason to shrink the problem.** Reducing N or χ to dodge an OOM is forbidden; the principled response is a better contraction algorithm or a factored representation.

5. **Every numpy contraction uses `optimize='greedy'`.** Spotting and fixing un-optimized contractions is a proactive duty, not a request.

6. **Systematic debugging for every failure; verification-before-completion for every claim.** No guessing; no "done" without fresh test evidence.

---

## 2. Current state (Phase 0)

Implemented and committed on the branch:

| Layer | Doc § | State |
|---|---|---|
| Classical PCN + QFT substrate | 1–7 | Built (pre-existing). |
| AST ↔ MPS encoder/decoder (sub-project A) | 10.1 | Built. 141 base tests; decoder optimized to one-sweep canonicalization. |
| Typing Hamiltonian (B) | 10.2 | Built. 5-species lattice (D_LOCAL = 65536); 8 STLC typing rules; factored expectation. |
| Evaluation Hamiltonian (C) | 10.3 | Built. Constraint-based; transition couplings; E1–E4 reduction acceptance passing. |
| MERA substrate (F) | 10.4 | Built. Binary 1D MERA; Fibonacci O(log N) scaling test passing. **Tested only at small physical dim (d = 3–4).** |
| LLM bridge (G) | 10.5 | Built. stdio JSON-RPC DSL; MockLLM + AnthropicLLM + OllamaLLM shims. |
| Constraint debugger (D) | 10.6 | Built. Generic `diagnose()` over a `NamedHamiltonianTerm` protocol. |
| STLC synthesis demo (E) | 10.7 | **Partial — 4/8 acceptance.** P1, P2, P5, P8 architecturally pass; P3, P4, P6, P7 blocked. |

### 2.1 Tracked findings (drift and limitations)

These are recorded here rather than in the architecture doc, per the decision to keep the doc as an aspirational north star.

- **F-1 — 5-species drift.** The architecture doc (§10.1) describes a 4-field encoder (`node_kind`, `type`, `binder_id`, `value`). The implementation has 5 species (`kind`, `type`, `bid`, `value`, `tobl`); D_LOCAL = 65536. All downstream specs were translated in-flight onto the 5-species lattice.

- **F-2 — structural-superposition limitation.** `HoleVar.candidates` is `list[str]` (binder names). A hole therefore occupies a single MPS site and can only be completed by a single `Var`. Synthesis problems whose answers are multi-node sub-trees (`f x` = `App(Var, Var)`, `if x<5 then x else x+1`) are unreachable: P3/P4/P6/P7 of sub-project E cannot lift past "smoke". Confirmed by direct sampling — only `f` or `x` are ever drawn for P3, never `f x`. **The architecture doc's §10.7 (`holes at arbitrary positions`) and the encoder spec (`HoleVar.candidates: list[str]`) are mutually inconsistent.** Resolving this is folded into sub-project M4.

- **F-3 — MERA tested only at small d.** Sub-project F's MERA substrate was built and verified at physical dim d = 3–4. Its layer-0 disentanglers and isometries operate on the physical leaf dimension; at D_LOCAL = 65536 they would be (65536², 65536²) tensors. **MERA-as-built cannot hold the logic layer's leaves.** This is the central technical problem M1 solves.

- **F-4 — performance.** Factored expectation at D_LOCAL = 65536 is the dominant cost in B/C/E. Acceptance suites run under relaxed timeouts. Not a correctness issue; an optimization backlog item.

---

## 3. End state and decomposition

The end state is the architecture doc's §10 (roadmap 10.1–10.11) + §11 (composition principles) + §12 (eighteen physics-derived extensions) + §13 (theoretical guarantees). The migration is decomposed into three phases.

### 3.1 Phase 1 — MERA retarget + calculus extension (sub-project M)

Sub-project M retargets the logic layer (A–E) from the 1D MPS substrate to the MERA substrate, and extends the calculus from STLC to an inductive calculus with `Nat`, `List`, equality, universal quantification, and recursion. Recursion gains O(log N) bond dimension via the MERA tree — realizing the architecture doc's §10.4 vision.

M is itself decomposed (it is a substrate retarget of four sub-projects, too large for one spec):

| Piece | Scope | Depends on |
|---|---|---|
| **M1** | **Factored MERA layer-0 + factored MERA expectation.** The foundation. Detailed in §5–§13 of this document. | F (built) |
| **M2** | Extended-calculus AST (`Nat`/`Zero`/`Succ`, `List`/`Nil`/`Cons`, `Eq`, `Forall`, `Fix`) + MERA-native encoder/decoder. Lexical depth → MERA layer depth; recursion → O(log N) via the tree. | M1 |
| **M3** | MERA-native typing + evaluation Hamiltonians. Retargets B and C onto the MERA substrate; adds typing rules (`T-Zero`, `T-Succ`, `T-Nil`, `T-Cons`, `T-Eq`, `T-Forall`, `T-Fix`) and evaluation rules (recursion unfolding, induction-as-ground-state). Reuses the factored term *definitions* from B/C; swaps the expectation backend to M1. | M2 |
| **M4** | MERA-native debugger + synthesis. Retargets D and E. **Absorbs sub-project H — structural superposition over AST sub-trees** (resolves finding F-2): `HoleVar.candidates` becomes `list[Node]`; the encoder allocates multi-site sub-tree superpositions; the decoder samples sub-tree structure. Lifts §10.7 from 4/8 toward 7–8/8. | M3 |

Dependency order within Phase 1: **M1 → M2 → M3 → M4**.

### 3.2 Phase 2 — hierarchical composition (§10.8–10.11)

| Sub-project | Doc § | Scope |
|---|---|---|
| **I** | 10.8 | Lemma library + promotion. Solved ground states cached as type-indexed primitives; `use_lemma` DSL constraint clamps a cached sub-MPS. |
| **J** | 10.9 | Abstraction discovery / wake-sleep. Cluster reduced density matrices of solved sub-states; promote recurring patterns to library primitives. |
| **K** | 10.10 | Cross-level message passing. Goal-graph DAG; parallel sub-QPCN dispatch; bottom-up result integration. |
| **L** | 10.11 | Hierarchical proof composition demo — the second publishable milestone. |

Dependency order: **I → {J, K} → L**. Phase 2 depends on Phase 1 complete (the composition layer caches and composes proof objects produced by the MERA-native logic layer).

### 3.3 Phase 3 — physics-derived extensions (§12)

All eighteen §12 extensions are roadmap entries, sequenced by the architecture doc's own §12.20 priority order. Each is its own spec → plan → implement cycle when reached. Commitment level here is *roadmap*, not *detailed*.

**Tier 1** (build on existing infrastructure):
1. §12.5 Holographic codes
2. §12.13 Bidirectional time evolution
3. §12.11 Modular Hamiltonian / entanglement spectrum
4. §12.6 Goldstone modes
5. §12.17 QCA framework
6. §12.16 Worldline path integral

**Tier 2** (genuine novelty, moderate cost):
7. §12.3 Topological degeneracy
8. §12.2 Topological invariants
9. §12.8 Dynamical phase transitions
10. §12.15 Witten index
11. §12.14 Quantum walks
12. §12.18 Noether discovery

**Tier 3** (built on stable foundations):
13. §12.10 Holographic compilation
14. §12.12 Quantum extremal surfaces
15. §12.1 Anomalies
16. §12.7 Replica method
17. §12.9 Self-modification via meta-Hamiltonian
18. §12.4 Conformal bootstrap

Dependency order: **Tier 1 → Tier 2 → Tier 3**, internally sequenced by §12.20.

### 3.4 Full migration chain

```
M1 → M2 → M3 → M4 → I → {J, K} → L → Tier1(1→2→3→4→5→6) → Tier2(7→…→12) → Tier3(13→…→18)
```

The LLM-vs-QPCN benchmark (existing task #50) becomes runnable once M4 completes (it needs §10.7 synthesis genuinely working and the OllamaLLM shim — the shim is already built).

### 3.5 What is NOT in this spec

- M2, M3, M4 internal design — own specs.
- I, J, K, L internal design — own specs.
- All eighteen §12 extensions' internal design — own specs.
- Updating `QFT_PCN_ARCHITECTURE.md` to absorb findings F-1…F-4 — explicitly out of scope; the doc stays aspirational.

---

## 4. Cross-cutting acceptance for the migration

Independent of any single sub-project, the migration as a whole must preserve:

1. **No regressions.** Every test green before a sub-project starts stays green after it completes (excluding tests a retarget explicitly supersedes, which must be migrated, not deleted).
2. **The §8.1 binding-as-entanglement structural markers** continue to pass on the MERA substrate (the MERA analog of `test_binder_bonds_have_channel_dimension`).
3. **No dense (65536, 65536) operator is ever materialized** in any committed production path.
4. **Cross-substrate agreement.** Where both MPS and MERA can represent the same encoded state, factored expectations agree to ≤ 1e-10.

---

## 5. Sub-project M1 — purpose

**Goal.** Make the MERA substrate able to carry the logic layer's 65536-dimensional physical leaves, and provide a factored-expectation API on MERA matching the MPS-side `_factored_expectation.py`. M1 is the foundation of the entire MERA retarget: M2 (encoder), M3 (Hamiltonians), and M4 (debugger/synthesis) all compute expectations through M1.

**The central problem.** F's MERA represents layer-0 disentanglers as rank-4 tensors `u` of shape `(d_0, d_0, d_0, d_0)` and isometries `w` of shape `(d_1, d_0, d_0)`, where `d_0` is the physical leaf dimension. At `d_0 = D_LOCAL = 65536`, a single disentangler is `65536⁴ ≈ 1.8e19` complex entries. Impossible. F's MERA works only because its tests use `d_0 = 3` or `4`.

**The principled solution.** The logic layer's local Hilbert space is a tensor product of five species: `H_leaf = H_kind ⊗ H_type ⊗ H_bid ⊗ H_value ⊗ H_tobl` with dims `(8, 8, 8, 16, 8)`. Layer-0 disentanglers and isometries that the encoder produces are themselves separable across species (the encoder writes per-species structure; mixing across species only happens at higher layers, in the small coarse-grained space). M1 represents and applies layer-0 tensors in **factored per-species form** — a layer-0 disentangler is a 5-tuple of small per-species disentanglers, never the dense product. Above layer 0, the coarse-grained dimension is `d_ℓ ≤ χ_layer` (≈ 16); those tensors are small dense arrays and need no factorization.

---

## 6. M1 driving principles

In addition to the project-wide principles (§1):

1. **Layer-0 tensors are factored 5-tuples, never dense.** A factored layer-0 disentangler is `(u_kind, u_type, u_bid, u_value, u_tobl)`; each `u_species` is a `(c², c²)` unitary where `c` is that species' cutoff. The dense `(d_0², d_0²)` disentangler is never formed.

2. **The factored structure is a layer-0 property only.** Above layer 0, the coarse-grained space (`d_ℓ ≤ χ_layer`) is small; tensors there are ordinary dense arrays. M1 does not factor layers ≥ 1.

3. **The leaf operator enters factored and stays factored through layer 0.** A factored leaf op (dict of per-species `(c, c)` matrices) ascends through the factored layer-0 disentangler+isometry producing a *small dense* op in the `d_1`-dimensional coarse space. From there, F's existing `_ascend_one_layer` handles layers ≥ 1 unchanged.

4. **Cross-substrate correctness is the anchor.** M1's factored MERA expectation is validated against the MPS-side `_factored_expectation.py` on encoded states both substrates represent. Agreement to ≤ 1e-10 is the acceptance gate.

5. **`_apply_factors_to_site` is the reuse target.** The species-axis einsum pattern in `_factored_expectation.py::_apply_factors_to_site` (reshape into per-species axes, `moveaxis`, contract active species, skip identities) is the proven mechanism. M1's layer-0 application reuses this pattern; it does not invent a new one.

---

## 7. M1 scope

### 7.1 In scope

- A factored representation of layer-0 MERA disentanglers and isometries (`FactoredMERALayer0`, or an extension of `MERATensor`).
- Construction of a factored layer-0 from per-species factors (used by M2's encoder).
- Factored ascent of a leaf operator through factored layer-0 into the `d_1`-dimensional coarse space.
- Four factored-expectation entry points on MERA, matching the MPS-side surface:
  - `factored_mera_local_expectation(state, leaf, op_factors)`
  - `factored_mera_two_site_expectation(state, leaf, op_factors_left, op_factors_right)`
  - `factored_mera_left_bond_expectation(state, leaf, site_op_factors, bond_projector)`
  - `factored_mera_right_bond_expectation(state, leaf, site_op_factors, bond_projector)`
- A cross-substrate correctness test suite (M1 vs `_factored_expectation.py`).

### 7.2 Out of scope (deferred to later M pieces or own specs)

- The MERA-native encoder (M2).
- Extended-calculus AST nodes (M2).
- Typing / evaluation Hamiltonians on MERA (M3).
- Factored MERA *evolution* (imaginary-time gate application) — M3 needs it; M1 provides only expectation. If M3's design shows evolution must be co-designed with expectation, that is M3's call.
- Layers ≥ 1 factorization — explicitly not needed (§6 principle 2).

### 7.3 Will not do

- Materialize a dense `(65536, 65536)` or `(65536², 65536²)` tensor anywhere.
- Modify `qft/mps.py`, `qft/evolution.py`, or `logic/_factored_expectation.py` (M1 is additive; the MPS path is the cross-check oracle and must stay untouched).
- Factor layers ≥ 1.

---

## 8. M1 design

### 8.1 The five species

From `logic/encoding.py` (unchanged, M1 consumes it):

| species | cutoff `c` |
|---|---|
| `kind` | 8 |
| `type` | 8 |
| `bid` | 8 |
| `value` | 16 |
| `tobl` | 8 |

`D_LOCAL = 8·8·8·16·8 = 65536`. Species order is fixed: `(kind, type, bid, value, tobl)`.

### 8.2 Factored layer-0 tensors

**Factored disentangler.** A layer-0 disentangler acting on a pair of leaves is represented as a 5-tuple `(U_kind, U_type, U_bid, U_value, U_tobl)`. Each `U_species` is a `(c², c²)` unitary acting on the *pair* of that species' registers across the two leaves. The dense pair disentangler is the tensor product `⊗_species U_species` (never formed). Per-species unitarity (`U_species† U_species = I_{c²}`) implies the dense disentangler is unitary.

**Factored isometry.** A layer-0 isometry `w` maps a pair `(d_0, d_0)` to one coarse site `d_1`. At `d_0 = 65536`, the coarse `d_1 ≤ χ_layer` is small (e.g. 16). The isometry is *not* factorable in general — it maps a 65536²-dim pair space into a 16-dim coarse space, intrinsically mixing species. **Resolution:** the layer-0 isometry is represented as a `(d_1, d_0², )`-shaped object only in its *action*, never materialized. Concretely, M1 stores the isometry as a function/closure that, given a factored pair operator, returns the `(d_1, d_1)` coarse operator — computed by contracting the pair operator's per-species factors against the isometry's *per-species partial contractions*. The isometry itself is constructed from a small set of `(d_1, c, c, c, c, c)`-shaped rank-6 tensors (one coarse index, five per-species pair-halves) — total size `d_1 · 8 · 8 · 8 · 16 · 8 = d_1 · 65536`, which at `d_1 = 16` is ≈ 1M entries. Tractable.

> **Design note for the implementer.** The isometry representation is the subtle part. The constraint: the isometry must (a) be applyable to a factored pair operator without forming the 65536² pair space, and (b) satisfy `w w† = I_{d_1}`. The rank-6 form `w[A, k, t, b, v, o]` (A = coarse index, k/t/b/v/o = per-species leaf-pair-collapsed indices) is one realization. The implementation plan must pin the exact contraction order; this spec fixes the *representation* (rank-6, never the dense pair space) and the *invariant* (`w w† = I`), and leaves the contraction-order optimization to the plan with `optimize='greedy'`.

### 8.3 Factored ascent of a leaf operator through layer 0

Input: a factored leaf operator — `dict[str, (c, c) ndarray]`, identity for unspecified species (same convention as `_factored_expectation.py::_normalize_factors`).

Step 1 — **pair embedding.** The leaf op acts on one leaf of a pair. Per species, embed `O_species` into the pair as `O_species ⊗ I_c` (leaf is left of pair) or `I_c ⊗ O_species` (leaf is right). Each per-species pair operator is `(c², c²)` — small.

Step 2 — **factored disentangler conjugation.** Per species: `O'_species = U_species · O_pair_species · U_species†`. Five independent small contractions.

Step 3 — **isometry projection into the coarse space.** Contract the five `O'_species` against the rank-6 isometry `w` and its conjugate to produce a single `(d_1, d_1)` dense operator. This is the only step that mixes species; the output is small (`d_1 ≤ χ_layer`).

Step 4 — **dense ascent through layers ≥ 1.** From the `(d_1, d_1)` operator upward, F's existing `MERA._ascend_one_layer` is used unchanged. Layers ≥ 1 are already small dense tensors in F's implementation.

Step 5 — **top contraction.** F's existing top-tensor contraction (the tail of `MERA.local_expectation`) produces the scalar.

### 8.4 The four expectation entry points

Each mirrors a `_factored_expectation.py` function, substrate swapped to MERA:

- `factored_mera_local_expectation(state, leaf, op_factors) -> float` — §8.3 ascent for one leaf.
- `factored_mera_two_site_expectation(state, leaf, op_factors_left, op_factors_right) -> float` — two adjacent leaves; both ascend; if they share a layer-0 pair, ascent is joint through that pair, otherwise independent until their causal cones meet. The implementation plan pins the causal-cone-merge contraction.
- `factored_mera_left_bond_expectation` / `factored_mera_right_bond_expectation` — the MERA analog of the MPS bond-projector expectations. On MERA, the "bond" is an inter-layer index; the bond projector acts on the coarse index between layer 0 and layer 1 in the leaf's causal cone. The implementation plan pins which index.

### 8.5 Files

```
src/qft_pcn/qft/
└── mera_factored.py          # FactoredMERALayer0, factored ascent, the 4 expectation entry points
src/qft_pcn/qft/tests/        # (or src/qft_pcn/tests/ — follow the existing convention)
└── test_mera_factored.py     # unit + cross-substrate tests
```

M1 is additive. It does not modify `mera.py` except possibly to expose `_ascend_one_layer` and the top-contraction tail as reusable (renaming a leading underscore, or adding a thin public wrapper) — the implementation plan decides; if `mera.py` is touched it is interface-exposure only, no behavior change, and F's MERA tests must still pass.

---

## 9. M1 error model

- `FactoredDimMismatch` — a per-species factor has wrong shape for its cutoff.
- `NonUnitaryFactor` — a disentangler factor fails `U† U = I` beyond tolerance.
- `IsometryViolation` (reuse `mera.py`'s existing class) — `w w† ≠ I_{d_1}`.
- `LeafOutOfRange` — leaf index outside `[0, N)`.

All raised eagerly at construction or call entry, never swallowed.

---

## 10. M1 acceptance criteria

M1 is complete when:

1. A `FactoredMERALayer0` can be constructed from per-species factors; per-species unitarity and the isometry invariant `w w† = I_{d_1}` are checked and tested.
2. `factored_mera_local_expectation` agrees with `_factored_expectation.py::factored_local_expectation` to ≤ 1e-10 on a set of encoded states representable on both substrates (the cross-substrate anchor).
3. `factored_mera_two_site_expectation` likewise agrees with the MPS two-site factored expectation.
4. The two bond-projector entry points agree with their MPS counterparts.
5. No test materializes a dense `(65536, …)` operator — verified by a memory-ceiling guard in the test process (`resource.setrlimit`, as already used in `conftest.py`).
6. F's existing MERA tests still pass (M1 did not break the substrate).
7. All four entry points respect the O(log N) causal cone — verified by the `CausalConeViolation` debug wrapper `mera.py` already provides.
8. Every claim of completion is backed by fresh pytest output.

---

## 11. M1 open questions

**None.** The one subtle point — the layer-0 isometry representation — is resolved in §8.2: rank-6 form, never the dense pair space, invariant `w w† = I_{d_1}`. The contraction-order optimization is delegated to the implementation plan (a plan-level decision, not a spec ambiguity).

---

## 12. Contract M1 exposes to M2/M3/M4

- M2's MERA-native encoder produces `FactoredMERALayer0` instances for the physical layer and ordinary dense tensors for layers ≥ 1.
- M3's MERA-native typing/evaluation Hamiltonians call the four `factored_mera_*_expectation` entry points exactly as B/C currently call the `_factored_expectation.py` functions — same signatures, substrate swapped.
- M4's debugger and synthesis consume M3's Hamiltonians; they touch M1 only transitively.

The signatures of the four entry points are frozen by §8.4 so M2/M3 can be specced against them.

---

## 13. Glossary (M1-local)

- **Layer-0** — the physical layer of the MERA tree, whose leaves carry the D_LOCAL = 65536 logic-layer Hilbert space.
- **Factored layer-0 tensor** — a disentangler or isometry represented as per-species factors, never as a dense D_LOCAL-scale array.
- **Coarse space** — the `d_ℓ`-dimensional space at layer ℓ ≥ 1, bounded by `χ_layer`; small enough for ordinary dense tensors.
- **Factored ascent** — lifting a factored leaf operator through factored layer-0 into a small dense operator in the coarse space.
- **Cross-substrate anchor** — the requirement that M1's MERA factored expectation agrees with the MPS factored expectation where both substrates can represent the same state.
