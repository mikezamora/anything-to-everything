# Spec: MERA-Native Logic Encoder (migration sub-project M1)

**Document type**: Implementation specification.
**Date**: 2026-05-22.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Supersedes**: §5–§13 of `docs/superpowers/specs/2026-05-22-mera-migration-design.md` (the old M1 "factored MERA layer-0", and merges in the old M2 "extended-calculus AST + MERA encoder"). The migration roadmap (§1–§4 of that document) stands; only Phase-1's internal decomposition changes — see §2 below.
**Status**: Approved design (leaf layout and ordering chosen by the user); ready for implementation plan.

---

## 0. How to read this spec

This specifies the **MERA-native logic encoder**: the extended-calculus AST, the way an AST is laid onto a MERA tree, the encoder and decoder, and the small expectation helper the downstream Hamiltonians need. It is "sub-project A, redone on the MERA substrate, with an extended calculus."

Everything here is the contract. If something needed during implementation is missing, **stop and ask** — do not guess.

No time, duration, or effort estimates appear in this document. Sequencing and dependency order are specified; calendar/effort are not. Standing project directive.

---

## 1. Driving principles (non-negotiable)

Inherited project-wide (see `2026-05-22-mera-migration-design.md` §1) and specialized here:

1. **The architecture is a multi-field synthesis. Do not stop digging at hard field-crossings.** When a fork offers an easy-but-wrong path and a harder principled path, take the principled path.

2. **Variable binding is genuine entanglement, never a classical lookup** (architecture doc §8.1). On the MERA substrate this means: a binder node's `bid` leaf and each of its use nodes' `bid` leaves are entangled *through the MERA tree tensors* on the path connecting them. For a concrete (hole-free) program this entanglement is a classical correlation realized as a product state; for a hole-bearing program it is genuine tree-carried entanglement. It is never a Python-side `dict[use, binder]` lookup.

3. **No large dense tensor is ever materialized.** Every MERA leaf is ≤16-dimensional by construction (§4). Every disentangler is at most `(16², 16²)`; every isometry at most `(16, 16²)`. There is no 65536-scale object anywhere — the one-species-per-leaf layout (§4) guarantees this structurally.

4. **OOM is a signal to optimize, not to shrink the problem.** Reducing N or χ to dodge an OOM is forbidden.

5. **Every numpy contraction uses `optimize='greedy'`.**

6. **Reuse F's MERA substrate; do not reinvent it.** `src/qft_pcn/qft/mera.py` provides `MERA`, `MERATensor`, `vacuum`, `from_product`, `local_expectation`, `two_site_expectation`, `apply_local_gate`, `apply_two_site_gate`, `inner`, `normalize`, `entanglement_entropy`, `_ascend_one_layer`, the causal-cone helpers, and the `MERAError` hierarchy. This sub-project consumes them. If a genuine bug or missing primitive is found in `mera.py`, fix it there; otherwise build on top.

---

## 2. What this supersedes, and the revised Phase-1 chain

The migration spec decomposed Phase 1 as `M1 (factored MERA layer-0) → M2 (extended-calculus AST + MERA encoder) → M3 (typing/eval Hamiltonians) → M4 (debugger/synthesis + structural superposition)`.

Writing the M1 implementation plan surfaced that **factored MERA layer-0 was solving a self-inflicted problem**: it only existed because the migration spec assumed one AST node per MERA leaf, making each leaf the full 65536-dimensional 5-species space and each layer-0 disentangler/isometry a 65536-scale tensor. The layer-0 isometry could not be represented in memory.

The resolution (chosen by the user): **one species-register per MERA leaf.** Each leaf carries a single species, dimension ≤16. No leaf is ever 65536-dimensional. Layer-0 tensors are then tiny and F's MERA substrate handles them with no factoring. The entire "factored layer-0 / factored expectation" apparatus is unnecessary and is removed from the migration.

**Revised Phase-1 chain:**

| Piece | Scope | Depends on |
|---|---|---|
| **M1** (this spec) | MERA-native logic encoder: extended-calculus AST, node-major species-leaf layout, AST→MERA encoder, MERA→AST decoder, k-adjacent-leaf expectation helper. | F (built) |
| **M2** | MERA-native typing + evaluation Hamiltonians (retarget B+C onto the species-leaf layout; calculus-extension rules). | M1 |
| **M3** | MERA-native debugger + synthesis (retarget D+E; absorbs structural superposition over AST sub-trees). | M2 |

Phases 2 and 3 of the migration roadmap are unchanged.

---

## 3. Scope

### 3.1 In scope

- The **extended-calculus AST**: the existing STLC nodes plus `Zero`, `Succ`, `NatLit` (Peano naturals), `Nil`, `Cons` (lists), `Eq` (propositional equality), `Forall` (schematic universal), `Fix` (recursion). Pure data classes plus parser/pretty-printer extensions.
- The **node-major species-leaf layout** (§4): the deterministic map from an N-node AST to `5N` species-leaves padded to a power of two.
- The **encoder** `encode_mera(ast, ...) -> (MERA, MeraEncodingMeta)`.
- The **decoder** `decode_mera(state, meta) -> DecodeResult` and `sample_mera(...)`.
- The **k-adjacent-leaf expectation helper** `mera_window_expectation(state, leaf0, k, op)` — generalizes F's `two_site_expectation` to `k` adjacent leaves, which M2's per-node (k=5) and two-node (k=10) Hamiltonian terms require.
- An acceptance test suite: round-trip of the §9.1 programs, the binding-as-entanglement structural marker on the tree, alpha-renaming invariance, PAD-leaf vacuum invariant, error paths.

### 3.2 Out of scope (later Phase-1 pieces / own specs)

- Typing rules and the typing Hamiltonian on MERA (M2).
- Evaluation Hamiltonian, recursion unfolding, induction-as-ground-state (M2).
- The constraint debugger and synthesis on MERA (M3).
- Structural superposition over multi-node AST sub-trees (M3).
- Imaginary-time evolution on MERA (M2/M3 decide where it lives).

### 3.3 Will not do

- Materialize any tensor at the old 65536 local dimension.
- One-AST-node-per-leaf layout (the superseded design).
- Five independent per-species MERA trees (cannot represent cross-species hole superpositions — see §5.4).
- Modify `src/qft_pcn/qft/mps.py` or the existing MPS-based `src/qft_pcn/logic/` modules (the MPS logic stack stays as the cross-check oracle and the still-shipped 4/8 synthesizer until M3 supersedes it).

---

## 4. Leaf layout

### 4.1 The five species

From `src/qft_pcn/logic/encoding.py` (consumed unchanged):

| species | cutoff | leaf basis usage |
|---|---|---|
| `kind`  | 8  | basis 0–7 |
| `type`  | 8  | basis 0–7 |
| `bid`   | 8  | basis 0–7 |
| `value` | 16 | basis 0–15 |
| `tobl`  | 8  | basis 0–7 |

### 4.2 Uniform leaf dimension

F's binary MERA requires a uniform leaf dimension. **Every leaf is 16-dimensional** (`MERA_LEAF_DIM = 16`, the maximum species cutoff). A species with cutoff 8 uses basis states 0–7 of its 16-dimensional leaf; states 8–15 are never populated for that species. This wastes a small constant factor and keeps the substrate uniform.

### 4.3 Node-major species-leaf layout

An AST is serialized in canonical pre-order to a list of `N` `NodeOccupancy` descriptors exactly as the MPS encoder does (`src/qft_pcn/logic/_serialize.py`'s walk order — reused). Node `i` then contributes **five consecutive leaves**:

```
leaf 5i+0 : kind   of node i
leaf 5i+1 : type   of node i
leaf 5i+2 : bid    of node i
leaf 5i+3 : value  of node i
leaf 5i+4 : tobl   of node i
```

The total leaf count is `5N`, padded with PAD-leaves (all species in their PAD basis state, the species' index 0) up to the next power of two `N_leaves = 2^L ≥ 5N`. `L = ceil(log2(5N))`.

**Why node-major.** The most common Hamiltonian term (M2's per-node typing rule) couples `kind`, `type`, `bid` of *one* node; node-major puts those within a 5-leaf window — the smallest possible MERA causal cone. Within-node cross-species entanglement (hole superpositions that correlate `type` and `bid`, §5.4) is then adjacent-leaf, the cheapest case for the tree. Binding entanglement (`bid` of a binder node ↔ `bid` of a use node) spans `5·|i−j|` leaves and is carried through `O(log N_leaves)` tree layers — fully supported, just not adjacent.

### 4.4 One tree, not five

The MERA is a **single tree over all `N_leaves` leaves**, not five independent per-species trees. Five separate trees cannot represent a hole superposition that correlates species (§5.4) — the architecture's hole mechanism requires it. The single tree's disentanglers can entangle across species because adjacent leaves (node-major) belong to different species of the same node.

---

## 5. Binding as entanglement on the MERA tree

This is the architectural soul (§1 principle 2). It must be realized, not gestured at.

### 5.1 The `bid` leaves

The `bid` leaf of node `i` is leaf index `5i+2`. Lexical binders (`Lam`, and the extended-calculus binders `Forall`, `Fix`) and variable uses (`Var`) carry a `bid` value identifying which binder is referenced. The encoder resolves every `Var` to its binder by lexical scope (reuse `src/qft_pcn/logic/_resolve.py`).

### 5.2 Concrete programs: product MERA

For a hole-free program, every leaf is in a definite basis state: a binder's `bid` leaf and each of its uses' `bid` leaves hold the *same* definite `bid` index (a depth-relative index, exactly the scheme the MPS encoder used — `bid_local = BID_0 + depth_from_innermost`). The encoded state is therefore a **product state**: F's `MERA.from_product` constructs it directly from the `5N`+pad leaf vectors. Disentanglers are identity; isometries are the canonical embeddings F's `vacuum`/`from_product` already build. No tree entanglement is needed because a classical correlation between definite states is a product state.

This is not a shortcut — it is correct. A concrete program's binding *is* a definite correlation; the entanglement only becomes non-trivial when the program is under-determined.

### 5.3 Hole-bearing programs: genuine tree entanglement

A `HoleVar(candidates=[...])` use site has a `bid` leaf in a superposition over the candidate binders' `bid` indices. The encoded state is then **not** a product state: the hole's `bid` leaf is entangled with the structure of the rest of the program. The encoder realizes this by constructing the MERA so that the `bid` leaf of the hole and the `bid`/`kind` leaves of the candidate binders share entanglement carried by the disentanglers and isometries on the tree paths connecting them.

The construction follows the MPS encoder's principle (Bell-pair-like superposition over candidates) lifted to the tree:
- The hole's `bid` leaf carries the equal-amplitude superposition `(1/√k) Σ_j |bid index of candidate j⟩`.
- The candidate binders' identities are correlated with that superposition through the tree tensors on the connecting paths.

The encoder may build this either analytically (writing the disentangler/isometry tensors in closed form along the connecting paths) or by a gate-based construction (apply CNOT-like two-leaf gates via `MERA.apply_two_site_gate` to entangle the hole's `bid` leaf with the candidates). The implementation plan picks one as production and uses the other as a cross-check (mirroring sub-project A's analytic-vs-gate cross-check).

### 5.4 Why one tree is mandatory

`HoleVar` candidates can differ in *both* `bid` and `type` (`[x:Int, y:Bool]`). The hole's encoded state is then `(1/√2)(|VAR,Int,bid_x⟩ + |VAR,Bool,bid_y⟩)` — the `type` leaf and `bid` leaf of the hole node are *entangled with each other*. Five independent per-species trees factorize across species and cannot represent this. The single tree (§4.4) can, because the hole node's `type` leaf (5i+1) and `bid` leaf (5i+2) are adjacent and a layer-0 disentangler entangles them.

### 5.5 The structural marker

Acceptance test (§9.4): for a hole-bearing program, the MERA's entanglement entropy across a tree cut separating the hole's `bid` leaf from the candidate binders' `bid` leaves is strictly positive. A classical-lookup shortcut (definite `bid` at the hole leaf) would give zero entropy. This is the MERA analog of the MPS encoder's `test_binder_bonds_have_channel_dimension`.

---

## 6. The encoder

### 6.1 API

```python
# src/qft_pcn/logic/mera_encoder.py

@dataclass
class MeraEncodingMeta:
    n_nodes: int                       # AST node count
    n_leaves: int                      # 5*n_nodes padded to a power of 2
    L: int                             # MERA layer count = log2(n_leaves)
    leaf_dim: int                      # 16
    species_of_leaf: list[str]         # length n_leaves; "kind"/"type"/.../"PAD"
    node_of_leaf: list[int]            # length n_leaves; AST node index or -1 for PAD
    site_to_ast_path: dict[int, tuple[int, ...]]
    binder_leaves: dict[int, int]      # AST node index of a binder -> its bid leaf
    use_to_binder: dict[int, int]      # bid leaf of a use -> bid leaf of its binder

def encode_mera(ast: Node,
                n_nodes_max: int = 32,
                chi_layer: int = 16) -> tuple[MERA, MeraEncodingMeta]:
    """Encode an AST into a unit-norm MERA over 5*n_nodes leaves (padded).

    Raises the same exception family as the MPS encoder
    (EncodingTooLarge, TooManyBinders, IntLiteralOutOfRange, IllScopedVar,
    UnsupportedNode) — reuse src/qft_pcn/logic/encoding.py.
    """
```

### 6.2 Pipeline

The encoder reuses the MPS encoder's front half unchanged:

1. `resolve_binders` (`_resolve.py`) — resolve every `Var`/`HoleVar` to its binder(s).
2. `serialize_preorder` (`_serialize.py`) — AST → `N` `NodeOccupancy` descriptors.
3. `compute_site_types` (`_types.py`) — per-node type tags.

Then the MERA-specific back half:

4. Expand each `NodeOccupancy` into five leaf basis vectors (kind, type, bid, value, tobl), each a 16-dim one-hot (or, for a hole leaf, an equal-amplitude superposition).
5. Pad with PAD-leaves to `N_leaves = 2^L`.
6. Build the MERA:
   - Concrete program → `MERA.from_product(leaf_vectors)`.
   - Hole-bearing program → start from `from_product`, then apply §5.3's entangling construction.
7. `state.normalize()`.
8. Assemble `MeraEncodingMeta`.

### 6.3 Extended-calculus nodes

The new AST nodes (`Zero`, `Succ`, `NatLit`, `Nil`, `Cons`, `Eq`, `Forall`, `Fix`) get `kind`-register basis values beyond the current 8. The MERA side defines a **new** constant `MERA_KIND_CUTOFF = 16` in `mera_encoding.py` (it does not mutate `encoding.py`'s `KIND_CUTOFF = 8`, which the MPS stack still imports unchanged); since the leaf is 16-dimensional, up to 16 kinds fit with no structural change. The exact new kind indices are pinned in §8. `Forall` and `Fix` are binders (they introduce a `bid`); `Zero`/`NatLit`/`Nil` are leaves; `Succ`/`Cons`/`Eq` are internal nodes with children. Type tags for `Nat`, `List τ`, `Eq a b`, `Prop` extend the `type` register similarly via a new `MERA_TYPE_CUTOFF = 16` (≤16 fit).

---

## 7. The decoder

```python
def decode_mera(state: MERA, meta: MeraEncodingMeta) -> DecodeResult
def sample_mera(state: MERA, meta: MeraEncodingMeta,
                n_samples: int = 1, rng=None) -> list[DecodeResult]
```

`decode_mera` measures each leaf (argmax of the per-leaf marginal, obtained via F's `MERA.local_expectation` against basis projectors, or a canonical-form sweep), groups leaves into nodes (5 per node), recovers each node's `(kind, type, bid, value, tobl)`, and rebuilds the AST by the structural parse the MPS decoder already implements (reuse the parse logic from `src/qft_pcn/logic/decoder.py`, swapping the per-site read for a per-5-leaf read). `sample_mera` uses MERA conditional sampling for hole-bearing states.

Round-trip on concrete programs is deterministic and exact (§9.1).

---

## 8. Constants and the k-leaf expectation helper

### 8.1 Constants

`src/qft_pcn/logic/mera_encoding.py` (new) defines:
- `MERA_LEAF_DIM = 16`.
- The extended `kind` basis: existing `KIND_PAD…KIND_BIN` (0–7) plus `KIND_ZERO, KIND_SUCC, KIND_NATLIT, KIND_NIL, KIND_CONS, KIND_EQ, KIND_FORALL, KIND_FIX` (8–15).
- The extended `type` basis: existing tags plus `TYPE_NAT, TYPE_LIST, TYPE_EQ, TYPE_PROP` and any remaining arrow slots, within 16.
- `SPECIES_ORDER = ("kind", "type", "bid", "value", "tobl")` and the per-node leaf offsets.

These extend `encoding.py`'s constants; they do not modify them (the MPS stack keeps importing `encoding.py` unchanged).

### 8.2 `mera_window_expectation`

```python
def mera_window_expectation(state: MERA, leaf0: int, k: int,
                            op: np.ndarray) -> complex:
    """<state | O | state> for an operator O on the k adjacent leaves
    [leaf0, leaf0+k). op has shape (16**k, 16**k).

    Generalizes MERA.two_site_expectation (k=2) to arbitrary k. M2's
    per-node typing terms use k=5; two-node terms use k=10.
    Contraction stays within the O(log N) causal cone of the window.
    """
```

The implementation ascends the window through the MERA causal cone, generalizing F's `two_site_expectation`. For `k` up to 10 and leaf dim 16, `16**k` is large (`16**10 ≈ 1.1e12`) — so the operator must **not** be materialized densely for large k. M2's terms are themselves factored (one small per-leaf operator per species-leaf); `mera_window_expectation` therefore also accepts a factored form:

```python
def mera_window_expectation_factored(state: MERA, leaf0: int,
                                     leaf_ops: dict[int, np.ndarray]) -> complex:
    """leaf_ops maps an absolute leaf index to a (16, 16) operator;
    unlisted leaves in the window are identity. No 16**k tensor is formed.
    """
```

The factored form is the one M2 actually calls. The dense `mera_window_expectation` exists for k≤2 cross-checks against F's `two_site_expectation`.

---

## 9. Acceptance tests

`src/qft_pcn/tests/test_mera_encoder.py`.

### 9.1 Round-trip

The five STLC demo programs plus three extended-calculus programs:

```
P1.  \x:Int. x
P2.  (\x:Int. x + 1)(2)
P3.  \f:Int->Int. \x:Int. f (f x)
P4.  if (1 < 2) then ((\x:Bool. x)(true)) else false
P5.  (\x:Int. (\y:Int. x + y)(3))(4)
P6.  Succ (Succ Zero)                              # Nat literal 2
P7.  Cons 1 (Cons 2 Nil)                           # List [1, 2]
P8.  \x:Nat. Eq x x                                # reflexivity instance
```

Each: `decode_mera(*encode_mera(parse(p))) ` is alpha-equivalent to `p`; the encoded MERA is unit-norm; residual < 1e-10.

### 9.2 Layout

`encode_mera` of an N-node AST produces a MERA with `5N` non-PAD leaves padded to a power of two; `meta.species_of_leaf` is the node-major repeating pattern; PAD leaves are all in PAD basis states.

### 9.3 Alpha-renaming

`\x:Int. x` and `\y:Int. y` encode to MERA states with `|inner|² > 1 − 1e-10`.

### 9.4 Binding-as-entanglement structural marker

For a hole-bearing program (`\x:Int. \y:Int. ?HOLE` with `HoleVar(["x","y"])`), the MERA's `entanglement_entropy` across a cut separating the hole's `bid` leaf from the candidate binders' `bid` leaves is `> 0.5`. A classical-lookup encoding would give 0. (Upper bound: `ln(candidate count)` plus the within-node cross-species contribution.)

### 9.5 PAD vacuum

Every PAD leaf has zero amplitude on any non-PAD basis state (checked via `MERA.local_expectation` against the project-away-from-PAD projector — the operator is 16×16, trivially fine).

### 9.6 Error paths

`EncodingTooLarge`, `TooManyBinders`, `IntLiteralOutOfRange`, `IllScopedVar`, `UnsupportedNode` raised on the corresponding malformed inputs.

### 9.7 Cross-substrate anchor

For each of P1–P5 (representable on both substrates), the MERA encoding and the existing MPS encoding decode to alpha-equivalent ASTs. This anchors the MERA encoder against the shipped, tested MPS encoder.

### 9.8 k-leaf expectation cross-check

`mera_window_expectation` with `k=2` agrees with F's `MERA.two_site_expectation` to ≤ 1e-10 on encoded states.

---

## 10. Error model

Reuse `src/qft_pcn/logic/encoding.py`'s exception family (`EncodingError` and subclasses). Add, in `mera_encoding.py`:
- `MeraLeafBudgetExceeded` — `5·n_nodes` padded exceeds a configured ceiling.

The encoder is total on well-scoped extended-calculus ASTs within the node budget; it raises rather than silently truncating.

---

## 11. File layout

```
src/qft_pcn/logic/
├── mera_encoding.py        # constants: MERA_LEAF_DIM, extended kind/type basis, SPECIES_ORDER
├── mera_encoder.py         # encode_mera, MeraEncodingMeta
├── mera_decoder.py         # decode_mera, sample_mera
├── ast.py                  # MODIFY: add Zero, Succ, NatLit, Nil, Cons, Eq, Forall, Fix + parser/pretty
└── _mera_window.py         # mera_window_expectation, mera_window_expectation_factored
src/qft_pcn/tests/
└── test_mera_encoder.py    # the §9 acceptance suite
```

`ast.py` is modified (extended-calculus nodes). All other files are new. `mps.py`, `evolution.py`, `mera.py`, and the MPS-based `logic/` modules are not modified — except `mera.py` may gain a thin public wrapper if `mera_window_expectation` needs an internal helper exposed (interface-only, no behavior change, F's MERA tests must still pass).

---

## 12. Acceptance criteria

M1 is complete when:

1. The eight round-trip programs (§9.1) encode→decode to alpha-equivalence with residual < 1e-10.
2. The layout test (§9.2) passes.
3. Alpha-renaming invariance (§9.3) passes.
4. The binding-as-entanglement structural marker (§9.4) passes — proving §1.1 is realized on the tree.
5. PAD vacuum (§9.5) passes.
6. All five error-path tests (§9.6) raise the right exception.
7. The cross-substrate anchor (§9.7) passes for P1–P5.
8. `mera_window_expectation` agrees with F's `two_site_expectation` at k=2 (§9.8).
9. F's existing MERA tests still pass; the MPS logic stack's tests still pass.
10. No test materializes a tensor larger than `16**2` densely (the factored window path is used for k>2). Verified by the `conftest.py` memory ceiling.
11. Every completion claim is backed by fresh pytest output.

---

## 13. Open questions

**None.** The leaf layout (one species per leaf), ordering (node-major), single-tree decision, and the concrete-vs-hole encoding split are all resolved above. The analytic-vs-gate choice for §5.3's hole entanglement is delegated to the implementation plan (a plan-level decision, not a spec ambiguity), exactly as sub-project A delegated it.

---

## 14. Contract M1 exposes to M2/M3

- M2's typing/evaluation Hamiltonians act on the species-leaf layout: a per-node typing rule is a factored operator on the 5 leaves `[5i, 5i+5)`; a two-node rule spans `[5i, 5i+5) ∪ [5j, 5j+5)`. M2 calls `mera_window_expectation_factored` with per-leaf 16×16 operators.
- `MeraEncodingMeta` exposes `binder_leaves`, `use_to_binder`, `node_of_leaf`, `species_of_leaf`, `site_to_ast_path` — everything M2's rule compiler and M3's debugger need to map energies back to AST locations.
- The extended `kind`/`type` basis in `mera_encoding.py` is the vocabulary M2's calculus-extension typing rules are written against.

---

## 15. Glossary (M1-local)

- **Species-leaf** — one MERA leaf carrying a single species register of one AST node.
- **Node-major layout** — leaf ordering where the five species-leaves of one AST node are consecutive.
- **Window** — `k` consecutive leaves; a per-node window is 5 leaves, a two-node window 10.
- **Concrete program** — a hole-free AST; encodes to a product MERA.
- **Hole-bearing program** — an AST with `HoleVar`/`TypeHole`; encodes to a genuinely tree-entangled MERA.
- **Binding-as-entanglement marker** — the §9.4 test: positive tree entanglement entropy across a hole-bid / binder-bid cut.
