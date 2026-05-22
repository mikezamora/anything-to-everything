# Spec: MERA-Native Typing + Evaluation Hamiltonians (migration sub-project M2)

**Document type**: Implementation specification.
**Date**: 2026-05-22.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Supersedes**: the MERA-retargeting clauses of sub-projects B and C. The
MPS-era specs `docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md`
(B) and `2026-05-21-eval-hamiltonian-design.md` (C) remain the rule-content
authority — M2 retargets their *substrate* from the 1D MPS lattice to the
MERA tree built in M1, keeps every rule's mathematical definition, and adds
the calculus-extension rules. The MPS Hamiltonians (`typing_hamiltonian.py`,
`evaluation_hamiltonian.py`) are NOT modified; they stay as the cross-check
oracle until M3 supersedes them.
**Depends on**: M1 — `docs/superpowers/specs/2026-05-22-mera-native-encoder-design.md`,
implemented in `src/qft_pcn/logic/{mera_encoder,mera_decoder,_mera_window,
mera_encoding}.py`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` — §8 (logic correspondence),
§10.2 (type→Hamiltonian), §10.3 (evaluation as ground-state finding, path b),
§10.4 (MERA for recursion), §13 (theoretical guarantees).
**Status**: Approved design; ready for implementation plan.

---

## 0. How to read this spec

This specifies the **MERA-native typing and evaluation Hamiltonians**: the
operators whose ground states are, respectively, "well-typed program" and
"well-typed AND fully-reduced program," evaluated against the MERA state M1
produces, plus the imaginary-time evolution that drives reduction on the tree,
plus the typing and evaluation rules for the extended-calculus nodes
(`Zero`, `Succ`, `NatLit`, `Nil`, `Cons`, `Eq`, `Forall`, `Fix`).

It is "sub-projects B and C, redone on the MERA substrate, with an extended
calculus."

Everything here is the contract. If something needed during implementation is
missing, **stop and ask** — do not guess. If a section feels complicated and
you want to "loosen it up," that is the signal to look harder for the right
encoding, not to soften the rule.

No time, duration, or effort estimates appear in this document. Sequencing and
dependency order are specified; calendar/effort are not. Standing project
directive.

---

## 1. Driving principles (non-negotiable)

Inherited project-wide (see `2026-05-22-mera-native-encoder-design.md` §1 and
the B/C specs §1) and specialized here. **None may be traded away for
implementation simplicity.** If you find yourself tempted to violate one, stop
and ask the human.

### 1.1 No time or effort estimates

Anywhere. Critical standing directive. Sequencing and dependency order are in
scope; calendar, duration, "effort," and "complexity" labels are not.

### 1.2 Binding is genuine entanglement, never a classical lookup

A typing rule that couples a binder to its uses (T-Var, and the new T-Fix
recursion-use rule) reads the binder/use correlation **through the MERA tree
structure** — it evaluates a factored operator on the `bid` leaves of the
binder node and the use node, and the correlation between those leaves is
carried by the disentanglers and isometries on the tree path connecting them
(M1 §5). It is **never** a Python-side `dict[use, binder]` lookup. M2 may read
`MeraEncodingMeta.use_to_binder` to know *which two leaves* a term acts on —
that is addressing, not evaluation — but the energy itself is always
`⟨ψ| O |ψ⟩` of a real Hermitian operator on the real tree-entangled state.

  - **Shortcut (forbidden):** "T-Var at use node `j`: look up
    `meta.use_to_binder[5j+2]`, read the binder's classical `param_ty` from
    `meta`, compare to the use node's classical `type`, add 1 if they differ."
    This smuggles classical type-checking into the operator and fails on
    hole-bearing MERA states where neither `type` nor `bid` is definite.
  - **Principled alternative (required):** T-Var is a factored operator on the
    5 leaves of the use node (kind/type/bid) plus, where the rule needs the
    binder's param_ty, the 5 leaves of the binder node — evaluated via M1's
    `mera_window_expectation_factored` against the genuinely-encoded MERA. On
    a concrete program the binder/use correlation is a definite (product-state)
    correlation; on a hole-bearing program it is genuine tree entanglement; the
    *same operator* handles both.

### 1.3 No dense operator at scale

Every MERA leaf is 16-dimensional (M1 §4.2). A per-node typing term spans 5
leaves; a two-node term spans 10. The dense operator on 10 leaves is
`16**10 ≈ 1.1e12` — it must **never** be materialized. Every Hamiltonian term
is a **factored operator**: a `dict[leaf_index -> (16,16) matrix]`, identity on
unlisted leaves, evaluated via M1's `mera_window_expectation_factored`. The
dense `mera_window_expectation` is used only for `k ≤ 2` cross-checks against
F's `two_site_expectation`.

  - **Shortcut (forbidden):** building a `16**5 × 16**5` dense per-node
    operator "because 5 leaves is small enough." It is not — `16**5 ≈ 1e6`,
    its square is `1e12` entries. Use the factored form even at k=5.

### 1.4 OOM means optimize, not shrink

Reducing the node budget, the leaf dimension, or `chi_layer` to dodge an
out-of-memory is forbidden. An OOM is a signal that a tensor is being
materialized that should be factored, or that an einsum is missing
`optimize='greedy'`.

### 1.5 `optimize='greedy'` on every einsum

Every `np.einsum` M2 introduces passes `optimize='greedy'`. No exceptions.

### 1.6 Constraint-based evaluation only (architecture §10.3 path b)

Reduction IS ground-state finding. The evaluation Hamiltonian penalizes any
non-normal-form configuration; imaginary-time evolution drives the MERA state
toward the ground state, which is the normal form. There is **no** explicit
beta-reduction rewriter, **no** classical interpreter, and **no** Lindblad
dissipative path (architecture §10.3 path a is rejected). If you find yourself
reaching for `substitute(body, x, arg)` or a density-matrix master equation,
stop.

  - **Shortcut (forbidden):** pattern-matching `App(Lam, arg)` in Python,
    substituting, and writing the result back into the MERA. That is a
    classical interpreter wearing a tree hat; it cannot reduce hole-bearing
    states and defeats the architecture.
  - **Principled alternative (required):** every evaluation term is a
    Hermitian penalty operator; reduction is observed by measuring the MERA
    after `mera_imaginary_evolve(state, H, ...)`.

### 1.7 The Hamiltonian operates on the MERA state via factored expectation

NO Python AST walk happens during Hamiltonian construction or evaluation. The
Hamiltonian is **structural**: `MeraTypingHamiltonian(meta)` and
`MeraEvalHamiltonian(meta)` are built from `MeraEncodingMeta` — the leaf
layout, the species-of-leaf table, the binder/use leaf bookkeeping — and
nothing else. The same instance evaluates any MERA state over the same layout.
`meta` is read for *addressing* (which leaves a term touches), never for the
program's classical type-checking result.

  - **Shortcut (forbidden):** a per-AST Hamiltonian whose coefficients are
    populated by walking the input program. It produces the right ⟨H⟩ for
    concrete inputs but cannot evaluate hole-bearing MERA states (M3's
    structural superposition) and cannot serve as the energy functional of
    imaginary-time evolution — which is the entire point.

### 1.8 Reuse M1's substrate and F's MERA; do not reinvent

M2 consumes `mera_window_expectation_factored`, `MeraEncodingMeta`,
`encode_mera`, `decode_mera`, and F's `MERA` primitives
(`apply_local_gate`, `apply_two_site_gate`, `copy`, `inner`, `normalize`,
`norm_sq`, `entanglement_entropy`). If a genuine missing primitive or bug is
found, fix it in `mera.py` / M1; otherwise build on top. M2 does not modify the
MPS logic stack.

---

## 2. What this supersedes, and the M1→M2→M3 chain

M1's revised Phase-1 chain (M1 spec §2):

| Piece | Scope | Depends on |
|---|---|---|
| **M1** (built) | MERA-native logic encoder: extended-calculus AST, node-major species-leaf layout, encoder/decoder, k-leaf window expectation. | F |
| **M2** (this spec) | MERA-native typing + evaluation Hamiltonians: retarget B+C onto the species-leaf layout; calculus-extension rules; MERA imaginary-time evolution. | M1 |
| **M3** | MERA-native debugger + synthesis: retarget D+E; structural superposition over AST sub-trees. | M2 |

M2 does not change M1's leaf layout, the encoder/decoder, or the window
helpers. It consumes them. It does not change the MPS-era B/C rule content;
it re-expresses each rule as a factored operator on the MERA leaf layout and
adds the extended-calculus rules.

---

## 3. Scope

### 3.1 In scope

- **`MeraTypingHamiltonian(meta)`** — mirrors B's `TypingHamiltonian` API
  (`.terms`, `.term_energy`, `.residuals`, `.total_energy`) but operates on a
  MERA state through M1's factored window expectation. Reuses B's per-species
  rule term *definitions*; what changes is the expectation backend (MERA
  factored window instead of the MPS factored-expectation path) and the
  leaf-addressing (5 leaves per node instead of 1 site per node).
- **`MeraEvalHamiltonian(meta)`** — mirrors C's `EvalHamiltonian` API.
  Constraint-based redex penalties (beta, arithmetic, comparison, if) plus the
  transition couplings that drive reduction.
- **`compose_mera_hamiltonians(*hams)`** — operator sum at the expectation
  level, mirroring `compose.py::compose_hamiltonians`.
- **MERA factored imaginary-time evolution** — `mera_imaginary_evolve` and a
  single-step `mera_trotter_step` — the MERA analog of
  `factored_evolution.py::factored_evolve`. Per-term factored gates applied to
  the MERA state via F's `apply_two_site_gate` / `apply_local_gate`.
- **Extended-calculus typing rules**: T-Zero, T-Succ, T-NatLit, T-Nil, T-Cons,
  T-Eq, T-Forall, T-Fix (§6).
- **Extended-calculus evaluation rules**: `Fix` recursion unfolding (§7.2) and
  induction-as-ground-state for `Forall` over `Nat`/`List` (§7.3).
- An acceptance test suite (§9): well-typed STLC + extended-calculus programs
  give ⟨H_typing⟩ ≈ 0; ill-typed give > 0 with attributable per-rule
  residuals; E1-E4-style reduction on the MERA substrate; a recursive `Fix`
  program reduces with O(log N) tree depth.

### 3.2 Out of scope (M3 / own specs)

- The constraint debugger on MERA (M3, retargets D).
- Synthesis with holes-in-evaluable-position on MERA (M3, retargets E).
- Structural superposition over multi-node AST sub-trees (M3).
- Polymorphism, general type variables (parametric `Forall` is in; full System
  F is not).
- IO / side effects (architecturally non-unitary; deferred indefinitely).

### 3.3 Will not do

- Modify the MPS logic stack (`typing_hamiltonian.py`,
  `evaluation_hamiltonian.py`, `compose.py`, `factored_evolution.py`,
  `_factored_expectation.py`, `_eval_terms.py`) or `mps.py`. They remain the
  cross-check oracle.
- Materialize any dense operator larger than `16**2` (the factored window path
  is used for k>2).
- Implement the Lindblad dissipative evaluation path (architecture §10.3 a).
- Implement an explicit beta-reduction rewriter or classical interpreter.
- Build a per-AST Hamiltonian (the operator is structural — §1.7).

---

## 4. The MERA leaf layout M2 builds on (M1 contract recap)

From M1 §4 and §14, fixed and consumed unchanged:

- **One species per leaf**, 16-dimensional. `MERA_LEAF_DIM = 16`.
- **Five leaves per AST node, node-major.** Node `i` occupies leaves
  `[5i, 5i+5)`:

  ```
  leaf 5i+0 : kind   of node i
  leaf 5i+1 : type   of node i
  leaf 5i+2 : bid    of node i
  leaf 5i+3 : value  of node i
  leaf 5i+4 : tobl   of node i
  ```

- Total leaf count `5N` padded to `N_leaves = 2^L`, with PAD-leaves in the
  PAD basis state (species index 0).
- `MeraEncodingMeta` exposes: `n_nodes`, `n_leaves`, `L`, `leaf_dim`,
  `species_of_leaf`, `node_of_leaf`, `site_to_ast_path`, `binder_leaves`
  (`AST node index -> its bid leaf`), `use_to_binder` (`use's bid leaf ->
  binder's bid leaf`), `nested_type_index`, `layout` (a `MeraLayout` with
  `leaf_of(node, species) -> int`).
- The extended kind/type basis (`mera_encoding.py`): base kinds 0–7
  (`KIND_PAD..KIND_BIN`) plus `KIND_ZERO=8, KIND_SUCC=9, KIND_NATLIT=10,
  KIND_NIL=11, KIND_CONS=12, KIND_EQ=13, KIND_FORALL=14, KIND_FIX=15`; base
  type tags plus `TYPE_NAT=8, TYPE_LIST=9, TYPE_EQ=10, TYPE_PROP=11`.

### 4.1 The two term geometries

Every M2 Hamiltonian term is one of two shapes:

- **Per-NODE term** — a factored operator on the 5 leaves `[5i, 5i+5)` of one
  node. Used by every one-site rule (T-Lit-Int, T-Zero, T-Succ, the arithmetic
  redex flag, …). Evaluated by `mera_window_expectation_factored(state,
  {leaf: op, ...})` with at most 5 entries.
- **Two-NODE term** — a factored operator spanning `[5i,5i+5) ∪ [5j,5j+5)`.
  Used by every rule that couples two AST nodes: T-App (fn node ↔ arg node),
  T-Var / T-Fix-use (use node ↔ binder node), the beta redex (APP node ↔ LAM
  node), the arithmetic redex (BIN node ↔ operand nodes). Evaluated by
  `mera_window_expectation_factored` with entries on both nodes' relevant
  leaves. The two node windows need not be adjacent; the factored expectation
  handles arbitrary leaf sets because per-leaf operators on distinct leaves
  commute (M1 `_mera_window.py` docstring).

Because the factored expectation accepts an arbitrary `dict[leaf -> (16,16)]`,
**there is no "long-range" problem** on the MERA: a term coupling node `i` and
node `j` is just a factored operator listing leaves from both. The MERA tree
carries the correlation; M2 only supplies the per-leaf operators and the leaf
indices (read from `meta.layout` / `meta.use_to_binder`).

### 4.2 Why no `tobl`-style obligation register hack is needed

The MPS-era B used a `tobl` register (B §1.4, §3.6) because the MPS could only
couple adjacent sites, so "child-of-X must have type Y" had to be collapsed to
a per-site projector against a pre-written obligation tag. On the MERA the
parent node and the child node are both addressable leaf windows, and a
two-node factored operator couples them directly. M2 therefore expresses
obligations as **genuine two-node terms** (T-App-Arg, T-If-Cond, T-Bin-Operand,
etc.), reading the parent's `type`/`value` leaves and the child's `type` leaf
in one factored operator. The `tobl` leaf still exists in the M1 layout (it is
the 5th leaf and the encoder writes it for cross-substrate fidelity), and M2's
rules MAY additionally read it, but the **obligation coupling is realized as a
two-node tree-carried term, not as a one-site read of a pre-baked tag**. This
is the principled path the MERA substrate makes available; taking the
one-site-`tobl`-read shortcut would discard the tree's coupling power and fail
on hole-bearing states whose `type` leaves are superposed.

---

## 5. `MeraTypingHamiltonian` — retargeting B

### 5.1 API

```python
# src/qft_pcn/logic/mera_typing_hamiltonian.py

@dataclass(frozen=True)
class MeraTypingTerm:
    """A single typing-rule term, addressable by (rule_id, node, arity)."""
    rule_id: str          # see §5.3 for the rule-name set
    node: int             # the PRIMARY AST node index the rule is anchored at
    arity: int            # 1 (per-node) or 2 (two-node)

class MeraTypingHamiltonian:
    """Structural typing Hamiltonian over a MERA leaf layout.

    Built from MeraEncodingMeta only (§1.7). The same instance evaluates
    <H_typing> on any MERA state encoding an AST of size <= meta.n_nodes.
    """
    def __init__(self, meta: MeraEncodingMeta): ...
    terms: list[MeraTypingTerm]
    def term_energy(self, state: MERA, term: MeraTypingTerm) -> float: ...
    def total_energy(self, state: MERA) -> float: ...
    def residuals(self, state: MERA) -> dict[tuple[str, int], float]: ...
```

`term_energy` returns `⟨ψ|H_term|ψ⟩`, a real non-negative number (every term
is a sum of projectors — PSD). `total_energy = Σ term_energy`.
`residuals` returns every per-term energy keyed `(rule_id, node)` — the
diagnostic granularity M3's debugger consumes.

`term_energy` raises `MeraTermNotFound` for a term not in `self.terms`
(mirrors B's `TermNotFound`).

### 5.2 Term geometry and the factored backend

Every rule is implemented as a function
`_energy_<rule>(state, meta, node) -> float` that:

1. Reads `meta.layout.leaf_of(node, species)` to find the leaf indices it
   needs (and, for two-node rules, the partner node via `meta.use_to_binder`
   or the layout's child-node bookkeeping — see §5.4).
2. Builds a `dict[leaf -> (16,16) np.ndarray]` of per-leaf projectors
   (identity on unlisted leaves).
3. Calls `mera_window_expectation_factored(state, leaf_ops)` and returns the
   real part.

The per-leaf projector helpers (`_project(16, idx)`, `_project_one_minus`,
`_project_set`) are 16×16 — identical in spirit to B's helpers but sized to
the uniform 16-dim leaf. The dense `16**5`/`16**10` operator is never formed
(§1.3).

For a rule whose operator is a *sum* of projector products (e.g. T-Obligation
ranges over `t`), the energy is the sum of the per-`t` factored-window
expectations — exactly as B's `_energy_t_obligation` sums over `t`.

### 5.3 The rule set

M2's typing rules are B's eight STLC rules **plus** eight extended-calculus
rules. The STLC rules keep B's exact mathematical content (B §3); only the
substrate changes. Rule names (stable; M3's debugger keys on them):

| rule_id | arity | anchored at | content |
|---|---|---|---|
| `T-Lit-Int` | 1 | the INT node | B §3.1 |
| `T-Lit-Bool` | 1 | the BOOL node | B §3.2 |
| `T-Bin-Arith` | 1 | the BIN node | B §3.3a |
| `T-Bin-Cmp` | 1 | the BIN node | B §3.3b |
| `T-Var` | 2 | the VAR node ↔ its binder node | §5.4 |
| `T-Abs` | 2 | the LAM node ↔ its body node | §5.4 |
| `T-App-Arrow` | 2 | the APP node ↔ its fn node | B §3.7 |
| `T-Obligation` | 2 | a parent node ↔ a child node | §5.5 |
| `T-Zero` | 1 | the ZERO node | §6.1 |
| `T-Succ` | 2 | the SUCC node ↔ its arg node | §6.2 |
| `T-NatLit` | 1 | the NATLIT node | §6.3 |
| `T-Nil` | 1 | the NIL node | §6.4 |
| `T-Cons` | 2 | the CONS node ↔ head/tail nodes | §6.5 |
| `T-Eq` | 2 | the EQ node ↔ lhs/rhs nodes | §6.6 |
| `T-Forall` | 2 | the FORALL node ↔ its body node | §6.7 |
| `T-Fix` | 2 | the FIX node ↔ its body node | §6.8 |

`H_typing = Σ_rules Σ_nodes H_rule(node)`. For a well-typed AST every term is
zero; for an ill-typed AST each locally-violating node contributes `+1` to
exactly one rule term (architecture §13.2 — energy-zero ground states satisfy
all constraints).

### 5.4 Two-node binder/use rules (T-Var, T-Abs, T-Fix)

T-Var (B §3.4): the VAR node's `type` must equal its binder's parameter type.
On the MERA this is a **two-node factored term** over the use node `j` and the
binder node `i = meta` ’s binder for `j`:

- The use node `j`'s `bid` leaf is `5j+2`; the binder node `i`'s `bid` leaf is
  `5i+2`. `meta.use_to_binder[5j+2] = 5i+2` gives the partner leaf — that is
  *addressing*, not lookup (§1.2).
- The rule penalizes the joint configuration "use node `j` is `KIND_VAR` AND
  binder node `i` has `type` tag `t` AND use node `j` has `type` tag `≠ t`,"
  summed over `t`. The factored operator lists: `5j+0` (kind = VAR),
  `5i+1` (type = t projector), `5j+1` (I − type = t projector). Three leaves;
  the bid leaves carry the binder/use correlation through the tree, so the
  term is non-trivial exactly when the encoded state binds `j` to `i`.

This is the §1.2 realization on the tree: the operator does not look up the
binder's type in `meta`; it reads leaf `5i+1` of the encoded state, and the
encoded state's binder/use entanglement (M1 §5) is what makes the joint
projector fire only on the genuinely-bound pair.

T-Abs (B §3.5): the LAM node's `type` must be an arrow whose `src` equals the
LAM's parameter type and whose `dst` equals the body node's `type`. Two-node
term over the LAM node and its body node (the body node index is the LAM's
first-child index in the pre-order layout — recoverable from
`meta.site_to_ast_path` / the layout, see §5.6).

T-Fix (§6.8): same binder/use machinery as T-Var, but the recursion variable
may be used inside the body; the rule couples the FIX node's parameter type to
each recursion-use node's type.

### 5.5 Obligation rules as two-node terms

The "child must have type X" constraints (LAM-body, APP-arg, IF-branches,
BIN-operands, SUCC-arg, CONS-head/tail, EQ-sides) become **genuine two-node
factored terms** (§4.2), each named `T-Obligation` and anchored at the parent
node (the `node` field of the `MeraTypingTerm` is the parent; the partner child
node is found from the layout). The term penalizes "parent node has
kind/value `K` AND child node has `type ≠ expected(K)`."

Because the MERA can couple non-adjacent leaf windows directly, there is no
need for a pre-baked `tobl` tag. The encoder still writes `tobl` (M1 layout),
and M2's `T-Obligation` MAY read it as a redundant consistency check, but the
load-bearing coupling is parent-node `type`/`value` leaves against child-node
`type` leaf. Per-`(parent, child)` pairs are enumerated from the layout.

### 5.6 Finding child nodes from the layout

The pre-order serialization (M1 reuses `_serialize.py`) lays node `i`'s
children contiguously after `i`. M2's term enumerator needs, per parent node,
the indices of its children. Two acceptable realizations:

- **(a)** Read child structure from `meta.site_to_ast_path`: a node whose
  `ast_path` extends the parent's by one element is a child. This is *layout
  addressing*, not an AST walk (§1.7) — `site_to_ast_path` is metadata M1
  already produced.
- **(b)** Extend `MeraEncodingMeta` with a `children_of_node: dict[int,
  list[int]]` field, populated by the M1 encoder's existing pre-order walk.

The implementation plan picks (b) — a small additive `MeraEncodingMeta` field
populated by the M1 encoder — because it is unambiguous and keeps M2's term
enumerator from re-deriving structure. This is a documented, additive change to
M1's `MeraEncodingMeta` and `encode_mera`; M1's existing tests must still pass
(the new field is populated, never read by M1).

### 5.7 What changes vs. B, precisely

- **Unchanged:** the rule mathematics — every projector content (which kind,
  which type tag, the `I − P` violation form) is copied from B §3.
- **Changed:** the expectation backend — `mera_window_expectation_factored`
  instead of `_factored_expectation.py`; per-leaf 16×16 operators instead of
  per-species 8/16-dim operators embedded into a 65536 local space.
- **Changed:** addressing — a node is 5 leaves, not 1 MPS site; a binder/use
  coupling is a two-node leaf set, not a bid *bond*. B's bid-bond
  `param_ty`-channel apparatus (B §5.2) is **gone** — on the MERA the binder's
  param type is just leaf `5i+1` of the binder node, read directly, with the
  tree carrying the correlation. This is strictly simpler and strictly more
  principled.
- **Added:** the eight extended-calculus rules (§6).

---

## 6. Extended-calculus typing rules

The surface calculus extends STLC with Peano naturals, lists, propositional
equality, schematic universals, and recursion (M1 §3.1). Each rule below is a
factored operator; "node" means an AST node, "leaf `5i+s`" its species leaf.
`P[species, idx]` is the 16×16 projector `|idx⟩⟨idx|` on that node's species
leaf; `Pbar = I − P`.

### 6.1 T-Zero (per-node)

`Zero : Nat`. At a `KIND_ZERO` node, the `type` leaf must be `TYPE_NAT`.

```
H_T-Zero(i) = P[kind=KIND_ZERO](5i) · Pbar[type=TYPE_NAT](5i+1)
```

Factored operator: `{5i: P_kind_ZERO, 5i+1: I − P_type_NAT}`.

### 6.2 T-Succ (two-node)

`Succ e : Nat` given `e : Nat`. Two-node term over the SUCC node `i` and its
arg node `c` (its single child). Two contributions, both penalty projectors:

```
H_T-Succ(i) = P[kind=KIND_SUCC](5i) · Pbar[type=TYPE_NAT](5i+1)        # result type
            + P[kind=KIND_SUCC](5i) · Pbar[type=TYPE_NAT](5c+1)        # arg type
```

Each is a separate factored-window expectation; the energy is their sum.

### 6.3 T-NatLit (per-node)

`NatLit n : Nat`. `NatLit` is sugar for `Succ^n Zero` but encodes to a single
leaf-node (M1 §3.1). The rule mirrors T-Zero:

```
H_T-NatLit(i) = P[kind=KIND_NATLIT](5i) · Pbar[type=TYPE_NAT](5i+1)
```

### 6.4 T-Nil (per-node)

`Nil : List τ` for some element type `τ`. The element type is carried in the
`value` leaf of the `KIND_NIL` node (the encoder writes the element type tag
into `value`; this is the one place `value` is type-bearing for a list node —
documented here as the M2/encoder contract). The rule checks only the head
type tag:

```
H_T-Nil(i) = P[kind=KIND_NIL](5i) · Pbar[type=TYPE_LIST](5i+1)
```

### 6.5 T-Cons (two-node)

`Cons h t : List τ` given `h : τ` and `t : List τ`. Two-node term over the
CONS node `i`, its head node `h`, and its tail node `t` (this is a
*three-node* leaf set — still a single factored-window expectation, the
factored form accepts any leaf set, §4.1). Contributions:

```
H_T-Cons(i) = P[kind=KIND_CONS](5i)  · Pbar[type=TYPE_LIST](5i+1)      # result is a List
            + P[kind=KIND_CONS](5i)  · Pbar[type=TYPE_LIST](5t+1)      # tail is a List
            + P[kind=KIND_CONS](5i)  · ( Σ_τ P[value=τ](5i+3)
                                          · Pbar[type=τ](5h+1) )       # head : τ
```

The third contribution sums over element-type tags `τ`, reading the CONS
node's `value` leaf (which carries the list element type, §6.4) against the
head node's `type` leaf — a genuine multi-leaf coupling, no `tobl` pre-bake.

### 6.6 T-Eq (two-node)

`Eq a b : Prop` given `a : τ` and `b : τ` for the same `τ`. Two-node term over
the EQ node `i`, its lhs node `a`, its rhs node `b`:

```
H_T-Eq(i) = P[kind=KIND_EQ](5i) · Pbar[type=TYPE_PROP](5i+1)           # result is a Prop
          + P[kind=KIND_EQ](5i) · ( Σ_τ P[type=τ](5a+1)
                                       · Pbar[type=τ](5b+1) )          # lhs/rhs same type
```

The second contribution penalizes lhs and rhs disagreeing on type — summed
over `τ`, it is zero exactly when `type(a) = type(b)`.

### 6.7 T-Forall (two-node)

`Forall x:τ. body : Prop` given `body : Prop` in the context extended with
`x:τ`. `Forall` is a binder (M1 §3.1; introduces a `bid`). Two-node term over
the FORALL node `i` and its body node `b`:

```
H_T-Forall(i) = P[kind=KIND_FORALL](5i) · Pbar[type=TYPE_PROP](5i+1)   # result is a Prop
              + P[kind=KIND_FORALL](5i) · Pbar[type=TYPE_PROP](5b+1)   # body is a Prop
```

The binder's parameter type `τ` (leaf `5i+1`'s companion — the encoder writes
the FORALL node's parameter type into its `value` leaf, mirroring `Lam`) is
checked by the recursion/use rule against each `Var` that references this
`Forall`; that is T-Var with the binder being a `Forall` node — T-Var (§5.4)
already handles any binder kind, so no separate rule is needed.

### 6.8 T-Fix (two-node)

`Fix f:τ. body : τ` given `body : τ` in the context extended with `f:τ`
(the standard recursion typing rule — the fixed point has the same type as the
recursion variable). `Fix` is a binder. Two-node term over the FIX node `i`
and its body node `b`:

```
H_T-Fix(i) = P[kind=KIND_FIX](5i) · ( Σ_τ P[value=τ](5i+3)
                                         · Pbar[type=τ](5i+1) )        # result : τ
           + P[kind=KIND_FIX](5i) · ( Σ_τ P[value=τ](5i+3)
                                         · Pbar[type=τ](5b+1) )        # body : τ
```

`τ` is the FIX's parameter type, written by the encoder into the FIX node's
`value` leaf. The recursion-variable uses inside `body` are `Var` nodes whose
binder is the FIX node; T-Var (§5.4) couples each such use's `type` to the FIX
node's parameter type through the tree — this is the binding-as-entanglement
principle (§1.2) applied to recursion.

### 6.9 Well-typedness

A program is well-typed iff the standard declarative typing derivation of the
extended calculus succeeds. For a well-typed AST every M2 term is zero
(architecture §13.2). For an ill-typed AST each locally-failing rule
contributes `+1` at the failing node; energies sum.

---

## 7. `MeraEvalHamiltonian` — retargeting C + recursion

### 7.1 API and the STLC redex rules

```python
# src/qft_pcn/logic/mera_evaluation_hamiltonian.py

@dataclass(frozen=True)
class MeraEvalTerm:
    rule_id: str
    node: int
    arity: int

class MeraEvalHamiltonian:
    def __init__(self, meta: MeraEncodingMeta,
                 lambda_beta: float = 1.0,
                 lambda_arith: float = 1.0,
                 lambda_if: float = 1.0,
                 lambda_fix: float = 1.0): ...
    terms: list[MeraEvalTerm]
    def term_energy(self, state: MERA, term: MeraEvalTerm) -> float: ...
    def total_energy(self, state: MERA) -> float: ...
    def residuals(self, state: MERA) -> dict[tuple[str, int], float]: ...
```

The STLC evaluation rules are C's, retargeted (C §3). Each is a non-negative
**redex-presence penalty**: a factored projector onto the *unreduced*
configuration, scaled by `λ > 0`. Rule names:

| rule_id | content (C reference) | leaf geometry |
|---|---|---|
| `R-Beta` | `P[kind=APP](app) · P[kind=LAM](fn)` | two-node: APP node ↔ fn node |
| `R-Arith` | `P[kind=BIN,value∈{+,−,*}](bin) · P[kind=INT](lhs) · P[kind=INT](rhs)` | three-node: BIN ↔ lhs ↔ rhs |
| `R-Cmp` | same with `value∈{<,==}`, operands INT | three-node |
| `R-If` | `P[kind=IF](if) · P[kind=BOOL](cond)` | two-node: IF node ↔ cond node |
| `R-Succ` | `P[kind=SUCC](succ) · P[kind=NATLIT](arg)` | two-node — `Succ (NatLit n)` folds to `NatLit (n+1)` |
| `R-Fix` | §7.2 | two-node: FIX node ↔ recursion-use node |

On the MERA the C-era "BIN op is at site k but operands are at k+1, k+2, which
are non-adjacent" problem (C §3.3) **disappears**: a three-node redex is one
factored-window expectation over the BIN, lhs, and rhs leaf sets — the factored
form accepts any leaf set. The C-era "sum of two overlapping two-site terms"
trick is not needed; M2 uses a single direct three-node penalty.

`H_eval = Σ λ_rule · Σ_nodes H_rule(node)`. All terms PSD; `⟨H_eval⟩ = 0` iff
the program is in normal form.

### 7.2 `Fix` recursion unfolding — where MERA pays off

`Fix f:τ. body` evaluates by unfolding: `Fix f. body  →  body[f := Fix f.
body]`. Classically this is unbounded; on a 1D MPS each unfold appends a copy
of `body` and the bid-bond carrying the recursion variable grows linearly with
unfold depth — `O(depth)` bond dimension. **On the MERA the recursion-use
coupling rides the tree, and the tree depth between the FIX node and a
recursion-use node is `O(log N_leaves)` — so a bounded number of unfolds keeps
bond dimension bounded by `χ_layer` regardless of unfold depth** (architecture
§10.4 acceptance: "a recursive Fibonacci program reaches its ground state with
bond dimension `O(log N)` rather than `O(N)`").

M2 realizes `Fix` evaluation **without an explicit unfold rewrite** (§1.6).
`R-Fix` is a constraint: it penalizes a configuration where a recursion-use
`Var` node still references the FIX node *and the FIX node has not yet been
contracted at this use*. Concretely:

- The `Fix` node and each recursion-use `Var` node are coupled by the tree
  (the use's `bid` leaf entangled with the FIX's `bid` leaf, M1 §5).
- `R-Fix` penalizes "FIX node present (`kind = KIND_FIX`) AND a bound
  recursion-use node still has `kind = KIND_VAR`" — i.e. the unfold has not
  propagated the body to the use site.
- Imaginary-time evolution under the transition couplings (§7.4) drives the
  recursion-use `Var` toward carrying the body's head structure, one MERA
  layer of contraction at a time. Because the contraction is tree-structured,
  the *number of layers* touched is `O(log N_leaves)`, not the unfold count.

The acceptance criterion (§9.4, §10) is: a recursive program (a small
`Fix`-defined function applied to a concrete argument that requires `k > 1`
recursive steps) imag-time-evolves to its normal form, and the maximum bond
dimension stays bounded by `χ_layer` throughout — the O(log N) tree-depth
claim, demonstrated.

The unfold is **bounded** in M2: the encoder allocates a finite node budget;
`Fix` programs whose normal form exceeds the budget raise
`MeraEvalBudgetExceeded` (§8) rather than diverging. M2's recursive demo
program is chosen to fit the budget.

### 7.3 Induction-as-ground-state for `Forall` over `Nat` / `List`

A `Forall n:Nat. P(n)` proposition is "true" in the QPCN sense iff its
encoded MERA state, evolved under `H_typing` extended with the **induction
constraint term**, relaxes to zero energy. The induction constraint is a
two-node penalty term `R-Induction` on a `Forall`-over-`Nat` (resp. `List`)
node:

- It penalizes a `KIND_FORALL` node over `TYPE_NAT` whose body proposition is
  *not* simultaneously satisfied at the base case (`body[n := Zero]`) and the
  step case (`body[n := Succ k]` given `body[n := k]`).
- On the MERA these two cases are sub-trees sharing the `Forall`'s `bid` leaf
  through the tree; the induction term is a factored operator coupling the
  `Forall` node's leaves to the base-case and step-case sub-tree root nodes.
- `⟨R-Induction⟩ = 0` iff base and step both hold — which is exactly the Peano
  induction principle. The ground state of `H_typing + R-Induction` is "the
  universally-quantified proposition holds."

M2 ships `R-Induction` for `Forall` over `Nat` and over `List` with
single-node base/step bodies (the analog of C's §5.4 "simple branches"
restriction). Deeper induction bodies are M3 scope. This is a deferral, not a
shortcut: the architecture supports the full version; M2 implements the
bounded-body variant and documents the restriction in §9.

### 7.4 Transition couplings that drive reduction

A pure redex-presence penalty makes the unreduced state high-energy but does
not by itself move amplitude to the reduced state — the imaginary-time
evolution needs **off-diagonal transition terms** that connect the unreduced
and reduced configurations. Mirroring the C-era `factored_evolution.py`
design, each redex rule contributes, in addition to its diagonal penalty, a
factored **transition gate**: a unitary two-leaf (or local) gate that moves
amplitude from the unreduced basis configuration toward the reduced one.

- `R-Beta`: a gate that, conditioned on APP-node = APP and fn-node = LAM,
  rotates the APP and LAM `kind` leaves toward `KIND_PAD` and propagates the
  arg's `value` leaf toward the recursion/use `Var` node's leaves.
- `R-Arith` / `R-Cmp`: a gate that, conditioned on BIN + two INT operands,
  rotates the BIN node's `value` leaf toward the computed result and the
  operand nodes toward `KIND_PAD`.
- `R-If`: a gate that selects the branch matching the cond `BOOL` value and
  collapses the other.
- `R-Succ`: `Succ (NatLit n) → NatLit (n+1)`.
- `R-Fix`: a gate that propagates one layer of the body into a recursion-use
  `Var` node.

Each transition gate is a factored operator (per-leaf 16×16, or a 256×256
two-leaf gate applied via `MERA.apply_two_site_gate`). The diagonal penalty
and the transition gate together make the reduced configuration the unique
zero-energy ground state and give imaginary-time evolution a non-zero matrix
element to relax through.

These transition gates live in `_mera_eval_terms.py` (§8 file layout) as pure
factored-gate builders; `MeraEvalHamiltonian` exposes them for the evolution
driver (§7.5). They are NOT a classical rewrite — they are unitary/Hermitian
operators; the *spectrum* selects the reduced state, not Python control flow
(§1.6).

### 7.5 MERA factored imaginary-time evolution

`mera_evolution_logic.py` provides the MERA analog of
`factored_evolution.py::factored_evolve`:

```python
def mera_trotter_step(state: MERA, ham, dt: float,
                      imaginary: bool = True) -> None:
    """One Trotter step: apply each term's factored gate to `state`
    in place. For imaginary time, the per-term gate is exp(-dt·H_term)
    in factored form; for real time, exp(-i·dt·H_term). Two-leaf gates
    go through MERA.apply_two_site_gate; local gates through
    apply_local_gate. After all gates, state.normalize()."""

def mera_imaginary_evolve(state: MERA, ham, dt: float, steps: int,
                          chi_layer: int | None = None) -> list[float]:
    """Repeat mera_trotter_step `steps` times in imaginary time.
    Returns the energy trajectory [<H>_0, <H>_1, ...]. Energy decreases
    monotonically (architecture §13.1)."""
```

`ham` is a `MeraEvalHamiltonian`, a `MeraTypingHamiltonian`, or a
`compose_mera_hamiltonians` result. Each term contributes a small factored
gate; the gate is `exp(-dt · λ · H_term)` built from the term's per-leaf
operators. Diagonal penalty terms give a diagonal gate (a per-leaf
phase/decay); transition terms (§7.4) give a genuine two-leaf rotation.

Bond dimension is capped at `chi_layer` (the MERA's layer dimension) on every
`apply_two_site_gate` — this is where the O(log N) `Fix` claim is enforced:
the tree truncation keeps recursion-use coupling bounded.

Convergence: by architecture §13.1, for a Hamiltonian with gap `Δ > 0`,
`mera_imaginary_evolve` relaxes to the ground state with error `O(e^{-Δτ})`.
The acceptance tests (§9) verify monotone energy decrease and convergence to
`⟨H⟩ ≈ 0` on the demo programs.

---

## 8. File layout and error model

### 8.1 Files

```
src/qft_pcn/logic/
├── mera_typing_hamiltonian.py     # MeraTypingHamiltonian, MeraTypingTerm,
│                                  #   STLC + extended-calculus typing rules
├── _mera_typing_rules.py          # per-rule factored-operator builders
│                                  #   (helpers shared by §5, §6)
├── mera_evaluation_hamiltonian.py # MeraEvalHamiltonian, MeraEvalTerm
├── _mera_eval_terms.py            # per-rule factored redex penalties +
│                                  #   transition gates (§7.4)
├── mera_compose.py                # compose_mera_hamiltonians
├── mera_evolution_logic.py        # mera_trotter_step, mera_imaginary_evolve
└── mera_encoder.py                # MODIFY: add children_of_node to
                                   #   MeraEncodingMeta (§5.6), populate it
src/qft_pcn/tests/
├── test_mera_typing_hamiltonian.py     # §9.1, §9.2
├── test_mera_eval_hamiltonian.py       # §9.3, §9.4
├── test_mera_compose.py                # §9.5
└── test_mera_evolution_logic.py        # §9.3-§9.4 evolution mechanics
```

`mera_encoder.py` gains one additive field on `MeraEncodingMeta`
(`children_of_node`) and the encoder populates it. No other M1 file changes.
The MPS logic stack and `mps.py` are untouched. `mera.py` may gain a thin
public wrapper only if a genuine internal primitive must be exposed
(interface-only, F's tests must still pass) — same proviso as M1 §11.

Public exports added to `src/qft_pcn/logic/__init__.py`:

```python
from .mera_typing_hamiltonian import MeraTypingHamiltonian, MeraTypingTerm
from .mera_evaluation_hamiltonian import MeraEvalHamiltonian, MeraEvalTerm
from .mera_compose import compose_mera_hamiltonians
from .mera_evolution_logic import mera_trotter_step, mera_imaginary_evolve
```

### 8.2 Error model

Reuse M1's / `encoding.py`'s exception family. New exceptions in
`mera_evaluation_hamiltonian.py`:

```python
class MeraEvalError(Exception): ...
class MeraEvalBudgetExceeded(MeraEvalError):
    """A Fix unfold's normal form exceeds the encoder's node budget."""
```

New in `mera_typing_hamiltonian.py`:

```python
class MeraTypingHamiltonianError(Exception): ...
class MeraTermNotFound(MeraTypingHamiltonianError): ...
```

`term_energy` raises `MeraTermNotFound` for an unknown term (mirrors B). The
Hamiltonians raise nothing at evaluation time otherwise — every term-energy is
a real non-negative number (PSD projector expectation). `compose_mera_
hamiltonians` raises `IncompatibleHamiltonians` when inputs disagree on
`meta.n_leaves` / `meta.n_nodes` (mirrors `compose.py`).

---

## 9. Acceptance tests

`src/qft_pcn/tests/test_mera_*.py`. Every completion claim is backed by fresh
pytest output (`superpowers:verification-before-completion`).

### 9.1 Well-typed programs give ⟨H_typing⟩ ≈ 0

The five STLC demo programs plus the three extended-calculus programs from M1
§9.1 (P1–P8), all well-typed:

```
P1.  \x:Int. x
P2.  (\x:Int. x + 1)(2)
P3.  \f:Int->Int. \x:Int. f (f x)
P4.  if (1 < 2) then ((\x:Bool. x)(true)) else false
P5.  (\x:Int. (\y:Int. x + y)(3))(4)
P6.  Succ (Succ Zero)
P7.  Cons 1 (Cons 2 Nil)
P8.  \x:Nat. Eq x x
```

plus two new well-typed extended-calculus programs:

```
P9.  forall n:Nat. Eq n n
P10. fix f:Nat->Nat. \x:Nat. x          # a (trivial) recursive function
```

For each: `state, meta = encode_mera(parse(p))`;
`H = MeraTypingHamiltonian(meta)`; `abs(H.total_energy(state)) < 1e-9`.

### 9.2 Ill-typed programs give ⟨H_typing⟩ > 0 with attributable residuals

Ill-typed variants, each isolating one rule:

```
IT1.  \x:Int. (x + true)               # T-Obligation: BIN rhs expects Int
IT2.  Succ true                        # T-Succ: arg not Nat
IT3.  Cons true Nil                    # T-Cons: head/element mismatch (or as designed)
IT4.  Eq 1 false                       # T-Eq: lhs/rhs type mismatch
IT5.  \x:Int. x  (decoded then type leaf surgically flipped)  # T-Lit / T-Var isolation
```

For each: `H.total_energy(state) > 0.5`; `H.residuals(state)` has exactly one
entry `> 0.5`, and its `(rule_id, node)` key matches the expected violation.
A `_mutate_leaf` test helper (analog of B's `_mutate_local_register`) swaps a
basis slice in a leaf tensor for the surgical-isolation tests.

### 9.3 Reduction acceptance on the MERA substrate (E1–E4 analog)

```
E1.  2 + 3                              -> 5
E2.  (\x:Int. x + 1)(2)                 -> 3
E3.  if (1 < 2) then 7 else 0           -> 7
E4.  (\x:Int. (\y:Int. x + y)(3))(4)    -> 7
```

For each: encode; `H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
MeraEvalHamiltonian(meta))`; `traj = mera_imaginary_evolve(state, H, dt=0.1,
steps=80)`; assert `traj` is monotone non-increasing (within 1e-6 wobble);
assert final `⟨H⟩ < 1e-2`; `decode_mera(state, meta)` is alpha-equivalent to
the expected normal form. The post-evolution decode uses the M1 decoder.

Also: a normal-form program (`\x:Int. x`) has `MeraEvalHamiltonian.total_energy
≈ 0`; an unreduced program (`2 + 3`) has `> 0.5`.

### 9.4 Recursive `Fix` program reduces with bounded bond dimension

```
E5.  ( fix f:Nat->Nat. \x:Nat. <body requiring >=2 unfolds> ) (NatLit 2)
```

The body is chosen so the normal form needs at least two recursive unfolds and
fits the node budget. Acceptance:

- `mera_imaginary_evolve` drives `⟨H⟩` to `< 1e-2`.
- `decode_mera` yields the correct normal form (a `NatLit`).
- **The maximum MERA layer bond dimension stays `≤ chi_layer` at every Trotter
  step** — verified by inspecting the MERA's layer dimensions during the
  evolution loop. This is the O(log N) tree-depth claim (architecture §10.4):
  recursion does not blow up the bond dimension.

### 9.5 Composer correctness

`compose_mera_hamiltonians(H_t, H_e).total_energy(state)` equals
`H_t.total_energy(state) + H_e.total_energy(state)` to `1e-9` on every demo
state. `IncompatibleHamiltonians` is raised when the inputs' metas disagree.

### 9.6 Structural / principle checks

- **Structural Hamiltonian (§1.7):** `MeraTypingHamiltonian(meta)` and
  `MeraEvalHamiltonian(meta)` are built from `meta` only; the same instance
  evaluates two unrelated programs of the same `n_nodes` correctly. A `dir()`
  scan asserts no attribute name contains `ast`, `tree`, `node_obj`, `walk`.
- **No classical interpreter (§1.6):** a static-source test asserts
  `mera_evaluation_hamiltonian.py` and `_mera_eval_terms.py` contain no
  `substitute`, `beta_reduce`, `evaluate`, and do not `import` the AST module
  for walking.
- **No dense operator at scale (§1.3):** a `conftest.py` memory ceiling (M1's)
  must not trip; a test asserts no Hamiltonian path materializes a tensor
  larger than `16**2`.
- **Factored window is the backend:** every rule's energy function is verified
  to call `mera_window_expectation_factored`, not the dense
  `mera_window_expectation`, for any window > 2 leaves.

### 9.7 Cross-substrate anchor

For P1–P5 (representable on both substrates) and the ill-typed IT1, the MERA
`MeraTypingHamiltonian` total energy agrees with the MPS `TypingHamiltonian`
total energy to `1e-6` (both are 0 for well-typed, both `> 0.5` for IT1). This
anchors M2 against the shipped, tested MPS Hamiltonian.

### 9.8 No regressions

F's MERA tests, M1's encoder tests, and the MPS logic stack's tests all still
pass after M2.

---

## 10. Acceptance criteria

M2 is complete when:

1. The ten well-typed programs (§9.1) give `⟨H_typing⟩ < 1e-9`.
2. The five ill-typed programs (§9.2) give `⟨H_typing⟩ > 0.5` with exactly one
   residual `> 0.5` matching the expected `(rule_id, node)`.
3. The four reduction programs (§9.3) imag-time-evolve to their normal forms
   with monotone energy decrease and final `⟨H⟩ < 1e-2`.
4. The recursive `Fix` program (§9.4) reduces to its normal form with the MERA
   layer bond dimension bounded by `chi_layer` throughout — the O(log N)
   tree-depth claim demonstrated.
5. `compose_mera_hamiltonians` (§9.5) is an exact operator sum.
6. The structural / no-interpreter / no-dense-operator checks (§9.6) pass.
7. The cross-substrate anchor (§9.7) passes for P1–P5 and IT1.
8. No regressions (§9.8): F, M1, and the MPS logic stack tests stay green.
9. Every completion claim is backed by fresh pytest output.

No claim of completion is acceptable without these tests running green in a
fresh shell.

---

## 11. Open questions

**None.** Resolved decisions, made visible:

- **Decision: obligations are two-node tree-carried terms, not one-site `tobl`
  reads.** The MERA can couple non-adjacent leaf windows directly (§4.2); the
  MPS-era `tobl` register was a 1D-adjacency workaround. M2 reads the `tobl`
  leaf only as a redundant check.
- **Decision: `MeraEncodingMeta` gains an additive `children_of_node` field.**
  The term enumerator needs parent→child structure; this is layout metadata
  the M1 pre-order walk already has (§5.6 (b)). M1's tests still pass.
- **Decision: `Fix` evaluation is constraint-based, no explicit unfold
  rewrite.** `R-Fix` is a penalty + transition gate; the spectrum + the tree's
  O(log N) depth do the work (§7.2).
- **Decision: induction-as-ground-state ships for single-node base/step
  bodies.** Deeper induction bodies are M3 scope — a documented deferral, not a
  shortcut.
- **Decision: no Lindblad path.** Architecture §10.3 path a is rejected;
  evaluation is constraint-based ground-state finding only.

If, while implementing, you find a real ambiguity this document does not
resolve — stop and ask the human. Do not paper over it.

---

## 12. Contract M2 exposes to M3

- `MeraTypingHamiltonian(meta)` and `MeraEvalHamiltonian(meta)` with the
  `.terms` / `.term_energy` / `.residuals` / `.total_energy` API. M3's
  debugger consumes `.residuals(state) -> dict[(rule_id, node), float]` and
  maps non-zero entries to AST locations via `meta.site_to_ast_path`.
- The stable rule-name set (§5.3, §7.1). M3 maps these to human-readable error
  templates.
- `compose_mera_hamiltonians` — M3's synthesis evolves under
  `H_typing + H_eval` plus a problem-specific source term.
- `mera_imaginary_evolve` — M3's synthesis driver. It accepts any composed
  MERA Hamiltonian; M3 supplies the hole-bearing initial state from M1's
  hole encoder.
- The additive `MeraEncodingMeta.children_of_node` field.
- The guarantee that the ground space of `H_typing + H_eval` is the well-typed
  normal-form space (architecture §13.2), and that every term is factored and
  local on a bounded leaf set (so MERA evolution scales).

What stays stable for M3 (no re-spec of M2 without changing these):

- The `(rule_id: str, node: int)` keying of `residuals`.
- The 16 typing rule names and the 6 evaluation rule names.
- The node-major 5-leaf layout (M1's contract, unchanged).
- `compose_mera_hamiltonians`'s operator-sum semantics.

---

## 13. Glossary (M2-local; see M1 and B/C glossaries for shared terms)

- **Per-node term** — a factored Hamiltonian operator on the 5 leaves of one
  AST node.
- **Two-node term** — a factored operator spanning the leaves of two (or more)
  AST nodes; the MERA tree carries the correlation.
- **Factored operator** — a `dict[leaf -> (16,16) matrix]`, identity on
  unlisted leaves; evaluated by `mera_window_expectation_factored`. Never
  materialized as a `16**k` dense matrix.
- **Redex-presence penalty** — a non-negative projector onto an unreduced
  configuration; the building block of `MeraEvalHamiltonian`.
- **Transition gate** — a unitary/Hermitian factored gate that connects an
  unreduced configuration to its reduced one; gives imaginary-time evolution a
  matrix element to relax through (§7.4).
- **Recursion unfolding (constraint-based)** — `Fix` evaluation realized as a
  penalty + transition gate, not an explicit rewrite; the MERA tree's
  O(log N) depth keeps bond dimension bounded across unfolds (§7.2).
- **Induction-as-ground-state** — a `Forall`-over-`Nat`/`List` proposition is
  "true" iff its encoded MERA, evolved under `H_typing + R-Induction`, relaxes
  to zero energy; `R-Induction` couples the base case and step case (§7.3).
- **Structural Hamiltonian** — built from `MeraEncodingMeta` alone; the same
  instance evaluates any program of the same size (§1.7).
