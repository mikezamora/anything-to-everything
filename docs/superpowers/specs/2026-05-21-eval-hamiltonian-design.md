# Spec: Evaluation Hamiltonian for the QPCN Logic Layer

**Document type**: Implementation specification (sub-project C of the §10 roadmap).
**Date**: 2026-05-21.
**Author**: Brainstorming session, this branch.
**Status**: Approved design, ready for implementation plan.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` (root). This spec realizes §10.3.
**Acceptance owner**: human review of the imaginary-time-evaluation test suite passing.

---

## 0. How to read this spec

This document is the contract for one sub-project. It exists because the §10 roadmap is too large for a single design cycle, so it was decomposed into seven sub-projects (A–G); this is **sub-project C: the evaluation Hamiltonian**. Sub-project A (`docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md`) produced the encoder, lattice species, and `EncodingMeta` we build against. Sub-project B (`docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md`) produces a `H_typing` whose ground state is the well-typed program; C extends that to a `H_eval` such that the ground state of `H_typing + H_eval` is the well-typed AND fully-reduced program.

The architecture document (§10.3) names two paths:
- **(a) Dissipative**: Lindblad operators on density matrices, MPO substrate. The doc rejects this.
- **(b) Constraint-based**: the *fully reduced* program is the ground state of an augmented Hamiltonian that penalizes any non-normal-form configuration. The doc strongly prefers this.

This spec is path (b). Evaluation IS ground-state finding; reduction is a side effect of imaginary-time evolution. No explicit beta-reduction rewriter is implemented anywhere in the codebase. The Hamiltonian's spectrum does the work.

Every section below is part of the contract. If something is missing here that you need to decide while implementing, **stop and ask** — see §1. The five other sub-projects (B, D, E, F, G) have their own specs and inherit the §1 principles below.

---

## 1. Driving principles (non-negotiable)

The whole reason this project exists is the central thesis of the architecture document: **a learning system whose substrate is a genuine quantum field theory will outperform classical deep networks on data with algebraic, conservation-law, or compositional structure** (`QFT_PCN_ARCHITECTURE.md` §1.1). Sub-project A made variable binding into entanglement and sub-project B made type checking into a gauge-constraint Hamiltonian. This sub-project makes **evaluation into ground-state finding**. If a subagent collapses any of those structural commitments, the entire program loses its identity.

A subagent reading this spec will be tempted to take shortcuts that destroy the correspondence. The following principles are non-negotiable.

### 1.1 Reduction is energy minimization, not rewriting

The Hamiltonian penalizes any non-normal-form configuration — any subexpression that *could* be reduced. The ground state is the normal form. Imaginary-time evolution drives the state toward the ground state. **No explicit beta-reduction rewrite is implemented anywhere.**

The "easy shortcut" — pattern-match on `App(Lam, arg)` in classical code, substitute, write the result back into the MPS — is rejected. That is a classical interpreter wearing a quantum hat. It defeats the architecture and short-circuits every downstream benefit (bidirectional reasoning, counterfactual queries via Hamiltonian perturbation, hole completion via superposition over reduction paths). If you find yourself reaching for `substitute(body, x, arg)`, stop. The Hamiltonian's spectrum does the work.

**Rule**: every term in `H_eval` is a Hermitian operator constructed once from `EncodingMeta`. Reduction is observed by measuring after `evolve(state, H_typing + H_eval, dt, steps, imaginary=True)`. No code path in `evaluation_hamiltonian.py` inspects an AST during evolution.

### 1.2 The Hamiltonian is structural, not program-specific

Built once against `EncodingMeta`, applied to any encoded program of the same `N`. Does NOT inspect the program's AST or the encoded MPS. Same Lagrangian principle as §1.2 of A's spec: H is a property of the lattice, not of the configuration.

The "easy shortcut" — building program-specific terms by walking the input AST after encoding ("at site 0 we have an APP, so I'll add a term that...") — is rejected. Sub-project E (synthesis) and sub-project G (LLM bridge) rely on a single fixed Hamiltonian being applied across many problem instances.

### 1.3 Composability with B is a hard constraint

`H_total = H_typing + H_eval` must work as a literal sum at the operator level. Both Hamiltonians act on the same lattice (A's `SPECIES`). `H_eval` must not break B's typing invariants — its terms must be zero on well-typed normal forms. Equivalently: **the ground space of `H_typing + H_eval` must be a subspace of the ground space of `H_typing`** (every fully-reduced program is well-typed) **and** of the ground space of `H_eval` (every fully-reduced program is a normal form).

The "easy shortcut" — adding negative-energy "rewards" for normal forms that happen to push the system away from well-typed configurations on intermediate sub-states — is rejected. C only adds non-negative penalty terms.

### 1.4 Local and two-site only

Each redex couples a small set of sites (the redex itself plus the binder bookkeeping). Arithmetic and conditional redices are bond-local: `BIN(IntLit, IntLit)` is a (BIN, lhs, rhs) triple where the BIN and its lhs are adjacent in pre-order, and lhs-rhs are adjacent in pre-order (`BIN, lhs, rhs` for unary BIN children — but BIN has two children, so the layout is `BIN_site, lhs_subtree, rhs_subtree`; we work at bond `(BIN, BIN+1)` and bond `(BIN, BIN+1+subtree_size(lhs))` — see §5.3 for the resolution).

Beta-reduction couples `APP_site`, `LAM_site` (which sits at `APP_site + 1`), arg subtree, and the binder's variable use sites. The bid bond from §A.5.4 carries the use-to-binder coupling; we **reuse** that channel structure for value propagation.

The "easy shortcut" — using global Hamiltonian terms that touch every site of a span — is rejected. TEBD evolution applies two-site gates; everything must factor through them.

### 1.5 Value channels are the new architectural commitment

Beta reduction requires the arg's *value* to flow into the binder's use sites. This is the analogue of variable substitution in classical evaluators. The existing `bid` channel mechanism (A §5.4) carries the *identity* of the binder, not the value. We add a parallel mechanism — the **value channel** — that uses the spare slots of the `value` register (indices 7..15 reserved by A §4.4) and a parallel bond structure to carry the literal value attached to a binder.

When `(λx:Int. x + 1)(2)` reduces:

1. The `arg = IntLit(2)` site contributes its value to the live-binder channel of `x`.
2. That value rides the bond chain to the `Var(x)` use site (the same chain the `bid` already rides).
3. At the use site, the `Var` is "promoted" to `IntLit(2)` — i.e. the local state shifts from `KIND_VAR` to `KIND_INT` with `value = 2 + INT_LIT_OFFSET = 9`.
4. The original `APP_site` and `LAM_site` collapse to `KIND_PAD` because they no longer carry information.

Each of these transitions is one local or two-site Hamiltonian term penalizing the unreduced configuration. The ground state has the transition completed.

The "easy shortcut" — pre-computing the value flow during encoding by setting the Var site's `value` register directly — is rejected. The Hamiltonian must do this, and only via its spectrum. (The encoder produces the *input* MPS; imaginary-time evolution produces the *output* MPS.)

### 1.6 Reuse the existing QFT machinery; do not reinvent

`src/qft_pcn/qft/hamiltonian.py` already provides `Hamiltonian` with `local_op(k)` and `bond_op(k)`. Sub-project B's `H_typing` is constructed in that framework. Sub-project C extends the same framework with additional terms.

The "easy shortcut" — building a custom evaluator that bypasses `HamiltonianConfig` and ships its own gates — is rejected. The TEBD machinery in `qft/evolution.py` knows how to drive `Hamiltonian` to the ground state; opting out forces re-derivation of TEBD.

If a real `Hamiltonian` extension point is needed (e.g. arbitrary user-defined two-site operators), we add it to `qft/hamiltonian.py` as a first-class feature — see §6 for the one concrete extension this sub-project requires.

### 1.7 No classical interpreter in the loop

A subagent will be tempted to evaluate the program in Python, check the answer, and write that answer back into the MPS. Stop. The whole point is that the Hamiltonian's spectrum yields the answer. **Tests verify the result by sampling the post-evolution MPS via sub-project A's `decode()`/`sample()`, not by comparing against a Python-evaluated reference.** A reference evaluator may exist for debugging purposes in `src/qft_pcn/logic/_reference_eval.py`, but it is never on the production path and is never imported by `evaluation_hamiltonian.py`.

### 1.8 Termination is well-defined

Convergence to the ground state = termination. Non-terminating programs (which do not exist in STLC without recursion; sub-project F will introduce recursion via MERA) would manifest as states that don't relax to a definite ground state. Sub-project C does not need to handle non-termination — but the principled framing (energy minimization) means F will not need to redesign C to support it; it will only need to provide the MERA substrate. **This is the structural argument for path (b) over path (a) of the architecture doc.**

---

## 2. Scope

### 2.1 In scope (this sub-project)

- An `H_eval` Hamiltonian built from `EncodingMeta`, composed of:
  - **Beta-redex penalty**: any `(APP_site, LAM_site)` configuration where the LAM is the App's function has positive energy unless the contraction has completed (LAM_site becomes PAD, APP_site becomes PAD, arg's value has flowed to the Var use site).
  - **Arithmetic-redex penalty**: any `(BIN, IntLit_lhs, IntLit_rhs)` configuration has positive energy unless the BIN site has been replaced by the result IntLit and the operand sites have become PAD.
  - **Comparison-redex penalty**: same for `BIN(<,IntLit,IntLit)` and `BIN(==, IntLit, IntLit)` producing BoolLit.
  - **If-redex penalty**: `(IF, BoolLit_cond, ...)` has positive energy unless the IF site has been replaced by the then-branch or else-branch (depending on the BoolLit value), and the unselected branch has collapsed to PAD.
- **Value channel** extension to the bond structure: a new register-bond structure parallel to A's `bid` channel that carries the value attached to a live binder.
- An extension to `qft/hamiltonian.py`: support arbitrary user-supplied additional one-site and two-site Hermitian terms (`CustomTerm` registry) so we can layer beta-redex, arithmetic, and conditional penalties without abusing `bare_mass`/`source`.
- A **composer** `compose_hamiltonians(H_typing, H_eval) -> Hamiltonian` that returns the structural sum.
- A test suite covering five evaluation programs (§7.1) that imag-time-evolve to their normal forms.

### 2.2 Out of scope (deferred to other sub-projects)

- Recursion / `let rec` / `fix` (sub-project F when MERA arrives).
- Side effects and IO (architecturally non-unitary; deferred indefinitely).
- General term-rewriting beyond beta + arithmetic + if (sub-project E may add more).
- Polymorphism (sub-project E).
- Holes in evaluable position (sub-project E).
- Synthesis demo end-to-end (sub-project E).
- The LLM bridge / DSL (sub-project G).
- The constraint debugger that surfaces per-term residual energies (sub-project D).

### 2.3 Will not do, even if asked later

- Implement explicit beta-reduction by rewriting the MPS in classical code. The §1.1 prohibition is permanent.
- Implement the dissipative Lindblad alternative from architecture doc §10.3(a). The choice is locked.
- Inspect the input AST during evolution.
- Materialize the full `d_local^2 × d_local^2 = 8192^2 × 8192^2` bond operator densely (we work in the factored structure — see §5.6).

---

## 3. Mathematical formulation of H_eval

`H_eval = H_beta + H_arith + H_if + H_collapse`

Each summand is a sum of local one-site and two-site terms. All terms are **non-negative** (PSD operators), so `⟨ψ | H_eval | ψ⟩ ≥ 0` for all `ψ`, with equality only on the normal-form ground space.

### 3.1 Notation

Site projectors on the `kind` register, embedded into the full `d_local` Hilbert space:

```
P_kind[K](site) = embed_op(|K⟩⟨K|_kind, species=0, dims=(8,8,8,16))
```

Examples: `P_PAD`, `P_VAR`, `P_LAM`, `P_APP`, `P_INT`, `P_BOOL`, `P_IF`, `P_BIN`. Similarly `P_value[v]` and `P_type[t]`.

For BIN sites we use the op-specific projector:

```
P_BIN_PLUS(site) = P_kind[BIN](site) · P_value[VALUE_PLUS](site)
P_BIN_MINUS(site), P_BIN_TIMES(site), P_BIN_LT(site), P_BIN_EQ(site)
```

A two-site projector is a tensor product (`np.kron`) of one-site projectors and lives in the `d_local^2 × d_local^2` bond Hilbert space.

For each redex-flavor we want a penalty term: "this configuration appears in the *unreduced* form". The term is just a positive multiple of the projector onto the unreduced configuration:

```
H_term = λ · P_unreduced
```

with `λ > 0` chosen large enough that imag-time evolution drives the configuration away. We will use `λ = 1.0` throughout and rely on test-driven tuning if the gap is too small (see §10.3).

### 3.2 Beta-redex penalty (H_beta)

A beta-redex is a configuration where site `k` has `kind = APP` and site `k + 1` has `kind = LAM`. (In pre-order serialization, `App(fn, arg)` writes APP at site `k`, then walks `fn` first; when `fn = Lam(...)`, the LAM site is `k + 1`.)

The unreduced configuration has positive energy. The reduced configuration (APP → PAD, LAM → PAD, the LAM's body has been spliced in, arg's value has flowed to the Var use sites) has zero energy.

```
H_beta(bond k) = λ_beta · P_APP(k) ⊗ P_LAM(k + 1)
```

This is a two-site Hermitian projector. It is non-zero only when site `k` is APP **and** site `k + 1` is LAM. Adding it to the bond Hamiltonian penalizes the unreduced redex by `λ_beta = 1.0`.

**The principled subtlety**: this term alone would penalize ALL beta-redices, including those for non-Lam functions (which is fine — they're never in scope) AND those where the App's arg is *not* yet evaluated to a value. But in STLC, `App(Lam(x, body), arg)` is itself a redex regardless of arg's reducedness — eager and lazy semantics differ here. We choose **eager / call-by-value**: the beta term is fully active even when arg is a complex sub-tree; the spectrum prefers configurations where the arg has been reduced *first* (because reducing the arg first lowers the arith/if energy faster than reducing the beta-redex first changes total energy).

This is a property of the spectrum, not of the term. The term `P_APP ⊗ P_LAM` is the same in both orderings. The reduction *order* is chosen by the imaginary-time relaxation dynamics — and that, in turn, is biased by the relative sizes of `λ_beta` vs `λ_arith`. We set `λ_arith ≥ λ_beta` so arithmetic redices reduce first; this matches call-by-value evaluation order.

### 3.3 Arithmetic-redex penalty (H_arith)

A *redex* for `BIN(+, IntLit_a, IntLit_b)` in pre-order serialization is the three-site sequence `(BIN at k, IntLit_a at k+1, IntLit_b at k+2)`. The reduced configuration is `(IntLit_{a+b} at k, PAD at k+1, PAD at k+2)`.

```
H_arith_unreduced(bond k, k+1)    = λ_arith · P_BIN_op(k) ⊗ P_INT(k + 1)
H_arith_unreduced(bond k+1, k+2)  = λ_arith · P_INT(k + 1) ⊗ P_INT(k + 2) · P_BIN_op_at_left(k+1???)
```

The challenge: the BIN-op identity is at site `k`, but the two operand IntLits are at `k+1` and `k+2`. A purely two-site Hamiltonian can only see two adjacent sites at a time. We resolve this with **three two-site terms acting on overlapping bonds**:

```
H_arith_pre(k):    P_BIN_PLUS(k) ⊗ P_INT(k + 1)    on bond (k, k+1)
H_arith_post(k):   P_INT(k + 1) ⊗ P_INT(k + 2)     on bond (k+1, k+2)
                   (alone, this is not arith-specific; combined with H_arith_pre via
                    the spectrum, configurations where both bonds carry their
                    projectors get total penalty 2λ — which is precisely the
                    "BIN+ with two IntLit children unreduced" configuration.)
```

This is a standard physics trick: a many-body penalty is realized as a *sum* of two-body penalties such that the overlap on the redex is the maximum and the partial overlaps (just one side present) carry partial penalty. The ground state with `H_arith = H_arith_pre + H_arith_post + H_arith_result` (the result term defined below) is exactly the normal-form configuration where one of the children has reduced.

**Why this works without over-penalizing.** Consider a configuration where site `k+1` is `IntLit` but site `k` is *not* `BIN_PLUS` (e.g. it's PAD because the BIN already reduced). Then `H_arith_pre = 0` because `P_BIN_PLUS(k)` is zero on PAD. Similarly `H_arith_post` is only nonzero when *both* sites are IntLit, which is exactly the unreduced state.

After reduction, the configuration is `(IntLit_{a+b}, PAD, PAD)`. Here `H_arith_pre = 0` (site k is IntLit, not BIN_PLUS), `H_arith_post = 0` (site k+1 is PAD), and the total is 0. Good.

**Result-correctness term (H_arith_result)**. The above terms encourage the redex to dissolve — they don't constrain *what value* the result takes. We need an additional term that says "if the result at site k is `IntLit_v` and the original was `BIN(+, IntLit_a, IntLit_b)`, then `v = a + b`."

This is the hardest term. The configuration we want to penalize is:

```
site k:   IntLit_v
site k+1: PAD   (was: IntLit_a)
site k+2: PAD   (was: IntLit_b)
```

with `v ≠ a + b`. But this is non-local: the original `a` and `b` are no longer on the lattice; the configuration that determines them is the *initial state* of the MPS, not the current state.

**Resolution**: we don't enforce arithmetic-correctness via a separate term. Instead, we work in a basis where the *original* IntLit sites are projected through an arithmetic gate to the result. The trick — which is the heart of the constraint-based evaluation scheme — is to use the **value-channel mechanism** (§5):

1. The value-channel carries the value of any IntLit through the bond.
2. At a BIN site, the value-channel from the lhs and the value-channel from the rhs are *combined* via an arithmetic gate into a result that lives in a `result-channel` on the right bond of the BIN.
3. The combined result is then projected back onto the BIN site's local `value` register (so the local state of the BIN site holds the reduced literal).
4. The original IntLit sites' value-channels are "consumed" — they exit the bond as `no_info`.

Now the result-correctness term is *local* on the BIN site:

```
H_arith_result(k) = λ_result · (1 - P_IntLit_AND_value_matches_channel_result(k))
```

This penalizes the BIN site for not carrying the result that the value-channels are computing. The channel computation is implemented as a unitary 2-site gate on the value-channel register; the projector is built from that gate. **See §5.3 for the construction.**

The full arithmetic term is `H_arith = H_arith_redex + H_arith_value_consume + H_arith_result`, where:
- `H_arith_redex` (positive penalty on unreduced BIN+IntLit+IntLit): drives the structural collapse.
- `H_arith_value_consume` (positive penalty if the operand IntLits' value-channels haven't been consumed): drives the operands to PAD.
- `H_arith_result` (positive penalty if the BIN site's local value doesn't match the channel-computed result): drives the BIN site's local literal to the correct value.

Each of these is constructed from one-site projectors and two-site projectors on adjacent bonds. The full construction is in §5.3.

### 3.4 Comparison-redex penalty

Same shape as arithmetic, but `BIN_LT(IntLit_a, IntLit_b)` reduces to `BoolLit_{a < b}` (similarly for `==`). The construction is identical to §3.3 with the `value` register's BOOL slots (`VALUE_FALSE = 0`, `VALUE_TRUE = 1`) playing the role of integer slots, and the channel-arithmetic gate computing `<` or `==` instead of `+/-/*`.

### 3.5 If-redex penalty (H_if)

An if-redex is `IF(BoolLit_b, then_branch, else_branch)` in pre-order: `(IF at k, BoolLit at k+1, then_subtree, else_subtree)`. The reduced configuration:

- If `b = true`: `(then's_root at k, PAD at k+1, then_subtree_minus_root_remains, else_subtree_all_PAD)`.
- If `b = false`: `(else's_root at k, PAD at k+1, then_subtree_all_PAD, else_subtree_remains)`.

The penalty term on the bond `(k, k+1)`:

```
H_if_redex(k) = λ_if · P_IF(k) ⊗ P_BOOL(k + 1)
```

penalizes the unreduced configuration.

The **harder constraint** is "the IF site has been promoted to the then-branch's root (or else-branch's root)". This requires:

1. The BoolLit's value-channel propagates rightward to the start of the then-subtree.
2. At the then-subtree's root (site `k+2`), the value-channel "decides" whether this subtree is selected — if selected, the value-channel carries an `active` flag; if not, the channel carries an `inactive` flag.
3. The IF site at `k` adopts the kind/type/value of whichever subtree-root is active.
4. The unselected subtree collapses to PAD (every site).

This is more complex than arithmetic because it requires propagating an `active`/`inactive` flag down an entire subtree. We resolve this with a **subtree-collapse channel** (§5.4) carried on bonds — a bit that says "I am inside an inactive subtree and must be PAD". Sites within the inactive subtree see their kind/value forced to PAD via a per-site coupling to the subtree-collapse channel.

### 3.6 Collapse penalty (H_collapse)

When the BIN or IF site has been reduced to a result, the original operand/branch sites need to collapse to PAD. The collapse term:

```
H_collapse(k, k+1) = λ_collapse · P_REDUCED_PARENT(k) ⊗ (I - P_PAD(k + 1))
                                                       (positive when child is not PAD)
```

penalizes configurations where a "reduced parent" (a BIN that has become an IntLit, or an IF that has been selected) still has a non-PAD child. `P_REDUCED_PARENT(k)` is the projector onto sites that *were* BIN or IF in the input but are now `IntLit / BoolLit / PAD`. This requires reading the **collapse channel** at site k — the channel encodes "this site has been reduced".

### 3.7 What's NOT penalized

- Well-typed configurations are not directly rewarded; that's B's job. C's terms must vanish on B's ground space when there is no remaining redex.
- Configurations with leftover non-redex structure (e.g. an unfinished arithmetic sub-expression whose operands aren't yet IntLit) are not directly penalized — they will be penalized indirectly, because the sub-expression's own redex term won't be satisfied.

### 3.8 Total energy of a normal form

A well-typed program in normal form has:
- No `App(Lam, _)` pair (every beta-redex resolved).
- No `Bin(op, IntLit, IntLit)` (every arithmetic redex resolved).
- No `If(BoolLit, _, _)` (every conditional redex resolved).
- All sites that *were* APP/LAM/BIN/IF parents are now PAD or the result literal.
- All sites that *were* unselected branches are now PAD.

For such a state, every term in `H_eval` is zero, and `⟨H_eval⟩ = 0`. This is the ground state condition.

---

## 4. Value propagation: the architectural commitment

The hardest single design choice in this sub-project. We summarize it here because §5's implementation will refer back.

### 4.1 The bid channel as a model

A's spec §5.4 introduced the **bid channel**: a bond-register structure that carries the identity (channel index → BinderHandle) of every binder live at a bond. Concretely, the `bid` sub-tensor at each site has shape `(|L_left| + 1, BID_CUTOFF, |L_right| + 1)`, and the channel basis is `{no_info, L[0], L[1], …}`.

### 4.2 The value channel

We add a parallel structure on a **new bond register**: the `value-channel`. Its basis at bond `i` is `{no_info, V[0], V[1], …}` where `V[k]` carries the integer or boolean *value* associated with the `k`-th live binder at bond `i`. The dimension of the value-channel matches the bid-channel: `|L_i| + 1`. The two channels are **separate factors of the bond Hilbert space**.

At each site:

- **A LAM site** introduces a binder with `no_info` on the value-channel (the binder has no value yet — it gets its value from the App's arg site).
- **An APP site** that is reducing pushes the arg's value onto the LAM's binder's value-channel.
- **A VAR site** reads the value-channel of its binder; this becomes the value that the Var "evaluates to" after reduction. Specifically, after reduction, the local kind/value of the Var site shifts to `KIND_INT/KIND_BOOL` with the channel's value.
- **An IntLit / BoolLit site** in the *argument* position (i.e. the arg of an APP that is reducing) has its value lifted into a value-channel on the bond going right out of the APP site.

The bond structure of the value-channel is constructed by H_eval gates, not by the encoder. The encoder produces an initial state where every value-channel is `no_info` (no reduction has happened yet). Imaginary-time evolution under `H_eval` lights up the appropriate channels.

### 4.3 Why this is "still" constraint-based

The value-channel is just another factor of the bond Hilbert space. The Hamiltonian's terms — projectors onto unreduced configurations — push the state into a subspace where the channels carry the right information. There is no rewriting; there is no classical control flow. The channels are degrees of freedom of the MPS, and the Hamiltonian's spectrum selects the channel configuration with the lowest energy.

### 4.4 Cost of the value channel

The bond dimension increases. A's spec sized bonds at `|L_i| + 1` on the bid register and 1 on every other register. We add `|L_i| + 1` on the value-channel register, multiplying the total bond by another factor of `|L_i| + 1`. For P5 from A's tests (3 live binders maximum), that takes the worst-case bond from `4` to `16`. `chi_max = 16` from A's spec stays just barely workable for sub-project A's programs; sub-project C demos use shallower nesting (one beta-redex at a time) so the new bound is `chi_max = 16` still — same as A — and is verified per program by §7.5.

For deeper programs sub-project E may raise `chi_max` to 32. We document this in §10.

### 4.5 What the value channel does NOT do

It does not encode arithmetic results that flow from a BIN site up to its parent. That flow is local: the BIN site's own local `value` register is overwritten by the arithmetic gate, no channel needed. The channel is only for **values that flow from an arg into a binder's use sites**, which is genuinely non-local (the use site may be many bonds away).

For arithmetic and conditional reduction, the operands are *adjacent* to the operator in pre-order, so the reduction is local. We don't need a channel for those — we just use two-site gates.

---

## 5. Implementation: term construction

### 5.1 Module structure

```
src/qft_pcn/logic/evaluation_hamiltonian.py
    EvalHamiltonianConfig       (top-level dataclass)
    build_eval_hamiltonian()    (factory)
    _build_beta_terms()
    _build_arith_terms()
    _build_if_terms()
    _build_collapse_terms()
    _value_channel_gate()       (the §5.3 channel-arithmetic gate)
    _subtree_collapse_gate()    (the §3.5 inactive-branch collapse gate)
src/qft_pcn/logic/compose.py
    compose_hamiltonians(H_typing, H_eval) -> Hamiltonian
```

### 5.2 The EvalHamiltonianConfig

```python
@dataclass
class EvalHamiltonianConfig:
    meta: EncodingMeta
    lambda_beta:     float = 1.0
    lambda_arith:    float = 1.0
    lambda_if:       float = 1.0
    lambda_collapse: float = 1.0
    lambda_result:   float = 1.0
    # Value-channel cap on each bond. Defaults to encoder's chi_max.
    chi_value:       int   = 16
```

The default `λ`s are 1.0; tests calibrate them per-program if the spectrum's relaxation rate is too slow (see §10 acceptance criteria).

### 5.3 Channel-arithmetic gate (the arith-result term)

For a BIN site at position `k`, the **arithmetic gate** is a unitary `U_arith` on the two-site Hilbert space `H_local(k) ⊗ H_local(k+1)` that takes:

```
input:  (BIN_op at k with value=op_code, IntLit at k+1 with value=v_lhs)
output: (BIN_op at k with value=op_code+intermediate(v_lhs), PAD at k+1)
```

with an intermediate that encodes "lhs has been read" via one of the reserved `value` slots 7..15. Then the second gate on bond `(k, k+1)` (actually `(k+1, k+2)` after relabeling — see below) absorbs the rhs:

```
input:  (BIN at k with intermediate(v_lhs), IntLit at k+1 with v_rhs)
                    ^-- this is the k=k+1 of the original layout
output: (IntLit_{v_lhs op v_rhs} at k, PAD at k+1)
```

Constructing this as a unitary gate is straightforward — the gate acts on the local kind/value registers of the two sites and is an explicit `(d_local^2 × d_local^2)` matrix. **But** we are not applying gates directly; we are building a Hamiltonian whose ground state implements them.

The **principled construction** of `H_arith_result` is:

```
H_arith_result(k, k+1) = λ_result · (I - U_arith_step1) · projector_onto_input
                       = λ_result · |Δ⟩⟨Δ|
```

where `|Δ⟩ = |input⟩ - |output⟩` for the gate U_arith_step1 acting on the relevant subspace. The square of this delta is a Hermitian projector onto the "uncorrected" subspace. Energy is zero exactly when the local state satisfies the gate's input-output relation.

**Concretely**: for each `(op, v_lhs)` pair, we build the local subspace projectors and combine them into the H_arith_result term. The full term is a sum over `op ∈ {+, -, *, <, ==}` and `v_lhs ∈ [−7, 8]` of such delta-projectors. This is a finite, bounded construction: 5 ops × 16 values = 80 terms per bond, each a fixed `64 × 64` matrix (acting on the relevant `(kind × value) × (kind × value)` sub-Hilbert-space) embedded into the full `d_local^2 × d_local^2` operator.

The result is Hermitian by construction (projectors are Hermitian) and non-negative (its eigenvalues are 0 or λ_result). Adding it to the bond Hamiltonian via `qft/hamiltonian.py`'s custom-term registry (§6) ships it into TEBD.

### 5.4 Subtree-collapse channel (for H_if)

When `If(BoolLit_true, then, else)` reduces, the **else** subtree must collapse to PAD. This subtree may span many sites; we use a bond-register flag `collapse ∈ {active, inactive}` that propagates rightward from the IF site through the then-subtree (carrying `active` for then and `inactive` for else), with the per-site collapse term:

```
H_subtree_collapse(k) = λ_collapse · n_{collapse_inactive}(k) · (I - P_PAD(k))
```

(One-site term, but it reads the bond-register-collapse value via the per-site density operator. Implementing this needs the same "embed channel state into local Hamiltonian via a constructor" trick we use for value channels.)

This is a known-hard term. **The simplification we adopt for sub-project C**: only support if-redices where both branches are simple (single-site). That means the IF site is at `k`, the BoolLit at `k+1`, the then-branch at `k+2`, the else-branch at `k+3`, and the collapse touches only one site per branch. For deeper branches, sub-project E will extend the construction with a real subtree-collapse channel. We document this restriction explicitly in §7 (the test programs respect it).

This is a deferral, not a shortcut: the architecture supports the full version (see §4.2); we just don't implement the deep-subtree variant until E.

### 5.5 Beta-redex full term

```
H_beta(k) = λ_beta · P_APP(k) ⊗ P_LAM(k + 1)
          + λ_beta · P_APP(k) ⊗ (I - P_PAD(k + 1))      (App's child is not yet PAD)
          + λ_beta · P_LAM(k + 1) ⊗ (... when child carries Var-Lam-mismatch ...)
```

The first term is the redex-presence penalty. The second penalizes the App site's still-being-present after the function-Lam has dissolved. The third term implements the value-channel coupling: at every Var site whose binder is the just-reduced LAM, the local kind/value should have transitioned from VAR to whatever the LAM's body site held (specifically, the value-channel of the LAM's binder).

The Var→Lam-body coupling is the only **non-adjacent** part of the beta-reduction. We handle it by propagating the LAM's body's local state into the Var's value-channel and then forcing the Var site's local state to match. Since the value-channel is a bond-register and the propagation is two-site at a time, the term factors:

```
H_beta_var_match(k) = λ_beta · (I - V) acting on (left_bond, Var_site)
```

where `V` is the unitary that reads the appropriate value-channel into the local Var-site's kind/value registers.

### 5.6 Composer

```python
def compose_hamiltonians(H_typing: Hamiltonian,
                        H_eval:   Hamiltonian) -> Hamiltonian:
    """Build H_total = H_typing + H_eval as a unified Hamiltonian.

    Both inputs must share the same species, N, and d_local. The result
    has local_op(k) = H_typing.local_op(k) + H_eval.local_op(k) and
    likewise for bond_op. No new species, no new sites.
    """
```

This is straightforward operator addition on the dense local/bond operators. The result is structurally identical to either input but with summed term contributions. TEBD on the composed Hamiltonian "just works".

### 5.7 Bond-dimension budget after evolution

After imag-time evolution, the bond dimension can grow up to `chi_max`. With `chi_value = 16` (per-bond cap on the value-channel) and the existing bid bond at most `7 + 1 = 8`, the worst-case bond dim is `8 × 16 = 128`, but in practice the encoder's initial state has bid-bond ≤ 4 and trivial value-channel; the truncation cap `chi_max = 16` is enforced in TEBD. The product structure of the channels means many configurations are sparse and survive truncation. §7.5 verifies the truncation error stays below 1e-4 per Trotter step on the demo programs.

### 5.8 Numerical considerations

- All Hamiltonian construction is done in `complex128`. All terms are explicitly Hermitianized via `0.5 * (H + H†)` after construction (defense in depth — should be Hermitian by construction).
- `λ` parameters: defaults of 1.0 are calibrated to match the typing-energy scale from sub-project B (which sets typing-violation energies around 1.0 per site). The total energy of a 30-node program with all redices unreduced is approximately `n_redices × λ`, bounded by `~30`. The energy gap between ground and first-excited state is at least `λ_arith ≈ 1.0`, giving an imag-time relaxation rate of about `1 / λ ≈ 1.0` per Trotter unit. With `dt_imag = 0.1`, we relax in ~10–30 Trotter steps. Tests use 50 steps to be safe.

---

## 6. Encoder requirements / extension to A and B

### 6.1 No changes to A's encoder

The encoder produces the initial MPS for evaluation. The initial state has all value-channels as `no_info` (because we just encoded; no reduction has happened). A's encoder is already correct for this purpose — the encoder writes the bid-channel structure for binder identity and leaves the value-channel (which doesn't exist in A's spec) implicit at dimension 1.

**However**: the encoder must be extended to produce **factored tensors that include a value-channel bond dimension of 1** (a placeholder that H_eval can grow). Concretely, A's `_tensors.py::build_site_tensors` constructs site tensors of shape `(bid_in, d_local, bid_out)`. We need them to be `(bid_in × value_chan_in, d_local, bid_out × value_chan_out)` with `value_chan_in = value_chan_out = 1` initially. The encoder writes a placeholder identity on the value-channel factor.

This is a small encoder change: at the bond-product step in `_tensors.py`, multiply the bond dimension by 1 (no-op, but record the structure in `EncodingMeta`). The bond is laid out as a tensor product of `bid_channel × value_channel`, with each factor having its own basis ordering.

**Change to A's `EncodingMeta`** (additive, non-breaking):

```python
@dataclass
class EncodingMeta:
    ...   # all A fields preserved
    value_channel_dim_per_bond: list[int] = field(default_factory=list)
    # length N - 1; value_channel_dim_per_bond[i] = bond dim on the
    # value-channel factor at bond i. Initially all 1; H_eval grows this
    # via evolution.
```

This new field is populated by the encoder as `[1] * (N - 1)` and updated by C's tooling if needed.

### 6.2 Required extension to `qft/hamiltonian.py`

`Hamiltonian.local_op(k)` and `bond_op(k)` are currently built from `HamiltonianConfig` (mass / kinetic / quartic / source / coupling parameters). C needs to add arbitrary user-supplied terms that don't fit this template (e.g. `P_APP ⊗ P_LAM`).

We extend `Hamiltonian` with a **custom-term registry**:

```python
@dataclass
class CustomTerm:
    """A user-supplied additional one-site or two-site term."""
    sites: tuple[int]        # (k,) for one-site, (k, k+1) for two-site
    operator: np.ndarray     # (d_local, d_local) or (d_local^2, d_local^2)
    name: str                # for diagnostics

# Hamiltonian gains:
def add_custom(self, term: CustomTerm) -> None: ...
def list_custom_terms(self) -> list[CustomTerm]: ...
```

`local_op(k)` sums the existing free-part terms plus all custom one-site terms at site k. `bond_op(k)` sums the existing kinetic-part terms plus all custom two-site terms at bond `(k, k+1)`.

The `Hamiltonian` is **fully backward-compatible**: existing tests in `qft/test_qft.py` continue to pass because `add_custom` is opt-in. Sub-project B's `build_typing_hamiltonian()` and C's `build_eval_hamiltonian()` both populate the custom-term registry; the composer in §5.6 simply concatenates the two registries (and adds the rest of the dataclass-level config, which should match across the two).

### 6.3 No changes to B beyond the species list it inherits from A

B's spec is parallel to this one and uses the same custom-term registry. The composer (§5.6) requires that B's and C's `Hamiltonian` instances agree on species and N. We require both sub-projects to produce `Hamiltonian` objects built from the same `EncodingMeta`, which guarantees this.

---

## 7. Acceptance tests

Located in `src/qft_pcn/tests/test_eval_hamiltonian.py`.

### 7.1 The five evaluation programs

```
E1.  2 + 3                              → 5
E2.  (λx:Int. x + 1)(2)                 → 3
E3.  if (1 < 2) then 7 else 0           → 7
E4.  (λx:Int. (λy:Int. x + y)(3))(4)    → 7
E5.  (λf:Int→Int. f 2)(λy:Int. y + 1)   → 3
```

Each program is encoded, evolved under `H_typing + H_eval`, then sampled. The sampled AST is asserted to be alpha-equivalent to the expected reduced form (which is just an `IntLit` or `BoolLit` in all five cases).

```python
def test_eval_E1_two_plus_three():
    src = "2 + 3"
    state, meta = encode(parse(src), N=16, chi_max=16)
    H_t = build_typing_hamiltonian(TypingHamiltonianConfig(meta=meta))
    H_e = build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta))
    H = compose_hamiltonians(H_t, H_e)
    # Imag-time evolve to ground state.
    evolve(state, H, dt=0.1, steps=80, imaginary=True, chi_max=16)
    state.normalize()
    # Sample.
    rng = np.random.default_rng(seed=42)
    [result] = sample(state, meta, n_samples=1, rng=rng)
    assert isinstance(result.ast, IntLit)
    assert result.ast.val == 5
    # Energy at ground = 0 (within numerical tolerance).
    e = energy(state, H)
    assert e < 1e-2, f"energy {e} should be ~0 at ground state"
```

`build_typing_hamiltonian` and `TypingHamiltonianConfig` come from sub-project B (assumed to exist).

### 7.2 Ground-state energy of a normal form is zero

For an already-normal program (no redices), `⟨H_eval⟩ = 0`.

```python
def test_normal_form_has_zero_eval_energy():
    """A program with no redices contributes zero energy from H_eval."""
    src = r"\x:Int. x"      # already in normal form (no betas, no arith, no ifs)
    state, meta = encode(parse(src), N=16, chi_max=16)
    H_e = build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta))
    e = energy(state, H_e)
    assert abs(e) < 1e-10, f"normal form has nonzero eval energy: {e}"
```

### 7.3 Unreduced program has positive H_eval energy

```python
def test_unreduced_has_positive_eval_energy():
    src = "2 + 3"
    state, meta = encode(parse(src), N=16, chi_max=16)
    H_e = build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta))
    e = energy(state, H_e)
    assert e > 0.5, (
        f"unreduced 2+3 should have energy >0.5 from H_eval, got {e}"
    )
```

### 7.4 Imag-time evolution decreases H_eval energy monotonically

```python
def test_imag_time_decreases_eval_energy():
    src = "2 + 3"
    state, meta = encode(parse(src), N=16, chi_max=16)
    H_e = build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta))
    energies = [energy(state, H_e)]
    for _ in range(20):
        trotter_step(state, H_e, dt=0.1, imaginary=True, chi_max=16)
        state.normalize()
        energies.append(energy(state, H_e))
    # Monotone decrease (allow tiny numerical wobble).
    for i in range(len(energies) - 1):
        assert energies[i + 1] <= energies[i] + 1e-6, (
            f"step {i}: energy went up from {energies[i]} to {energies[i+1]}"
        )
    assert energies[-1] < energies[0] * 0.1, (
        f"after 20 steps, energy {energies[-1]} should be << initial {energies[0]}"
    )
```

### 7.5 Bond dimension stays within chi_max throughout

```python
def test_bond_dim_within_chi_max_during_evolution():
    src = r"(\x:Int. x + 1)(2)"
    state, meta = encode(parse(src), N=16, chi_max=16)
    H = compose_hamiltonians(
        build_typing_hamiltonian(TypingHamiltonianConfig(meta=meta)),
        build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta)),
    )
    for _ in range(50):
        trotter_step(state, H, dt=0.1, imaginary=True, chi_max=16)
        state.normalize()
        for b in state.bond_dimensions():
            assert b <= 16, f"bond dim {b} exceeds chi_max=16"
```

### 7.6 Composer correctness

```python
def test_compose_hamiltonians_is_sum():
    """⟨ψ|H_total|ψ⟩ = ⟨ψ|H_typing|ψ⟩ + ⟨ψ|H_eval|ψ⟩ on the same MPS."""
    src = "2 + 3"
    state, meta = encode(parse(src), N=16, chi_max=16)
    H_t = build_typing_hamiltonian(TypingHamiltonianConfig(meta=meta))
    H_e = build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta))
    H = compose_hamiltonians(H_t, H_e)
    e_total = energy(state, H)
    e_t = energy(state, H_t)
    e_e = energy(state, H_e)
    assert abs(e_total - (e_t + e_e)) < 1e-10
```

### 7.7 Hermiticity of every H_eval term

```python
def test_all_h_eval_terms_are_hermitian():
    src = "2 + 3"
    state, meta = encode(parse(src), N=16, chi_max=16)
    H_e = build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta))
    for k in range(H_e.N):
        op = H_e.local_op(k)
        assert np.allclose(op, op.conj().T, atol=1e-10), \
            f"local_op({k}) is not Hermitian"
    for k in range(H_e.N - 1):
        op = H_e.bond_op(k)
        assert np.allclose(op, op.conj().T, atol=1e-10), \
            f"bond_op({k}) is not Hermitian"
```

### 7.8 H_eval terms are PSD (non-negative spectrum)

```python
def test_all_h_eval_terms_are_psd():
    src = "2 + 3"
    state, meta = encode(parse(src), N=16, chi_max=16)
    H_e = build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta))
    for k in range(H_e.N):
        op = H_e.local_op(k)
        eigs = np.linalg.eigvalsh(op)
        assert eigs.min() > -1e-10, \
            f"local_op({k}) has negative eigenvalue {eigs.min()}"
    for k in range(H_e.N - 1):
        op = H_e.bond_op(k)
        eigs = np.linalg.eigvalsh(op)
        assert eigs.min() > -1e-10, \
            f"bond_op({k}) has negative eigenvalue {eigs.min()}"
```

### 7.9 Term-isolation tests

For each of beta, arith, if, we test that turning off all other λs still drives the corresponding redex toward zero:

```python
def test_beta_term_alone_reduces_beta_redex():
    cfg = EvalHamiltonianConfig(meta=meta, lambda_arith=0.0,
                                lambda_if=0.0, lambda_collapse=0.0)
    # ... evolve and verify only beta-redices reduce; arith/if leftover
```

(Similar for arith-only and if-only.)

### 7.10 No-classical-evaluator check

A static test that asserts the implementation does not import or call any reference evaluator:

```python
def test_no_classical_evaluator_imported():
    """H_eval construction must not depend on a Python AST evaluator."""
    import src.qft_pcn.logic.evaluation_hamiltonian as eh
    src_text = inspect.getsource(eh)
    assert "evaluate" not in src_text.lower(), (
        "evaluation_hamiltonian.py mentions 'evaluate' — has a classical "
        "interpreter slipped in?"
    )
    assert "substitute" not in src_text.lower()
    assert "beta_reduce" not in src_text.lower()
    # Hamiltonian construction does not import the AST data classes for
    # walking; it only consumes EncodingMeta.
    assert "from .ast" not in src_text
    assert "import ast" not in src_text  # we use the term in a comment
```

This test is the §1.1 / §1.7 enforcer.

### 7.11 Performance budget

```python
@pytest.mark.timeout(30)
def test_eval_E1_performance_budget():
    src = "2 + 3"
    state, meta = encode(parse(src), N=16, chi_max=16)
    H = compose_hamiltonians(
        build_typing_hamiltonian(TypingHamiltonianConfig(meta=meta)),
        build_eval_hamiltonian(EvalHamiltonianConfig(meta=meta)),
    )
    evolve(state, H, dt=0.1, steps=80, imaginary=True, chi_max=16)
    # Must complete in 30s on a developer laptop.
```

---

## 8. File layout

```
src/qft_pcn/logic/
├── evaluation_hamiltonian.py          # main module (built in this sub-project)
├── compose.py                          # compose_hamiltonians()
├── _eval_terms.py                      # internal: beta, arith, if, collapse term builders
└── _channel_gates.py                   # internal: value-channel arithmetic gate, subtree-collapse gate

src/qft_pcn/qft/
├── hamiltonian.py                      # +CustomTerm, +add_custom, +list_custom_terms (extension)

src/qft_pcn/tests/
└── test_eval_hamiltonian.py            # all tests from §7
```

Public exports from `src/qft_pcn/logic/__init__.py` (added by this sub-project):

```python
from .evaluation_hamiltonian import (
    EvalHamiltonianConfig,
    build_eval_hamiltonian,
)
from .compose import compose_hamiltonians
```

No deletions of A's exports. No changes to A's `encoder.py` except the small `EncodingMeta` extension in §6.1.

---

## 9. Error model

Custom exceptions live in `src/qft_pcn/logic/evaluation_hamiltonian.py`:

```python
class EvaluationError(EncodingError):
    """Base for sub-project C errors."""

class IncompatibleHamiltonians(EvaluationError):
    """compose_hamiltonians() received H_typing and H_eval that disagree
    on species, N, or d_local."""

class ValueChannelOverflow(EvaluationError):
    """A bond's value-channel grew beyond chi_value during evolution."""
```

`IncompatibleHamiltonians` is raised eagerly by `compose_hamiltonians`. `ValueChannelOverflow` is a runtime check after each Trotter step; if the bond dim on the value-channel exceeds `chi_value`, the bond is truncated to `chi_value` and a warning logged. (We do NOT raise — bond truncation is a normal TEBD operation. The exception class exists only for the `strict=True` mode used in tests.)

---

## 10. Acceptance criteria

The sub-project is complete when:

1. The five evaluation tests in §7.1 (E1–E5) pass: imag-time evolution drives each program to its normal form, the post-evolution `sample()` returns the correct AST, and `⟨H⟩ < 1e-2` at the ground state.
2. The normal-form energy test in §7.2 passes (`⟨H_eval⟩ = 0` on already-normal programs).
3. The unreduced-energy test in §7.3 passes (`⟨H_eval⟩ > 0.5` on `2 + 3`).
4. The monotone-decrease test in §7.4 passes (energy decreases at every Trotter step, ending < 10% of initial).
5. The bond-dim test in §7.5 passes (bond dim stays ≤ chi_max = 16 throughout evolution).
6. The composer test in §7.6 passes (composed Hamiltonian's energy equals sum of components').
7. The Hermiticity test in §7.7 passes (every local/bond op is Hermitian to 1e-10).
8. The PSD test in §7.8 passes (no negative eigenvalues to 1e-10).
9. The term-isolation tests in §7.9 pass (each redex type reduces independently when others' λs are 0).
10. The no-classical-evaluator test in §7.10 passes (no `evaluate`/`substitute`/`beta_reduce` in the module).
11. The performance budget test in §7.11 passes (E1 completes in 30s).
12. All previously passing tests in `src/qft_pcn/tests/` still pass (no regression in A or B).
13. The custom-term registry extension to `qft/hamiltonian.py` does not break any existing `test_qft.py` test.
14. `EncodingMeta.value_channel_dim_per_bond` field has been added and is populated correctly by A's encoder (initial all-1s) — verified by extending A's existing tests where they read `EncodingMeta`.

No claim of completion is acceptable without these tests actually running green in a fresh shell. (See `superpowers:verification-before-completion`.)

---

## 11. Open questions

**None.** Every design choice above is locked. If you, while implementing, find a real ambiguity that this document does not resolve — stop and ask the human. Do not paper over it.

A note on **λ tuning**: the defaults (`λ_* = 1.0`) are calibrated against sub-project B's typing-violation scale, also `~1.0`. If the spectrum's relaxation rate is too slow on the test programs (visible as the §7.4 monotone-decrease test passing but with the final energy not < 10% of initial), the implementer is free to:

- Raise individual `λ`s by up to 5× without re-asking.
- Lower `dt_imag` and increase step count by the same factor.
- Both of these are calibration, not architectural changes. If a λ greater than 10× is needed, that's a signal the term construction is off — stop and ask.

---

## 12. Contracts for downstream sub-projects

### 12.1 Contract for D (constraint debugger)

D consumes per-term energy contributions to produce structured error reports. C must expose:

```python
def per_term_energies(state: MPS, H_eval: Hamiltonian) -> dict[str, float]:
    """Return {term_name: ⟨ψ|H_term|ψ⟩} for each CustomTerm in H_eval.

    term_name follows the convention:
      "beta:site_{k}"
      "arith:site_{k}:op_{plus}"
      "if:site_{k}"
      "collapse:site_{k}"
      "arith_result:site_{k}:op_{plus}"
    """
```

This is implementable from the custom-term registry directly: for each `CustomTerm`, compute `⟨ψ|term.operator|ψ⟩` using `MPS.local_expectation` or `MPS.two_site_expectation`.

D's UI consumes this dict and surfaces the highest-energy terms with their human-readable names.

### 12.2 Contract for E (STLC synthesis demo)

E builds on the **composed** Hamiltonian `H_typing + H_eval` plus possibly a problem-specific source term that encodes the synthesis target. E imag-time-evolves and samples to obtain valid completions.

C provides:

- `compose_hamiltonians(H_typing, H_eval)` returning a `Hamiltonian` suitable for `evolve()`.
- The guarantee that the ground space of `H_total` is the well-typed normal-form space.
- The guarantee that `H_total` is local + two-site (so TEBD scales).
- `chi_value = chi_max = 16` is sufficient for §7's programs; E may request larger via `EvalHamiltonianConfig(chi_value=32)` if its programs are deeper.

E does not need to know about value channels or any internal C structure; the composed Hamiltonian is the entire interface.

### 12.3 Contract for F (MERA)

F replaces the 1D MPS with a hierarchical tree. F's TEBD analogue (binary-tree TEBD) must handle the same `Hamiltonian` interface (local_op + bond_op via custom terms). C's terms remain valid because they are local and two-site; F's task is to lift the bond-op application to the binary-tree topology.

The value channels carry across cleanly under MERA: the channel-bond is just another bond factor, and MERA's RG-style coarse-graining handles bond factors as a matter of course.

---

## 13. Glossary (local)

- **Redex** — a "reducible expression" — a subterm that can take a beta / arithmetic / conditional reduction step. The unreduced redex carries the corresponding penalty energy.
- **Normal form** — a program with no redices remaining. Sub-project C's ground state condition.
- **H_eval** — the evaluation Hamiltonian built by this sub-project; sum of `H_beta + H_arith + H_if + H_collapse`.
- **H_typing** — sub-project B's typing Hamiltonian. We compose with it but do not modify it.
- **H_total** — `H_typing + H_eval`. The Hamiltonian that drives the full reduction-to-well-typed-normal-form.
- **Value channel** — a new bond-register factor (§4) carrying the value associated with each live binder. Parallel to A's bid channel.
- **Subtree-collapse channel** — a one-bit-per-bond flag carrying "the subtree downstream of this bond is inactive (must be PAD)"; used by H_if.
- **Custom term** — a user-supplied additional Hermitian operator added to the `qft/hamiltonian.py` Hamiltonian via the new registry (§6.2).
- **Reduction order** — the order in which redices reduce (eager vs lazy). We choose call-by-value (eager arg-reduction-first) and enforce it via relative λ sizes.
- **Channel-arithmetic gate** — the unitary `U_arith` that combines two operand values from a bond's value-channel into the BIN site's local literal (§5.3).
- **PSD term** — positive semi-definite Hermitian operator. Eigenvalues are non-negative. The basic building block of every penalty in this sub-project.
- **Composer** — `compose_hamiltonians(H_t, H_e)` (§5.6). Adds two `Hamiltonian` objects by summing their custom-term registries and config-level terms.

---

## 14. Where this fits in the larger arc

This sub-project produces the Hamiltonian whose ground state encodes evaluation. It composes with sub-project B (which provides the typing Hamiltonian) to give the full Hamiltonian for sub-project E (the STLC synthesis demo). Sub-project D (debugger) consumes per-term energies from this sub-project for error reporting.

Failing to realize §1.1 here — by sneaking in a classical interpreter — would silently break E. E's synthesis depends on the ground state being computed by the Hamiltonian's spectrum, not by an external evaluator. A classical evaluator would produce the right answer on the five test programs of §7.1 (which is why §7.10 is a static-source check, not a behavioral one), but it would produce *no* meaningful answer on a program with holes or ambiguity, because those programs don't have a single classical evaluation result. That is the practical reason §1.1 is non-negotiable.
