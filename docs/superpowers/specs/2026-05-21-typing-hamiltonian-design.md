# Spec: STLC Type-System → Hamiltonian Compiler for the QPCN Logic Layer

**Document type**: Implementation specification (sub-project B of the §10 roadmap).
**Date**: 2026-05-21.
**Author**: Brainstorming session, this branch.
**Status**: Approved design, ready for implementation plan.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Architecture doc**: `QFT_PCN_ARCHITECTURE.md` (root). This spec realizes §10.2.
**Depends on**: sub-project A (spec `docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md`, implemented in `src/qft_pcn/logic/`).
**Acceptance owner**: human review of the well-typed vs. ill-typed test matrix passing.

---

## 0. How to read this spec

This document is the contract for sub-project B. It exists because the §10 roadmap is too large for a single design cycle, so it was decomposed into seven sub-projects (A–G); this is **sub-project B: the typing-rule Hamiltonian compiler**. Sub-project A is the AST↔MPS encoder/decoder it builds on; C will add an evaluation Hamiltonian on top; D consumes the per-term residual energies B exposes; E is the synthesis demo.

Every section below is part of the contract. If something is missing here that you need to decide while implementing, **stop and ask**. Do not fill in by guessing — see §1.

This spec is intentionally written in the same style and at the same level of rigor as sub-project A's spec. If you find yourself wanting to "loosen up" a section because it's complicated, that is the signal to look harder for the right encoding, not to soften the rule.

---

## 1. Driving principles (non-negotiable)

Sub-project B realizes the second clause of the architecture's central thesis (§8.1 of `QFT_PCN_ARCHITECTURE.md`): **type checking is gauge-invariance verification**. The Hamiltonian we build here is *not* a type-checker wearing a quantum hat. It is the actual Hermitian operator whose ground state is "well-typed program," and whose excited-state energy gap counts typing-rule violations.

A subagent reading this spec will be tempted to take shortcuts that destroy the correspondence. The following principles are non-negotiable. **None of them may be traded away for implementation simplicity.** If you find yourself tempted to violate one, stop and ask the human.

### 1.1 The Hamiltonian is structural, not data-dependent

There is **one** `H_typing` per N-site lattice. The same Hamiltonian applies to any AST of size `≤ N` that the encoder can produce. Its definition reads only the *register basis* (kind/type/bid/value/tobl tags 0..d−1 per site) and the *channel structure* of the bid bond — it does **not** inspect the program AST, nor any AST-derived classical metadata, at construction time.

The "easy shortcut" — building a per-AST Hamiltonian whose matrix coefficients are populated from `EncodingMeta.something(ast)` — is rejected. It would silently smuggle classical type-checking into the Hamiltonian's *construction* (a Python lookup parameterizing the operator) while pretending the *evaluation* (`⟨ψ|H|ψ⟩`) is quantum. Sub-project E (synthesis with holes) needs `H_typing` to work uniformly on *superposed* ASTs — those have no single classical structural metadata to look up.

### 1.2 No typing rule is implemented as classical Python lookup

Every typing rule below (§3) becomes a SUM OF LOCAL/TWO-SITE OPERATORS on the MPS lattice. The energy reported by `⟨ψ|H_term|ψ⟩` for term `H_term` is the *quantum* expectation of a real Hermitian operator on a real quantum state — not the result of recursing through the AST in Python and adding `1` for each violation.

The "easy shortcut" — walking the AST, classical type-checking it, and producing a synthetic `H = (n_violations) · I` — is forbidden. It produces the right ⟨H⟩ for concrete inputs but
1. cannot be combined with H_eval (sub-project C) into a single physical Hamiltonian whose ground state is well-typed AND reduced;
2. cannot evaluate against a hole-bearing MPS where the AST does not exist concretely;
3. cannot be used as the energy functional of imaginary-time evolution (sub-project E), which is the *point* of the architecture.

### 1.3 Binder→use type coupling is realized via the bid bond

T-Var is "the type at a Var site equals the type of its binder's parameter." With A's bid bond carrying *which* binder, the principled extension carries also the binder's `param_ty` on that same channel. The implementation does this by enriching A's bid-bond channel basis from `{|no_info⟩, |c_1⟩, …, |c_L⟩}` to `{|no_info⟩} ∪ { |c_i, t⟩ : i ∈ [1, L], t ∈ TYPE_CUTOFF }` — i.e., each binder channel carries an explicit param_ty tag *as a real bond degree of freedom*.

The "easy shortcut" — taking `param_ty` from `EncodingMeta` and baking it into the operator's matrix elements at construction time — is rejected for the same reason as §1.1: it makes `H_typing` AST-dependent.

### 1.4 Non-adjacent typing constraints are realized via a per-site obligation register

Several typing rules couple non-adjacent sites: e.g., T-App's `arg.type = fn.src`, where `fn` is at site `k+1` (adjacent to APP at `k`) but `arg` may be at site `k + 1 + size(fn)`, which is *not* adjacent to APP. The principled realization is **not** a long-range Python-side calculation; it is a **per-site `tobl` (type-obligation) field species** whose value at site `k` records the type the parent expects of the subtree rooted at `k`.

The encoder writes `tobl(k)` at encode time as part of the pre-order walk (the parent always knows what type it expects of each child, locally). The Hamiltonian's `T-Obligation` term then becomes a *one-site* projector at every site: penalty `1` if `tobl(k) ≠ T_NONE` and `type(k) ≠ tobl(k)`.

This collapses every "child-of-X must have type Y" rule to one structural rule: `tobl` is a real quantum DOF, the encoder writes a real amplitude into it, and the Hamiltonian reads a real expectation. For hole-bearing programs (sub-project E), `tobl(k)` can be a superposition over candidate expected types and the constraint propagates accordingly.

The "easy shortcut" — implementing T-App's arg constraint with a long-range Python computation that goes "look up where arg is, read its type, compare to fn's src" — is forbidden. The `tobl` register is the principled bond-free realization of bonded obligations.

### 1.5 Residual energies are diagnostic, not aggregate

Each typing rule, applied at each site, is a separately named term: `H_typing_rule_at_site_k`. Sub-project D (debugger) reads `⟨ψ|H_typing_rule_at_site_k|ψ⟩` per term to localize which rule fired at which site. The implementation MUST expose this granularity. A monolithic `H_typing` that only knows its total expectation is useless for D.

### 1.6 Reuse the existing QFT machinery; do not reinvent

The existing `src/qft_pcn/qft/{fock,mps,hamiltonian}.py` provides `FieldSpecies`, `embed_op`, `MPS.local_expectation`, `MPS.two_site_expectation`. Sub-project B extends the species list to 5 and adds a `TypingHamiltonian` class that holds typing-rule *terms*, each of which can produce a dense `local_op(k)` or `bond_op(k)` of the multi-species local Hilbert space.

To stay tractable at `d_local = 65536` (see §4.6), term-energies are evaluated through a **factored expectation path** (§6.3) that exploits the tensor-product structure of per-species operators and never materializes a full `65536 × 65536` matrix in memory. The existing `MPS.local_expectation` is used only for the factored sub-operators (e.g., a one-site projector on the `type` register, embedded into the full local space by `embed_op` — that materializes one species' worth, which is much smaller).

The "easy shortcut" — building `H_typing` as a `Hamiltonian(HamiltonianConfig(...))` and shoehorning typing rules into `density_couplings` — is rejected. That config is for bosonic dynamics; typing rules are projector-like and need their own term machinery.

---

## 2. Scope

### 2.1 In scope (this sub-project)

- A new field species `tobl` (type-obligation), cutoff = `TYPE_CUTOFF = 8`.
- A small extension to sub-project A's encoder: write `tobl(k)` at every site, and write `param_ty` onto each live-binder bid-bond channel.
- A `TypingHamiltonian` class that holds typing-rule terms keyed by `(rule_id, site)`.
- A factored-expectation evaluation path: `TypingHamiltonian.term_energy(state, rule, site) -> float` and `TypingHamiltonian.total_energy(state) -> float`.
- A `residuals(state) -> dict[(str, int), float]` API that returns all per-term energies for sub-project D.
- A test suite covering: (a) ten programs with well-typed and ill-typed variants; (b) per-rule isolation (a program ill-typed under exactly one rule has nonzero residual on exactly that rule); (c) the encoder extension's correctness (channel param_ty and tobl values agree with the AST's classical typing derivation).

### 2.2 Out of scope (deferred to other sub-projects)

- Evaluation / beta-reduction Hamiltonian (sub-project C / §10.3).
- Imaginary-time evolution under `H_typing` to find well-typed completions (sub-project E / §10.7).
- TEBD integration: B does NOT make `H_typing` compatible with `evolution.trotter_step`. The Trotter step requires materializing dense local/bond ops; B defers that to sub-project E (which can handle the cost or restrict the active species).
- MERA upgrade (sub-project F / §10.4).
- LLM bridge (sub-project G / §10.5).
- Type inference (still unsupported; the encoder requires Lam params to be annotated).
- Recursion / `let rec` / `fix` (forwarded to F).
- Polymorphism, type variables.

### 2.3 Will not do, even if asked later

- Take `param_ty` or `tobl` values from `EncodingMeta` at Hamiltonian-construction time and bake them as operator coefficients. (§1.1)
- Implement typing-rule violations as a classical Python type-checker that returns a synthetic Hamiltonian with `n_violations · I`. (§1.2)
- Add long-range many-site Hamiltonian terms (the `tobl` mechanism makes them unnecessary).
- Reuse `value` register slots for typing purposes (it's owned by A's literal/op-code encoding and will be owned by C for intermediate values).

---

## 3. The typing rules, formalized as Hamiltonian terms

The surface language is exactly the one from sub-project A's spec §3:

```
e ::= x | n | b | λx:τ. e | e e | if e then e else e | e ⊕ e
τ ::= Int | Bool | τ → τ
⊕ ∈ {+, -, *, <, ==}
```

Each rule below produces an operator `H_rule` that is a sum of *local* projectors over sites. Each per-site/per-bond projector has eigenvalue `1` on the rule-violating subspace and `0` on the rule-satisfying subspace. Therefore `⟨ψ|H_rule|ψ⟩` equals the *probability* (under the MPS measurement distribution) that the rule fires at any site. For concrete (product) inputs, this probability is exactly the number of violations.

Notation:

- `P_kind=K` denotes the local projector at any one site onto the `kind = K` basis state, embedded in the full local Hilbert space via `embed_op` on the kind species.
- `P_type=T` likewise for `type`.
- `P_tobl=T` likewise for `tobl`.
- `P_value=V` likewise for `value`.
- `I_local` is the identity on the local Hilbert space.
- All operators are Hermitian (sums of projectors are Hermitian); all are positive semi-definite.

### 3.1 T-Lit-Int (one-site)

At every site `k ∈ [0, N)`:

```
H_lit_int(k) = P_kind=KIND_INT(k) · (I_local - P_type=TYPE_INT(k))
```

Interpretation: if `kind = INT` at this site, the type tag must be `T_INT`. Contributes `+1` per site where `kind=INT ∧ type ≠ T_INT`. Total:

```
H_T_lit_int = Σ_k H_lit_int(k)
```

### 3.2 T-Lit-Bool (one-site)

At every site `k`:

```
H_lit_bool(k) = P_kind=KIND_BOOL(k) · (I_local - P_type=TYPE_BOOL(k))
```

### 3.3 T-Bin-Arith and T-Bin-Cmp (one-site, output type by op code)

The BIN site discriminates on its `value` register's op code:

```
H_bin_arith(k) = P_kind=KIND_BIN(k)
              · (P_value=VALUE_PLUS(k) + P_value=VALUE_MINUS(k) + P_value=VALUE_TIMES(k))
              · (I_local - P_type=TYPE_INT(k))

H_bin_cmp(k)   = P_kind=KIND_BIN(k)
              · (P_value=VALUE_LT(k) + P_value=VALUE_EQ(k))
              · (I_local - P_type=TYPE_BOOL(k))
```

The operand-type constraints (lhs : Int, rhs : Int) are NOT in T-Bin — they are realized by T-Obligation (§3.6). The encoder writes `tobl(lhs_root) = T_INT` and `tobl(rhs_root) = T_INT` at every BIN site's children. The same applies for T-Bin-Cmp's operands.

### 3.4 T-Var (two-site, between the VAR site and the binder's bond channel)

This is the rule that *forces* the bid-bond extension (§5.2). At a Var site `v`, the type must equal the binder's `param_ty`. The binder's param_ty rides on its bid channel, which is part of the *bond* between the site `v` and its left neighbor `v−1`.

The two-site operator on bond `(v−1, v)` reads, for each channel `c` carrying a binder with param_ty `t`:

```
H_var_at_v = P_kind=KIND_VAR(v) · Σ_t [ Π_c,t(v−1, v) · (I_local(v) - P_type=t(v)) ]
```

where `Π_c,t(v−1, v)` is the bond projector onto "the bid bond's channel `c` carries param_ty `t`." Because param_ty is a *real bond DOF* (§5.2), this projector is realized as a two-site projector on the joint `(d_local × d_local)` Hilbert space of sites `v−1` and `v` — specifically, it projects onto MPS-bond-states where the bid-bond's channel-with-param-ty is exactly `|c, t⟩`. The encoder's tensor construction ensures that the only nonzero bond states are the ones where `t = binder.param_ty` for binder occupying channel `c`.

Summed over sites:

```
H_T_var = Σ_v H_var_at_v
```

Implementation detail: `H_var_at_v` is a two-site operator on bond `(v−1, v)`. Its term-energy is computed via `MPS.two_site_expectation` with the operator built in factored form on the kind/type/bid species (value and tobl are identity-passing). Sites with `v = 0` (no left bond) cannot be Var sites by definition (the encoder writes site `0` as the root expression's kind, which is never Var in a well-scoped program — a free Var raises `IllScopedVar`). The Hamiltonian still includes `H_var_at_0` as a term; its energy is automatically 0 because `P_kind=KIND_VAR(0) = 0` on the encoder's output.

### 3.5 T-Abs (two-site, between LAM and the bond's binder channel)

A `Lam x:S. body` site has `type = TArrow(S, B)` where `B = type(body)`. The encoder writes both `LAM.type` (its flat-arrow tag) and the binder's `param_ty = S` onto the outgoing bid channel. **T-Abs verifies these are consistent**: the param_ty riding on the LAM's outgoing channel must equal `src(LAM.type)`.

For each flat arrow tag `a` with `src(a) = s`:

```
H_abs_at_lam(l) = P_kind=KIND_LAM(l) · P_type=a(l) · (I - P_outgoing_channel_param_ty=s(l, l+1))
```

The operator `P_outgoing_channel_param_ty=s(l, l+1)` is a two-site projector on the bond `(l, l+1)` (specifically on its bid register) that fires when the outgoing channel's param_ty is `s`. For `a = T_ARR_NESTED`, the `src` is unknown at the rule level — T-Abs *passes* (the obligation is recorded in `meta.nested_type_index` and is the encoder's responsibility; the typing Hamiltonian does not penalize `T_ARR_NESTED` sites).

The body-type half of T-Abs (`body.type = dst(LAM.type)`) is handled by T-Obligation (§3.6): the encoder writes `tobl(body_root) = dst(LAM.type)`. So T-Abs at the LAM site itself is just the src-consistency check above.

### 3.6 T-Obligation (one-site, every site)

This is the rule that handles every "child must have type X" constraint where X is determined by the parent's local state. The encoder writes `tobl(k)` for every site at encode time, with the following meanings:

- `tobl(root) = T_NONE` — the AST root has no parent expectation.
- For LAM's body: `tobl(body_root) = dst(LAM.type)`.
- For APP's fn: `tobl(fn_root) = T_NONE` — APP doesn't impose a single type on fn (fn must be an arrow with a specific dst, handled by T-App-Arrow §3.7). Encoder writes T_NONE here intentionally.
- For APP's arg: `tobl(arg_root) = src(fn.type)`, where `fn.type` is computed at encode time from the AST.
- For IF's cond: `tobl(cond_root) = T_BOOL`.
- For IF's then-branch: `tobl(then_root) = type(IF)`.
- For IF's else-branch: `tobl(else_root) = type(IF)`.
- For BIN's lhs/rhs: `tobl(operand_root) = T_INT` (always — both arith and cmp ops require Int operands).
- For all other sites (PAD, non-root-of-subtree sites): `tobl(k) = T_NONE`.

Note that "root of subtree" is the FIRST site of a child subtree, by pre-order layout. The encoder visits children in canonical order, so `child_k_root_site_offset` is computable as it walks. The encoder writes `tobl` exactly at these positions.

The one-site Hamiltonian term:

```
H_obligation(k) = Σ_{t ≠ T_NONE} P_tobl=t(k) · (I_local - P_type=t(k))
```

i.e., for each non-`T_NONE` tobl tag `t`, project onto sites where `tobl = t` and `type ≠ t`. Summed over sites:

```
H_T_obligation = Σ_k H_obligation(k)
```

Note: this rule does NOT distinguish "which AST rule did this come from" — it lumps T-Abs-body, T-App-arg, T-If-*, T-Bin-* together under one mechanism. For diagnostic purposes (§1.5), the per-site label is enough: D reads `⟨H_obligation(k)⟩` per site and combines it with the encoder's `meta.tobl_per_site[k]` to report "site k expected type T but had type T'." This is acceptable because the rule that was violated is unambiguously identifiable from `tobl(k)` and the AST kind at site `k`'s parent (which D reads from `meta.site_to_ast_path`).

### 3.7 T-App-Arrow (two-site, between APP site and the fn-root site)

T-App says: `f : A → B` and `a : A` imply `f a : B`. The B-part is `type(APP) = dst(fn.type)`, which is encoded as a two-site constraint between APP at site `k` and fn at site `k + 1` (adjacent by pre-order). The A-part (arg.type = fn.src) is handled by T-Obligation (§3.6) via `tobl(arg_root) = fn.src`.

For each pair `(y, a)` with `dst(a) = y`:

```
H_app_arrow(k) = P_kind=KIND_APP(k) · Σ_y P_type=y(k) · (I - Σ_{a: dst(a)=y} P_type=a(k+1))
```

i.e., if site `k` is APP with type `y`, site `k+1` must have a type whose dst is `y`. The legal types at `k+1` for a given `y` are the flat arrow tags with `dst(a) = y`, namely:

| y           | legal a (`type(k+1)`)                                       |
|-------------|-------------------------------------------------------------|
| `T_INT`     | `T_ARR_II, T_ARR_BI, T_ARR_NESTED*`                         |
| `T_BOOL`    | `T_ARR_IB, T_ARR_BB, T_ARR_NESTED*`                         |
| `T_ARR_*`   | `T_ARR_NESTED*`                                             |
| `T_NONE`    | (impossible: APP always has a non-NONE type when well-typed) |

`T_ARR_NESTED*` is always allowed (we cannot inspect its internal structure from the flat tag, so we permissively accept it; the encoder's `meta.nested_type_index` carries the full Ty and could enforce it in a strengthened version, but at the typing-Hamiltonian level we conservatively accept).

Summed:

```
H_T_app_arrow = Σ_k H_app_arrow(k)
```

The `(arg.type = fn.src)` half: as noted, the encoder writes `tobl(arg_root) = fn.src`, and T-Obligation enforces it. No separate `H_T_app_arg` term is needed.

### 3.8 T-If (no separate rule, fully covered by T-Obligation)

The `tobl` mechanism handles:
- `cond.type = Bool` via `tobl(cond_root) = T_BOOL`,
- `then.type = T` via `tobl(then_root) = type(IF)`,
- `else.type = T` via `tobl(else_root) = type(IF)`.

The `(then.type = else.type)` invariant is implied by both being equal to `type(IF)`. There is no separate `H_T_if` term.

### 3.9 Full Hamiltonian

```
H_typing = H_T_lit_int + H_T_lit_bool + H_T_bin_arith + H_T_bin_cmp
         + H_T_var + H_T_abs + H_T_obligation + H_T_app_arrow
```

For a well-typed AST, `⟨ψ|H_typing|ψ⟩ = 0` exactly (machine precision). For an ill-typed AST, each violating site contributes `+1` to exactly one rule's term-energy.

### 3.10 What "well-typed" means here

A program is "well-typed" iff the standard STLC declarative typing derivation succeeds. The encoder, given a well-typed AST, writes registers such that every per-rule term-energy is zero. The encoder, given an ill-typed AST, writes registers that *reflect the typing derivation's local rules* — and the Hamiltonian counts where the derivation fails locally. (For instance, if a programmer writes `(λx:Int. x + true)`, the encoder will write `tobl(true_lit_site) = T_INT` (the BIN's rhs obligation) and `type(true_lit_site) = T_BOOL`; `H_T_obligation` fires `+1` at that site.)

For programs with mixed errors, the energies sum: ill-typed at three places gives `⟨H_typing⟩ = 3`.

---

## 4. Field species

### 4.1 The species list

Sub-project B extends A's `SPECIES` from 4 species to 5:

```python
SPECIES_B: tuple[FieldSpecies, ...] = (
    FieldSpecies(name="kind",  cutoff=8,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="type",  cutoff=8,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="bid",   cutoff=8,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="value", cutoff=16, bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="tobl",  cutoff=8,  bare_mass=0.0, kinetic=0.0),
)
```

The first four match A's `SPECIES`. The fifth is new.

### 4.2 `tobl` basis

Reuses the `type` register's basis encoding 1:1:

| dim | name      | meaning |
|---:|-----------|---------|
| 0  | TOBL_NONE | no obligation |
| 1  | TOBL_INT  | parent expects `Int` |
| 2  | TOBL_BOOL | parent expects `Bool` |
| 3  | TOBL_ARR_II | parent expects `Int → Int` |
| 4  | TOBL_ARR_IB | parent expects `Int → Bool` |
| 5  | TOBL_ARR_BI | parent expects `Bool → Int` |
| 6  | TOBL_ARR_BB | parent expects `Bool → Bool` |
| 7  | TOBL_ARR_NESTED | parent expects a higher-order arrow (full Ty in `EncodingMeta.nested_tobl_index[k]`) |

Cutoff = 8 = `TYPE_CUTOFF`. The basis matches the `type` register so that `H_T_obligation` can compare `tobl(k)` and `type(k)` symbolically.

### 4.3 Local Hilbert space

Per site: `d_local_B = 8 × 8 × 8 × 16 × 8 = 65536`.

This is 8× A's `d_local_A = 8192`. A dense `d_local_B × d_local_B` operator weighs ≈ 64 GB and is therefore **never materialized** in B's code. All Hamiltonian-term operators are kept in factored form on the per-species sub-Hilbert spaces and applied via the factored-expectation path (§6.3).

The MPS site tensors themselves are kept in factored form (the same factorization A uses), so the encoded MPS does not require 64 GB either — each per-species sub-tensor is small.

### 4.4 Local-basis ordering

Per A's convention (leftmost species changes slowest):

```
flat_index(s_kind, s_type, s_bid, s_value, s_tobl)
  = s_kind * (T * B * V * O)
  + s_type * (B * V * O)
  + s_bid  * (V * O)
  + s_value * O
  + s_tobl
```

where `T=8, B=8, V=16, O=8`.

This extends A's 4-species ordering by inserting `tobl` as the *innermost* (fastest-varying) species. The encoder code can reuse the original 4-species basis computation by treating the result as the multi-index over the outer four species and writing the chosen `tobl` index in the inner range `[s_tobl, s_tobl + 1)` of an 8-wide stride.

### 4.5 Default values

At every site by default, `tobl = TOBL_NONE = 0`. The encoder writes `tobl(k) ≠ 0` only at parent-imposed-obligation positions (§3.6). PAD sites have `tobl = TOBL_NONE`.

### 4.6 Memory budget for site tensors

A's per-site factored tensor has shape `(χ_l, 8192, χ_r)`. With the bid-bond extension (§5.2), `χ` grows by a factor of `8` on bonds with live binders (each channel's param_ty tag adds a factor of 8). Worst-case site tensor (factored): `32 × 65536 × 32 ≈ 67 M complex entries ≈ 1 GB`. We use the same sparse construction A uses (§5.4) — only the nonzero slices are materialized.

In practice the per-site tensor has the form `(χ_l, kind_idx, type_idx, bid, value_idx, tobl_idx, χ_r)` with `kind_idx, type_idx, value_idx, tobl_idx` all *definite* (one nonzero per site) and only `bid` × bond carrying real entanglement. Total nonzero entries per site = `χ_l × BID_CUTOFF × χ_r ≤ 32 × 8 × 32 = 8192`.

---

## 5. Encoder extension (A's encoder + new responsibilities)

Sub-project B requires sub-project A's encoder to do three extra things. We document them here as a small follow-up to A. Implementing this extension is part of B's plan (Task 1 of B's plan).

### 5.1 Write `tobl(k)` at every site

The encoder's existing pre-order walk passes a "current type obligation" parameter through the descent. At each child invocation, the parent passes the type it expects of that child (or `T_NONE` if no expectation). The child writes that value into its local `tobl` register and recurses.

Specifically (using A's `_serialize.serialize_preorder` machinery):

| parent kind | child position | tobl(child_root)            |
|-------------|----------------|------------------------------|
| (root)      | —              | `T_NONE`                     |
| `LAM`       | body           | `dst(LAM.type)` as a flat tag |
| `APP`       | fn             | `T_NONE` *                   |
| `APP`       | arg            | `src(fn.type)` as a flat tag * |
| `IF`        | cond           | `T_BOOL`                     |
| `IF`        | then-branch    | `type(IF)`                   |
| `IF`        | else-branch    | `type(IF)`                   |
| `BIN`       | lhs            | `T_INT`                      |
| `BIN`       | rhs            | `T_INT`                      |

\* For APP, the encoder uses the AST-computed `fn.type` (which it already computes for `type_tag` in A's `_types.compute_site_types`). When `fn.type = T_ARR_NESTED`, the encoder writes `tobl(arg_root) = T_ARR_NESTED` AND stores the corresponding full `Ty` in `meta.nested_tobl_index[arg_root]`. The Hamiltonian's T-Obligation treats `T_ARR_NESTED` tobl as a permissive constraint (accepts any T_ARR_NESTED-tagged type at the child; full match is deferred to a refined version).

### 5.2 Write `param_ty` onto each live-binder bid-bond channel

A's bond bid-register basis is `{|no_info⟩, |c_1⟩, …, |c_L⟩}`. B extends this to:

```
{ |no_info⟩ } ∪ { |c_i, t⟩ : i ∈ [1, L], t ∈ [TYPE_CUTOFF] }
```

i.e., each binder channel is replaced by an 8-slot orbit indexed by param_ty. The new basis size per bond is `1 + 8|L|`.

The encoder writes, for each live binder `b` on a bond, a definite param_ty: `t = ty_to_tag(b.param_ty)` (A's `_types.ty_to_tag` already exists). The bond projector onto `|c, t⟩` for the actual binder is amplitude `1`; all other `|c, t'⟩` for `t' ≠ t` are amplitude `0`.

Concretely, A's `_tensors._bid_bond_tensor_at_site` builds a tensor of shape `(L_in + 1, BID_CUTOFF, L_out + 1)`. B's extension changes the bond-channel dimensions to `(1 + 8 L_in, BID_CUTOFF, 1 + 8 L_out)`. Index 0 is `no_info`; indices `1 + 8(i-1) + t = 1 + 8(i-1) + t` (for `i ∈ [1, L], t ∈ [TYPE_CUTOFF]`) is `|c_i, t⟩`.

For each binder `c_i` on a bond, only the slot `1 + 8(i-1) + ty_to_tag(c_i.param_ty)` is populated (amplitude `1` for that slot); the other 7 slots in `c_i`'s orbit are 0. The encoder's pass-through rules and LAM/VAR rules from A's §5.4 generalize naturally: a LAM site introducing binder `b` writes into its outgoing channel's `|c_b, ty_to_tag(b.param_ty)⟩` slot; a VAR site reading channel `c_b` reads from `|c_b, t⟩` for the SPECIFIC param_ty tag of its binder (because the channel only has one slot populated).

### 5.3 New `EncodingMeta` fields

A's `EncodingMeta` is extended with two new fields:

```python
@dataclass
class EncodingMeta:
    # ... existing fields ...
    tobl_per_site: list[int]          # tobl(k) for k in [0, N)
    nested_tobl_index: dict[int, Ty]  # site -> full Ty when tobl(k) = TOBL_ARR_NESTED
    # The channel param_ty is recoverable from live_binders_per_bond + the
    # binders' param_ty (which is in the AST). For convenience we also expose:
    channel_param_ty_per_bond: list[list[int]]  # parallel to live_binders_per_bond
```

Note: `channel_param_ty_per_bond` is *metadata*, not a substitute for the bond DOF. The Hamiltonian's T-Var rule (§3.4) reads the bond DOF, not this metadata. The metadata is exposed for sub-project D (diagnostics) and for testing.

### 5.4 Backward compatibility

The encoder's existing 4-species output continues to be supported when `with_typing_extension=False` (default `True` after this extension). For B's tests, the encoder is invoked with `with_typing_extension=True`. For A's existing tests, `with_typing_extension=False` is set in a compatibility shim or the tests are updated to reflect the new 5-species output (which is equivalent on the original 4 species for `tobl=TOBL_NONE` everywhere; the 5th register is a product factor that contributes identity).

**Decision: the extension is the new default.** A's tests are updated to expect 5 species. The old 4-species code path is removed. All callers of `encode()` see the extended output. This keeps the codebase from carrying two parallel encoders.

### 5.5 chi_max for B's encoder

Bond dim per binder = `1 + 8|L|`. For acceptance tests, |L| ≤ 3 → bond dim ≤ 25. For A's `TooManyBinders` cap (|L| ≤ 6 → bond dim ≤ 49), we'd need chi_max ≥ 64.

Default chi_max for B's encoder: `chi_max=32` (covers |L| ≤ 3). For programs with deeper binders, callers pass `chi_max=64` explicitly.

A's existing default `chi_max=16` becomes inadequate for the extended encoder when |L| ≥ 2. We update A's default to `chi_max=32` AS PART OF B's encoder-extension task. This is a small breaking change to A's API; documented in the plan.

---

## 6. Hamiltonian construction

### 6.1 API

```python
# src/qft_pcn/logic/typing_hamiltonian.py

@dataclass(frozen=True)
class TypingTerm:
    """A single typing-rule term, addressable by (rule_id, site)."""
    rule_id: str             # one of: "T-Lit-Int", "T-Lit-Bool", "T-Bin-Arith",
                             #         "T-Bin-Cmp", "T-Var", "T-Abs",
                             #         "T-Obligation", "T-App-Arrow"
    site: int                # the lattice site at which this term acts
                             # (for two-site terms, the LEFT site of the bond)
    arity: int               # 1 (one-site) or 2 (two-site)


class TypingHamiltonian:
    """The structural typing Hamiltonian for an N-site lattice.

    Built once for any program of size <= N (per §1.1). The same instance
    evaluates ⟨H_typing⟩ on any MPS produced by the extended encoder.
    """

    SPECIES: tuple[FieldSpecies, ...]   # the 5-species list (§4.1)
    N: int
    terms: list[TypingTerm]             # all the (rule, site) terms

    def __init__(self, N: int): ...

    # Per-term energy (the diagnostic API used by sub-project D).
    def term_energy(self, state: MPS, term: TypingTerm) -> float: ...

    # Total energy.
    def total_energy(self, state: MPS) -> float: ...

    # All per-term residuals, keyed by (rule_id, site).
    def residuals(self, state: MPS) -> dict[tuple[str, int], float]: ...

    # Operators (for sub-project E's TEBD integration; not used by B itself).
    # These materialize dense operators and are expensive at d=65536; callers
    # restrict to programs small enough to handle the cost, or fall back to
    # term_energy for energy-only evaluation.
    def local_op_dense(self, site: int) -> np.ndarray: ...   # (d, d)
    def bond_op_dense(self, site: int) -> np.ndarray: ...    # (d^2, d^2)
```

Optional `term_id` resolution: clients call `H.term_energy(state, TypingTerm("T-Var", 5, 2))` to get the residual for T-Var at site 5 (the bond `(4, 5)`).

### 6.2 Term enumeration

The constructor enumerates terms via a deterministic procedure:

- For each site `k ∈ [0, N)`: add T-Lit-Int(k), T-Lit-Bool(k), T-Bin-Arith(k), T-Bin-Cmp(k), T-Obligation(k) as one-site terms.
- For each `k ∈ [1, N)`: add T-Var(k) as a two-site term on bond `(k−1, k)`.
- For each `k ∈ [0, N)`: add T-Abs(k) as a two-site term on bond `(k, k+1)` (skip `k = N−1`).
- For each `k ∈ [0, N−1)`: add T-App-Arrow(k) as a two-site term on bond `(k, k+1)`.

Total term count: roughly `5N + 3(N−1) ≈ 8N`. For `N=32`, ≈ 256 terms. Each term's energy is independent of the others, so they can be evaluated in any order.

### 6.3 Factored expectation evaluation

The naïve implementation materializes a `65536 × 65536` operator per term (64 GB) — infeasible. The factored path exploits two structural facts:

1. **Each typing rule's operator factors as a tensor product over per-species local operators.** E.g., `H_lit_int(k) = P_kind=INT(k) ⊗ (I_type - P_type=INT(k)) ⊗ I_bid ⊗ I_value ⊗ I_tobl`. Each factor is a small dense matrix (8×8 or 16×16).

2. **`MPS.local_expectation` and `MPS.two_site_expectation` accept the embedded operator, but they don't need it to be a single dense matrix — they only need to contract it against the MPS site tensors.** We provide an embedded helper `embed_factored_local(op_factors)` that builds the relevant dense operator on-the-fly *only for the species that are non-identity* in the factorization, and uses `embed_op` to lift it. This produces an 8×8, 8×8×8×8 = 64×64, or 8192×8192 operator at worst — never the full 65536×65536.

The implementation pattern is:

```python
def _local_term_energy(self, state: MPS, op_factors: dict[str, np.ndarray],
                       site: int) -> float:
    """op_factors maps species name -> small dense matrix on that species'
    sub-space. Species not in op_factors are treated as identity."""
    # Build a dense local operator on the SUB-Hilbert space of the species
    # that participate, via successive embed_op calls. Then take expectation
    # by contracting only those species' contributions to the local basis.
    active_species = list(op_factors.keys())
    # ... build op_local of shape (d_active, d_active) via embed_op ...
    # ... apply via state.local_expectation with the operator embedded into
    # the full d_local basis using a per-species kronecker product ...
    return float(state.local_expectation(site, op_local_full))
```

For the largest term (T-Var, which acts on kind/type/bid simultaneously, dim 8×8×8 = 512 sub-Hilbert space), the embedded operator is `512×512` factored into the full `8192 × 8192` sub-space (taking value and tobl identity), then into the full `65536 × 65536` is wasteful BUT: we only need the *expectation value*, not the operator itself. So we use the factored `MPS.two_site_expectation` with the operator built only on the active species' joint space, expanded to the full local space via `embed_op` *on the fly* (returning the dense `65536×65536` matrix would still be 64 GB).

**Resolution**: implement a custom factored expectation helper `_factored_local_expectation(state, site, op_factors)` and `_factored_two_site_expectation(state, site, op_factors)` in `typing_hamiltonian.py`. These contract per-species against the MPS site tensor's local-register slices directly, never materializing the full dense operator.

Concretely, for a one-site operator factored as `(O_kind, O_type, I, I, I)`:

```python
# state.tensors[site] has shape (chi_l, d_local, chi_r)
# View the local index as the multi-index (s_kind, s_type, s_bid, s_value, s_tobl).
# Reshape: A[k] -> shape (chi_l, 8, 8, 8, 16, 8, chi_r).
# Apply O_kind on axis 1, O_type on axis 2.
A = state.tensors[site].reshape(chi_l, 8, 8, 8, 16, 8, chi_r)
A_new = np.einsum('kK,Khjbtm->khjbtm', O_kind, A)
A_new = np.einsum('hH,kHjbtm->khjbtm', O_type, A_new)
# Then contract A_new against state.tensors[site].conj() via the environment.
# This is local_expectation but on the factored representation.
```

This avoids materializing the `65536 × 65536` dense operator and runs in ~`d_local × χ²` time per term, which is `65536 × 32² ≈ 67M` ops per term — comfortable.

### 6.4 `total_energy` and `residuals` are evaluated by iterating `term_energy`

```python
def total_energy(self, state: MPS) -> float:
    return sum(self.term_energy(state, t) for t in self.terms)

def residuals(self, state: MPS) -> dict[tuple[str, int], float]:
    return {(t.rule_id, t.site): self.term_energy(state, t) for t in self.terms}
```

No caching at the framework level; D may cache as needed.

### 6.5 `local_op_dense` and `bond_op_dense`

For E's TEBD integration: these materialize the dense operator at one site / one bond by summing the relevant term-factored operators. Dense form is `(65536, 65536)` ≈ 64 GB — these methods raise `MemoryError` unless the caller is in a configuration where this is tolerable (in practice, sub-project E will need either reduced cutoffs or a dedicated factored-Trotter step). **B's tests do NOT call these methods.** They are documented and skeletoned for E to flesh out.

---

## 7. Acceptance tests

Located in `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`. Uses the existing pytest convention.

### 7.1 The five well-typed programs (residual = 0)

```
WT1.  λx:Int. x                                        — T-Var, T-Abs
WT2.  (λx:Int. x + 1)(2)                               — T-App, T-Bin-Arith, T-Var
WT3.  λf:Int->Int. λx:Int. f (f x)                     — nested binders, T-App chain
WT4.  if (1 < 2) then ((λx:Bool. x)(true)) else false  — T-If, T-Bin-Cmp, T-App, T-Var
WT5.  (λx:Int. (λy:Int. x + y)(3))(4)                  — interior shadowing, T-App nest
```

For each WTi:

```python
def test_well_typed_residual_zero_WT{i}():
    p = parse(WT_i_source)
    state, meta = encode(p, N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    energy = H.total_energy(state)
    assert abs(energy) < 1e-10, (
        f"{WT_i_source}: expected ⟨H_typing⟩ = 0, got {energy}; "
        f"residuals: {H.residuals(state)}"
    )
```

### 7.2 The five ill-typed programs (residual > 0)

Each program isolates exactly one rule's violation:

```
IT1.  λx:Int. (x + true)
        Violation: T-Obligation at the 'true' lit site (parent BIN's rhs
        expected T_INT, got T_BOOL). Expected: total_energy ≈ 1, residual on
        T-Obligation at the 'true' site = 1.

IT2.  λx:Int. (if x then 1 else 0)
        Violation: T-Obligation at 'x' (IF's cond expected T_BOOL, got T_INT).
        Expected: total_energy ≈ 1.

IT3.  λx:Int. ((1)(x))
        Violation: T-App-Arrow at the APP site — fn (the IntLit 1) has
        type T_INT, not an arrow type. Expected: total_energy ≈ 1, residual on
        T-App-Arrow at the APP site = 1.

IT4.  λx:Int->Bool. (x 1) + 1
        Violation: T-Obligation at the rhs of the outer + (which is IntLit 1,
        type Int — wait, that's correct). The real violation is that (x 1)
        has type Bool but the BIN(+, _, IntLit) expects T_INT on the lhs.
        Expected: total_energy ≈ 1, residual on T-Obligation at the (x 1)
        APP site = 1.

IT5.  (λx:Bool. x + 1)(true)
        Violations: TWO. (a) T-Obligation at the inner 'x' (BIN's lhs
        expected T_INT, got T_BOOL since x:Bool). (b) T-Var-mismatch: x
        is bound to Bool, but BIN's lhs obligation is Int. Note that
        T-Var itself checks "x's type = x's binder's param_ty" which IS
        satisfied (x.type = Bool, binder.param_ty = Bool). The violation
        is T-Obligation at the 'x' use site. Expected: total_energy ≈ 1.
```

Tests:

```python
def test_ill_typed_residual_localized_IT{i}():
    p = parse(IT_i_source)
    state, meta = encode(p, N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    energy = H.total_energy(state)
    assert energy > 0.5, f"{IT_i_source}: expected ⟨H⟩ > 0, got {energy}"
    residuals = H.residuals(state)
    # Check exactly one term fires with energy near 1; everything else is ~0.
    big = [k for k, v in residuals.items() if v > 0.5]
    small = [k for k, v in residuals.items() if 1e-10 < v <= 0.5]
    assert len(big) == 1, f"expected exactly 1 term firing, got {big + small}"
    assert big[0] == EXPECTED_VIOLATION_KEY_FOR_IT[i]
```

`EXPECTED_VIOLATION_KEY_FOR_IT` is hard-coded per program above.

### 7.3 Per-rule isolation: T-Var

Construct an MPS by hand that has `kind = VAR, type = T_BOOL` at site 1, with the bid bond carrying a channel with `param_ty = T_INT`. The encoder won't produce this naturally (well-typed encoder respects the program), but the test constructs the MPS tensor directly by modifying an encoded P1's site-1 tensor:

```python
def test_t_var_isolated_violation():
    """Manually inject a T-Var violation by overwriting the VAR site's type."""
    state, meta = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    # Site 1 is the VAR; overwrite its type from T_INT to T_BOOL.
    state = _mutate_local_register(state, site=1, register="type",
                                   old=TYPE_INT, new=TYPE_BOOL)
    H = TypingHamiltonian(N=8)
    residuals = H.residuals(state)
    # T-Var at site 1 should fire.
    assert residuals[("T-Var", 1)] > 0.99
    # T-Obligation at site 1 should also fire (the obligation is presumably
    # also TYPE_INT) — that's fine; we just check T-Var fires.
    assert residuals[("T-Lit-Int", 1)] < 1e-10  # site 1 is VAR, not INT
```

`_mutate_local_register` is a test helper that swaps two basis-state slices in a site tensor — defined in the test module.

### 7.4 Per-rule isolation: T-App-Arrow

```python
def test_t_app_arrow_isolated_violation():
    """Encode (λx:Int. x)(1): well-typed. Now overwrite APP site's type
    from T_INT to T_BOOL — the T-App-Arrow rule should fire (fn's type
    T_ARR_II has dst=T_INT, but APP claims dst=T_BOOL)."""
    state, meta = encode(parse(r"(\x:Int. x)(1)"), N=8, chi_max=32)
    state = _mutate_local_register(state, site=0, register="type",
                                   old=TYPE_INT, new=TYPE_BOOL)
    H = TypingHamiltonian(N=8)
    residuals = H.residuals(state)
    assert residuals[("T-App-Arrow", 0)] > 0.99
```

### 7.5 T-Lit-Int / T-Lit-Bool isolation

```python
def test_t_lit_int_isolated_violation():
    """Overwrite an IntLit's type to T_BOOL."""
    state, meta = encode(parse(r"\x:Int. 3"), N=8, chi_max=32)
    state = _mutate_local_register(state, site=1, register="type",
                                   old=TYPE_INT, new=TYPE_BOOL)
    H = TypingHamiltonian(N=8)
    assert H.residuals(state)[("T-Lit-Int", 1)] > 0.99


def test_t_lit_bool_isolated_violation():
    state, meta = encode(parse(r"\x:Int. true"), N=8, chi_max=32)
    state = _mutate_local_register(state, site=1, register="type",
                                   old=TYPE_BOOL, new=TYPE_INT)
    H = TypingHamiltonian(N=8)
    assert H.residuals(state)[("T-Lit-Bool", 1)] > 0.99
```

### 7.6 T-Bin-Arith / T-Bin-Cmp isolation

```python
def test_t_bin_arith_isolated_violation():
    """A BIN(+, ...) whose output type is T_BOOL — should fire T-Bin-Arith."""
    state, meta = encode(parse(r"\x:Int. 1 + 2"), N=8, chi_max=32)
    # Site 1 is BIN(+); change its type to T_BOOL.
    state = _mutate_local_register(state, site=1, register="type",
                                   old=TYPE_INT, new=TYPE_BOOL)
    H = TypingHamiltonian(N=8)
    assert H.residuals(state)[("T-Bin-Arith", 1)] > 0.99
```

### 7.7 Encoder extension correctness

```python
def test_encoder_writes_tobl_correctly():
    """For every well-typed program, the encoder's tobl values agree with
    the rule table in §5.1."""
    for src in [WT1, WT2, WT3, WT4, WT5]:
        p = parse(src)
        state, meta = encode(p, N=32, chi_max=32)
        # The encoder exposes tobl_per_site:
        expected = _compute_expected_tobl_from_ast(p, N=32)
        assert meta.tobl_per_site == expected, (
            f"{src}: tobl_per_site = {meta.tobl_per_site}; expected {expected}"
        )


def test_encoder_writes_channel_param_ty_correctly():
    """For each live binder on each bond, the channel's param_ty tag matches
    the binder's param_ty in the AST."""
    p = parse(r"\f:Int->Int. \x:Int. f x")
    state, meta = encode(p, N=16, chi_max=32)
    # Bond 0 (between Lam_f and Lam_x): one binder (f), param_ty = Int->Int.
    assert meta.channel_param_ty_per_bond[0] == [TYPE_ARR_II]
    # Bond 1 (between Lam_x and App): two binders (f, x), both Int variants.
    assert meta.channel_param_ty_per_bond[1] == [TYPE_ARR_II, TYPE_INT]
```

### 7.8 Hamiltonian-construction is structural (§1.1 check)

```python
def test_typing_hamiltonian_is_structural():
    """The Hamiltonian must be constructable without an AST. The same
    instance evaluates correctly on multiple ASTs."""
    H = TypingHamiltonian(N=32)
    # Don't pass an AST or any meta to the constructor.
    # Now evaluate on two unrelated ASTs.
    state1, _ = encode(parse(r"\x:Int. x"), N=32, chi_max=32)
    state2, _ = encode(parse(r"(\x:Int. x + 1)(2)"), N=32, chi_max=32)
    assert abs(H.total_energy(state1)) < 1e-10
    assert abs(H.total_energy(state2)) < 1e-10


def test_typing_hamiltonian_does_not_walk_ast():
    """The Hamiltonian must not store or reference any AST data."""
    H = TypingHamiltonian(N=32)
    # Verify no AST-derived attributes leak into the H instance:
    forbidden_attr_substrings = ["ast", "node", "tree", "binder_handle",
                                 "var_ref", "occupancy"]
    for attr_name in dir(H):
        if attr_name.startswith("_"): continue
        for sub in forbidden_attr_substrings:
            assert sub not in attr_name.lower(), (
                f"H.{attr_name} suggests AST dependency"
            )
```

### 7.9 Residual decomposition (§1.5 check)

```python
def test_residuals_sum_to_total_energy():
    """Σ term_energies == total_energy (within rounding)."""
    for src in [WT2, IT1, IT2, IT3]:
        p = parse(src)
        state, _ = encode(p, N=32, chi_max=32)
        H = TypingHamiltonian(N=32)
        total = H.total_energy(state)
        residuals_sum = sum(H.residuals(state).values())
        assert abs(total - residuals_sum) < 1e-10


def test_residuals_addressable_by_rule_and_site():
    """The per-term residuals expose both rule_id and site."""
    state, _ = encode(parse(WT2_source), N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    r = H.residuals(state)
    for (rule_id, site), val in r.items():
        assert isinstance(rule_id, str)
        assert 0 <= site < 32
        assert val >= 0.0   # Hermitian projector → nonneg expectation
```

### 7.10 Performance budget

```python
@pytest.mark.timeout(10)
def test_total_energy_under_budget():
    """Encoding WT5 and evaluating its full Hamiltonian energy completes
    in under 5 seconds."""
    p = parse(WT5_source)
    state, _ = encode(p, N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    for _ in range(5):
        _ = H.total_energy(state)
```

If this test fails, the factored-expectation path (§6.3) was not implemented properly. Do not relax the budget without a written justification.

### 7.11 Cross-check against classical type-checker

A small reference type-checker `_classical_type_check(ast) -> dict[(int, str), bool]` in the test module assigns "would this rule be violated at site k of the AST" by traversing the AST in Python. The Hamiltonian's `residuals(state)` must agree:

```python
def test_hamiltonian_agrees_with_classical_checker():
    for src in [WT1, WT2, WT3, WT4, WT5, IT1, IT2, IT3, IT4, IT5]:
        p = parse(src)
        classical = _classical_type_check(p, N=32)  # site -> (rule, violated)
        state, _ = encode(p, N=32, chi_max=32)
        H = TypingHamiltonian(N=32)
        q = H.residuals(state)
        for (rule, site), violated in classical.items():
            energy = q.get((rule, site), 0.0)
            if violated:
                assert energy > 0.5, f"{src}: {rule}@{site} expected fire"
            else:
                assert energy < 1e-10, f"{src}: {rule}@{site} false positive"
```

This is the strongest agreement test: it verifies that the quantum Hamiltonian's per-rule per-site energies match the classical type-checker's outcomes EXACTLY. The classical checker is the spec; the Hamiltonian must agree.

### 7.12 No regressions in sub-project A's test suite

Sub-project B's encoder extension changes A's encoder. After the extension, all 100+ tests in A's suite (test_logic_*.py) must still pass.

```bash
.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_*.py -v
# expect: every test green
```

---

## 8. File layout

```
src/qft_pcn/logic/
├── encoding.py                 # extend SPECIES list, add TOBL_*,
│                               #   nested_tobl_index, channel_param_ty_per_bond
│                               #   to EncodingMeta
├── _serialize.py               # _emit() walks now passes parent-imposed
│                               #   obligations; NodeOccupancy gains tobl field
├── _tensors.py                 # build_site_tensors generalized to write
│                               #   tobl on tobl register and param_ty on
│                               #   each bid channel
├── _channels.py                # compute_channel_param_ty(sites) added
├── _typing_extension.py        # NEW: helpers to compute parent obligations
│                               #   during the pre-order walk
├── encoder.py                  # encode() now writes 5-species output;
│                               #   new fields populated in EncodingMeta
└── typing_hamiltonian.py       # NEW: TypingHamiltonian class, TypingTerm,
                                #   per-term factored-expectation evaluators
src/qft_pcn/tests/
└── test_logic_typing_hamiltonian.py    # all tests from §7
```

No changes to `src/qft_pcn/qft/*` are required: typing-Hamiltonian works against the existing `MPS` and `FieldSpecies` classes. The factored-expectation helpers live in B's new module, not in `qft/`.

Public exports added to `src/qft_pcn/logic/__init__.py`:

```python
from .encoding import (
    TOBL_NONE, TOBL_INT, TOBL_BOOL,
    TOBL_ARR_II, TOBL_ARR_IB, TOBL_ARR_BI, TOBL_ARR_BB,
    TOBL_ARR_NESTED, TOBL_CUTOFF,
)
from .typing_hamiltonian import TypingHamiltonian, TypingTerm
```

Public exports added to `src/qft_pcn/__init__.py`:

```python
from .logic import TypingHamiltonian, TypingTerm
```

---

## 9. Error model

The typing-Hamiltonian itself raises no errors at evaluation time: every term-energy is a real, non-negative number (it's the expectation of a projector). For inputs the encoder rejected, the encoder's existing exceptions (`EncodingTooLarge`, `TooManyBinders`, `IntLiteralOutOfRange`, `IllScopedVar`, `UnsupportedNode`) apply unchanged.

New exceptions (in `encoding.py`):

```python
class TyOutOfFlatRange(EncodingError):
    """The encoder needs to write a flat type tag for an obligation but
    the Ty does not map to any flat tag (T_ARR_NESTED is fine; this is
    only raised for unhandled cases like T_NONE for a non-PAD site)."""
    def __init__(self, site: int, ty: Ty):
        super().__init__(f"site {site}: Ty {ty!r} has no flat tag")
```

Internal sanity checks in `typing_hamiltonian.py`:

```python
class TypingHamiltonianError(Exception): ...

class TermNotFound(TypingHamiltonianError):
    def __init__(self, term: TypingTerm):
        super().__init__(f"term {term} not in this Hamiltonian's term list")
```

`TypingHamiltonian.term_energy(state, term)` raises `TermNotFound` if `term` isn't in `self.terms`. (Avoids silent zeros for typos.)

---

## 10. Acceptance criteria

The sub-project is complete when:

1. **Encoder extension**: A's encoder, with the §5 changes, produces MPSes whose `EncodingMeta.tobl_per_site` and `EncodingMeta.channel_param_ty_per_bond` match the AST's classical typing derivation. Tested by §7.7.
2. **Well-typed programs (5)**: `⟨H_typing⟩ < 1e-10` for each of WT1..WT5. Tested by §7.1.
3. **Ill-typed programs (5)**: `⟨H_typing⟩ > 0.5` for each of IT1..IT5, with exactly one term contributing > 0.5 and that term's `(rule_id, site)` matching the expected violation. Tested by §7.2.
4. **Per-rule isolation**: each rule (T-Lit-Int, T-Lit-Bool, T-Bin-Arith, T-Bin-Cmp, T-Var, T-App-Arrow) can be isolated by surgically violating only its precondition; the corresponding term fires and no others do. Tested by §§7.3–7.6.
5. **Structural Hamiltonian (§1.1)**: `TypingHamiltonian(N)` builds without reference to any AST; the same instance correctly evaluates multiple unrelated ASTs. Tested by §7.8.
6. **Residual decomposition (§1.5)**: `Σ residuals = total_energy`, and the keys are (rule_id, site) tuples consumable by D. Tested by §7.9.
7. **Performance**: `total_energy(state)` on WT5 completes in < 5 seconds, 5 iterations under 10 seconds. Tested by §7.10.
8. **Classical agreement**: per-term residuals agree with a classical type-checker on all 10 demo programs. Tested by §7.11.
9. **No regressions in sub-project A**: all existing tests in `src/qft_pcn/tests/test_logic_*.py` still pass after the encoder extension. Tested by §7.12.

No claim of completion is acceptable without these tests actually running green in a fresh shell. (See `superpowers:verification-before-completion`.)

---

## 11. Open questions

**None.** Every design choice above is locked. If you, while implementing, find a real ambiguity that this document does not resolve — stop and ask the human. Do not paper over it.

Notable resolved decisions (just so they're visible):

- **Decision: tobl is a new field species, not a bond DOF.** Bond DOFs are not directly addressable by the existing `MPS.local_expectation` / `MPS.two_site_expectation` machinery, and the `tobl` value is a per-site amplitude written by the encoder. Adding it as a 5th species is the cleanest realization.
- **Decision: param_ty IS a bond DOF (carried on bid channels).** Unlike `tobl`, the binder's param_ty is genuinely shared between the LAM site and all its Var-use sites, which is the textbook definition of bond entanglement. We carry it on the bid channels as an extra factor.
- **Decision: factored-expectation evaluation; no dense Hamiltonian operators at d=65536.** Materializing the dense operator costs 64 GB per term and is infeasible. The factored path is mathematically equivalent and `O(d_local × χ²)` in time.
- **Decision: 5-species encoder is the new default; the 4-species code path is removed.** A's tests are updated to expect 5 species (where `tobl=TOBL_NONE` everywhere on sites without obligation, contributing a product factor of 1 in the local Hilbert space).
- **Decision: chi_max default for B's encoder is 32** (was 16 in A). A's default is raised to 32 too. This is a small breaking change to A's existing tests, which pass chi_max=16 — they're updated to pass chi_max=32.

---

## 12. Contract for downstream sub-projects

### 12.1 What sub-project C (evaluation Hamiltonian) gets from B

- The 5-species `SPECIES_B` list and basis constants. C extends it further (likely adds a `eval_state` species for reduction intermediates).
- `EncodingMeta.tobl_per_site` and `nested_tobl_index` (defines what types each site should evaluate TO, useful as a guide for the eval Hamiltonian).
- The `TypingHamiltonian` instance — C's `EvaluationHamiltonian` may need to consult typing constraints during evaluation (e.g., to validate types preserved by reduction).
- The factored-expectation infrastructure in `typing_hamiltonian.py` — C will need similar machinery for its own large-d operators.

### 12.2 What sub-project D (debugger) gets from B

- The `TypingHamiltonian.residuals(state) -> dict[(str, int), float]` API. D iterates this, identifies non-zero entries, and uses `meta.site_to_ast_path[site]` to localize the violation in the original AST. The string keys are stable rule names that D maps to human-readable error templates:
  - `"T-Lit-Int"` → "Site {k} (AST path {path}) is an integer literal but tagged with type {type}, not Int."
  - `"T-Var"` → "Site {k} is a Var of type {type}, but its binder is declared with parameter type {param_ty}."
  - etc.

### 12.3 What sub-project E (synthesis demo) gets from B

- The full `H_typing` operator available via term-by-term factored evaluation. E's imaginary-time evolution under `H_typing + H_eval` requires the operator to be applicable to MPS in TEBD style (`trotter_step`). **B does NOT provide TEBD-compatibility**; E will need to either (a) handle the dense-operator cost by restricting to small N and reduced species cutoffs, or (b) implement factored Trotter gates. This is documented as E's prerequisite — B's contract is energy-only.
- The encoder extension produces MPS that support superposition over `tobl` and `param_ty` channels — i.e., E's holes naturally extend to "what type should this be?" superpositions, which the Hamiltonian's energy gradient will guide.

### 12.4 What stays stable

The following will not change in C, D, E without re-spec'ing B:

- The 5-species list and basis ordering (kind, type, bid, value, tobl).
- The `tobl_per_site` field on `EncodingMeta` and its semantics (§5.1).
- The `channel_param_ty_per_bond` field on `EncodingMeta`.
- The `(rule_id: str, site: int)` keying of `residuals(state)`.
- The 8 rule names: `"T-Lit-Int"`, `"T-Lit-Bool"`, `"T-Bin-Arith"`, `"T-Bin-Cmp"`, `"T-Var"`, `"T-Abs"`, `"T-Obligation"`, `"T-App-Arrow"`.

---

## 13. Glossary (local; refer to A's glossary for shared terms)

- **Obligation** — at a site `k`, the type the parent AST node expects of the subtree rooted at `k`. Encoded into the `tobl` register at `k` at encode time. `T_NONE` means "no expectation" (e.g., at the root).
- **Channel param_ty** — for each live-binder channel on the bid bond, the flat type tag of that binder's `param_ty` annotation. Carried as a bond DOF (real quantum DOF on the bid register) alongside the channel ID.
- **Per-term residual** — `⟨ψ|H_term|ψ⟩` for a single named term `(rule_id, site)`. Always non-negative; sums to `total_energy`.
- **Structural Hamiltonian** — a Hamiltonian whose definition depends only on (a) the lattice length `N`, (b) the species list, and (c) the typing rules — not on the specific AST encoded. Cf. §1.1.
- **Factored expectation** — `⟨ψ|O|ψ⟩` computed without materializing `O` as a dense matrix on the full local Hilbert space; instead, `O` is held as a tensor product of small per-species matrices, and the expectation is computed by direct contraction against the MPS site tensors. See §6.3.
- **Flat type tag** — one of the 8 values `T_NONE, T_INT, T_BOOL, T_ARR_II, T_ARR_IB, T_ARR_BI, T_ARR_BB, T_ARR_NESTED` from A's `encoding.py`. The same encoding is used for `tobl` and channel `param_ty` tags.
- **Live binder** — same as in A's spec: a binder whose `lam_site ≤ i` and whose last var-use site `> i` for bond `i`. The bid bond at bond `i` has `1 + 8 · |live binders|` slots after B's extension.
