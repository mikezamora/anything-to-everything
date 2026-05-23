# I-Task-10 Blocker Fix Plans (M2/M3 substrate gaps)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every task and superpowers:systematic-debugging on unexpected failures. Run superpowers:verification-before-completion before any completion claim. Each task is strict TDD: write the failing test, watch it fail, implement, watch it pass, commit.

**Goal:** Close the five substrate gaps in M2 / M3 / I that block I Task 10 — the two-stage acceptance demo (spec §8.12 / arch §10.8). Each gap is treated as one self-contained task with a dedicated implementer-subagent prompt. Blocker #2 (parser surface syntax for `forall` / `Eq` / `add` / `Zero` / `Nat`) is in flight on a parallel branch and is noted here as deferred — do not re-investigate it.

**Tech Stack:** Python 3.11, numpy 1.26, pytest. Built on M1 (`mera_encoder.py`), M2 (`MeraEvalHamiltonian`, `MeraTypingHamiltonian`), M3 (`mera_synthesis/runner.py`), and F's `qft/mera.py`.

**Spec anchors:** §1.1 (variable binding = entanglement), §1.5 (operator-algebraic promotion), §1.6 (no classical rewrite — reduction is ground-state finding), §7.1 (factored redex penalties), §7.4 (transition gates), §7.5 (imaginary-time evolution), §8.12 (two-stage acceptance), arch §10.8 (lemma library).

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Driving principles (non-negotiable; embed verbatim in every subagent prompt, each paired with the shortcut it forbids)

1. **No time / duration / effort estimates anywhere.** Critical standing directive.
2. **Binding is genuine entanglement, never a classical lookup (§1.1).** *Forbidden:* enumerate the Forall-bound variable's candidate values, run the body for each, AND/OR the results. *Principled:* the Forall-bound leaves stay genuinely symbolic (the leaf carries a uniform superposition over its codomain) for the lifetime of evolution; reduction rules that would touch it become identity.
3. **No dense operator at scale (§1.3).** Every new penalty / gate is factored: `dict[leaf -> (16,16)]` or a 256x256 two-leaf gate. No `16**k` for k>2.
4. **No classical rewrite (§1.6).** *Forbidden:* "pattern-match `add x Zero`, splice `x`, write back." *Principled:* a diagonal projector onto the unreduced redex configuration plus a non-Hermitian one-way transition gate that drives the redex leaves toward their reduced configuration. The spectrum + imaginary-time evolution do the work.
5. **Operator-algebraic promotion (§1.5).** *Forbidden:* "decode the lemma to AST, paste it at the hole, re-run the typing Hamiltonian." *Principled:* `use_lemma` compiles to a tensor-network clamp; freezing is a per-leaf gate-application skip, never an AST rewrite.
6. **`optimize='greedy'` on every einsum/tensordot.**
7. **Reuse M1 / M2 / F primitives; do not modify the MPS logic stack or the MERA primitives unless a genuine missing primitive blocks the design.**

---

## Pre-existing worktree state

Branch `claude/qft-pcn-hybrid-architecture-ihCIR`, HEAD post-`6506037`. Three concurrent implementer agents run on disjoint scopes (perf-fix on `mera_*`, parser extension on `ast.py`, K Task 4 on `composition/dispatcher.py`). Unrelated modified files (`tauri-app/`, root `*.md`, `lib/`, viz) exist; leave them alone. Stage only the files each task names.

---

## Blocker #2 — DEFERRED

Surface parser syntax for `forall` / `Eq` / `add` / `Zero` / `Nat` in `src/qft_pcn/logic/ast.py::parse` is being handled by a separate in-flight subagent. Do NOT re-investigate in this plan; the other plans here assume that after the parser-extension agent lands, `parse("forall x:Nat. Eq (add x Zero) x")` returns the expected `Forall(...)` AST and parser tests cover the surface forms. If the parser-extension agent has not landed when an implementer picks up a task below, the implementer constructs the test ASTs directly via the existing `Forall`, `Eq`, `Zero`, `NatLit`, `Bin(op='+')` constructors (the AST classes already exist).

---

## Blocker #1 — `relax_program` driver in M3

`src/qft_pcn/logic/mera_synthesis/runner.py` currently exposes only `synthesize(SynthesisProblem)` — a witness-augmented sketch-completion driver, not a general program-relaxation driver. I Task 10's two-stage acceptance needs a thinner entry point: encode an `ast_src` directly, compose the M2 Hamiltonians (typing + eval + any `use_lemma` constraints), imag-time evolve to a residual gate `eps`, return the relaxed state, the meta, the composed `H`, the residual, and the number of Trotter steps taken.

**Files:**
- Modify: `src/qft_pcn/logic/mera_synthesis/runner.py` — add `relax_program` (new top-level entry point) and `RelaxResult` dataclass.
- Modify: `src/qft_pcn/logic/mera_synthesis/__init__.py` — re-export `relax_program`, `RelaxResult`.
- Test: `src/qft_pcn/logic/mera_synthesis/tests/test_relax_program.py` (create the tests directory if absent — match the existing convention used by `composition/tests/`).

### Step 1: Write the failing tests

Create `src/qft_pcn/logic/mera_synthesis/tests/test_relax_program.py`:

```python
"""Tests for relax_program (I Task 10 driver, spec §8.12)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import IntLit, Bin
from src.qft_pcn.logic.mera_synthesis import relax_program, RelaxResult


def test_relax_program_returns_named_result_for_closed_arith():
    src = Bin(op="+", lhs=IntLit(2), rhs=IntLit(3))
    res = relax_program(src, constraints=(), eps=1e-3,
                        max_trotter_steps=200, dt=0.05, chi_layer=16)
    assert isinstance(res, RelaxResult)
    assert res.state is not None
    assert res.meta.n_nodes >= 3
    assert res.residual < 1e-3
    assert 1 <= res.trotter_steps <= 200


def test_relax_program_stops_when_residual_below_eps():
    src = Bin(op="+", lhs=IntLit(1), rhs=IntLit(1))
    res = relax_program(src, constraints=(), eps=1e-2,
                        max_trotter_steps=500, dt=0.05, chi_layer=16)
    # Early termination: it did NOT consume the full budget.
    assert res.trotter_steps < 500
    assert res.residual < 1e-2


def test_relax_program_runs_full_budget_when_eps_unreachable():
    # eps=0 forces full-budget consumption.
    src = Bin(op="+", lhs=IntLit(2), rhs=IntLit(3))
    res = relax_program(src, constraints=(), eps=0.0,
                        max_trotter_steps=40, dt=0.05, chi_layer=16)
    assert res.trotter_steps == 40
```

### Step 2: Run the tests, verify they fail — `ImportError: cannot import name 'relax_program'`.

### Step 3: Implement

In `src/qft_pcn/logic/mera_synthesis/runner.py`, add (top-level, after `synthesize`):

```python
from dataclasses import dataclass
from ..ast import Node
from ..mera_encoder import MeraEncodingMeta
from src.qft_pcn.qft.mera import MERA


@dataclass(frozen=True)
class RelaxResult:
    """Result record of relax_program (I Task 10, spec §8.12).

    state          -- the relaxed MERA after imag-time evolution.
    meta           -- the encoding meta for `state` (the §1.2 addressing data).
    hamiltonian    -- the composed H whose ground state is being approached.
    residual       -- final <H>(state); the §8.12 acceptance gate is residual < eps.
    trotter_steps  -- number of Trotter steps actually applied (may be <
                      max_trotter_steps if early-termination on residual).
    """
    state: MERA
    meta: MeraEncodingMeta
    hamiltonian: object
    residual: float
    trotter_steps: int


def relax_program(
    ast_src: Node,
    *,
    constraints: tuple = (),
    eps: float = 1e-3,
    dt: float = 0.05,
    max_trotter_steps: int = 200,
    chi_layer: int = 16,
    n_nodes_max: int = 32,
    frozen_leaves: set[int] | None = None,
    lemma_library=None,
) -> RelaxResult:
    """Encode `ast_src`, compose H = H_typing + H_eval (+ promoted lemma
    constraints), and run imag-time evolution to residual < eps or until
    max_trotter_steps is consumed (spec §8.12, arch §10.8).

    Returns a RelaxResult. No re-encoding of completions, no AST sampling;
    this is the program-relaxation driver, not the synthesis runner.

    `constraints` is a tuple of DSL constraint dicts; presently only
    {"kind": "use_lemma", ...} is honored. Each is compiled via the
    Promoter (mode='init_clamp' by default), and the union of clamped
    leaves is added to `frozen_leaves`. `lemma_library` is required when
    `constraints` is non-empty.
    """
    from ..mera_encoder import encode_mera
    from ..mera_typing_hamiltonian import MeraTypingHamiltonian
    from ..mera_evaluation_hamiltonian import MeraEvalHamiltonian
    from ..mera_evolution_logic import mera_imaginary_evolve_state
    from .hamiltonian import ComposedMeraSynthesisHamiltonian

    state, meta = encode_mera(ast_src, n_nodes_max=n_nodes_max,
                              chi_layer=chi_layer)

    # Compose H_typing + H_eval. Weights of 1.0 each — relaxation is
    # judged by residual, not by ranking against alternatives.
    h_typing = MeraTypingHamiltonian(meta)
    h_eval = MeraEvalHamiltonian(meta)
    H = ComposedMeraSynthesisHamiltonian(
        weighted_sub_hams=[(h_typing, 1.0), (h_eval, 1.0)],
        extra_terms=[])

    # Apply use_lemma clamps (§1.5 operator-algebraic promotion). Each
    # clamp's leaves are added to frozen_leaves so subsequent imag-time
    # evolution does not deform them.
    frozen: set[int] = set(frozen_leaves or ())
    if constraints:
        if lemma_library is None:
            raise ValueError(
                "constraints non-empty but lemma_library is None")
        from src.qft_pcn.composition.promoter import Promoter
        promoter = Promoter(lemma_library, mode="init_clamp")
        for c in constraints:
            if c.get("kind") != "use_lemma":
                raise NotImplementedError(
                    f"unsupported constraint kind: {c.get('kind')!r}")
            promoted = promoter.compile_constraint(c)
            frozen |= promoter.apply_init_clamp(state, meta, promoted)

    # Imag-time evolution with early termination at residual < eps. The
    # driver delegates the inner stepping loop to mera_imaginary_evolve_state
    # (per spec §7.5), but consumes the budget in small chunks so it can
    # check the residual gate between chunks. This is purely a control-flow
    # wrapper -- it never touches the physics.
    chunk = max(1, min(10, max_trotter_steps))
    steps_taken = 0
    residual = float(H.total_energy(state))
    while steps_taken < max_trotter_steps and residual > eps:
        remaining = max_trotter_steps - steps_taken
        this_chunk = min(chunk, remaining)
        _, state = mera_imaginary_evolve_state(
            state, H, dt=dt, steps=this_chunk, chi_layer=chi_layer,
            frozen_leaves=(frozen if frozen else None))
        steps_taken += this_chunk
        residual = float(H.total_energy(state))

    return RelaxResult(state=state, meta=meta, hamiltonian=H,
                       residual=residual, trotter_steps=steps_taken)
```

In `src/qft_pcn/logic/mera_synthesis/__init__.py` add the re-exports:

```python
from .runner import synthesize, relax_program, RelaxResult
```

### Step 4: Run the tests, verify they pass.

If the inner evolution driver lacks a `frozen_leaves` parameter (Blocker #6 not yet landed), implement #6 first or have `relax_program` raise a clear error when `frozen` is non-empty. The tests above pass `constraints=()`, so neither path is exercised; the parameter wiring is in place for the integration demo.

### Step 5: Commit

```
feat(mera_synthesis): relax_program driver + RelaxResult for I Task 10

The two-stage acceptance demo (spec §8.12, arch §10.8) needs a thin
program-relaxation driver -- encode an AST, compose typing+eval H,
imag-time evolve to residual < eps, return (state, meta, H, residual,
trotter_steps). Distinct from the synthesis runner: no sketch holes,
no witness augmentation, no completion ranking; just relaxation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

**Dependencies:** Soft dependency on Blocker #6 (frozen_leaves kwarg). The `frozen_leaves=` argument is wired in `relax_program` but the tests in Step 1 do not exercise it. The two-stage demo run (with `use_lemma`) requires #6 to land first.

---

## Blocker #3 — `R-AddZero` reduction rule (`add x Zero -> x`)

`MeraEvalHamiltonian` (`src/qft_pcn/logic/mera_evaluation_hamiltonian.py`) only reduces closed arith on concrete `NatLit/IntLit` operands. The demo needs an OPEN reduction rule: when a `BIN(+)` node has a `NatLit(0)` (== `Zero`) on one side and an arbitrary sub-expression on the other, the redex reduces to that sub-expression. With a `Forall`-bound `Var` on one side this is the only reduction that fires (no closed arith available, the bound `x` stays symbolic).

The mechanism stays operator-algebraic: a diagonal projector onto the unreduced configuration (one operand is a Zero / `NatLit(0)`, regardless of the other operand's kind), and per-leaf one-way transition gates driving the BIN node to take on the non-zero operand's leaf values.

### Operator-algebraic design

`add x Zero` (or `add Zero x`) reduces to `x`. After reduction, every leaf of the BIN node carries the same basis index as the corresponding leaf of the non-zero operand sub-tree's ROOT node, and the BIN node's two child sub-trees collapse to PAD.

Encoding facts (read from `mera_encoding.py`):
- `KIND_ZERO = 8`, `KIND_NATLIT = 10`, `KIND_BIN = ...`, `KIND_VAR = ...`.
- `NatLit(0)` is encoded as `KIND_NATLIT` with `value` leaf at the slot for value 0 (the encoder's `NatLit` path).
- `Zero` is encoded as `KIND_ZERO`.
- A `Forall`-bound `Var` has `kind = KIND_VAR` with its `bid` leaf entangled to the `Forall` node's `bid` leaf (M1 §5).

The redex-presence projector is

    P_R-AddZero(node, lhs, rhs) =
        P[kind=BIN](node)            ⊗
        P[value in {+}](node_value)  ⊗
        ( P[kind in {ZERO, NATLIT_zero}](lhs_kind) ⊗ I(rhs_kind) +
          I(lhs_kind) ⊗ P[kind in {ZERO, NATLIT_zero}](rhs_kind) -
          P[kind in {ZERO, NATLIT_zero}](lhs_kind) ⊗ P[kind in {ZERO, NATLIT_zero}](rhs_kind) )

i.e. EXACTLY ONE of {lhs, rhs} is a zero. `NATLIT_zero` is the joint two-leaf projector `P[kind=KIND_NATLIT] ⊗ P[value=NATLIT_VALUE_ZERO]` on that operand. The inclusion-exclusion combination is built as a sum of **three** factored terms — never materialized as a `16^k` operator (§1.3). When ONE term fires (`lhs` is zero, say), the redex is present.

Transition gates (one rule firing on `lhs == zero`):

  - On the BIN node: drive each species leaf toward the `rhs` sub-tree's root-node leaf value — `kind`, `type`, `bid`, `value`, `tobl` — five `single_leaf_transition_gate` calls, one per species. Target indices are read by `_leaf_argmax(state, rhs_node, species)` for the concrete-kind species, and by `_leaf_argmax_in(...)` restricted to non-PAD for any species in superposition.
  - On the `lhs` sub-tree: collapse every node to PAD via `_collapse_moves`.
  - On the `rhs` sub-tree: collapse every node EXCEPT the root to PAD; the root's leaves are *promoted* into the BIN node (so the rhs root's own leaves drain to PAD AFTER the BIN node's leaves are fully on the rhs root's values — same staging as `_beta_app_unfinished` already enforces for R-Beta).

The symmetric case (`rhs == zero`) is structurally identical with `lhs` and `rhs` swapped.

Critically — and this is what makes #3 cleanly interoperate with #5 — when the **non-zero operand is a Forall-protected `Var`**, the promotion target for the BIN node's leaves is the Var's leaves. The `bid` leaf in particular receives the entangled bid index, so the BIN-node-now-a-Var carries the same binding entanglement as the original Var: the proof holds for the same universally bound `x`. No classical "substitute `x` for the BIN node" — the leaves are physically copied with their entanglement structure intact (§1.1).

### Files to touch

- Modify: `src/qft_pcn/logic/_mera_eval_terms.py` (+~50 lines) — add `add_zero_penalty_ops` (the diagonal redex-presence projectors, returned as a *list* of factored ops representing the inclusion-exclusion sum).
- Modify: `src/qft_pcn/logic/mera_evaluation_hamiltonian.py` (~80 lines) — add `RULE_R_ADD_ZERO = "R-AddZero"`, append to `_ALL_RULES`, extend `_penalty_ops` and `term_gates`, add `_add_zero_unfinished` (the cleanup-liveness guard mirroring `_arith_bin_unfinished`), add `_add_zero_promote_targets: dict[int, dict[str, int]]` snapshot store mirroring `_if_keep_targets`.
- Test: `src/qft_pcn/tests/test_mera_eval_add_zero.py`.

### Edit pointers

- `_mera_eval_terms.py`: insert after `succ_penalty_ops` (line ~95). New function:

```python
def add_zero_penalty_ops(bin_kind_leaf, bin_value_leaf,
                         lhs_kind_leaf, lhs_value_leaf,
                         rhs_kind_leaf, rhs_value_leaf,
                         lam):
    """Three-factor inclusion-exclusion sum projecting onto an add-zero
    redex configuration (spec §7.1 extended): a BIN(+) node with EXACTLY
    ONE of {lhs, rhs} a Zero / NatLit(0). Returns a LIST of three factored
    operators (each dict[leaf -> (16,16)]); the caller sums their
    expectations.

    The three terms are:
        T1 = P[BIN] (x) P[+] (x) P[zero](lhs_kind, lhs_value) (x) I(rhs)
        T2 = P[BIN] (x) P[+] (x) I(lhs) (x) P[zero](rhs_kind, rhs_value)
        T3 = -P[BIN] (x) P[+] (x) P[zero](lhs) (x) P[zero](rhs)
    so T1 + T2 + T3 = projector onto "EXACTLY one zero operand".

    P[zero] on an operand is the factored two-leaf product
        ( P[KIND_ZERO] + P[KIND_NATLIT] . P[value=NATLIT_VALUE_ZERO] )
    expanded as a small dict[leaf -> (16,16)] over (kind_leaf,
    value_leaf). The cross term P[NATLIT]·P[value=0] is itself a
    two-leaf factored projector; the inclusion-exclusion combinator
    above keeps the operator factored at every step (no 16**k operator).
    """
    # ... implementation as sketched ...
```

The two-leaf `P[zero]` joint projector is a small (4-entry) dict; combined with the BIN-node factors it stays a `dict[leaf -> (16,16)]` per term. The `lam` scaling is applied once on `bin_kind_leaf` (matching `arith_penalty_ops` convention).

- `mera_evaluation_hamiltonian.py`:
  - Line 42-50 (rule constants block): add `RULE_R_ADD_ZERO = "R-AddZero"`, append to `_ALL_RULES`.
  - Line 263-272 (`_lambda_for`): add a branch returning `self.lambda_arith` for `RULE_R_ADD_ZERO` (re-use the arith coupling).
  - Line 292-329 (`_penalty_ops`): add a branch for `RULE_R_ADD_ZERO` returning the LIST of three factored ops from `add_zero_penalty_ops`.
  - Line ~518 (after `_arith_bin_unfinished`): add `_add_zero_unfinished(self, state, term) -> bool` mirroring the staged-reduction liveness guard.
  - Line ~563 (`term_gates`): add the `RULE_R_ADD_ZERO` branch building the five-species promote moves + the collapse moves, populating `self._add_zero_promote_targets[node]` on first firing.
  - Line 159-160 (`_enumerate_terms`): the existing loop already enumerates one term per `(rule, node)` for every rule in `_ALL_RULES`, so adding the rule to `_ALL_RULES` automatically gives every node an `R-AddZero` term. The redex-presence gate (`term_energy < 1e-9 and not unfinished`) means it costs nothing where it does not apply.

### API additions

- `add_zero_penalty_ops(bin_kind_leaf, bin_value_leaf, lhs_kind_leaf, lhs_value_leaf, rhs_kind_leaf, rhs_value_leaf, lam) -> list[dict[int, np.ndarray]]` (note: returns a LIST, unlike the single-op builders).
- `RULE_R_ADD_ZERO = "R-AddZero"` exposed at module level.
- `MeraEvalHamiltonian._add_zero_promote_targets: dict[int, dict[str, int]]`.
- `MeraEvalHamiltonian._add_zero_unfinished(state, term) -> bool`.

### Test design

Create `src/qft_pcn/tests/test_mera_eval_add_zero.py`:

```python
"""Tests for R-AddZero (spec §8.12 substrate)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import (
    Bin, IntLit, NatLit, Zero, Forall, Eq, Var, TNat,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian, RULE_R_ADD_ZERO,
)
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_imaginary_evolve_state,
)


def test_add_zero_penalty_nonzero_for_concrete_add_zero():
    src = Bin(op="+", lhs=NatLit(5), rhs=NatLit(0))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    # The R-AddZero term on the BIN node should report non-zero energy.
    bin_terms = [t for t in H.terms
                 if t.rule_id == RULE_R_ADD_ZERO]
    energies = [H.term_energy(state, t) for t in bin_terms]
    assert any(e > 1e-6 for e in energies)


def test_add_zero_relaxes_closed_case_to_normal_form():
    src = Bin(op="+", lhs=NatLit(5), rhs=NatLit(0))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=80, chi_layer=16)
    # Post-relaxation the BIN node has been replaced by the lhs literal:
    # H.total_energy approaches 0 (no remaining redexes).
    assert H.total_energy(state) < 1e-3


def test_add_zero_relaxes_open_case_with_free_var():
    # forall x:Nat. (x + 0). The R-AddZero rule must reduce this even
    # though `x` is a free Var (no closed arith available).
    body = Bin(op="+", lhs=Var(name="x"), rhs=Zero())
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    e0 = H.total_energy(state)
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=120, chi_layer=16)
    # The BIN node has been replaced by the Var; eval residual drops to ~0
    # (typing residual is handled separately).
    eval_residual = sum(
        H.term_energy(state, t) for t in H.terms
        if t.rule_id == RULE_R_ADD_ZERO)
    assert eval_residual < 1e-3
    assert H.total_energy(state) < e0
```

### Dependencies / sequencing

- Depends on Blocker #5 (`Forall`-aware semantics) for the open-case test to fully converge: without #5, the existing R-Arith rule may *also* try to fire on the `Var` operand (which has no `KIND_INT` weight, so in practice it stands down; but the cleaner story is that #5's "Forall-protected" leaf flag turns off ALL rules whose target leaves are protected). Mark the open-case test as `pytest.xfail` until #5 lands, OR have the test rely only on the diagonal energy term descending.
- No dependency on #1, #4, #6.
- Independent of #2 (parser): tests construct ASTs directly.

### Commit

```
feat(mera_eval): R-AddZero reduction (add x Zero -> x)

Spec §7.1 / §8.12 substrate. Adds a diagonal redex-presence projector
for the "exactly one operand is Zero" configuration (inclusion-exclusion
sum of three factored two-leaf projectors -- no 16**k operator) and the
five-species promotion + child-collapse transition gates. Reduces both
closed (NatLit(5) + NatLit(0)) and open (Var + Zero under Forall) cases
to normal form via imag-time evolution. No classical rewrite anywhere.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Blocker #4 — `R-Eq-Refl` reflexivity rule

`Eq` is currently structural-only — `KIND_EQ = 13` is a tagged kind with no associated reduction rule. The proof obligation for `Eq lhs rhs` when `lhs ≡ rhs` is a zero-energy ground state: the proposition reduces to `True` (a `BoolLit` / `True` literal, encoded as `KIND_BOOL` + `value=VALUE_TRUE`). The Hamiltonian must penalize the configuration "Eq node with `lhs` not structurally equal to `rhs`" and emit zero penalty exactly when the two operand sub-trees are leaf-for-leaf identical.

### Operator-algebraic design

The cleanest formulation is a **leaf-pair equality projector** for each `(lhs_leaf, rhs_leaf)` species pair, with the Eq term's total penalty proportional to `sum_species (I − P_equal_pair(species))`. `P_equal_pair` on a single species (e.g. kind) is the projector onto the symmetric subspace `Σ_i |i,i⟩⟨i,i|`. As a `(16*16, 16*16) = (256, 256)` two-leaf operator it sits at the cap of the §1.3 budget; the spec already authorizes 256x256 two-leaf operators for `fix_transition_gate` so this is in-budget.

Alternative factored form to keep every operator a `dict[leaf -> (16,16)]`: write `P_equal_pair = I - sum_{i!=j} |i,j⟩⟨i,j|`. The sum has 16*15 = 240 rank-1 terms but each is a factored two-leaf product `|i⟩⟨i| ⊗ |j⟩⟨j|`, so the diagonal expectation `<state | P_equal_pair | state>` is computed as `1 - sum_{i!=j} <P_i>_lhs * <P_j>_rhs` — pure single-leaf marginals (16 + 16 single-leaf reads, then 240 multiplies). No operator over `16^2`. Use this form; the 256x256 form is a fallback if performance issues surface.

The Hamiltonian penalty is

    H_R-Eq-Refl(eq_node, lhs_root, rhs_root) =
        lambda_eq * P[kind=KIND_EQ](eq_node)
            ⊗ sum_{species in (kind, type, bid, value, tobl)}
                  ( I_pair - P_equal_pair(species)(lhs_root, rhs_root) )

`<H>_state = 0` iff EVERY species on the lhs and rhs root nodes carries the same basis index — i.e. the two roots are leaf-equal. For deeper trees, recursively extend the sum across the lhs/rhs sub-tree node pairs (paired by structural position via `meta.children_of_node`), so structural equality drives `<H>` to 0 for arbitrarily deep equal sub-trees. The recursion is **structural**, not dynamic: the term enumeration walks the lhs/rhs sub-trees of an Eq node ONCE at construction time and emits one factored term per (species, paired-node-position) — pure addressing (§1.2).

When `lhs ≡ rhs` the proposition reduces to `BoolLit(True)`. The transition gates promote the Eq node to a BoolLit:
  - `kind` leaf: `KIND_EQ → KIND_BOOL`.
  - `value` leaf: `VALUE_NONE → VALUE_TRUE`.
  - other species: drive to BoolLit's canonical values.
  - lhs and rhs sub-trees collapse to PAD (mirrors R-If's branch-keep collapse).

The transition gate fires only once the diagonal `H_R-Eq-Refl` has dropped to ~0 (i.e. the two sides are confirmed leaf-equal); otherwise it would corrupt the proof obligation. This is just the redex-presence gate already used by every other R-rule.

### Files to touch

- Modify: `src/qft_pcn/logic/_mera_eval_terms.py` (+~40 lines) — add `eq_refl_penalty_ops(eq_kind_leaf, lhs_leaves_by_species, rhs_leaves_by_species, lam) -> list[dict]`. Each element is a factored two-leaf op covering one (species, depth=0) pair; sub-tree pairs are emitted per depth by the Hamiltonian itself.
- Modify: `src/qft_pcn/logic/mera_evaluation_hamiltonian.py` (~70 lines) — `RULE_R_EQ_REFL = "R-Eq-Refl"`, append to `_ALL_RULES`, extend `_penalty_ops` to walk the lhs/rhs sub-tree node pair (via `meta.children_of_node[eq_node]`'s first two kids) and emit a per-(node-pair, species) term, extend `term_gates` to emit the Eq-node promotion + lhs/rhs collapse once the diagonal residual is below `1e-9`.
- Test: `src/qft_pcn/tests/test_mera_eval_eq_refl.py`.

### API additions

- `eq_refl_penalty_ops(eq_kind_leaf, lhs_leaves, rhs_leaves, lam) -> list[dict[int, np.ndarray]]`.
- `RULE_R_EQ_REFL`.
- New helper `MeraEvalHamiltonian._eq_paired_subtree(self, eq_node) -> list[tuple[int, int]]` returning the structural pairing of lhs/rhs sub-tree nodes (BFS in lockstep). Pairs whose structure diverges (one side has children, the other does not) are dropped — the diagonal penalty will read non-zero on the differing leaves and the rule simply does not fire on that Eq node, which is correct: the proposition is unprovable by reflexivity.
- New `MeraEvalHamiltonian._eq_refl_promote_targets: dict[int, dict[str, int]]` snapshot store.

### Test design

Create `src/qft_pcn/tests/test_mera_eval_eq_refl.py`:

```python
"""Tests for R-Eq-Refl (spec §7.1 extended, §8.12 substrate)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import (
    Eq, NatLit, Zero, Var, Forall, TNat, Bin,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian, RULE_R_EQ_REFL,
)
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_imaginary_evolve_state,
)


def test_eq_refl_zero_on_structurally_equal_operands():
    # Eq(NatLit(3), NatLit(3)) has the R-Eq-Refl term reading 0.
    src = Eq(lhs=NatLit(3), rhs=NatLit(3))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    refl_terms = [t for t in H.terms if t.rule_id == RULE_R_EQ_REFL]
    assert all(H.term_energy(state, t) < 1e-9 for t in refl_terms)


def test_eq_refl_nonzero_on_unequal_operands():
    src = Eq(lhs=NatLit(3), rhs=NatLit(4))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    refl_terms = [t for t in H.terms if t.rule_id == RULE_R_EQ_REFL]
    assert any(H.term_energy(state, t) > 1e-6 for t in refl_terms)


def test_eq_refl_relaxes_after_add_zero():
    # The composite goal: forall x:Nat. Eq (x + 0) x.
    # After R-AddZero replaces (x+0) with x, R-Eq-Refl reads 0 and
    # promotes the Eq node to BoolLit(True). Combined eval residual ~0.
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=200, chi_layer=16)
    refl_residual = sum(H.term_energy(state, t) for t in H.terms
                        if t.rule_id == RULE_R_EQ_REFL)
    assert refl_residual < 1e-3
```

### Dependencies / sequencing

- Independent of #1, #6.
- Test #3 (`test_eq_refl_relaxes_after_add_zero`) depends on Blocker #3 (R-AddZero) — schedule #4 AFTER #3.
- Test #3 also depends on Blocker #5 (Forall-aware) to handle the `Var` symbolically. Without #5 the test may still pass: the structural equality projector evaluates the SAME free-Var leaves on lhs and rhs, which trivially match (same `bid` index on both sides — the Var occurrences share a binder, so M1's bid leaves are entangled the same way). But if relaxation collapses the `Var` to a concrete value mid-evolution, the lhs and rhs branches risk diverging temporarily; #5 guarantees they stay symbolic and identical.
- Independent of #2 (parser): tests build ASTs directly.

### Commit

```
feat(mera_eval): R-Eq-Refl reflexivity (Eq lhs rhs reduces when lhs == rhs)

Spec §7.1 extended / §8.12 substrate. Diagonal projector built as a
factored sum over (species, paired-subtree-node) of single-leaf-marginal
inequality probes -- no 16^k operator. Promotion gate replaces the Eq
node with BoolLit(True) when the diagonal residual hits zero, mirroring
R-If's branch-keep promote/collapse staging.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Blocker #5 — `Forall`-aware ground-state semantics

The hardest of the five. A `Forall x:T. body` introduces a binder whose use-sites must stay **symbolic** during imag-time evolution: the proof must hold uniformly for every value of `x`. Naive relaxation would collapse a `Var` leaf's superposition toward a concrete value (whatever direction the typing / eval gates push it) — and then the proof would be about that specific value, not the universal claim.

### Operator-algebraic design — the load-bearing piece

The standing M1 design (§1.1) already encodes the Var's binding as entanglement between the Var's `bid` leaf and the binder's `bid` leaf via the rank-k structural-hole superposition encoder (`encode_hole_state`). But the Var's `value` and `kind` leaves are concrete: a Var carries `kind=KIND_VAR` and `value=VALUE_NONE`. The "symbolic" character of the Var is exactly this: its kind / value leaves are NOT instantiated to a literal, and no rule may instantiate them during evolution.

The principle: a `Forall`-bound leaf is **protected** — every transition gate whose target is a protected leaf becomes identity for the duration of evolution. Protection is a per-leaf flag carried in `MeraEncodingMeta`.

This is NOT a classical enumeration. The Var's leaves are not "ranging over" candidate values via a Python loop — they remain physically in the encoded MERA state, just unmodified. Any rule that would have collapsed them (R-Beta substituting the Forall-bound `x` for a literal, R-Arith trying to extract an INT value from them, etc.) stands down, leaving the proof obligation to be discharged by rules that touch only the OTHER leaves (R-AddZero promoting the BIN node TO the Var's leaves, copying their entangled state but never modifying them; R-Eq-Refl reading the Var's leaves on both sides of an Eq and confirming structural identity).

The "Forall-protected" flag is set on:
  - The Forall node's own body-binding-leaves (the bid leaf carrying the binder identity).
  - Every Var node in the Forall's scope whose binder is this Forall.
  - All five species of those Var nodes (kind, type, bid, value, tobl).

When a transition gate's target leaf is in the protected set, the gate is replaced by identity. This is the same mechanism Blocker #6 uses for lemma-clamp freezing — see #6 for the surgical hook.

### Why this is not a classical lookup

A classical lookup would be: "scan the body, find every occurrence of `x`, substitute a value, run." The proposed design instead: the bound `x` occurrences exist as Var leaves in the encoded MERA; their **state** never changes; the body's reductions (R-AddZero, etc.) read from them via the *operator algebra* (promotion gates targeting OTHER leaves with the Var's leaf indices as the read value, computed via `_leaf_argmax` / `_leaf_argmax_in`). The Var leaves themselves are inert — but they are still part of the MERA state, with their bid entanglement to the Forall node intact. When the BIN(+) node in `x + 0` is replaced by `x`, the BIN node's `bid` leaf is driven (via `single_leaf_transition_gate`) to the same basis index the Var's bid leaf carries — and that index is itself entangled (through the MERA tree) with the Forall's bid leaf. The proof artifact is the entanglement structure, not the value (§1.1 verbatim).

### Files to touch

- Modify: `src/qft_pcn/logic/mera_encoder.py` (~40 lines) — extend `MeraEncodingMeta` with `forall_protected_leaves: set[int]`. Populate it inside `encode_mera` by walking the AST: for each `Forall` node, find the binder's bid leaf and the bid leaf of every Var whose name matches and whose closest enclosing binder is this Forall (the existing `use_to_binder` map already records the bid leaves for every Var). For each such Var, add all five species' leaves to the set. Also add the Forall's bid leaf itself.
- Modify: `src/qft_pcn/logic/mera_evolution_logic.py` (~20 lines) — extend `mera_trotter_step` and `mera_imaginary_evolve_state` with `frozen_leaves: set[int] | None = None`. (This is the SAME surgical addition as Blocker #6 — both blockers use the same hook; #5 populates it from `meta.forall_protected_leaves`, #6 populates it from `Promoter.apply_init_clamp`'s return value.)
- Modify: `src/qft_pcn/logic/mera_evaluation_hamiltonian.py` (~10 lines) — `term_gates` accepts an OPTIONAL `frozen_leaves` parameter and skips any gate whose target leaf is in the set (or returns the gate as a numerically-identity replacement, which costs nothing at the evolution layer). The natural cleanest design: keep `term_gates` unchanged and skip at the evolution-driver layer (Blocker #6's hook), because the evolution driver already groups gates per leaf — dropping a leaf is a single dict-key skip.
- Test: `src/qft_pcn/tests/test_mera_forall_protection.py`.

### API additions

- `MeraEncodingMeta.forall_protected_leaves: set[int]` (default `field(default_factory=set)` so old call sites keep working).
- `mera_imaginary_evolve_state(..., frozen_leaves: set[int] | None = None)` (shared with Blocker #6).
- New helper `src/qft_pcn/logic/mera_encoder.py:_collect_forall_protected_leaves(ast, meta) -> set[int]`.

### Test design

Create `src/qft_pcn/tests/test_mera_forall_protection.py`:

```python
"""Tests for Forall-protected leaves (spec §1.1, §8.12 substrate)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import (
    Forall, Var, Bin, Zero, Eq, NatLit, TNat,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve_state


def test_forall_protected_leaves_populated():
    src = Forall(param="x", param_ty=TNat(), body=Var(name="x"))
    state, meta = encode_mera(src)
    assert hasattr(meta, "forall_protected_leaves")
    assert isinstance(meta.forall_protected_leaves, set)
    # The Forall's own bid leaf AND the Var-x's 5 leaves are protected.
    assert len(meta.forall_protected_leaves) >= 6


def test_forall_protected_var_leaves_unchanged_through_evolution():
    body = Bin(op="+", lhs=Var(name="x"), rhs=Zero())
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    snapshots_before = {
        leaf: np.array(state.leaves[leaf]).copy()
        for leaf in meta.forall_protected_leaves
    }
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=80, chi_layer=16,
        frozen_leaves=meta.forall_protected_leaves)
    for leaf, before in snapshots_before.items():
        after = np.array(state.leaves[leaf])
        assert np.allclose(before, after, atol=1e-12), (
            f"protected leaf {leaf} drifted")


def test_forall_proof_holds_uniformly():
    # forall x:Nat. Eq (x + 0) x: after evolution, the eval residual is
    # below eps AND the Var-x leaves are still symbolic (not collapsed to
    # any concrete value).
    body = Eq(lhs=Bin(op="+", lhs=Var(name="x"), rhs=Zero()),
              rhs=Var(name="x"))
    src = Forall(param="x", param_ty=TNat(), body=body)
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=200, chi_layer=16,
        frozen_leaves=meta.forall_protected_leaves)
    assert H.total_energy(state) < 1e-2
    # The Var-x's value leaf is still VALUE_NONE (not pushed toward any
    # literal): the proof is universal, not for a specific x.
    from src.qft_pcn.logic.encoding import VALUE_NONE
    # Find the Var node (the only KIND_VAR in scope).
    var_node = next(
        n for n in range(meta.n_nodes)
        if meta.layout.leaf_of(n, "kind") in meta.forall_protected_leaves
        and n != 0)   # node 0 is the Forall
    v_leaf = meta.layout.leaf_of(var_node, "value")
    v = np.asarray(state.leaves[v_leaf])[0, :, 0]
    assert abs(v[VALUE_NONE]) ** 2 > 0.999
```

### Dependencies / sequencing

- Blocker #5 depends on Blocker #6's `frozen_leaves=` kwarg landing first (or being landed together — #5 cannot be tested in isolation without it).
- Blocker #3 depends on #5 for its open-case test, so the order is: #6 → #5 → #3 → #4.
- Independent of #1 (which just wires `frozen_leaves` through `relax_program`).
- Independent of #2 (parser): tests build ASTs directly.

### Commit

```
feat(mera_encoder): Forall-protected leaves preserve universal binders

Spec §1.1: variable binding is genuine entanglement, not a classical
lookup. A Forall-bound Var must stay symbolic for the lifetime of
imag-time evolution -- otherwise relaxation would collapse it to a
specific value and the proof would be about that value, not universal.

This commit adds `MeraEncodingMeta.forall_protected_leaves`: the set of
leaves (the Forall's bid + every bound Var's 5 species leaves) that the
evolution driver must skip when applying gates. Combined with #6's
`frozen_leaves=` hook in mera_imaginary_evolve_state, the bound x's
state is bitwise unchanged across evolution while the body relaxes
around it.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Blocker #6 — `frozen_leaves` in MERA imag-time evolution

The Promoter (`src/qft_pcn/composition/promoter.py:apply_init_clamp`) writes lemma leaf tensors into the host MERA and returns a `set[int]` of clamped leaves. But the downstream call to `mera_imaginary_evolve_state` (or `mera_trotter_step`) has no way to receive that set — subsequent relaxation freely overwrites the clamped leaves, defeating the §8.6 acceptance gate ("clamp + freeze").

This is the smallest of the five blockers — a surgical addition to the evolution loop.

### Operator-algebraic design

A gate is **skipped** when its target leaf is in `frozen_leaves`. Skipping means: the leaf vector is not multiplied by the gate matrix. This is operator-algebraically the projector-trivial restriction of the Hamiltonian's domain — formally `H_frozen = P_unfrozen H P_unfrozen + (1-P_unfrozen)`, where `P_unfrozen` is the projector onto the subspace where the frozen leaves are exactly their clamped value. In practice this is implemented as a single conditional in the gate-grouping loop of `mera_trotter_step`: when a gate's target leaf is frozen, drop it before grouping.

For pair gates with one frozen and one unfrozen leaf, both legs are dropped (the gate is not separable into a single-leg residue; safest is to skip the whole gate — the frozen leaf is part of a proved lemma, the unfrozen leaf does not yet belong to a redex that the lemma alone resolves; if relaxation needs to act on it, a non-frozen-touching gate will do it later).

This is §1.5 verbatim: the freeze is a tensor-network operation (a per-leaf skip in the gate dispatch loop), never an AST or graph rewrite.

### Files to touch

- Modify: `src/qft_pcn/logic/mera_evolution_logic.py` (~20 lines) — add `frozen_leaves: set[int] | None = None` to `mera_trotter_step` and `mera_imaginary_evolve_state`; in `mera_trotter_step`'s gate-grouping loop, skip any gate whose target leaf(es) intersect `frozen_leaves`.
- Test: `src/qft_pcn/tests/test_mera_evolution_frozen.py`.

### Edit pointers

`src/qft_pcn/logic/mera_evolution_logic.py`:

- Line 30 `def mera_trotter_step(state, ham, dt, imaginary=True, chi_layer=None, cache=None)` → add `frozen_leaves: set[int] | None = None` after `cache`.
- Line 82 `for term in ham.terms:` — after `gates = ham.term_gates(...)`, filter gates:

```python
        if frozen_leaves:
            gates = [(leaves, gate) for (leaves, gate) in gates
                     if not (set(leaves) & frozen_leaves)]
```

- Line 166 `def mera_imaginary_evolve_state(state, ham, dt, steps, chi_layer=None)` → add `frozen_leaves: set[int] | None = None`; pass through to `mera_trotter_step`.
- Line 152 `def mera_imaginary_evolve` likewise (it just delegates).

### API additions

- `mera_trotter_step(..., frozen_leaves: set[int] | None = None)`.
- `mera_imaginary_evolve_state(..., frozen_leaves: set[int] | None = None)`.
- `mera_imaginary_evolve(..., frozen_leaves: set[int] | None = None)`.

### Test design

Create `src/qft_pcn/tests/test_mera_evolution_frozen.py`:

```python
"""Tests for frozen_leaves in MERA imag-time evolution (§5.2a, §8.6)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import Bin, IntLit
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_trotter_step, mera_imaginary_evolve_state,
)


def test_frozen_leaf_bitwise_unchanged_one_step():
    src = Bin(op="+", lhs=IntLit(2), rhs=IntLit(3))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    # Freeze the BIN node's value leaf -- the rule that would normally
    # drive it to slot 5 is now barred.
    bin_val_leaf = meta.layout.leaf_of(0, "value")
    frozen = {bin_val_leaf}
    snap = np.array(state.leaves[bin_val_leaf]).copy()
    new = mera_trotter_step(state, H, dt=0.05, chi_layer=16,
                            frozen_leaves=frozen)
    assert np.allclose(new.leaves[bin_val_leaf], snap)


def test_frozen_leaves_bitwise_unchanged_many_steps():
    src = Bin(op="+", lhs=IntLit(2), rhs=IntLit(3))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    leaf = meta.layout.leaf_of(0, "kind")
    frozen = {leaf}
    snap = np.array(state.leaves[leaf]).copy()
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=60, chi_layer=16,
        frozen_leaves=frozen)
    assert np.allclose(state.leaves[leaf], snap, atol=1e-12)


def test_non_frozen_leaves_still_evolve():
    # The frozen-leaf machinery does not turn off ALL evolution.
    src = Bin(op="+", lhs=IntLit(2), rhs=IntLit(3))
    state, meta = encode_mera(src)
    H = MeraEvalHamiltonian(meta)
    e0 = H.total_energy(state)
    # Freeze only a PAD leaf (encoder-allocated tail leaf) -- no gate
    # ever targets it, so energy descent is unaffected.
    pad_leaves = [i for i, sp in enumerate(meta.species_of_leaf)
                  if sp == "PAD"]
    frozen = {pad_leaves[0]} if pad_leaves else set()
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.05, steps=60, chi_layer=16,
        frozen_leaves=frozen)
    assert H.total_energy(state) < e0
```

### Dependencies / sequencing

- This blocker has **no dependencies on the others**. It is the first to land (smallest surface, unblocks #1 and #5).
- After #6 lands, schedule: #5 (Forall-protected), #3 (R-AddZero), #4 (R-Eq-Refl), #1 (relax_program).

### Commit

```
feat(mera_evolution): frozen_leaves= kwarg in imag-time evolution

Spec §5.2a / §8.6: the Promoter's apply_init_clamp writes a cached
lemma's leaf tensors into the host MERA's lemma window and returns the
set of clamped leaves. Subsequent imag-time evolution must NOT deform
those leaves; this commit adds the surgical hook -- frozen_leaves= in
mera_trotter_step and mera_imaginary_evolve_state, threaded through the
gate-grouping loop as a per-leaf skip. Operator-algebraic: a per-leaf
gate-dispatch restriction, never an AST rewrite (§1.5).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Suggested landing order

1. **#6** (frozen_leaves) — smallest, no deps. Unblocks #1, #5.
2. **#5** (Forall-protected) — depends on #6; populates the protected set; needed for #3's open-case test and #4's universal test.
3. **#3** (R-AddZero) — depends on #5 for the open case.
4. **#4** (R-Eq-Refl) — depends on #3 (the composite `Eq (x+0) x` test) and #5.
5. **#1** (relax_program) — depends on #6 for `frozen_leaves` wire-through; the open-case relaxation test depends on #3+#4+#5 to actually converge.
6. **#2** (parser surface syntax) — handled by a parallel subagent; lands independently.

After all five land, re-attempt I Task 10's verbatim two-stage acceptance demo.
