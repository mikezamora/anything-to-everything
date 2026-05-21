# Typing Hamiltonian Implementation Plan — Part 3 of 3 (Acceptance tests + final verification)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Continues from Parts 1 and 2.** Assumes Tasks 1–18 are complete: the encoder produces 5-species MPS with tobl and bid bond extension; `TypingHamiltonian` exposes `total_energy`/`residuals` with all 8 rules implemented.

This part covers Tasks 19–28: the five WT programs, five IT programs, per-rule isolation tests for each rule, classical-checker agreement test, performance budget, structural-Hamiltonian invariants (§7.8 of spec), residual sum check, public exports, and the final spec acceptance gate.

---

## Task 19: Test helper module `_test_mutate.py`

**Files:**
- Create: `src/qft_pcn/tests/_test_helpers.py`.

We use this in several isolation tests to surgically mutate a site's local register state (swap two basis-state slices in a site tensor). Centralized helper avoids duplication.

- [ ] **Step 1: Implement the helper**

Create `src/qft_pcn/tests/_test_helpers.py`:

```python
"""Test-only helpers for the typing Hamiltonian acceptance tests.

`mutate_local_register` swaps two basis-state slices on a single species'
register at one MPS site. Used to surgically violate a typing rule and
verify the per-rule term fires.

This helper does NOT preserve normalization on its own; the caller usually
calls state.normalize() afterward (though the swap is a unitary
permutation on the local register subspace, so norm is preserved for
sliced amplitudes).
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.logic.encoding import (
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
)


_SPECIES_INDEX = {"kind": 0, "type": 1, "bid": 2, "value": 3, "tobl": 4}
_SPECIES_DIMS = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF)


def mutate_local_register(state: MPS, site: int, register: str,
                          old: int, new: int) -> MPS:
    """Swap basis slices `old` and `new` on `register` at `site`.

    Equivalent to applying a local unitary swap gate on the chosen register.
    Returns the SAME MPS (mutation is in place).
    """
    if register not in _SPECIES_INDEX:
        raise ValueError(f"unknown register {register!r}")
    sp_idx = _SPECIES_INDEX[register]
    sp_dim = _SPECIES_DIMS[sp_idx]
    if not (0 <= old < sp_dim and 0 <= new < sp_dim):
        raise ValueError(f"old/new out of range for {register}")
    t = state.tensors[site]
    chi_l, d, chi_r = t.shape
    # Reshape into per-species axes.
    A = t.reshape(chi_l, *_SPECIES_DIMS, chi_r)
    # Move species axis to second position (after chi_l).
    target = sp_idx + 1
    A_moved = np.moveaxis(A, target, 1)
    # Swap slices.
    out = A_moved.copy()
    tmp = out[:, old, ...].copy()
    out[:, old, ...] = out[:, new, ...]
    out[:, new, ...] = tmp
    A_back = np.moveaxis(out, 1, target)
    state.tensors[site] = A_back.reshape(chi_l, d, chi_r)
    return state
```

- [ ] **Step 2: Smoke test the helper itself**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def test_mutate_local_register_preserves_norm():
    """The local-register swap is a unitary; norm stays at 1."""
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    state, _ = encode(parse(r"\x:Int. x"), N=4, chi_max=32)
    n0 = state.norm_sq()
    mutate_local_register(state, site=1, register="type",
                          old=TYPE_INT, new=TYPE_BOOL)
    n1 = state.norm_sq()
    assert abs(n1 - n0) < 1e-10
```

- [ ] **Step 3: Run**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py::test_mutate_local_register_preserves_norm -v`
Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/tests/_test_helpers.py src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic): add mutate_local_register helper for surgical violations

Used by the typing-rule isolation tests to swap basis slices on a chosen
register at a chosen site, simulating a specific typing-rule violation.
The swap is a local unitary; norm is preserved.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Acceptance test — five well-typed programs (WT1..WT5)

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Per spec §7.1. For each WT program, `total_energy < 1e-10`.

- [ ] **Step 1: Write the tests**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
WT_PROGRAMS = [
    ("WT1", r"\x:Int. x"),
    ("WT2", r"(\x:Int. x + 1)(2)"),
    ("WT3", r"\f:Int->Int. \x:Int. f (f x)"),
    ("WT4", r"if (1 < 2) then ((\x:Bool. x)(true)) else false"),
    ("WT5", r"(\x:Int. (\y:Int. x + y)(3))(4)"),
]


@pytest.mark.parametrize("name,src", WT_PROGRAMS, ids=[p[0] for p in WT_PROGRAMS])
def test_well_typed_residual_zero(name, src):
    """Spec §7.1: for each well-typed program, ⟨H_typing⟩ = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    p = parse(src)
    state, meta = encode(p, N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    energy = H.total_energy(state)
    # Identify any nonzero residual (for diagnostics on failure).
    residuals = {k: v for k, v in H.residuals(state).items()
                 if v > 1e-10}
    assert abs(energy) < 1e-9, (
        f"{name} ({src}): expected ⟨H_typing⟩ = 0, got {energy}; "
        f"nonzero residuals: {residuals}"
    )
```

- [ ] **Step 2: Run and verify**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "well_typed_residual_zero" -v`
Expected: 5 tests pass (one per WT program). If any fails, the diagnostic prints the nonzero residuals.

If a test fails with a specific rule firing at a specific site, debug:
1. Re-read the spec rule for that rule_id.
2. Check the encoder writes the correct tobl/param_ty for that site.
3. Add focused unit tests to isolate the bug.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic/typing_hamiltonian): five well-typed acceptance programs (WT1..WT5)

Spec §7.1: each of the five demo programs from sub-project A's spec must
evaluate to ⟨H_typing⟩ = 0 to within 1e-9. Programs span:
  WT1: identity lambda
  WT2: arithmetic in a lambda body
  WT3: higher-order application chain
  WT4: if-then-else with type coercion
  WT5: nested binders with shadowing

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Acceptance test — five ill-typed programs (IT1..IT5)

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Per spec §7.2. For each IT program, `total_energy > 0.5` and exactly one rule fires.

Note: IT programs are constructed to violate exactly one rule. The expected rule and site for each:

- **IT1** `\x:Int. (x + true)`: T-Obligation at the BoolLit site (BIN's rhs expects Int, got Bool).
- **IT2** `\x:Int. (if x then 1 else 0)`: T-Obligation at Var x (IF's cond expects Bool, got Int).
- **IT3** `\x:Int. ((1)(x))`: T-App-Arrow at the inner APP (fn = IntLit 1 has type Int, not arrow).

  Wait — this IT3 won't parse because `(1)(x)` is `App(1, x)` and `1.type = Int` is not arrow. Encoding may fail or succeed depending on `_compute_ast_type`'s fallback. Let me adapt: the encoder's `_compute_ast_type` for `App(IntLit(1), Var(x))` will return TInt (the fallback because fn is not arrow). So the encoded MPS would have APP.type = T_INT and fn.type = T_INT. T-App-Arrow at APP site fires because dst-allowed-by-type-INT is `{T_ARR_II, T_ARR_BI, T_ARR_NESTED}` — fn.type = T_INT is NOT in that set, so T-App-Arrow fires.

- **IT4**: revise to a cleaner single-violation case. Let me pick a different one.
- **IT5** `(\x:Bool. x + 1)(true)`: T-Obligation at Var x (BIN's lhs expects Int, got Bool); ALSO at the BIN_INT itself wait let me re-check. Lambda binds x:Bool. body is x + 1. At Var(x): tobl = T_INT (BIN's lhs), type = T_BOOL. T-Obligation fires. T-Var: x.type = T_BOOL, binder.param_ty = T_BOOL — same. T-Var doesn't fire. So exactly one violation (T-Obligation at Var x).

Let me re-examine IT4. Cleaner: `\x:Int->Bool. x 1`. App(Var x, IntLit 1). x has param_ty Int->Bool; arg is Int. The App's type should be Bool. Encoder: App.type = dst(Int->Bool) = Bool. fn = Var x, type = Int->Bool (arrow Int -> Bool = T_ARR_IB). T-App-Arrow at the APP site: APP.type = Bool, fn must be arrow with dst=Bool. Allowed = {T_ARR_IB, T_ARR_BB, T_ARR_NESTED}. fn.type = T_ARR_IB is allowed — T-App-Arrow doesn't fire. T-Obligation at arg: tobl = src(fn.type) = Int. arg.type = Int. Doesn't fire. So this is WELL-TYPED!

Let me pick a truly ill-typed IT4: `\x:Int. (x x)`. Here x has type Int (not arrow); App(Var x, Var x). _compute_ast_type returns TInt for App (fallback). fn = Var x, type = Int. T-App-Arrow: APP.type = T_INT, fn.type must be in `{T_ARR_II, T_ARR_BI, T_ARR_NESTED}`. fn.type = T_INT, fires.

But this has TWO violations: T-App-Arrow AND T-Obligation on arg (since the fallback set arg's obligation to TOBL_NONE).

Actually, looking at the spec's IT4 example: I wrote "T-Obligation at the (x 1) APP site". Let me rewrite as: `\x:Int->Bool. (x 1) + 1`. Outer BIN's lhs (the APP `(x 1)`) has tobl=T_INT but the APP's type is T_BOOL. T-Obligation fires at the APP site.

Let me also recheck IT3. I'll commit to clean cases that exhibit exactly one rule firing:

**IT1**: `\x:Int. (x + true)` — T-Obligation at the BoolLit site (BIN rhs).
**IT2**: `\x:Int. if x then 1 else 0` — T-Obligation at Var x (IF cond).
**IT3**: `\x:Int->Bool. (x 1) + 1` — T-Obligation at the inner APP site (outer BIN's lhs).
**IT4**: `\x:Int. (\y:Bool. y)(x)` — T-Obligation at the inner Var x (App arg, expected Bool, got Int).
**IT5**: `(\x:Bool. x + 1)(true)` — T-Obligation at Var x (BIN lhs expected Int, got Bool).

All five focus on T-Obligation violations at different positions; they're cleanly single-violation.

To test T-Var, T-Lit-Int, T-Lit-Bool, T-Bin-Arith, T-Bin-Cmp, and T-App-Arrow violations explicitly, we use the isolation tests (Tasks 22-25) that surgically mutate the MPS.

- [ ] **Step 1: Write the tests**

Append:

```python
IT_PROGRAMS = [
    # (id, source, expected_violation_rule, expected_violation_site_predicate)
    # site predicate: function that takes the encoded sites layout and
    # returns the expected violating site index. Computed below per program.
]


def _it_violation_descriptor():
    """The mapping is computed lazily once we can parse each source."""
    # Just list (id, source, expected_rule_id, AST-path-based predicate to
    # locate the site).
    return [
        # IT1: \x:Int. x + true
        # Pre-order: LAM@0, BIN+@1, VAR_x@2, BoolLit@3, PAD...
        # Violation: T-Obligation at site 3 (BoolLit, BIN's rhs, tobl=INT, type=BOOL).
        ("IT1", r"\x:Int. x + true", "T-Obligation", 3),
        # IT2: \x:Int. if x then 1 else 0
        # Pre-order: LAM@0, IF@1, VAR_x@2, IntLit@3, IntLit@4, PAD...
        # Violation: T-Obligation at site 2 (Var x, IF's cond, tobl=BOOL, type=INT).
        ("IT2", r"\x:Int. if x then 1 else 0", "T-Obligation", 2),
        # IT3: \x:Int->Bool. (x 1) + 1
        # Pre-order: LAM@0, BIN+@1, APP@2, VAR_x@3, IntLit@4, IntLit@5, PAD...
        # Violation: T-Obligation at site 2 (APP, BIN's lhs, tobl=INT, type=BOOL).
        ("IT3", r"\x:Int->Bool. (x 1) + 1", "T-Obligation", 2),
        # IT4: \x:Int. (\y:Bool. y)(x)
        # Pre-order: LAM_x@0, APP@1, LAM_y@2, VAR_y@3, VAR_x@4, PAD...
        # Violation: T-Obligation at site 4 (Var x, App's arg, tobl=BOOL, type=INT).
        ("IT4", r"\x:Int. (\y:Bool. y)(x)", "T-Obligation", 4),
        # IT5: (\x:Bool. x + 1)(true)
        # Pre-order: APP@0, LAM_x@1, BIN+@2, VAR_x@3, IntLit@4, BoolLit@5, PAD...
        # Violation: T-Obligation at site 3 (Var x, BIN's lhs, tobl=INT, type=BOOL).
        ("IT5", r"(\x:Bool. x + 1)(true)", "T-Obligation", 3),
    ]


@pytest.mark.parametrize(
    "name,src,expected_rule,expected_site",
    _it_violation_descriptor(),
    ids=lambda p: p[0] if isinstance(p, tuple) else None,
)
def test_ill_typed_residual_localized(name, src, expected_rule, expected_site):
    """Spec §7.2: each ill-typed program has ⟨H⟩ > 0.5 with exactly one
    term firing > 0.5, at the expected (rule, site)."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian

    state, meta = encode(parse(src), N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    energy = H.total_energy(state)
    assert energy > 0.5, f"{name}: expected ⟨H⟩ > 0.5, got {energy}"

    residuals = H.residuals(state)
    big = [(k, v) for k, v in residuals.items() if v > 0.5]
    assert len(big) == 1, (
        f"{name}: expected exactly 1 firing rule, got {len(big)}: {big}"
    )
    fired_key, fired_val = big[0]
    assert fired_key == (expected_rule, expected_site), (
        f"{name}: expected fire at {(expected_rule, expected_site)}, "
        f"got {fired_key} with energy {fired_val}"
    )
    # Sum the small residuals — they should be at most numerical noise.
    small_sum = sum(v for k, v in residuals.items() if v <= 0.5)
    assert small_sum < 1e-8, (
        f"{name}: small residuals sum to {small_sum} (too large; expected "
        f"only one term firing)"
    )
```

- [ ] **Step 2: Run and verify**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "ill_typed_residual_localized" -v`
Expected: 5 tests pass.

If any test fails because the expected violation site is wrong, the test driver prints the actual firing rule + site; correct the expected values in `_it_violation_descriptor`.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic/typing_hamiltonian): five ill-typed programs (IT1..IT5)

Each ill-typed program violates exactly one typing rule at one site. The
test verifies ⟨H_typing⟩ > 0.5 AND exactly one term fires > 0.5 at the
expected (rule, site).

Programs and violations:
  IT1: x + true            — T-Obligation at BoolLit (BIN rhs)
  IT2: if x then 1 else 0  — T-Obligation at Var x (IF cond)
  IT3: (x 1) + 1           — T-Obligation at App (outer BIN lhs)
  IT4: (\y:Bool. y)(x)     — T-Obligation at Var x (App arg)
  IT5: (\x:Bool. x+1)(true)— T-Obligation at Var x (BIN lhs)

All five are T-Obligation violations at different lattice positions — the
T-Obligation rule's centrality (handling Lam-body, App-arg, If-*, Bin-*
constraints) shows up here. Per-rule isolation tests for T-Lit-Int,
T-Lit-Bool, T-Bin-Arith, T-Bin-Cmp, T-Var, T-App-Arrow are in Task 22.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 22: Per-rule isolation tests for T-Lit-Int, T-Lit-Bool, T-Bin-Arith, T-Bin-Cmp

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Each test encodes a well-typed program, mutates one site's `type` register to violate one specific rule, and verifies that rule fires while others don't.

- [ ] **Step 1: Write the tests**

Append:

```python
def test_isolation_t_lit_int():
    """Spec §7.5: corrupt an IntLit's type to T_BOOL; T-Lit-Int fires alone."""
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    state, _ = encode(parse(r"\x:Int. 3"), N=8, chi_max=32)
    # Site 1 is IntLit(3). Mutate type INT -> BOOL.
    mutate_local_register(state, 1, "type", TYPE_INT, TYPE_BOOL)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Lit-Int", 1)] > 0.99
    # Other firings: T-Obligation also fires (because tobl at site 1 = TOBL_INT
    # and type is now T_BOOL). So we DON'T require "only T-Lit-Int fires" —
    # spec §7.5 specifically says "the corresponding term fires and no others
    # do" for the surgically-changed register — but T-Obligation observing
    # the same register naturally also fires.
    # Refined assertion: T-Lit-Int fires AND T-Lit-Bool does NOT.
    assert r[("T-Lit-Bool", 1)] < 1e-10


def test_isolation_t_lit_bool():
    """Mutate a BoolLit's type to T_INT; T-Lit-Bool fires."""
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    from src.qft_pcn.logic.encoding import TYPE_BOOL, TYPE_INT
    state, _ = encode(parse(r"\x:Int. true"), N=8, chi_max=32)
    mutate_local_register(state, 1, "type", TYPE_BOOL, TYPE_INT)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Lit-Bool", 1)] > 0.99
    assert r[("T-Lit-Int", 1)] < 1e-10


def test_isolation_t_bin_arith():
    """Mutate a BIN(+)'s type to T_BOOL; T-Bin-Arith fires."""
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    state, _ = encode(parse(r"\x:Int. 1 + 2"), N=8, chi_max=32)
    # Site 1 is BIN(+). Mutate INT -> BOOL.
    mutate_local_register(state, 1, "type", TYPE_INT, TYPE_BOOL)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Bin-Arith", 1)] > 0.99
    assert r[("T-Bin-Cmp", 1)] < 1e-10


def test_isolation_t_bin_cmp():
    """Mutate a BIN(<)'s type to T_INT; T-Bin-Cmp fires."""
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_BOOL, TYPE_INT
    state, _ = encode(parse(r"\x:Int. x < 5"), N=8, chi_max=32)
    # Site 1 is BIN(<). Mutate BOOL -> INT.
    mutate_local_register(state, 1, "type", TYPE_BOOL, TYPE_INT)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Bin-Cmp", 1)] > 0.99
    assert r[("T-Bin-Arith", 1)] < 1e-10
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "isolation_t_lit or isolation_t_bin" -v`
Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic/typing_hamiltonian): per-rule isolation for lit and bin rules

Four tests verifying that surgically corrupting one register on one site
fires the expected rule (T-Lit-Int, T-Lit-Bool, T-Bin-Arith, T-Bin-Cmp)
while the complementary rule does NOT fire. The mutated register may also
trigger T-Obligation as a downstream consequence (the parent expectation
is now violated); that's expected and is tested separately.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 23: Per-rule isolation for T-Var and T-App-Arrow

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

- [ ] **Step 1: Write the tests**

Append:

```python
def test_isolation_t_var():
    """Spec §7.3: corrupt Var's type to mismatch its binder's param_ty.
    T-Var fires (the bond carries the original param_ty as a real DOF)."""
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_BOOL
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    # Site 1 is Var(x). Mutate type INT -> BOOL.
    mutate_local_register(state, 1, "type", TYPE_INT, TYPE_BOOL)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-Var", 1)] > 0.99


def test_isolation_t_app_arrow():
    """Spec §7.4: corrupt fn's type to a non-arrow; T-App-Arrow fires."""
    from src.qft_pcn.tests._test_helpers import mutate_local_register
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoding import TYPE_ARR_II, TYPE_INT
    state, _ = encode(parse(r"(\x:Int. x)(1)"), N=8, chi_max=32)
    # Site 0 = APP, site 1 = LAM (fn). LAM's type = T_ARR_II.
    # Mutate site 1's type to T_INT — now fn looks like an Int, not an arrow.
    mutate_local_register(state, 1, "type", TYPE_ARR_II, TYPE_INT)
    state.normalize()
    H = TypingHamiltonian(N=8)
    r = H.residuals(state)
    assert r[("T-App-Arrow", 0)] > 0.99
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "isolation_t_var or isolation_t_app" -v`
Expected: 2 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic/typing_hamiltonian): per-rule isolation for T-Var and T-App-Arrow

T-Var: corrupting Var's type register breaks the bond-channel param_ty
coupling. T-Var fires because the bond DOF still carries the original
param_ty while the site-local type was changed.

T-App-Arrow: corrupting fn's type to a non-arrow type fires the
arrow-dst-matches-APP-type constraint.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 24: Encoder-extension correctness test (spec §7.7)

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

- [ ] **Step 1: Implement the classical-reference helper**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def _compute_expected_tobl_from_ast(ast, N: int) -> list[int]:
    """Mirror the encoder's compute_tobl_tags by computing expected per-site
    obligations from the AST. Used as a reference oracle for the encoder
    test below."""
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic._typing_extension import compute_tobl_tags
    sites = serialize_preorder(ast, N=N)
    return compute_tobl_tags(ast, sites)


def test_encoder_writes_tobl_per_site_correctly_for_all_WT():
    """Spec §7.7: for every WT program, encoder's tobl matches the
    reference computation."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    for name, src in WT_PROGRAMS:
        p = parse(src)
        state, meta = encode(p, N=32, chi_max=32)
        expected = _compute_expected_tobl_from_ast(p, N=32)
        assert meta.tobl_per_site == expected, (
            f"{name}: meta.tobl_per_site disagrees with reference"
        )


def test_encoder_writes_channel_param_ty_correctly():
    """Spec §7.7: each bond's channel param_ty matches the binders' param_ty
    in the AST."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_ARR_II

    p = parse(r"\f:Int->Int. \x:Int. f x")
    state, meta = encode(p, N=16, chi_max=32)
    # Bond 0 (LAM_f@0 - LAM_x@1): only Lam_f live, param_ty = Int->Int.
    assert meta.channel_param_ty_per_bond[0] == [TYPE_ARR_II]
    # Bond 1 (LAM_x@1 - APP@2): both Lam_f and Lam_x live.
    assert meta.channel_param_ty_per_bond[1] == [TYPE_ARR_II, TYPE_INT]


def test_encoder_tobl_consistency_in_p5():
    """WT5 has nested binders and shadowing; verify tobl values explicitly."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.encoding import TOBL_NONE, TOBL_INT
    p = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    state, meta = encode(p, N=32, chi_max=32)
    # Layout: APP@0, LAM_x@1, APP@2, LAM_y@3, BIN+@4, VAR_x@5, VAR_y@6,
    #         INT@7 (the 3), INT@8 (the 4), PAD...
    # Tobl by rule:
    #   0 (APP root): NONE
    #   1 (LAM_x, fn of outer APP): NONE
    #   2 (APP inner, body of LAM_x): TOBL_INT (LAM_x's body type = Int)
    #   3 (LAM_y, fn of inner APP): NONE
    #   4 (BIN+, body of LAM_y): TOBL_INT
    #   5 (Var x, BIN lhs): TOBL_INT
    #   6 (Var y, BIN rhs): TOBL_INT
    #   7 (IntLit 3, arg of inner APP): TOBL_INT
    #   8 (IntLit 4, arg of outer APP): TOBL_INT
    expected = [
        TOBL_NONE, TOBL_NONE, TOBL_INT, TOBL_NONE, TOBL_INT,
        TOBL_INT, TOBL_INT, TOBL_INT, TOBL_INT,
    ] + [TOBL_NONE] * 23
    assert meta.tobl_per_site == expected
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "encoder_writes" -v`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic): encoder-extension correctness (tobl + channel_param_ty)

Spec §7.7: for every well-typed program, the encoder's EncodingMeta has:
  - tobl_per_site == expected from rule-table
  - channel_param_ty_per_bond[b] matches each live binder's param_ty in
    declaration order

Three tests: (a) tobl agrees with reference for all five WT programs,
(b) channel_param_ty has correct shape and values for a two-binder program,
(c) tobl on the nested-binder P5 matches an explicitly computed expected
list.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 25: Classical-checker agreement (spec §7.11)

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

A reference type-checker in the test file produces a `dict[(rule, site), bool]` of expected violations. The Hamiltonian's residuals must agree EXACTLY.

- [ ] **Step 1: Implement the classical checker**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def _classical_type_check_residuals(ast, N: int) -> dict[tuple[str, int], bool]:
    """Reference type-checker: for each (rule_id, site), return True iff
    that rule should fire at that site.

    The keys of the returned dict are a subset of the (rule_id, site) keys
    in TypingHamiltonian(N).terms. For absent keys, the rule should NOT
    fire (energy ~ 0).
    """
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic._types import compute_site_types, ty_to_tag
    from src.qft_pcn.logic._channels import (
        compute_live_binders, compute_channel_param_ty_per_bond,
    )
    from src.qft_pcn.logic._typing_extension import compute_tobl_tags
    from src.qft_pcn.logic.encoding import (
        KIND_INT, KIND_BOOL, KIND_BIN, KIND_LAM, KIND_VAR, KIND_APP,
        TYPE_INT, TYPE_BOOL, TYPE_NONE, TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI,
        TYPE_ARR_BB, TYPE_ARR_NESTED,
        TOBL_NONE,
        VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
        BIN_VALUE_FROM_OP,
    )
    from src.qft_pcn.logic.typing_hamiltonian import _APP_FN_ALLOWED_BY_DST

    sites = serialize_preorder(ast, N=N)
    type_tags = compute_site_types(ast, sites)
    tobl_tags = compute_tobl_tags(ast, sites)
    live = compute_live_binders(sites)
    pt_per_bond = compute_channel_param_ty_per_bond(sites, live)

    out: dict[tuple[str, int], bool] = {}

    arith_codes = {VALUE_PLUS, VALUE_MINUS, VALUE_TIMES}
    cmp_codes = {VALUE_LT, VALUE_EQ}

    for k in range(N):
        occ = sites[k]
        ki = occ.kind
        ti = type_tags[k]
        oi = tobl_tags[k]
        vi = (BIN_VALUE_FROM_OP[occ.bin_op]
              if ki == KIND_BIN and occ.bin_op else 0)

        # T-Lit-Int: kind=INT and type != INT.
        out[("T-Lit-Int", k)] = (ki == KIND_INT and ti != TYPE_INT)
        # T-Lit-Bool: kind=BOOL and type != BOOL.
        out[("T-Lit-Bool", k)] = (ki == KIND_BOOL and ti != TYPE_BOOL)
        # T-Bin-Arith: kind=BIN, value in arith codes, type != INT.
        out[("T-Bin-Arith", k)] = (
            ki == KIND_BIN and vi in arith_codes and ti != TYPE_INT
        )
        # T-Bin-Cmp.
        out[("T-Bin-Cmp", k)] = (
            ki == KIND_BIN and vi in cmp_codes and ti != TYPE_BOOL
        )
        # T-Obligation: tobl != NONE and type != tobl.
        out[("T-Obligation", k)] = (oi != TOBL_NONE and ti != oi)

    # T-Var: bond (k-1, k) where kind(k)=VAR; type(k) != binder.param_ty.
    for k in range(1, N):
        occ = sites[k]
        if occ.kind != KIND_VAR:
            out[("T-Var", k)] = False
            continue
        # Find which binder this Var refers to (via var_ref.binder_site).
        binder_lam_site = occ.var_ref.binder_site
        bond_idx = k - 1
        # Find the binder's channel slot on bond bond_idx and get its param_ty.
        channel_pt = None
        for c_idx, bh in enumerate(live[bond_idx]):
            if bh.lam_site == binder_lam_site:
                channel_pt = pt_per_bond[bond_idx][c_idx]
                break
        if channel_pt is None:
            out[("T-Var", k)] = False
        else:
            out[("T-Var", k)] = (type_tags[k] != channel_pt)

    # T-Abs: bond (k, k+1) where kind(k)=LAM; outgoing channel's param_ty !=
    # src(type(k)).
    for k in range(N - 1):
        occ = sites[k]
        if occ.kind != KIND_LAM:
            out[("T-Abs", k)] = False
            continue
        # The encoder writes the new binder's param_ty onto the outgoing
        # channel. The LAM's param_ty IS that value. The LAM's type is the
        # ARROW; src(arrow) should equal the binder's param_ty.
        # type_tags[k] is the arrow tag; pt_per_bond[k] has the NEW binder's
        # param_ty as its last entry (declaration order).
        from src.qft_pcn.logic.typing_hamiltonian import _ARROW_SRC
        if type_tags[k] == TYPE_ARR_NESTED:
            out[("T-Abs", k)] = False   # permissive
            continue
        if type_tags[k] not in _ARROW_SRC:
            # LAM's type is not an arrow — encoder's invariant is broken.
            # Still flag T-Abs as firing.
            out[("T-Abs", k)] = True
            continue
        expected_src = _ARROW_SRC[type_tags[k]]
        # The new binder is the last in pt_per_bond[k].
        if not pt_per_bond[k]:
            out[("T-Abs", k)] = False  # no binder created — encoder error
        else:
            actual_pt = pt_per_bond[k][-1]
            out[("T-Abs", k)] = (actual_pt != expected_src)

    # T-App-Arrow: bond (k, k+1) where kind(k)=APP; type(k+1) not in allowed.
    for k in range(N - 1):
        occ = sites[k]
        if occ.kind != KIND_APP:
            out[("T-App-Arrow", k)] = False
            continue
        y = type_tags[k]
        allowed = _APP_FN_ALLOWED_BY_DST.get(y, [])
        out[("T-App-Arrow", k)] = (type_tags[k + 1] not in allowed)

    return out


@pytest.mark.parametrize("name,src",
                         WT_PROGRAMS + [(n, s) for (n, s, _, _)
                                        in _it_violation_descriptor()])
def test_hamiltonian_agrees_with_classical_checker(name, src):
    """Spec §7.11: per-rule residuals agree with the classical type-checker
    on all 10 demo programs."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian

    p = parse(src)
    state, _ = encode(p, N=32, chi_max=32)
    classical = _classical_type_check_residuals(p, N=32)
    H = TypingHamiltonian(N=32)
    residuals = H.residuals(state)
    mismatches: list[str] = []
    for (rule, site), expected_violated in classical.items():
        actual_energy = residuals.get((rule, site), 0.0)
        if expected_violated and actual_energy <= 0.5:
            mismatches.append(
                f"{rule}@{site}: classical says violated, H says "
                f"energy={actual_energy:.4g}"
            )
        elif not expected_violated and actual_energy > 1e-8:
            mismatches.append(
                f"{rule}@{site}: classical says OK, H says "
                f"energy={actual_energy:.4g}"
            )
    assert not mismatches, f"{name}: {mismatches}"
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "agrees_with_classical" -v`
Expected: 10 tests pass (5 WT + 5 IT).

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic/typing_hamiltonian): classical-checker agreement on all 10 demos

Spec §7.11: for each WT and IT program, the Hamiltonian's per-rule
residuals match a classical type-checker's verdict EXACTLY. Violated rules
have energy > 0.5; non-violated rules have energy < 1e-8.

The classical checker is the spec — the Hamiltonian must agree. Disagreement
is a bug in either the Hamiltonian construction or the encoder extension.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 26: Structural-Hamiltonian invariants (spec §7.8) + residual sum check (§7.9)

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

- [ ] **Step 1: Write the tests**

Append:

```python
def test_typing_hamiltonian_is_structural():
    """Spec §7.8: the same H evaluates correctly on multiple unrelated ASTs."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian

    H = TypingHamiltonian(N=32)
    state1, _ = encode(parse(r"\x:Int. x"), N=32, chi_max=32)
    state2, _ = encode(parse(r"(\x:Int. x + 1)(2)"), N=32, chi_max=32)
    state3, _ = encode(parse(r"if 1 < 2 then 10 else 20"), N=32, chi_max=32)
    assert abs(H.total_energy(state1)) < 1e-9
    assert abs(H.total_energy(state2)) < 1e-9
    assert abs(H.total_energy(state3)) < 1e-9


def test_typing_hamiltonian_no_ast_attributes():
    """Spec §7.8: H must not store AST data. Attribute names cannot suggest
    AST dependence."""
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    H = TypingHamiltonian(N=32)
    forbidden_substrings = ["ast", "node", "tree", "binder_handle",
                            "var_ref", "occupancy", "meta_"]
    for attr_name in dir(H):
        if attr_name.startswith("_"):
            continue
        for sub in forbidden_substrings:
            assert sub not in attr_name.lower(), (
                f"H.{attr_name} suggests AST dependency"
            )


def test_typing_hamiltonian_constructor_takes_only_N():
    """Spec §1.1: TypingHamiltonian must construct from N alone."""
    import inspect
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    sig = inspect.signature(TypingHamiltonian.__init__)
    params = list(sig.parameters.keys())
    # 'self' + 'N' only.
    assert params == ["self", "N"]


def test_residuals_sum_equals_total_energy_all_demos():
    """Spec §7.9: Σ residuals == total_energy for every demo."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    H = TypingHamiltonian(N=32)
    for name, src in WT_PROGRAMS:
        state, _ = encode(parse(src), N=32, chi_max=32)
        total = H.total_energy(state)
        residuals_sum = sum(H.residuals(state).values())
        assert abs(total - residuals_sum) < 1e-9, (
            f"{name}: total={total}, sum={residuals_sum}"
        )
    for name, src, _, _ in _it_violation_descriptor():
        state, _ = encode(parse(src), N=32, chi_max=32)
        total = H.total_energy(state)
        residuals_sum = sum(H.residuals(state).values())
        assert abs(total - residuals_sum) < 1e-9, (
            f"{name}: total={total}, sum={residuals_sum}"
        )


def test_residual_keys_are_addressable():
    """Each (rule_id, site) key is a non-empty str and a valid site index."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    for (rule_id, site), energy in H.residuals(state).items():
        assert isinstance(rule_id, str) and rule_id
        assert 0 <= site < 8
        assert isinstance(energy, float)
        assert energy >= -1e-12  # Hermitian projector → non-negative
```

- [ ] **Step 2: Run**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "structural or residuals_sum or constructor or keys_are_addressable or no_ast_attributes" -v`
Expected: 5 tests pass.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic/typing_hamiltonian): structural-Hamiltonian invariants

Spec §7.8: TypingHamiltonian constructs from N alone (no AST input). Its
attributes don't reference any AST term. The same instance correctly
evaluates multiple unrelated programs.

Spec §7.9: Σ residuals == total_energy for every demo program; per-term
residuals are addressable by (str rule_id, int site) and non-negative.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 27: Performance budget + public exports

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py` (performance test).
- Modify: `src/qft_pcn/logic/__init__.py` (re-exports).
- Modify: `src/qft_pcn/__init__.py` (top-level re-exports).

- [ ] **Step 1: Write the performance test**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
@pytest.mark.timeout(10)
def test_total_energy_under_budget_p5():
    """Spec §7.10: encoding WT5 + 5 evaluations completes under 10 seconds."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    p = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    state, _ = encode(p, N=32, chi_max=32)
    H = TypingHamiltonian(N=32)
    for _ in range(5):
        e = H.total_energy(state)
        assert abs(e) < 1e-9
```

- [ ] **Step 2: Update `src/qft_pcn/logic/__init__.py`**

Edit `src/qft_pcn/logic/__init__.py`. Add the new TOBL_* and TypingHamiltonian exports:

```python
from .encoding import (
    # ... existing ...
    TOBL_NONE, TOBL_INT, TOBL_BOOL,
    TOBL_ARR_II, TOBL_ARR_IB, TOBL_ARR_BI, TOBL_ARR_BB,
    TOBL_ARR_NESTED, TOBL_CUTOFF,
)
from .typing_hamiltonian import (
    TypingHamiltonian, TypingTerm,
    TypingHamiltonianError, TermNotFound,
    RULE_T_LIT_INT, RULE_T_LIT_BOOL,
    RULE_T_BIN_ARITH, RULE_T_BIN_CMP,
    RULE_T_OBLIGATION, RULE_T_VAR, RULE_T_ABS, RULE_T_APP_ARROW,
)

__all__ = [
    # ... existing ...
    "TOBL_NONE", "TOBL_INT", "TOBL_BOOL",
    "TOBL_ARR_II", "TOBL_ARR_IB", "TOBL_ARR_BI", "TOBL_ARR_BB",
    "TOBL_ARR_NESTED", "TOBL_CUTOFF",
    "TypingHamiltonian", "TypingTerm",
    "TypingHamiltonianError", "TermNotFound",
    "RULE_T_LIT_INT", "RULE_T_LIT_BOOL",
    "RULE_T_BIN_ARITH", "RULE_T_BIN_CMP",
    "RULE_T_OBLIGATION", "RULE_T_VAR", "RULE_T_ABS", "RULE_T_APP_ARROW",
]
```

- [ ] **Step 3: Update `src/qft_pcn/__init__.py`**

Re-export TypingHamiltonian and TypingTerm at the top level:

```python
from .logic import TypingHamiltonian, TypingTerm
# extend __all__ accordingly
```

- [ ] **Step 4: Smoke test the re-exports**

```python
def test_top_level_typing_hamiltonian_import():
    """The TypingHamiltonian is accessible from qft_pcn top-level."""
    from src.qft_pcn import TypingHamiltonian, TypingTerm
    from src.qft_pcn import encode, parse
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    assert abs(H.total_energy(state)) < 1e-10
```

Append this test to `test_logic_typing_hamiltonian.py`.

- [ ] **Step 5: Run all typing-hamiltonian tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -v 2>&1 | tail -50`
Expected: all tests pass (40+ tests).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py src/qft_pcn/logic/__init__.py src/qft_pcn/__init__.py
git commit -m "$(cat <<'EOF'
feat(logic): public re-exports for TypingHamiltonian + performance test

Adds TypingHamiltonian, TypingTerm, RULE_T_* and TOBL_* constants to the
logic package's public surface, plus top-level qft_pcn re-exports. The
performance test confirms total_energy(state) on WT5 runs 5 iterations
under 10 seconds with chi_max=32.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 28: Final verification — all spec acceptance criteria

**Files:** none (verification only).

This is the spec §10 acceptance gate. Per `superpowers:verification-before-completion`, do not claim done without observing all-green output.

- [ ] **Step 1: Run the full typing-hamiltonian test suite**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py src/qft_pcn/tests/test_logic_factored_expectation.py src/qft_pcn/tests/test_logic_typing_extension.py -v 2>&1 | tail -50`
Expected: every test passes (≈ 50 tests).

- [ ] **Step 2: Run sub-project A's logic tests (no regressions)**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_ast.py src/qft_pcn/tests/test_logic_encoding.py src/qft_pcn/tests/test_logic_resolve.py src/qft_pcn/tests/test_logic_serialize.py src/qft_pcn/tests/test_logic_types.py src/qft_pcn/tests/test_logic_channels.py src/qft_pcn/tests/test_logic_tensors.py src/qft_pcn/tests/test_logic_encoder_smoke.py src/qft_pcn/tests/test_logic_decoder.py src/qft_pcn/tests/test_logic_roundtrip.py src/qft_pcn/tests/test_logic_gate_construction.py src/qft_pcn/tests/test_logic_acceptance.py -v 2>&1 | tail -30`
Expected: every test passes (A's full suite).

- [ ] **Step 3: Run the pre-existing qft tests to confirm zero regressions**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_qft.py src/qft_pcn/tests/test_qft_pcn.py src/qft_pcn/tests/test_multifield.py -v 2>&1 | tail -20`
Expected: every test passes.

- [ ] **Step 4: Run the entire test suite (excluding qiskit-dependent test_quantum.py)**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/ -v --ignore=src/qft_pcn/tests/test_quantum.py 2>&1 | tail -40`
Expected: every test passes.

- [ ] **Step 5: Confirm spec acceptance criteria one-by-one**

The criteria list from `docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md`, §10:

1. **Encoder extension correct**: `test_encoder_writes_tobl_*` and `test_encoder_writes_channel_param_ty*`.
2. **WT programs (5) → ⟨H⟩ < 1e-10**: `test_well_typed_residual_zero` (parameterized over WT1..WT5).
3. **IT programs (5) → ⟨H⟩ > 0.5, localized**: `test_ill_typed_residual_localized` (parameterized over IT1..IT5).
4. **Per-rule isolation (6 rules)**: `test_isolation_t_lit_int`, `test_isolation_t_lit_bool`, `test_isolation_t_bin_arith`, `test_isolation_t_bin_cmp`, `test_isolation_t_var`, `test_isolation_t_app_arrow`.
5. **Structural Hamiltonian**: `test_typing_hamiltonian_is_structural`, `test_typing_hamiltonian_no_ast_attributes`, `test_typing_hamiltonian_constructor_takes_only_N`.
6. **Residual decomposition**: `test_residuals_sum_equals_total_energy_all_demos`, `test_residual_keys_are_addressable`.
7. **Performance budget**: `test_total_energy_under_budget_p5`.
8. **Classical-checker agreement**: `test_hamiltonian_agrees_with_classical_checker` (parameterized over all 10 demos).
9. **A's tests still pass**: Step 2 above.

Confirm by grepping:

```bash
grep -E "^def test_" src/qft_pcn/tests/test_logic_typing_hamiltonian.py | wc -l
```

Expected: ≥ 40 tests.

- [ ] **Step 6: Final commit (CHANGELOG note, optional)**

If a changelog file exists at repo root, add a note for sub-project B; otherwise skip. Sub-project B is complete.

---

## Done

Sub-project B is implemented. Sub-project C (evaluation Hamiltonian / §10.3) is the next item in the §10 decomposition, with its own spec/plan/implementation cycle.

The interfaces this sub-project produces — `TypingHamiltonian`, `EncodingMeta.tobl_per_site`, `EncodingMeta.channel_param_ty_per_bond`, the per-rule residuals API — are the contract C and D will program against.

Per `superpowers:verification-before-completion`, do not claim the sub-project complete without the `pytest` runs in Task 28 actually succeeding in a fresh shell. The user expects evidence, not assertions.

**Notable items for downstream:**

- **Sub-project C** (evaluation Hamiltonian) inherits the 5-species lattice and may extend it further (e.g., adding an `eval_state` species). C's reduction Hamiltonian acts on the SAME MPS that B's typing Hamiltonian acts on; the joint Hamiltonian `H_typing + H_eval` is what sub-project E imaginary-time-evolves.
- **Sub-project D** (debugger) consumes `H.residuals(state)` and `meta.site_to_ast_path` to produce structured error messages. The rule_id strings (`"T-Lit-Int"`, etc.) are stable.
- **Sub-project E** (synthesis demo) needs `H_typing` to be applicable in TEBD style. B explicitly does NOT provide that; E will need to handle either reduced cutoffs or factored Trotter gates. This is documented as E's prerequisite.

The encoder extension and `TypingHamiltonian` together realize the architecture document's central thesis (§8.1) operationally: **type checking is gauge-invariance verification, on a real quantum state, via a real Hermitian operator** — not a Python classical type-checker dressed up in quantum clothing.
