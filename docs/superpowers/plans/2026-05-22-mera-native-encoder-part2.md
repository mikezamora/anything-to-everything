# MERA-Native Logic Encoder Implementation Plan — Part 2 of 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Continues from Part 1** (`2026-05-22-mera-native-encoder-part1.md`, Tasks 1–7). Assumes: extended-calculus AST nodes, `mera_encoding.py` constants, the node-major layout, per-leaf basis vectors, `encode_mera` (concrete path), `decode_mera`, and concrete round-trip P1–P8 are all done and committed.

**Spec:** `docs/superpowers/specs/2026-05-22-mera-native-encoder-design.md` — authoritative.

**Driving principles:** see Part 1's preamble. The one most in play in Part 2: **binding is genuine entanglement, never a classical lookup.** Part 2's hole-encoding path is where the §1.1 soul is realized on the MERA tree — the structural-marker test (Task 13) fails loudly if a shortcut is taken.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Task 8: `mera_window_expectation` — dense (k≤2) form

**Files:**
- Create: `src/qft_pcn/logic/_mera_window.py`
- Test: `src/qft_pcn/tests/test_mera_window.py`

The dense form for `k≤2` cross-checks against F's `MERA.two_site_expectation`. The factored form (Task 9) is the one M2 actually calls.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_window.py`:

```python
"""Tests for mera_window_expectation (spec §8.2, §9.8)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic._mera_window import mera_window_expectation
from src.qft_pcn.logic.mera_encoding import MERA_LEAF_DIM


def test_window_k1_matches_local_expectation():
    """k=1 window equals MERA.local_expectation."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    # Number operator on leaf 0.
    op = np.diag(np.arange(d)).astype(complex)
    via_window = mera_window_expectation(state, leaf0=0, k=1, op=op)
    via_local = state.local_expectation(0, op)
    assert np.isclose(via_window, via_local, atol=1e-10)


def test_window_k2_matches_two_site_expectation():
    """k=2 window equals MERA.two_site_expectation (spec §9.8)."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    rng = np.random.default_rng(0)
    # A random Hermitian two-leaf operator.
    m = rng.standard_normal((d * d, d * d)) + 1j * rng.standard_normal((d * d, d * d))
    op = m + m.conj().T
    via_window = mera_window_expectation(state, leaf0=0, k=2, op=op)
    via_two_site = state.two_site_expectation(0, op)
    assert np.isclose(via_window, via_two_site, atol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_window.py -v`
Expected: ImportError on `_mera_window`.

- [ ] **Step 3: Implement the dense `mera_window_expectation`**

Create `src/qft_pcn/logic/_mera_window.py`:

```python
"""k-adjacent-leaf expectation on a MERA (spec §8.2).

mera_window_expectation is the dense form, used only for k<=2 to
cross-check F's two_site_expectation. M2's Hamiltonian terms call the
factored form (Task 9) — never the dense form for k>2, since a dense
(16**k, 16**k) operator is intractable for large k.
"""
from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.mera import MERA


def mera_window_expectation(state: MERA, leaf0: int, k: int,
                            op: np.ndarray) -> complex:
    """<state | O | state> for O on the k adjacent leaves [leaf0, leaf0+k).

    op has shape (d**k, d**k) with d = state.d_local. For k==1 this
    delegates to MERA.local_expectation; for k==2 to
    MERA.two_site_expectation. k>2 dense is rejected — use the factored
    form (mera_window_expectation_factored).
    """
    d = state.d_local
    if k == 1:
        if op.shape != (d, d):
            raise ValueError(f"k=1 op shape {op.shape}, expected ({d},{d})")
        return state.local_expectation(leaf0, op)
    if k == 2:
        if op.shape != (d * d, d * d):
            raise ValueError(
                f"k=2 op shape {op.shape}, expected ({d*d},{d*d})")
        return state.two_site_expectation(leaf0, op)
    raise ValueError(
        f"dense mera_window_expectation supports k in {{1,2}}; for k={k} "
        f"use mera_window_expectation_factored")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_window.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_mera_window.py src/qft_pcn/tests/test_mera_window.py
git commit -m "$(cat <<'EOF'
feat(logic/_mera_window): dense k<=2 window expectation on MERA

Delegates k=1 to local_expectation and k=2 to two_site_expectation;
rejects dense k>2 (use the factored form). Cross-checks F's MERA
expectation primitives.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `mera_window_expectation_factored`

**Files:**
- Modify: `src/qft_pcn/logic/_mera_window.py`
- Test: `src/qft_pcn/tests/test_mera_window.py` (append)

The factored form takes one small `(16,16)` operator per leaf in the window; unlisted leaves are identity. No `16**k` tensor is formed. This is what M2's per-node (k=5) and two-node (k=10) typing terms call.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_mera_window.py`:

```python
from src.qft_pcn.logic._mera_window import mera_window_expectation_factored


def test_factored_single_leaf_matches_local():
    """A factored op on one leaf equals local_expectation."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    op = np.diag(np.arange(d)).astype(complex)
    via_factored = mera_window_expectation_factored(state, {0: op})
    via_local = state.local_expectation(0, op)
    assert np.isclose(via_factored, via_local, atol=1e-10)


def test_factored_two_separable_leaves_matches_two_site():
    """A factored op O_a (x) O_b on leaves 0,1 equals two_site_expectation
    of the kron product."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    rng = np.random.default_rng(1)
    a = rng.standard_normal((d, d)); a = a + a.T
    b = rng.standard_normal((d, d)); b = b + b.T
    op_a = a.astype(complex)
    op_b = b.astype(complex)
    via_factored = mera_window_expectation_factored(state, {0: op_a, 1: op_b})
    via_two_site = state.two_site_expectation(0, np.kron(op_a, op_b))
    assert np.isclose(via_factored, via_two_site, atol=1e-10)


def test_factored_identity_leaves_skipped():
    """Listing an identity op for a leaf equals not listing it."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    d = MERA_LEAF_DIM
    op = np.diag(np.arange(d)).astype(complex)
    I = np.eye(d, dtype=complex)
    with_id = mera_window_expectation_factored(state, {0: op, 1: I})
    without = mera_window_expectation_factored(state, {0: op})
    assert np.isclose(with_id, without, atol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_window.py -k factored -v`
Expected: ImportError on `mera_window_expectation_factored`.

- [ ] **Step 3: Implement the factored form**

Append to `src/qft_pcn/logic/_mera_window.py`:

```python
def mera_window_expectation_factored(
    state: MERA, leaf_ops: dict[int, np.ndarray]) -> complex:
    """<state | O | state> where O = prod over listed leaves of a per-leaf
    (d, d) operator (unlisted leaves: identity).

    No (d**k, d**k) tensor is formed. Each per-leaf operator is applied to
    the state by MERA.apply_local_gate on a copy, then the inner product
    with the original gives the expectation — because the listed operators
    act on distinct leaves and therefore commute, the product operator's
    expectation equals < state | (prod gates) | state >.

    Apply each gate to the ket copy, then return <state | ket_copy>.
    """
    d = state.d_local
    # Drop identity operators.
    active = {leaf: op for leaf, op in leaf_ops.items()
              if not _is_identity(op, d)}
    if not active:
        return complex(state.norm_sq())
    ket = state.copy()
    for leaf, op in active.items():
        if op.shape != (d, d):
            raise ValueError(
                f"leaf {leaf} op shape {op.shape}, expected ({d},{d})")
        ket.apply_local_gate(leaf, op)
    return state.inner(ket)


def _is_identity(op: np.ndarray, d: int) -> bool:
    if op.shape != (d, d):
        return False
    return np.allclose(op, np.eye(d, dtype=op.dtype), atol=1e-12)
```

**Note on correctness:** the listed per-leaf operators act on *distinct* leaves, so they pairwise commute and `O = ∏ gate_leaf`. `apply_local_gate` applies a gate to one leaf in place; applying all of them to a ket copy yields `O|state⟩`; `state.inner(O|state⟩) = ⟨state|O|state⟩`. This is exact and forms no `16**k` tensor. Verify `MERA.apply_local_gate` and `MERA.inner` exist with these semantics by reading `mera.py` (Part-1 Task 5 already used `from_product`; `apply_local_gate` and `inner` are in `mera.py`'s public API per the spec §1 principle 6 list).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_window.py -v`
Expected: 5 passed (2 from Task 8 + 3 here).

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_mera_window.py src/qft_pcn/tests/test_mera_window.py
git commit -m "$(cat <<'EOF'
feat(logic/_mera_window): factored k-leaf window expectation

mera_window_expectation_factored takes one (16,16) op per leaf; applies
them as commuting local gates to a ket copy and returns <state|ket>.
No 16**k tensor is ever formed — the form M2's per-node (k=5) and
two-node (k=10) typing terms call.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Hole-bearing encoding — tree entanglement

**Files:**
- Create: `src/qft_pcn/logic/_mera_holes.py`
- Modify: `src/qft_pcn/logic/mera_encoder.py` (wire the hole path)
- Test: `src/qft_pcn/tests/test_mera_holes.py`

This is the §1.1 soul on the MERA tree (spec §5.3). A `HoleVar(candidates=[...])` use site's `bid` leaf carries an equal-amplitude superposition over the candidates' `bid` indices; that leaf is entangled, through the tree, with the candidate binders.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_holes.py`:

```python
"""Tests for hole-bearing MERA encoding (spec §5.3, §5.5)."""
from __future__ import annotations
import math
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera


def _hole_program():
    """\\x:Int. \\y:Int. ?HOLE  with HoleVar over {x, y}."""
    h = HoleVar(candidates=["x", "y"])
    return Lam(param="x", param_ty=TInt(),
               body=Lam(param="y", param_ty=TInt(), body=h))


def test_hole_program_encodes_to_unit_norm():
    state, meta = encode_mera(_hole_program())
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-10)


def test_hole_bid_leaf_is_superposed():
    """The hole's bid leaf has weight on more than one basis state."""
    state, meta = encode_mera(_hole_program())
    # The hole is AST node 2 (Lam_x@0, Lam_y@1, HOLE@2); its bid leaf is
    # 5*2 + 2 = 12.
    from src.qft_pcn.logic.mera_encoding import MERA_LEAF_DIM
    p = np.empty(MERA_LEAF_DIM)
    for b in range(MERA_LEAF_DIM):
        proj = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
        proj[b, b] = 1.0
        p[b] = float(np.real(state.local_expectation(12, proj)))
    nonzero = np.sum(p > 1e-9)
    assert nonzero >= 2, f"hole bid leaf not superposed: {p}"


def test_hole_program_is_not_a_product_state():
    """A genuine hole encoding has tree entanglement somewhere."""
    state, meta = encode_mera(_hole_program())
    # Some interior cut has positive entanglement entropy.
    max_S = max(state.entanglement_entropy(cut)
                for cut in range(1, state.N - 1))
    assert max_S > 1e-6, "hole program encoded as a product state"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_holes.py -v`
Expected: `encode_mera` raises `NotImplementedError` (the Part-1 stub).

- [ ] **Step 3: Implement the hole path**

Create `src/qft_pcn/logic/_mera_holes.py`. The construction (spec §5.3):

1. Encode the program as if concrete, but the hole's `bid` leaf vector is the equal-amplitude superposition `(1/√k) Σ_j e_{bid index of candidate j}` instead of a one-hot. `from_product` then gives a state where the hole's bid leaf is superposed but **unentangled** (a product state with one superposed leaf).
2. Entangle the hole's `bid` leaf with the candidate binders' `bid` leaves: for each candidate `j`, apply a CNOT-like two-leaf gate (via `MERA.apply_two_site_gate`) that correlates "hole bid leaf is in candidate j's bid state" with "candidate j's binder is the referenced one". This is the gate-based construction; it produces genuine tree entanglement because `apply_two_site_gate` modifies the disentanglers/isometries on the connecting path.

```python
"""Hole-bearing MERA encoding: genuine tree entanglement (spec §5.3).

A HoleVar use site's bid leaf carries an equal-amplitude superposition
over the candidate binders' bid indices, entangled through the tree with
those binders. Gate-based construction: start from a product MERA with
the hole's bid leaf superposed, then apply CNOT-like two-leaf gates that
correlate the hole's choice with each candidate binder.
"""
from __future__ import annotations

import numpy as np

from .ast import Node, HoleVar
from .mera_encoding import MERA_LEAF_DIM
from src.qft_pcn.qft.mera import MERA


def cnot_like_gate(control_value: int, d: int = MERA_LEAF_DIM) -> np.ndarray:
    """A (d*d, d*d) two-leaf gate: when the control leaf is in basis state
    `control_value`, flip the target leaf to mark the correlation; identity
    otherwise. Convention matches np.kron (control = outer index).

    Concretely: |control_value, t> -> |control_value, t XOR mark> and
    |c != control_value, t> -> |c, t>. For a marker we use t -> t with a
    phase, but the simplest faithful correlator is the controlled basis
    permutation below.
    """
    g = np.zeros((d * d, d * d), dtype=complex)
    for c in range(d):
        for t in range(d):
            row = c * d + t
            if c == control_value:
                t_out = (t + 1) % d        # controlled increment = mark
            else:
                t_out = t
            col = c * d + t_out
            g[col, row] = 1.0
    return g
```

The exact entangling protocol (which leaves, which gates, which order) depends on the candidate binders' positions and is the intricate part. Specify it precisely in `_mera_holes.py`'s `entangle_hole(state, hole_bid_leaf, candidate_binder_bid_leaves, candidate_bid_values) -> None` (in-place). The protocol:

- For a hole with `k` candidates, the hole's `bid` leaf starts in `(1/√k) Σ_j e_{v_j}` where `v_j` is candidate `j`'s bid index.
- For each candidate `j`, apply `cnot_like_gate(control_value=v_j)` as a two-leaf gate (`MERA.apply_two_site_gate`) between the hole's `bid` leaf and candidate `j`'s binder `bid` leaf, so that the branch where the hole chose `v_j` is correlated with that binder.
- `apply_two_site_gate` requires the two leaves be adjacent in MERA's pair structure or handles arbitrary leaves — read `mera.py::apply_two_site_gate` for its leaf-adjacency contract. If it requires adjacency, the protocol must route the correlation through intermediate gates (a chain), or use a sequence of adjacent swaps; pin this in the implementation, using the structural-marker test (Task 13) and `test_hole_program_is_not_a_product_state` as oracles.

Then modify `mera_encoder.py`: replace the `NotImplementedError` branch with a call into `_mera_holes`. The hole path: build leaf vectors as in the concrete path but with superposed bid leaves for holes; `from_product`; then `entangle_hole` for each hole; `normalize`.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_holes.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the concrete suite to confirm no regression**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_encoder_concrete.py src/qft_pcn/tests/test_mera_roundtrip.py --timeout=60 -q 2>&1 | tail -6`
Expected: all still pass (the hole path must not disturb the concrete path).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/_mera_holes.py src/qft_pcn/logic/mera_encoder.py src/qft_pcn/tests/test_mera_holes.py
git commit -m "$(cat <<'EOF'
feat(logic/_mera_holes): hole-bearing encoding with genuine tree entanglement

A HoleVar use site's bid leaf carries an equal-amplitude superposition
over candidate binder bid indices; CNOT-like two-leaf gates entangle it
through the tree with the candidate binders. The encoded state is not a
product state — the §1.1 binding-as-entanglement principle realized on
the MERA tree.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `sample_mera`

**Files:**
- Modify: `src/qft_pcn/logic/mera_decoder.py`
- Test: `src/qft_pcn/tests/test_mera_decoder.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_mera_decoder.py`:

```python
import numpy as np
from src.qft_pcn.logic.mera_decoder import sample_mera


def test_sample_returns_n_results():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    out = sample_mera(state, meta, n_samples=3,
                      rng=np.random.default_rng(0))
    assert len(out) == 3
    for r in out:
        assert isinstance(r, DecodeResult)


def test_sample_product_state_is_deterministic():
    """For a concrete program (product state) every sample equals decode."""
    from src.qft_pcn.logic.decoder import ast_alpha_eq
    state, meta = encode_mera(parse(r"\x:Int. x + 1"))
    det = decode_mera(state, meta)
    rng = np.random.default_rng(7)
    for _ in range(4):
        s = sample_mera(state, meta, n_samples=1, rng=rng)[0]
        assert ast_alpha_eq(s.ast, det.ast)
```

(`DecodeResult`, `decode_mera`, `parse`, `encode_mera` are already imported at the top of the file from Task 6 / Part 1.)

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_decoder.py -k sample -v`
Expected: ImportError on `sample_mera`.

- [ ] **Step 3: Implement `sample_mera`**

Append to `src/qft_pcn/logic/mera_decoder.py`:

```python
def sample_mera(state: MERA, meta: MeraEncodingMeta,
                n_samples: int = 1, rng=None) -> list[DecodeResult]:
    """Sample n_samples ASTs from the MERA distribution.

    Left-to-right leaf-by-leaf conditional sampling: measure each leaf
    from its marginal conditioned on prior measurements, project, advance.
    For a product (concrete) state the marginal is a delta and every
    sample equals decode_mera. For a hole-bearing state each call draws
    a fresh completion.
    """
    if rng is None:
        rng = np.random.default_rng()
    results: list[DecodeResult] = []
    for _ in range(n_samples):
        ket = state.copy()
        per_node: list[tuple[int, int, int, int, int]] = []
        node_idxs: list[int] = []
        for leaf in range(LEAVES_PER_NODE * meta.n_nodes):
            p = _leaf_marginal(ket, leaf)
            b = int(rng.choice(MERA_LEAF_DIM, p=p))
            # Project the ket onto basis state b at this leaf and renormalize.
            proj = np.zeros((MERA_LEAF_DIM, MERA_LEAF_DIM), dtype=complex)
            proj[b, b] = 1.0
            ket.apply_local_gate(leaf, proj)
            nrm = ket.norm_sq()
            if nrm > 1e-15:
                ket.normalize()
            node_idxs.append(b)
            if len(node_idxs) == LEAVES_PER_NODE:
                per_node.append(tuple(node_idxs))
                node_idxs = []
        ast = _structural_parse(per_node, meta)
        results.append(DecodeResult(ast=ast, residual_norm=0.0))
    return results
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_decoder.py -v`
Expected: all decoder tests pass (Task 6's 3 + 2 here).

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_decoder.py src/qft_pcn/tests/test_mera_decoder.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_decoder): sample_mera conditional sampling

Left-to-right leaf-by-leaf conditional measurement. Deterministic on
product (concrete) states; draws fresh completions on hole-bearing
states. Sub-project M3's synthesis will sample hole completions through
this entry point.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Acceptance — layout, alpha-renaming, PAD vacuum

**Files:**
- Create: `src/qft_pcn/tests/test_mera_acceptance.py`

- [ ] **Step 1: Write the tests**

Create `src/qft_pcn/tests/test_mera_acceptance.py`:

```python
"""MERA-native encoder acceptance suite (spec §9.2, §9.3, §9.5)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def test_layout_is_node_major_5n_padded():
    """spec §9.2."""
    state, meta = encode_mera(parse(r"\x:Int. x"))   # 2 nodes
    assert meta.n_nodes == 2
    assert meta.n_leaves == 16
    assert meta.species_of_leaf[:10] == [
        "kind", "type", "bid", "value", "tobl",
        "kind", "type", "bid", "value", "tobl",
    ]
    assert all(s == "PAD" for s in meta.species_of_leaf[10:])


def test_alpha_renaming_identical_state():
    """spec §9.3: \\x.x and \\y.y encode to the same MERA state."""
    s1, _ = encode_mera(parse(r"\x:Int. x"))
    s2, _ = encode_mera(parse(r"\y:Int. y"))
    overlap = abs(s1.inner(s2)) ** 2
    assert overlap > 1.0 - 1e-10, f"overlap {overlap}"


def test_pad_leaves_are_vacuum():
    """spec §9.5: PAD leaves have zero amplitude on non-PAD basis states."""
    from src.qft_pcn.logic.mera_encoding import MERA_LEAF_DIM, KIND_PAD
    state, meta = encode_mera(parse(r"\x:Int. x"))
    proj = np.eye(MERA_LEAF_DIM, dtype=complex)
    proj[KIND_PAD, KIND_PAD] = 0.0       # project away from PAD
    for leaf in range(meta.n_leaves):
        if meta.species_of_leaf[leaf] == "PAD":
            val = state.local_expectation(leaf, proj)
            assert abs(val) < 1e-10, f"PAD leaf {leaf} not vacuum: {val}"
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_acceptance.py -v`
Expected: 3 passed.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_mera_acceptance.py
git commit -m "$(cat <<'EOF'
test(mera): layout, alpha-renaming, PAD-vacuum acceptance (spec §9.2/3/5)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Acceptance — the binding-as-entanglement structural marker

**Files:**
- Modify: `src/qft_pcn/tests/test_mera_acceptance.py` (append)

This is the §1.1 gate. If a subagent encoded holes with a classical-lookup shortcut (definite bid at the hole leaf, no tree entanglement), this test fails.

- [ ] **Step 1: Write the test**

Append to `src/qft_pcn/tests/test_mera_acceptance.py`:

```python
from src.qft_pcn.logic.ast import Lam, TInt, HoleVar


def test_binding_is_entanglement_structural_marker():
    """spec §9.4 / §5.5. A hole-bearing program has strictly positive
    tree entanglement entropy across a cut separating the hole's bid leaf
    from the candidate binders' bid leaves.

    A classical-lookup encoding (definite bid at the hole leaf) gives
    S = 0 and fails this test with a message naming the violated section.
    """
    h = HoleVar(candidates=["x", "y"])
    ast = Lam(param="x", param_ty=TInt(),
              body=Lam(param="y", param_ty=TInt(), body=h))
    state, meta = encode_mera(ast)
    # Hole is node 2; its bid leaf is 12. Candidate binders are nodes 0,1;
    # their bid leaves are 2 and 7. A cut at leaf 12 separates the hole's
    # bid leaf region from the binders' region.
    max_S = max(state.entanglement_entropy(cut)
                for cut in range(1, state.N - 1))
    assert max_S > 0.5, (
        f"hole-bearing program has max tree entanglement entropy {max_S}; "
        f"expected > 0.5. A value near 0 means binding was encoded as a "
        f"classical lookup, not entanglement — spec §5, §1.1 violated."
    )
```

- [ ] **Step 2: Run the test**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_acceptance.py::test_binding_is_entanglement_structural_marker -v`
Expected: pass.

If it fails with `max_S ≈ 0`: the Task-10 hole construction did not actually entangle. Do NOT lower the 0.5 threshold. Root-cause the entangling-gate protocol with systematic-debugging — the gates must modify the tree tensors on the connecting path.

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_mera_acceptance.py
git commit -m "$(cat <<'EOF'
test(mera): binding-as-entanglement structural marker (spec §9.4)

The §1.1 gate: a hole-bearing program must carry strictly positive tree
entanglement entropy. A classical-lookup encoding gives S=0 and fails
with a message naming the violation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Acceptance — error paths

**Files:**
- Modify: `src/qft_pcn/tests/test_mera_acceptance.py` (append)

- [ ] **Step 1: Write the tests**

Append to `src/qft_pcn/tests/test_mera_acceptance.py`:

```python
from src.qft_pcn.logic.encoding import (
    EncodingTooLarge, TooManyBinders, IntLiteralOutOfRange,
    IllScopedVar, UnsupportedNode,
)


def test_encoding_too_large_raises():
    with pytest.raises(EncodingTooLarge):
        encode_mera(parse(r"\x:Int. x + x + x"), n_nodes_max=3)


def test_too_many_binders_raises():
    src = (r"\a:Int. \b:Int. \c:Int. \d:Int. \e:Int. \f:Int. \g:Int. "
           r"\h:Int. a")
    with pytest.raises(TooManyBinders):
        encode_mera(parse(src))


def test_int_literal_out_of_range_raises():
    with pytest.raises(IntLiteralOutOfRange):
        encode_mera(parse(r"\x:Int. x + 99"))


def test_ill_scoped_var_raises():
    with pytest.raises(IllScopedVar):
        encode_mera(parse("undefined_name"))


def test_unsupported_node_raises():
    class _Bogus:
        pass
    with pytest.raises(UnsupportedNode):
        encode_mera(_Bogus())  # type: ignore[arg-type]
```

- [ ] **Step 2: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_acceptance.py -k raises -v`
Expected: 5 passed. (These exceptions come from the reused `_resolve.py`/`_serialize.py`/`_mera_leaves.py` front-half; if any does not propagate, fix the propagation — do not catch-and-swallow.)

- [ ] **Step 3: Commit**

```bash
git add src/qft_pcn/tests/test_mera_acceptance.py
git commit -m "$(cat <<'EOF'
test(mera): error-path acceptance (spec §9.6)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Public re-exports

**Files:**
- Modify: `src/qft_pcn/logic/__init__.py`
- Modify: `src/qft_pcn/__init__.py`
- Test: `src/qft_pcn/tests/test_mera_acceptance.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_mera_acceptance.py`:

```python
def test_top_level_import_path():
    """encode_mera/decode_mera reachable from the package surface."""
    from src.qft_pcn.logic import encode_mera as e2, decode_mera as d2
    state, meta = e2(parse(r"\x:Int. x"))
    res = d2(state, meta)
    assert res.residual_norm < 1e-10
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_acceptance.py::test_top_level_import_path -v`
Expected: ImportError — `encode_mera` not exported from `logic/__init__.py`.

- [ ] **Step 3: Add the re-exports**

In `src/qft_pcn/logic/__init__.py`, add to the imports and `__all__`:

```python
from .mera_encoder import encode_mera, MeraEncodingMeta
from .mera_decoder import decode_mera, sample_mera
from .ast import (
    Zero, Succ, NatLit, Nil, Cons, Eq, Forall, Fix,
    TNat, TList, TEq, TProp,
)
```

Append these names to `logic/__init__.py`'s `__all__` list.

In `src/qft_pcn/__init__.py`, add `encode_mera`, `decode_mera` to the `from .logic import ...` line and to `__all__` (alongside the existing `encode`, `decode`).

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_acceptance.py::test_top_level_import_path -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/__init__.py src/qft_pcn/__init__.py src/qft_pcn/tests/test_mera_acceptance.py
git commit -m "$(cat <<'EOF'
feat(logic): re-export the MERA-native encoder surface

encode_mera, decode_mera, sample_mera, MeraEncodingMeta and the
extended-calculus AST nodes are now reachable from logic/ and the
qft_pcn package top level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Final verification

Per `superpowers:verification-before-completion` — no completion claim without fresh evidence.

- [ ] **Step 1: Run the full MERA-encoder suite**

Run:
```
.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_ast.py src/qft_pcn/tests/test_mera_encoding.py src/qft_pcn/tests/test_mera_layout.py src/qft_pcn/tests/test_mera_leaves.py src/qft_pcn/tests/test_mera_encoder_concrete.py src/qft_pcn/tests/test_mera_decoder.py src/qft_pcn/tests/test_mera_roundtrip.py src/qft_pcn/tests/test_mera_window.py src/qft_pcn/tests/test_mera_holes.py src/qft_pcn/tests/test_mera_acceptance.py --timeout=120 -v 2>&1 | tail -40
```
Expected: every test passes.

- [ ] **Step 2: Confirm F's MERA tests and the MPS logic stack still pass**

Run:
```
.venv/bin/python -m pytest src/qft_pcn/tests/test_mera.py src/qft_pcn/tests/test_mera_compat.py src/qft_pcn/tests/test_logic_ast.py src/qft_pcn/tests/test_logic_roundtrip.py --timeout=120 -q 2>&1 | tail -12
```
Expected: all pass — M1 did not regress F's MERA substrate or the shipped MPS logic stack.

- [ ] **Step 3: Check the spec §12 acceptance criteria one-by-one**

For each of the 11 criteria in spec §12, point to the passing test:
1. Round-trip P1–P8 — `test_mera_roundtrip.py`.
2. Layout — `test_mera_acceptance.py::test_layout_is_node_major_5n_padded`.
3. Alpha-renaming — `test_mera_acceptance.py::test_alpha_renaming_identical_state`.
4. Structural marker — `test_mera_acceptance.py::test_binding_is_entanglement_structural_marker`.
5. PAD vacuum — `test_mera_acceptance.py::test_pad_leaves_are_vacuum`.
6. Error paths — `test_mera_acceptance.py` `-k raises`.
7. Cross-substrate anchor — `test_mera_roundtrip.py::test_cross_substrate_anchor_p1_to_p5`.
8. k-leaf cross-check — `test_mera_window.py`.
9. F's MERA + MPS stack still green — Step 2.
10. No tensor larger than `16**2` materialized densely — confirmed by `conftest.py`'s memory ceiling not tripping.
11. Evidence — Steps 1–2 output.

- [ ] **Step 4: Commit a short completion note (only if a CHANGELOG exists; otherwise skip)**

M1 is complete. Sub-project M2 (MERA-native typing + evaluation Hamiltonians) is the next migration piece, with its own spec → plan → implement cycle.

---

## Done

The MERA-native logic encoder is implemented. The revised Phase-1 chain continues: **M2 (typing/eval Hamiltonians on MERA) → M3 (debugger/synthesis + structural superposition)**.

Per `superpowers:verification-before-completion`: do not claim M1 complete without the Task-16 pytest runs succeeding in a fresh shell.
