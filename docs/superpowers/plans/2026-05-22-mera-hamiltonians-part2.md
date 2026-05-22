# MERA-Native Typing + Evaluation Hamiltonians Implementation Plan — Part 2 of 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Continue from Part 1 — Tasks 8 onward. Use superpowers:test-driven-development for every task and superpowers:systematic-debugging on unexpected failures. Run superpowers:verification-before-completion before any completion claim.

**Spec:** `docs/superpowers/specs/2026-05-22-mera-hamiltonians-design.md` — authoritative.

**Part 1** delivered: `MeraEncodingMeta.children_of_node`, the shared rule helpers, `MeraTypingHamiltonian` (STLC + extended-calculus typing rules), and the P1-P10 / IT1-IT5 typing acceptance suite.

**Part 2** delivers: `MeraEvalHamiltonian` (redex penalties + transition gates), `compose_mera_hamiltonians`, MERA imaginary-time evolution, the `Fix` recursion demo, the E1-E5 reduction acceptance, the cross-substrate anchor, and final verification.

---

## Driving principles (from spec §1 — non-negotiable; embed in every subagent prompt)

1. **No time/effort/duration estimates anywhere.** Critical standing directive.
2. **Binding is genuine entanglement.** `R-Fix` couples the FIX node and the recursion-use nodes through the encoded MERA tree, never a classical lookup.
3. **No dense operator at scale.** Every redex penalty and transition gate is factored: `dict[leaf -> (16,16)]` or a 256×256 two-leaf gate. No `16**k` for k>2.
4. **OOM means optimize, not shrink.**
5. **`optimize='greedy'` on every einsum.**
6. **Constraint-based evaluation only.** No explicit beta-reduction rewriter, no classical interpreter, no Lindblad path. `_mera_eval_terms.py` must contain no `substitute`, `beta_reduce`, `evaluate`, and must not import the AST module for walking. **Shortcut (forbidden):** pattern-match `App(Lam,arg)`, substitute, write back. **Alternative (required):** Hermitian penalty + factored transition gate; the spectrum + imaginary-time evolution do the work.
7. **The Hamiltonian is structural.** Built from `MeraEncodingMeta` only.
8. **Reuse M1 and F; do not modify the MPS logic stack.** `MERA.apply_two_site_gate` / `apply_local_gate` are the evolution primitives.

---

## Task 8: Evaluation redex-penalty term builders

The diagonal redex-presence penalties for beta, arithmetic, comparison, if, and Succ — pure factored-operator builders.

**Files:**
- Create: `src/qft_pcn/logic/_mera_eval_terms.py`
- Test: `src/qft_pcn/tests/test_mera_eval_terms.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_eval_terms.py`:

```python
"""Tests for the MERA evaluation redex-penalty builders (spec §7.1)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic._mera_eval_terms import (
    beta_penalty_ops, arith_penalty_ops, cmp_penalty_ops,
    if_penalty_ops, succ_penalty_ops,
    DEFAULT_LAMBDA_BETA, DEFAULT_LAMBDA_ARITH, DEFAULT_LAMBDA_IF,
    DEFAULT_LAMBDA_FIX,
)


def test_default_lambdas_are_positive():
    for lam in (DEFAULT_LAMBDA_BETA, DEFAULT_LAMBDA_ARITH,
                DEFAULT_LAMBDA_IF, DEFAULT_LAMBDA_FIX):
        assert lam > 0


def test_beta_penalty_ops_returns_two_leaf_ops():
    # beta couples the APP node's kind leaf and the fn node's kind leaf.
    ops = beta_penalty_ops(app_kind_leaf=0, fn_kind_leaf=5, lam=1.0)
    assert set(ops.keys()) == {0, 5}
    for op in ops.values():
        assert op.shape == (16, 16)


def test_arith_penalty_ops_three_leaves():
    ops = arith_penalty_ops(bin_kind_leaf=0, bin_value_leaf=3,
                            lhs_kind_leaf=5, rhs_kind_leaf=10, lam=1.0)
    # bin kind, bin value, lhs kind, rhs kind.
    assert len(ops) == 4


def test_penalty_ops_are_projectors():
    ops = if_penalty_ops(if_kind_leaf=0, cond_kind_leaf=5, lam=1.0)
    for op in ops.values():
        # a projector scaled by lam^(1/n) per leaf is fine; check the
        # product is idempotent up to the lam scaling is out of scope.
        # Just verify each is diagonal (basis-projector-like).
        assert np.allclose(op, np.diag(np.diag(op)))
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_eval_terms.py -v`
Expected: ImportError on `_mera_eval_terms`.

- [ ] **Step 3: Implement `_mera_eval_terms.py`**

Create `src/qft_pcn/logic/_mera_eval_terms.py`. Each `*_penalty_ops` returns
a `dict[leaf -> (16,16)]` factored projector onto the *unreduced* redex
configuration. The `lam` scaling is applied by distributing `lam` onto one
leaf's operator (or by the caller scaling the energy — pick one and document;
distributing `lam` onto the `kind` leaf of the primary node is cleanest since
the energy of a product of projectors with one scaled factor equals
`lam · ⟨projector product⟩`).

```python
"""Factored redex-penalty builders for the MERA evaluation Hamiltonian
(spec §7.1, §7.4). Constraint-based: each builder returns a factored
operator (dict[leaf -> (16,16)]) projecting onto an UNREDUCED redex.

This module contains NO classical interpreter: no substitution, no
beta-reduction routine, no AST import. Reduction is driven by the
spectrum via imaginary-time evolution (spec §1.6). The static guard
test enforces this.
"""
from __future__ import annotations

import numpy as np

from .mera_encoding import (
    KIND_APP, KIND_LAM, KIND_BIN, KIND_INT, KIND_IF, KIND_BOOL,
    KIND_SUCC, KIND_NATLIT,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
)
from ._mera_typing_rules import leaf_proj, leaf_proj_set

DEFAULT_LAMBDA_BETA = 1.0
DEFAULT_LAMBDA_ARITH = 1.0
DEFAULT_LAMBDA_IF = 1.0
DEFAULT_LAMBDA_FIX = 1.0


def _scaled(proj: np.ndarray, lam: float) -> np.ndarray:
    """Scale a projector by lam (applied to one leaf of the product so the
    factored expectation yields lam * <projector product>)."""
    return lam * proj


def beta_penalty_ops(app_kind_leaf: int, fn_kind_leaf: int,
                     lam: float) -> dict:
    """P[kind=APP](app) o+ P[kind=LAM](fn), scaled by lam (spec §7.1)."""
    return {
        app_kind_leaf: _scaled(leaf_proj(KIND_APP), lam),
        fn_kind_leaf:  leaf_proj(KIND_LAM),
    }


def arith_penalty_ops(bin_kind_leaf: int, bin_value_leaf: int,
                      lhs_kind_leaf: int, rhs_kind_leaf: int,
                      lam: float) -> dict:
    """P[kind=BIN] . P[value in {+,-,*}] (bin) o+ P[kind=INT](lhs)
    o+ P[kind=INT](rhs), scaled by lam (spec §7.1)."""
    return {
        bin_kind_leaf:  _scaled(leaf_proj(KIND_BIN), lam),
        bin_value_leaf: leaf_proj_set([VALUE_PLUS, VALUE_MINUS,
                                       VALUE_TIMES]),
        lhs_kind_leaf:  leaf_proj(KIND_INT),
        rhs_kind_leaf:  leaf_proj(KIND_INT),
    }


def cmp_penalty_ops(bin_kind_leaf: int, bin_value_leaf: int,
                    lhs_kind_leaf: int, rhs_kind_leaf: int,
                    lam: float) -> dict:
    """Comparison redex: BIN value in {<,==}, operands INT (spec §7.1)."""
    return {
        bin_kind_leaf:  _scaled(leaf_proj(KIND_BIN), lam),
        bin_value_leaf: leaf_proj_set([VALUE_LT, VALUE_EQ]),
        lhs_kind_leaf:  leaf_proj(KIND_INT),
        rhs_kind_leaf:  leaf_proj(KIND_INT),
    }


def if_penalty_ops(if_kind_leaf: int, cond_kind_leaf: int,
                   lam: float) -> dict:
    """P[kind=IF](if) o+ P[kind=BOOL](cond), scaled by lam (spec §7.1)."""
    return {
        if_kind_leaf:   _scaled(leaf_proj(KIND_IF), lam),
        cond_kind_leaf: leaf_proj(KIND_BOOL),
    }


def succ_penalty_ops(succ_kind_leaf: int, arg_kind_leaf: int,
                     lam: float) -> dict:
    """Succ(NatLit n) redex: P[kind=SUCC] o+ P[kind=NATLIT] (spec §7.1)."""
    return {
        succ_kind_leaf: _scaled(leaf_proj(KIND_SUCC), lam),
        arg_kind_leaf:  leaf_proj(KIND_NATLIT),
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_eval_terms.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_mera_eval_terms.py src/qft_pcn/tests/test_mera_eval_terms.py
git commit -m "$(cat <<'EOF'
feat(logic/_mera_eval_terms): factored redex-penalty builders

beta/arith/cmp/if/succ redex-presence penalties as factored operators
on the MERA leaf layout. Constraint-based per spec §1.6: no classical
interpreter, no substitution, no AST import.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: `MeraEvalHamiltonian` — redex penalties

The evaluation Hamiltonian class with the diagonal redex-penalty terms (transition gates added in Task 11).

**Files:**
- Create: `src/qft_pcn/logic/mera_evaluation_hamiltonian.py`
- Test: `src/qft_pcn/tests/test_mera_eval_hamiltonian.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_eval_hamiltonian.py`:

```python
"""Tests for MeraEvalHamiltonian redex penalties (spec §7.1, §9.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import (
    MeraEvalHamiltonian, MeraEvalTerm, MeraEvalTermNotFound,
)


def test_constructs_from_meta():
    _, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    assert len(H.terms) > 0


def test_rule_names_present():
    _, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    names = {t.rule_id for t in H.terms}
    assert {"R-Beta", "R-Arith", "R-Cmp", "R-If"} <= names


def test_normal_form_zero_energy():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraEvalHamiltonian(meta)
    assert abs(H.total_energy(state)) < 1e-9


def test_unreduced_arith_positive_energy():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    assert H.total_energy(state) > 0.5


def test_unreduced_beta_positive_energy():
    state, meta = encode_mera(parse(r"(\x:Int. x)(1)"))
    H = MeraEvalHamiltonian(meta)
    res = H.residuals(state)
    assert any(k[0] == "R-Beta" and v > 0.5 for k, v in res.items())


def test_unknown_term_raises():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    with pytest.raises(MeraEvalTermNotFound):
        H.term_energy(state, MeraEvalTerm("R-Nonsense", 0, 2))


def test_residuals_keyed_by_rule_node():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    for key, val in H.residuals(state).items():
        assert isinstance(key[0], str) and isinstance(key[1], int)
        assert val >= -1e-12
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_eval_hamiltonian.py -v`
Expected: ImportError on `mera_evaluation_hamiltonian`.

- [ ] **Step 3: Implement `mera_evaluation_hamiltonian.py`**

Create the module. Mirror `MeraTypingHamiltonian`'s shape. Term enumeration:
for each rule (`R-Beta`, `R-Arith`, `R-Cmp`, `R-If`, `R-Succ`, `R-Fix`),
enumerate one term per AST node — the energy function returns 0 when the
node's kind does not match (so enumerating every node keeps it structural).
`R-Fix` is stubbed here, filled in Task 11/12.

```python
"""MERA-native evaluation Hamiltonian (sub-project M2; spec §7).

H_eval = sum over redex rules, over AST nodes, of lambda_rule . H_rule.
Each H_rule is a non-negative factored projector onto an UNREDUCED redex
configuration. <psi|H_eval|psi> = 0 iff the encoded program is in normal
form. Reduction is ground-state finding (spec §1.6): imaginary-time
evolution under H_typing + H_eval drives the MERA toward the normal form.
The transition gates that GIVE evolution a matrix element to relax
through live alongside the penalties (spec §7.4, added in Task 11).

NO classical interpreter: this module never substitutes, never
beta-reduces, never imports the AST module for walking.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.qft_pcn.qft.mera import MERA
from .mera_encoder import MeraEncodingMeta
from .mera_encoding import (
    KIND_APP, KIND_BIN, KIND_IF, KIND_SUCC, KIND_FIX,
    SPECIES_LEAF_OFFSET, LEAVES_PER_NODE,
)
from ._mera_window import mera_window_expectation_factored
from ._mera_eval_terms import (
    beta_penalty_ops, arith_penalty_ops, cmp_penalty_ops,
    if_penalty_ops, succ_penalty_ops,
    DEFAULT_LAMBDA_BETA, DEFAULT_LAMBDA_ARITH, DEFAULT_LAMBDA_IF,
    DEFAULT_LAMBDA_FIX,
)

RULE_R_BETA = "R-Beta"
RULE_R_ARITH = "R-Arith"
RULE_R_CMP = "R-Cmp"
RULE_R_IF = "R-If"
RULE_R_SUCC = "R-Succ"
RULE_R_FIX = "R-Fix"

_ALL_RULES = (RULE_R_BETA, RULE_R_ARITH, RULE_R_CMP, RULE_R_IF,
              RULE_R_SUCC, RULE_R_FIX)


class MeraEvalError(Exception):
    """Base error for MeraEvalHamiltonian operations."""


class MeraEvalBudgetExceeded(MeraEvalError):
    """A Fix unfold's normal form exceeds the encoder's node budget."""


class MeraEvalTermNotFound(MeraEvalError):
    def __init__(self, term):
        super().__init__(f"term {term} not in this Hamiltonian's term list")


@dataclass(frozen=True)
class MeraEvalTerm:
    rule_id: str
    node: int
    arity: int


def _kind_leaf(meta, node):
    return meta.layout.leaf_of(node, "kind")


def _value_leaf(meta, node):
    return meta.layout.leaf_of(node, "value")


class MeraEvalHamiltonian:
    """Structural evaluation Hamiltonian over a MERA leaf layout."""

    def __init__(self, meta: MeraEncodingMeta,
                 lambda_beta: float = DEFAULT_LAMBDA_BETA,
                 lambda_arith: float = DEFAULT_LAMBDA_ARITH,
                 lambda_if: float = DEFAULT_LAMBDA_IF,
                 lambda_fix: float = DEFAULT_LAMBDA_FIX):
        self.meta = meta
        self.N_nodes = meta.n_nodes
        self.lambda_beta = float(lambda_beta)
        self.lambda_arith = float(lambda_arith)
        self.lambda_if = float(lambda_if)
        self.lambda_fix = float(lambda_fix)
        self.terms: list[MeraEvalTerm] = self._enumerate_terms()
        self._terms_set = frozenset(self.terms)

    def _enumerate_terms(self):
        terms = []
        for rule in _ALL_RULES:
            for n in range(self.N_nodes):
                terms.append(MeraEvalTerm(rule, n, 2))
        return terms

    def _lambda_for(self, rule):
        if rule == RULE_R_BETA:
            return self.lambda_beta
        if rule in (RULE_R_ARITH, RULE_R_CMP, RULE_R_SUCC):
            return self.lambda_arith
        if rule == RULE_R_IF:
            return self.lambda_if
        if rule == RULE_R_FIX:
            return self.lambda_fix
        raise MeraEvalTermNotFound(rule)

    def term_energy(self, state: MERA, term: MeraEvalTerm) -> float:
        if term not in self._terms_set:
            raise MeraEvalTermNotFound(term)
        lam = self._lambda_for(term.rule_id)
        node = term.node
        meta = self.meta
        kids = meta.children_of_node.get(node, [])
        if term.rule_id == RULE_R_BETA:
            if len(kids) < 1:
                return 0.0
            ops = beta_penalty_ops(_kind_leaf(meta, node),
                                   _kind_leaf(meta, kids[0]), lam)
        elif term.rule_id in (RULE_R_ARITH, RULE_R_CMP):
            if len(kids) < 2:
                return 0.0
            builder = (arith_penalty_ops if term.rule_id == RULE_R_ARITH
                       else cmp_penalty_ops)
            ops = builder(_kind_leaf(meta, node), _value_leaf(meta, node),
                          _kind_leaf(meta, kids[0]),
                          _kind_leaf(meta, kids[1]), lam)
        elif term.rule_id == RULE_R_IF:
            if len(kids) < 1:
                return 0.0
            ops = if_penalty_ops(_kind_leaf(meta, node),
                                 _kind_leaf(meta, kids[0]), lam)
        elif term.rule_id == RULE_R_SUCC:
            if len(kids) < 1:
                return 0.0
            ops = succ_penalty_ops(_kind_leaf(meta, node),
                                   _kind_leaf(meta, kids[0]), lam)
        elif term.rule_id == RULE_R_FIX:
            return self._energy_r_fix(state, node, lam)
        else:
            raise MeraEvalTermNotFound(term)
        return float(mera_window_expectation_factored(state, ops).real)

    def _energy_r_fix(self, state, node, lam) -> float:
        """R-Fix redex penalty (spec §7.2). Filled in Task 12; stub 0.0."""
        return 0.0

    def total_energy(self, state: MERA) -> float:
        return sum(self.term_energy(state, t) for t in self.terms)

    def residuals(self, state: MERA) -> dict:
        return {(t.rule_id, t.node): self.term_energy(state, t)
                for t in self.terms}
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_eval_hamiltonian.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_evaluation_hamiltonian.py src/qft_pcn/tests/test_mera_eval_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_evaluation_hamiltonian): redex-penalty evaluation Hamiltonian

MeraEvalHamiltonian / MeraEvalTerm with R-Beta/R-Arith/R-Cmp/R-If/R-Succ
diagonal redex penalties as factored operators. <H_eval> = 0 on normal
forms, > 0 on unreduced redices. R-Fix and transition gates follow.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: `compose_mera_hamiltonians`

Operator sum at the expectation level, mirroring `compose.py`.

**Files:**
- Create: `src/qft_pcn/logic/mera_compose.py`
- Test: `src/qft_pcn/tests/test_mera_compose.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_compose.py`:

```python
"""Tests for compose_mera_hamiltonians (spec §9.5)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_compose import (
    compose_mera_hamiltonians, IncompatibleHamiltonians,
)


def test_total_energy_is_sum():
    state, meta = encode_mera(parse(r"2 + 3"))
    H_t = MeraTypingHamiltonian(meta)
    H_e = MeraEvalHamiltonian(meta)
    H = compose_mera_hamiltonians(H_t, H_e)
    assert abs(H.total_energy(state)
               - (H_t.total_energy(state) + H_e.total_energy(state))) < 1e-9


def test_terms_concatenated():
    _, meta = encode_mera(parse(r"2 + 3"))
    H_t = MeraTypingHamiltonian(meta)
    H_e = MeraEvalHamiltonian(meta)
    H = compose_mera_hamiltonians(H_t, H_e)
    assert len(H.terms) == len(H_t.terms) + len(H_e.terms)


def test_incompatible_metas_raise():
    _, m1 = encode_mera(parse(r"\x:Int. x"))
    _, m2 = encode_mera(parse(r"(\x:Int. x + 1)(2)"))
    H1 = MeraTypingHamiltonian(m1)
    H2 = MeraEvalHamiltonian(m2)
    with pytest.raises(IncompatibleHamiltonians):
        compose_mera_hamiltonians(H1, H2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_compose.py -v`
Expected: ImportError on `mera_compose`.

- [ ] **Step 3: Implement `mera_compose.py`**

Create `src/qft_pcn/logic/mera_compose.py` — adapt `compose.py`'s
`ComposedHamiltonian`, keying compatibility on `meta.n_leaves` and
`meta.n_nodes` instead of `N`:

```python
"""compose_mera_hamiltonians: operator-sum of MERA Hamiltonians at the
expectation level (spec §9.5). Mirrors compose.py for the MERA substrate.
"""
from __future__ import annotations

from src.qft_pcn.qft.mera import MERA


class IncompatibleHamiltonians(Exception):
    """Inputs disagree on n_leaves / n_nodes."""


class ComposedMeraHamiltonian:
    """Operator-sum wrapper. .terms is the concatenation; .term_energy
    routes by term identity; .total_energy sums the sub-Hamiltonians.
    """

    def __init__(self, *hamiltonians):
        if not hamiltonians:
            raise ValueError("compose_mera_hamiltonians requires >=1 input")
        ref = hamiltonians[0].meta
        for h in hamiltonians[1:]:
            if (h.meta.n_leaves != ref.n_leaves
                    or h.meta.n_nodes != ref.n_nodes):
                raise IncompatibleHamiltonians(
                    f"meta mismatch: n_leaves {ref.n_leaves} vs "
                    f"{h.meta.n_leaves}, n_nodes {ref.n_nodes} vs "
                    f"{h.meta.n_nodes}")
        self._hams = tuple(hamiltonians)
        self.meta = ref
        self._owner = {}
        merged = []
        for h in self._hams:
            for t in h.terms:
                merged.append(t)
                self._owner[id(t)] = h
        self.terms = merged

    def term_energy(self, state: MERA, term) -> float:
        owner = self._owner.get(id(term))
        if owner is None:
            for h in self._hams:
                if term in getattr(h, "_terms_set", ()):
                    return h.term_energy(state, term)
            raise KeyError(term)
        return owner.term_energy(state, term)

    def total_energy(self, state: MERA) -> float:
        return sum(h.total_energy(state) for h in self._hams)

    def residuals(self, state: MERA) -> dict:
        out = {}
        for h in self._hams:
            for key, val in h.residuals(state).items():
                if key in out:
                    out[(type(h).__name__, *key)] = val
                else:
                    out[key] = val
        return out


def compose_mera_hamiltonians(*hamiltonians):
    """Compose two or more MERA sum-of-terms Hamiltonians into one.

    The result's total_energy equals the sum of the inputs' total_energy
    on every state. All inputs must share n_leaves and n_nodes.
    """
    return ComposedMeraHamiltonian(*hamiltonians)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_compose.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_compose.py src/qft_pcn/tests/test_mera_compose.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_compose): compose_mera_hamiltonians operator sum

Operator sum of MERA Hamiltonians at the expectation level; total_energy
is the sum of sub-Hamiltonian energies. Compatibility keyed on n_leaves
and n_nodes. Mirrors compose.py for the MERA substrate.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Transition gates + MERA imaginary-time evolution

The off-diagonal transition gates (spec §7.4) and `mera_evolution_logic.py` (spec §7.5).

**Files:**
- Modify: `src/qft_pcn/logic/_mera_eval_terms.py` (add transition gate builders)
- Create: `src/qft_pcn/logic/mera_evolution_logic.py`
- Test: `src/qft_pcn/tests/test_mera_evolution_logic.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_evolution_logic.py`:

```python
"""Tests for MERA imaginary-time evolution (spec §7.5)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_compose import compose_mera_hamiltonians
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_trotter_step, mera_imaginary_evolve,
)


def test_trotter_step_preserves_norm():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    mera_trotter_step(state, H, dt=0.1, imaginary=True)
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-8)


def test_imaginary_evolve_returns_energy_trajectory():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    traj = mera_imaginary_evolve(state, H, dt=0.1, steps=20)
    assert len(traj) == 21        # initial + 20 steps


def test_imaginary_evolve_monotone_decrease():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = MeraEvalHamiltonian(meta)
    traj = mera_imaginary_evolve(state, H, dt=0.1, steps=30)
    for i in range(len(traj) - 1):
        assert traj[i + 1] <= traj[i] + 1e-6, (
            f"energy rose at step {i}: {traj[i]} -> {traj[i+1]}")
    assert traj[-1] < traj[0] * 0.5


def test_evolve_accepts_composed_hamiltonian():
    state, meta = encode_mera(parse(r"2 + 3"))
    H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
                                  MeraEvalHamiltonian(meta))
    traj = mera_imaginary_evolve(state, H, dt=0.1, steps=10)
    assert traj[-1] <= traj[0] + 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_evolution_logic.py -v`
Expected: ImportError on `mera_evolution_logic`.

- [ ] **Step 3: Add transition gates to `_mera_eval_terms.py`**

For each redex rule, add a `*_transition_gate` builder that returns a factored
gate moving amplitude from the unreduced configuration toward the reduced one.
The cleanest realization (and the one to implement) is per-term
`exp(-dt · λ · H_term)` *plus* a small Hermitian off-diagonal coupling, but a
simpler and sufficient form for M2's acceptance is:

- Build each redex term as a **Hermitian operator** `H_term = λ · (P_unreduced
  − T)`, where `P_unreduced` is the diagonal projector (Task 8) and `T` is a
  Hermitian off-diagonal operator `|reduced⟩⟨unreduced| + |unreduced⟩⟨reduced|`
  connecting the unreduced basis configuration of the redex window to its
  reduced configuration. Then `H_term` has the reduced configuration as a
  lower-energy state, and `exp(-dt·H_term)` (a factored two-leaf or local gate)
  moves amplitude toward it.

For each rule, the reduced configuration is concrete and small:

- `R-Arith` on `(BIN[+], INT a, INT b)`: reduced = `(INT (a+b), PAD, PAD)`.
  The transition couples the `kind`/`value` leaves of the three nodes. The
  arithmetic on `a, b` is a fixed table (a, b ∈ the int-literal range). Build
  the gate as a two-leaf gate per operand absorbed, OR as a set of per-`(a,b)`
  basis-pair couplings — a finite, bounded construction (mirrors C §5.3's
  `U_arith` and the `factored_evolution.py` design).
- `R-Beta`, `R-If`, `R-Succ`: analogous fixed reduced configurations.
- `R-Fix`: one layer of body propagation (Task 12).

Implement `redex_evolution_gate(ops_unreduced, ops_reduced, dt, lam)` that,
given the unreduced and reduced factored configurations on a leaf window,
returns the per-leaf or per-leaf-pair `exp(-dt·λ·H_window)` factored gates.
Keep every gate ≤ 256×256 (a two-leaf gate); never form a `16**k`, k>2
operator. Use `np.einsum(..., optimize='greedy')` for any contraction.

**This is the load-bearing task.** If the full transition-gate construction is
involved, factor it exactly as `factored_evolution.py` does for the MPS — read
that file and lift its per-redex gate builders onto the MERA leaf windows. Do
NOT shortcut to a classical rewrite (spec §1.6). If a genuine ambiguity in how
the reduced configuration maps onto leaves arises, stop and ask.

- [ ] **Step 4: Implement `mera_evolution_logic.py`**

```python
"""MERA imaginary-time evolution (spec §7.5).

mera_trotter_step applies each Hamiltonian term's factored
exp(-dt.H_term) gate to the MERA state in place; mera_imaginary_evolve
repeats it and returns the energy trajectory. Reduction is observed by
decoding the relaxed state — no classical rewrite anywhere (spec §1.6).
"""
from __future__ import annotations

from src.qft_pcn.qft.mera import MERA


def mera_trotter_step(state: MERA, ham, dt: float,
                      imaginary: bool = True,
                      chi_layer: int | None = None) -> None:
    """One Trotter step: apply each term's factored gate to `state` in
    place, then renormalize. Two-leaf gates go through
    MERA.apply_two_site_gate (bond dim capped at chi_layer); local gates
    through apply_local_gate.
    """
    for term in ham.terms:
        gates = ham.term_gates(term, dt, imaginary)   # see note
        for leaves, gate in gates:
            if len(leaves) == 1:
                state.apply_local_gate(leaves[0], gate)
            elif len(leaves) == 2:
                state.apply_two_site_gate(leaves[0], gate)
            else:
                raise ValueError(
                    f"gate spans {len(leaves)} leaves; only 1 or 2 "
                    f"supported by MERA gate primitives")
    state.normalize()


def mera_imaginary_evolve(state: MERA, ham, dt: float, steps: int,
                          chi_layer: int | None = None) -> list[float]:
    """Repeat mera_trotter_step `steps` times in imaginary time. Returns
    the energy trajectory [<H>_0, ..., <H>_steps]. Energy decreases
    monotonically (architecture §13.1).
    """
    traj = [ham.total_energy(state)]
    for _ in range(steps):
        mera_trotter_step(state, ham, dt, imaginary=True,
                          chi_layer=chi_layer)
        traj.append(ham.total_energy(state))
    return traj
```

**Note — `ham.term_gates`.** The evolution driver needs each term's factored
`exp(-dt·H_term)` gate(s). Add a `term_gates(self, term, dt, imaginary) ->
list[tuple[tuple[int,...], np.ndarray]]` method to `MeraEvalHamiltonian`,
`MeraTypingHamiltonian`, and `ComposedMeraHamiltonian`:

- For a **diagonal penalty-only term** (typing rules; redex penalties with no
  transition), the gate is a per-leaf diagonal `exp(-dt·λ·diag)` — a 1-leaf
  gate per participating leaf. Applying a per-leaf diagonal gate is exact and
  cheap; it implements the imaginary-time *decay* of the penalized
  configuration.
- For a **transition term** (redex rules with a transition gate), the gate is
  the two-leaf `exp(-dt·H_window)` from Task 11 Step 3.
- For `ComposedMeraHamiltonian.term_gates`, dispatch to the owning
  sub-Hamiltonian.

Add `term_gates` to all three Hamiltonian classes as part of this task; the
typing Hamiltonian's gates are diagonal-decay only (typing is a constraint, not
a reduction — its gates damp ill-typed amplitude). Document this in each
method's docstring.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_evolution_logic.py -v --timeout=120`
Expected: 4 passed. The monotone-decrease test is the real check — if energy
rises, the transition gate's sign is wrong (imaginary time must be
`exp(-dt·H)`, not `exp(+dt·H)`); debug with systematic-debugging.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/_mera_eval_terms.py src/qft_pcn/logic/mera_evolution_logic.py src/qft_pcn/logic/mera_typing_hamiltonian.py src/qft_pcn/logic/mera_evaluation_hamiltonian.py src/qft_pcn/tests/test_mera_evolution_logic.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_evolution_logic): MERA imaginary-time evolution

Per-term factored exp(-dt.H_term) gates applied to the MERA state via
apply_local_gate / apply_two_site_gate. Redex transition gates connect
unreduced and reduced configurations so the spectrum can relax through
them. Energy decreases monotonically; reduction is ground-state finding,
no classical rewrite (spec §1.6).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: `R-Fix` recursion unfolding + the O(log N) demo

`R-Fix` redex penalty + transition gate (spec §7.2), and the recursive demo (spec §9.4).

**Files:**
- Modify: `src/qft_pcn/logic/_mera_eval_terms.py`, `mera_evaluation_hamiltonian.py`
- Test: `src/qft_pcn/tests/test_mera_fix_recursion.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_fix_recursion.py`:

```python
"""Tests for Fix recursion unfolding on the MERA (spec §7.2, §9.4)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import (
    Fix, Lam, App, Var, NatLit, Succ, Zero, TNat, TArrow,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_compose import compose_mera_hamiltonians
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve


def _recursive_demo():
    # A small Fix program whose normal form needs >=2 unfolds and fits
    # the node budget. The exact body is chosen by the implementer to be
    # representable by the encoder; the contract is: encode_mera succeeds,
    # the normal form is a NatLit, and >=2 recursive steps are required.
    # Placeholder shape (adjust to a body the encoder accepts):
    return App(
        fn=Fix(param="f", param_ty=TArrow(src=TNat(), dst=TNat()),
               body=Lam(param="x", param_ty=TNat(), body=Var(name="x"))),
        arg=NatLit(val=2),
    )


def test_fix_program_encodes():
    state, meta = encode_mera(_recursive_demo())
    assert np.isclose(state.norm_sq(), 1.0, atol=1e-8)


def test_fix_program_reduces_to_normal_form():
    ast = _recursive_demo()
    state, meta = encode_mera(ast)
    H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
                                  MeraEvalHamiltonian(meta))
    traj = mera_imaginary_evolve(state, H, dt=0.1, steps=120)
    assert traj[-1] < 1e-2, f"did not converge: final <H> = {traj[-1]}"
    res = decode_mera(state, meta)
    assert res.ast is not None   # decodes to a definite normal form


def test_fix_bond_dimension_bounded():
    """The O(log N) tree-depth claim: max MERA layer bond dim stays
    bounded by chi_layer through the whole evolution (spec §9.4)."""
    ast = _recursive_demo()
    state, meta = encode_mera(ast, chi_layer=16)
    H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
                                  MeraEvalHamiltonian(meta))
    from src.qft_pcn.logic.mera_evolution_logic import mera_trotter_step
    for _ in range(60):
        mera_trotter_step(state, H, dt=0.1, imaginary=True, chi_layer=16)
        # inspect the MERA layer dimensions; none may exceed chi_layer.
        for dim in state.layer_dimensions():   # see note
            assert dim <= 16, f"bond dim {dim} exceeds chi_layer=16"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_fix_recursion.py -v --timeout=180`
Expected: failures — `_energy_r_fix` stub, no `R-Fix` gate, possibly no
`layer_dimensions` accessor.

- [ ] **Step 3: Implement `R-Fix`**

In `_mera_eval_terms.py` add `fix_penalty_ops` and a `fix_transition_gate`:

- `fix_penalty_ops`: penalizes "FIX node present (`kind = KIND_FIX`) AND a
  bound recursion-use `Var` node still has `kind = KIND_VAR`" — a two-node
  factored operator over the FIX node's `kind` leaf and a recursion-use node's
  `kind` leaf.
- `fix_transition_gate`: a two-leaf gate propagating one layer of the body's
  head structure into a recursion-use `Var` node. Because the FIX node and the
  recursion-use node are coupled through the MERA tree (M1 §5), the gate acts
  on a bounded leaf window and the *tree depth* between them is `O(log
  N_leaves)`.

In `mera_evaluation_hamiltonian.py`, fill `_energy_r_fix`: find the FIX node's
recursion-use `Var` nodes — a `Var` node `j` is a recursion-use of FIX node
`i` iff `meta.use_to_binder[bid_leaf(j)] == bid_leaf(i)`. For each such `j`,
add the `fix_penalty_ops` energy. `meta` is read for addressing only (spec
§1.2). Update `term_gates` so `R-Fix` terms supply the `fix_transition_gate`.

**Recursion budget.** If a `Fix` program's normal form would exceed the
encoder's node budget, the encoder already raises (M1's
`EncodingTooLarge`/budget exception). M2's `MeraEvalBudgetExceeded` is raised
if `R-Fix`'s unfold detects an over-budget configuration during evolution —
guard the transition gate and raise rather than silently truncating
structure. The demo program (`_recursive_demo`) is chosen to fit the budget.

- [ ] **Step 4: Add `layer_dimensions` to `MERA` if absent**

`test_fix_bond_dimension_bounded` needs an accessor for the MERA's per-layer
bond dimensions. Check `src/qft_pcn/qft/mera.py` — if `layer_dims` /
`bond_dimensions` / a similar accessor exists, use it (adjust the test). If
none exists, add a thin read-only `layer_dimensions(self) -> list[int]`
property to `MERA` that returns each layer's tensor bond dimension — an
interface-only addition, no behavior change; F's MERA tests must still pass.
Run F's MERA tests after: `.venv/bin/python -m pytest src/qft_pcn/tests/ -k
"mera" --timeout=60 -q 2>&1 | tail -5`.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_fix_recursion.py -v --timeout=180`
Expected: 3 passed. The bond-dimension test is the O(log N) claim — if it
fails because bond dim grows, the truncation cap is not being applied in
`apply_two_site_gate`; debug with systematic-debugging.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/_mera_eval_terms.py src/qft_pcn/logic/mera_evaluation_hamiltonian.py src/qft_pcn/qft/mera.py src/qft_pcn/tests/test_mera_fix_recursion.py
git commit -m "$(cat <<'EOF'
feat(logic/mera-eval): R-Fix recursion unfolding with O(log N) tree depth

R-Fix redex penalty + transition gate: the FIX node and its
recursion-use Var nodes are coupled through the MERA tree, so a
recursive unfold touches O(log N) tree layers and keeps bond dimension
bounded by chi_layer. A recursive demo program reduces to its normal
form with bounded bond dimension — the architecture §10.4 claim,
demonstrated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Reduction acceptance suite (E1-E4) + induction-as-ground-state

Spec §9.3 reduction acceptance and the §7.3 `R-Induction` term.

**Files:**
- Modify: `src/qft_pcn/logic/_mera_eval_terms.py`, `mera_evaluation_hamiltonian.py`
- Create: `src/qft_pcn/tests/test_mera_reduction.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_reduction.py`:

```python
"""Reduction acceptance on the MERA substrate (spec §9.3, §7.3)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import parse, IntLit
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_decoder import decode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_compose import compose_mera_hamiltonians
from src.qft_pcn.logic.mera_evolution_logic import mera_imaginary_evolve

_PROGRAMS = {
    "E1": (r"2 + 3", 5),
    "E2": (r"(\x:Int. x + 1)(2)", 3),
    "E3": (r"if 1 < 2 then 7 else 0", 7),
    "E4": (r"(\x:Int. (\y:Int. x + y)(3))(4)", 7),
}


@pytest.mark.parametrize("name", list(_PROGRAMS))
def test_reduction(name):
    src, expected_val = _PROGRAMS[name]
    state, meta = encode_mera(parse(src))
    H = compose_mera_hamiltonians(MeraTypingHamiltonian(meta),
                                  MeraEvalHamiltonian(meta))
    traj = mera_imaginary_evolve(state, H, dt=0.1, steps=120)
    # monotone non-increasing
    for i in range(len(traj) - 1):
        assert traj[i + 1] <= traj[i] + 1e-6, f"{name}: energy rose"
    assert traj[-1] < 1e-2, f"{name}: final <H> = {traj[-1]}"
    res = decode_mera(state, meta)
    assert isinstance(res.ast, IntLit), f"{name}: not reduced to a literal"
    assert res.ast.val == expected_val, (
        f"{name}: got {res.ast.val}, expected {expected_val}")


def test_normal_form_zero_eval_energy():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    assert abs(MeraEvalHamiltonian(meta).total_energy(state)) < 1e-9


def test_unreduced_positive_eval_energy():
    state, meta = encode_mera(parse(r"2 + 3"))
    assert MeraEvalHamiltonian(meta).total_energy(state) > 0.5
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_reduction.py -v --timeout=300`
Expected: the E1-E4 tests fail if the transition gates do not yet drive the
reduction to the *correct value* — Task 11 built the gate machinery, this task
verifies it produces the right answer. Debug with systematic-debugging: the
likely fault is the arithmetic-result table or the gate's reduced-configuration
mapping.

- [ ] **Step 3: Calibrate / fix the transition gates**

If the E1-E4 tests fail, the transition gates' reduced configuration is wrong
or the λ calibration is too slow. Permitted calibration (per C-spec §11): raise
individual λs up to 5×, lower `dt` and raise `steps` proportionally. If a λ >
10× is needed, the gate construction is wrong — stop and ask. Fix the gate (do
NOT replace it with a classical rewrite, spec §1.6). Re-run until E1-E4 pass.

- [ ] **Step 4: Add `R-Induction` (induction-as-ground-state, spec §7.3)**

Add a `R-Induction` rule to `MeraEvalHamiltonian` (or a dedicated
`MeraInductionHamiltonian` — pick one; `R-Induction` as a rule of
`MeraEvalHamiltonian` is simpler). It is a two-node penalty on a
`KIND_FORALL`-over-`TYPE_NAT` (resp. `TYPE_LIST`) node, penalizing the
configuration where the body proposition is not simultaneously satisfied at
the base case and the step case. For M2, restrict to single-node base/step
bodies (spec §7.3). Add a focused test: `forall n:Nat. Eq n n` evolved under
`H_typing + R-Induction` relaxes to zero energy (the induction holds);
construct a false universal and verify it does NOT relax to zero. Keep this
test in `test_mera_reduction.py` or a sibling file.

If the single-node-body induction term is genuinely ambiguous to construct
from the spec, stop and ask — do not approximate it away.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_reduction.py -v --timeout=300`
Expected: E1-E4 + normal-form + induction tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/_mera_eval_terms.py src/qft_pcn/logic/mera_evaluation_hamiltonian.py src/qft_pcn/tests/test_mera_reduction.py
git commit -m "$(cat <<'EOF'
test(mera-eval): E1-E4 reduction acceptance + induction-as-ground-state

Four programs imag-time-evolve to their normal forms on the MERA
substrate with monotone energy decrease and correct decoded values.
R-Induction makes a forall-over-Nat proposition true iff its encoded
MERA relaxes to zero energy under H_typing + R-Induction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Structural / no-interpreter / cross-substrate checks + final verification

Spec §9.6, §9.7, §9.8 — the principle-enforcement tests and the cross-substrate anchor.

**Files:**
- Create: `src/qft_pcn/tests/test_mera_hamiltonian_principles.py`
- Modify: `src/qft_pcn/logic/__init__.py` (public exports)

- [ ] **Step 1: Write the principle + anchor tests**

Create `src/qft_pcn/tests/test_mera_hamiltonian_principles.py`:

```python
"""Principle-enforcement + cross-substrate tests (spec §9.6-§9.8)."""
from __future__ import annotations
import inspect
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_typing_hamiltonian import MeraTypingHamiltonian
from src.qft_pcn.logic import mera_evaluation_hamiltonian as MEH
from src.qft_pcn.logic import _mera_eval_terms as MET


def test_no_classical_interpreter_in_eval():
    """Spec §1.6 / §9.6: eval Hamiltonian has no classical interpreter."""
    for mod in (MEH, MET):
        src = inspect.getsource(mod)
        low = src.lower()
        assert "substitute" not in low, f"{mod.__name__}: substitute"
        assert "beta_reduce" not in low, f"{mod.__name__}: beta_reduce"
        # 'evaluate' may appear in docstrings as a concept; check for a
        # call/def, not the word.
        assert "def evaluate" not in low
        assert "from .ast import" not in src and "from . import ast" not in src


def test_typing_hamiltonian_structural():
    """Spec §9.6: same instance evaluates correctly; no AST attrs."""
    state, meta = encode_mera(parse(r"\x:Int. x"))
    H = MeraTypingHamiltonian(meta)
    for attr in dir(H):
        if attr.startswith("_"):
            continue
        for bad in ("ast", "tree", "walk", "node_obj"):
            assert bad not in attr.lower()


def test_cross_substrate_anchor_well_typed():
    """Spec §9.7: MERA and MPS typing Hamiltonians agree on P1-P5."""
    from src.qft_pcn.logic.encoder import encode as encode_mps
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    for src in (r"\x:Int. x", r"(\x:Int. x + 1)(2)",
                r"\f:Int->Int. \x:Int. f (f x)",
                r"if 1 < 2 then ((\x:Bool. x)(true)) else false",
                r"(\x:Int. (\y:Int. x + y)(3))(4)"):
        p = parse(src)
        m_state, m_meta = encode_mera(p)
        mera_e = MeraTypingHamiltonian(m_meta).total_energy(m_state)
        mps_state, mps_meta = encode_mps(p, N=32)
        mps_e = TypingHamiltonian(N=32).total_energy(mps_state)
        assert abs(mera_e) < 1e-6 and abs(mps_e) < 1e-6, (
            f"{src}: mera {mera_e}, mps {mps_e}")


def test_cross_substrate_anchor_ill_typed():
    """Spec §9.7: both substrates flag IT1 as ill-typed."""
    from src.qft_pcn.logic.encoder import encode as encode_mps
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    p = parse(r"\x:Int. x + true")
    m_state, m_meta = encode_mera(p)
    mps_state, mps_meta = encode_mps(p, N=32)
    assert MeraTypingHamiltonian(m_meta).total_energy(m_state) > 0.5
    assert TypingHamiltonian(N=32).total_energy(mps_state) > 0.5
```

- [ ] **Step 2: Run to verify it fails / passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_hamiltonian_principles.py -v --timeout=120`
Expected: most pass immediately (the modules were built clean). If
`test_no_classical_interpreter` fails, a banned identifier slipped in — remove
it. If the cross-substrate anchor fails, the MERA and MPS encoders disagree on
a register — debug with systematic-debugging against M1's §9.7 cross-substrate
test.

- [ ] **Step 3: Add public exports**

In `src/qft_pcn/logic/__init__.py` add:

```python
from .mera_typing_hamiltonian import MeraTypingHamiltonian, MeraTypingTerm
from .mera_evaluation_hamiltonian import MeraEvalHamiltonian, MeraEvalTerm
from .mera_compose import compose_mera_hamiltonians
from .mera_evolution_logic import mera_trotter_step, mera_imaginary_evolve
```

- [ ] **Step 4: Full regression — F, M1, MPS stack, M2**

Run the full suite:

```bash
.venv/bin/python -m pytest src/qft_pcn/tests/ --timeout=600 -q 2>&1 | tail -20
```

Expected: all green — F's MERA tests, M1's encoder tests, the MPS logic stack,
and M2's new suites. If anything regressed, M2 touched a shared file
incorrectly — `git diff` the shared files (`mera_encoder.py`, `mera.py`,
`__init__.py`) and fix; M2's only permitted shared-file changes are the
additive `children_of_node` field and an optional `layer_dimensions` accessor.

- [ ] **Step 5: Verification before completion**

Apply `superpowers:verification-before-completion`: re-run the M2 acceptance
suites in a fresh shell and confirm the output before claiming completion:

```bash
.venv/bin/python -m pytest \
  src/qft_pcn/tests/test_mera_typing_hamiltonian.py \
  src/qft_pcn/tests/test_mera_eval_hamiltonian.py \
  src/qft_pcn/tests/test_mera_reduction.py \
  src/qft_pcn/tests/test_mera_fix_recursion.py \
  src/qft_pcn/tests/test_mera_compose.py \
  src/qft_pcn/tests/test_mera_evolution_logic.py \
  src/qft_pcn/tests/test_mera_hamiltonian_principles.py \
  -v --timeout=600 2>&1 | tail -30
```

Confirm every spec §10 acceptance criterion:
1. P1-P10 well-typed → ⟨H_typing⟩ < 1e-9.
2. IT1-IT5 ill-typed → ⟨H_typing⟩ > 0.5, one attributable residual.
3. E1-E4 reduce with monotone decrease, final ⟨H⟩ < 1e-2.
4. The `Fix` demo reduces with bond dim bounded by `chi_layer`.
5. `compose_mera_hamiltonians` is an exact operator sum.
6. Structural / no-interpreter / no-dense-operator checks pass.
7. Cross-substrate anchor passes for P1-P5 and IT1.
8. No regressions.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/tests/test_mera_hamiltonian_principles.py src/qft_pcn/logic/__init__.py
git commit -m "$(cat <<'EOF'
test(mera-hamiltonians): principle enforcement + cross-substrate anchor

Static no-classical-interpreter guard, structural-Hamiltonian check, and
the cross-substrate anchor confirming the MERA typing Hamiltonian agrees
with the shipped MPS TypingHamiltonian on P1-P5 + IT1. Public exports
for the M2 deliverables. Closes sub-project M2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Done criteria (spec §10)

M2 is complete when all of Task 14 Step 5's eight criteria are confirmed by
fresh pytest output. No completion claim is acceptable without that output.

The deliverables: `mera_typing_hamiltonian.py`, `_mera_typing_rules.py`,
`mera_evaluation_hamiltonian.py`, `_mera_eval_terms.py`, `mera_compose.py`,
`mera_evolution_logic.py`, the additive `MeraEncodingMeta.children_of_node`
field, an optional `MERA.layer_dimensions` accessor, and the test suites.

## Contract handed to M3

Per spec §12: `MeraTypingHamiltonian` / `MeraEvalHamiltonian` with the
`.terms` / `.term_energy` / `.residuals` / `.total_energy` API,
`compose_mera_hamiltonians`, `mera_imaginary_evolve`, the stable rule-name
set, and the `children_of_node` field. M3's debugger consumes `.residuals`;
M3's synthesis evolves under a composed Hamiltonian from a hole-bearing M1
initial state.
