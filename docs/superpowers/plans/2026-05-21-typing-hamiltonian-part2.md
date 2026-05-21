# Typing Hamiltonian Implementation Plan — Part 2 of 3 (TypingHamiltonian)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Continues from Part 1.** Assumes Tasks 1–9 are complete: encoder produces 5-species MPS with tobl, bid bond carries per-channel param_ty as a real DOF, EncodingMeta exposes `tobl_per_site` and `channel_param_ty_per_bond`.

**Driving principle reminder (spec §1):**
- One Hamiltonian per N-site lattice. NO AST data in the H's construction.
- Each typing rule = a real Hermitian operator. Per-term residuals expose (rule_id, site).
- Operators are stored in **factored form** (per-species small matrices) and evaluated via a custom factored-expectation path — the dense 65536×65536 operator is never materialized.

This part covers Tasks 10–18: factored-expectation primitives, per-rule term builders, the `TypingHamiltonian` class, `term_energy`, `total_energy`, `residuals`.

---

## Task 10: Factored-expectation primitives (`_factored_expectation.py`)

**Files:**
- Create: `src/qft_pcn/logic/_factored_expectation.py`.
- Create: `src/qft_pcn/tests/test_logic_factored_expectation.py`.

The factored-expectation helper computes `⟨ψ | O | ψ⟩` for an operator `O` represented as a tensor product `O_kind ⊗ O_type ⊗ O_bid ⊗ O_value ⊗ O_tobl` of per-species small matrices (each species' operator is `(cutoff × cutoff)`). It does this by reshaping the MPS site tensor into per-species axes and contracting per-species, never materializing the dense `65536 × 65536` operator.

- [ ] **Step 1: Write the failing tests**

Create `src/qft_pcn/tests/test_logic_factored_expectation.py`:

```python
"""Tests for logic/_factored_expectation.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.qft_pcn.logic._factored_expectation import (
    factored_local_expectation, factored_two_site_expectation,
    embed_factored_to_dense,
)
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.encoding import (
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    KIND_LAM, KIND_VAR, KIND_PAD, TYPE_INT, TYPE_ARR_II, BID_0, BID_NONE,
)


def _ident(d: int) -> np.ndarray:
    return np.eye(d, dtype=complex)


def _project_basis(d: int, idx: int) -> np.ndarray:
    """|idx><idx| in a d-dim Hilbert space."""
    p = np.zeros((d, d), dtype=complex)
    p[idx, idx] = 1.0
    return p


def test_factored_local_expectation_identity_is_one():
    """For a normalized MPS, <I> = 1 on every site."""
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    op_factors = {
        "kind":  _ident(KIND_CUTOFF),
        "type":  _ident(TYPE_CUTOFF),
        "bid":   _ident(BID_CUTOFF),
        "value": _ident(VALUE_CUTOFF),
        "tobl":  _ident(TOBL_CUTOFF),
    }
    for k in range(8):
        e = factored_local_expectation(state, k, op_factors)
        assert abs(e - 1.0) < 1e-10, f"site {k}: <I> = {e}, expected 1"


def test_factored_local_expectation_kind_projector_at_lam():
    """<P_kind=LAM>(0) = 1 for an encoded \\x:Int. x (site 0 IS LAM)."""
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    op_factors = {
        "kind": _project_basis(KIND_CUTOFF, KIND_LAM),
        # Others default to identity.
    }
    e = factored_local_expectation(state, 0, op_factors)
    assert abs(e - 1.0) < 1e-10


def test_factored_local_expectation_kind_projector_at_var():
    """<P_kind=LAM>(1) = 0 for an encoded \\x:Int. x (site 1 is VAR)."""
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    op_factors = {"kind": _project_basis(KIND_CUTOFF, KIND_LAM)}
    e = factored_local_expectation(state, 1, op_factors)
    assert abs(e) < 1e-10


def test_factored_local_expectation_at_pad_kind_pad():
    """<P_kind=PAD>(k) = 1 for any PAD site."""
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    op_factors = {"kind": _project_basis(KIND_CUTOFF, KIND_PAD)}
    for k in range(2, 8):
        e = factored_local_expectation(state, k, op_factors)
        assert abs(e - 1.0) < 1e-10


def test_factored_local_expectation_type_projector():
    """<P_type=TYPE_ARR_II>(0) = 1 for \\x:Int. x (LAM site has type Int->Int)."""
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    op_factors = {"type": _project_basis(TYPE_CUTOFF, TYPE_ARR_II)}
    e = factored_local_expectation(state, 0, op_factors)
    assert abs(e - 1.0) < 1e-10


def test_factored_two_site_expectation_identity_is_one():
    """For a normalized MPS, <I tensor I> = 1 on every bond."""
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    op_factors_l = {"kind": _ident(KIND_CUTOFF)}
    op_factors_r = {"kind": _ident(KIND_CUTOFF)}
    for k in range(7):
        e = factored_two_site_expectation(state, k, op_factors_l, op_factors_r)
        assert abs(e - 1.0) < 1e-10


def test_factored_two_site_expectation_lam_app_kind_pattern():
    """For (\\x.x)(1) encoded: site 0 = APP, site 1 = LAM.
    <P_kind=APP(0) ⊗ P_kind=LAM(1)> = 1."""
    from src.qft_pcn.logic.encoding import KIND_APP
    state, _ = encode(parse(r"(\x:Int. x)(1)"), N=8, chi_max=32)
    op_l = {"kind": _project_basis(KIND_CUTOFF, KIND_APP)}
    op_r = {"kind": _project_basis(KIND_CUTOFF, KIND_LAM)}
    e = factored_two_site_expectation(state, 0, op_l, op_r)
    assert abs(e - 1.0) < 1e-10


def test_factored_matches_dense_local_op():
    """For a small d test, factored expectation matches dense embed_op result."""
    state, _ = encode(parse(r"\x:Int. x"), N=4, chi_max=32)
    # Build O = P_kind=LAM ⊗ I (only kind register is non-identity).
    factors = {"kind": _project_basis(KIND_CUTOFF, KIND_LAM)}
    factored_value = factored_local_expectation(state, 0, factors)
    # Dense version: lift to full local Hilbert space via embed_op.
    from src.qft_pcn.qft.fock import embed_op
    O_dense = embed_op(_project_basis(KIND_CUTOFF, KIND_LAM), 0,
                       (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF,
                        VALUE_CUTOFF, TOBL_CUTOFF))
    dense_value = state.local_expectation(0, O_dense)
    assert abs(factored_value - dense_value) < 1e-8
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_factored_expectation.py -v`
Expected: ImportError on `_factored_expectation`.

- [ ] **Step 3: Implement `_factored_expectation.py`**

Create `src/qft_pcn/logic/_factored_expectation.py`:

```python
"""Factored-expectation primitives for the typing Hamiltonian.

The local Hilbert space has dimension 65536. A dense operator on it would
require ~64 GB of memory per term. The factored approach exploits the
tensor-product structure of typing-rule operators (each is a product
O_kind ⊗ O_type ⊗ O_bid ⊗ O_value ⊗ O_tobl of small per-species matrices)
and contracts species-by-species against the MPS site tensor.

Time cost: ~ d_local * chi^2 per site, comparable to MPS.local_expectation
on small operators.
"""

from __future__ import annotations

import numpy as np

from src.qft_pcn.qft.mps import MPS
from .encoding import (
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF, D_LOCAL,
)

_SPECIES_NAMES = ("kind", "type", "bid", "value", "tobl")
_SPECIES_DIMS = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF)


def _normalize_factors(op_factors: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Return a 5-tuple of per-species matrices, with identity for unset ones."""
    factors: list[np.ndarray] = []
    for name, d in zip(_SPECIES_NAMES, _SPECIES_DIMS):
        if name in op_factors:
            op = op_factors[name]
            if op.shape != (d, d):
                raise ValueError(
                    f"op_factors[{name!r}] has shape {op.shape}; expected ({d}, {d})"
                )
            factors.append(op.astype(complex, copy=False))
        else:
            factors.append(np.eye(d, dtype=complex))
    return tuple(factors)


def _reshape_site_tensor(t: np.ndarray) -> np.ndarray:
    """Reshape (chi_l, D_LOCAL, chi_r) -> (chi_l, kind, type, bid, value,
    tobl, chi_r)."""
    chi_l, d, chi_r = t.shape
    assert d == D_LOCAL, f"expected D_LOCAL={D_LOCAL}, got {d}"
    return t.reshape(chi_l, *_SPECIES_DIMS, chi_r)


def _apply_factored_op_to_axis(A: np.ndarray, op: np.ndarray,
                                axis_kind: int, axis_type: int, axis_bid: int,
                                axis_value: int, axis_tobl: int,
                                species_index: int) -> np.ndarray:
    """Apply op (cutoff x cutoff) to A on the species axis."""
    # A has shape (chi_l, kind, type, bid, value, tobl, chi_r).
    # axes 1..5 correspond to species 0..4.
    target = species_index + 1
    # Use einsum to contract op[s_out, s_in] with A[..., s_in, ...].
    # We need a general approach. We'll move the target axis to position 1,
    # apply op, move back.
    A_moved = np.moveaxis(A, target, 1)
    out_shape = A_moved.shape   # (chi_l, cutoff, ...)
    A_flat = A_moved.reshape(out_shape[0], out_shape[1], -1)
    A_applied = np.einsum('ij,bjr->bir', op, A_flat)
    A_back = A_applied.reshape(out_shape)
    return np.moveaxis(A_back, 1, target)


def factored_local_expectation(state: MPS, site: int,
                                op_factors: dict[str, np.ndarray]) -> float:
    """<state | O_site | state> where O = ⊗ op_factors[species].

    Species not in op_factors are treated as identity.

    Returns a real number (since the Hamiltonian terms are Hermitian).
    """
    factors = _normalize_factors(op_factors)
    env = np.ones((1, 1), dtype=complex)
    for k in range(state.N):
        t = state.tensors[k]    # (chi_l, D_LOCAL, chi_r)
        if k == site:
            # Apply each species operator to the corresponding axis.
            A = _reshape_site_tensor(t)
            B = A
            for sp_idx, op in enumerate(factors):
                B = _apply_factored_op_to_axis(
                    B, op,
                    axis_kind=1, axis_type=2, axis_bid=3,
                    axis_value=4, axis_tobl=5,
                    species_index=sp_idx,
                )
            B_flat = B.reshape(t.shape)
            env = np.einsum('ij,isk,jsl->kl', env, B_flat, t.conj())
        else:
            env = np.einsum('ij,isk,jsl->kl', env, t, t.conj())
    return float(np.real(env[0, 0]))


def factored_two_site_expectation(
    state: MPS, site: int,
    op_factors_left: dict[str, np.ndarray],
    op_factors_right: dict[str, np.ndarray],
) -> float:
    """<state | O_site ⊗ O_{site+1} | state> for a separable two-site op.

    The two factor dicts give the per-species operators for the left site
    (at `site`) and the right site (at `site+1`). Species not in a dict are
    treated as identity on that side.

    For NON-separable two-site operators, decompose them into a sum of
    separable terms and call this helper once per term. The typing-Hamiltonian
    rules in §3 all factor naturally into a small sum of separable terms.

    Returns a real number.
    """
    if not 0 <= site < state.N - 1:
        raise ValueError(f"site {site} invalid for two-site op")

    factors_l = _normalize_factors(op_factors_left)
    factors_r = _normalize_factors(op_factors_right)

    env = np.ones((1, 1), dtype=complex)
    for k in range(state.N):
        t = state.tensors[k]
        if k == site:
            A = _reshape_site_tensor(t)
            B = A
            for sp_idx, op in enumerate(factors_l):
                B = _apply_factored_op_to_axis(
                    B, op, 1, 2, 3, 4, 5, sp_idx,
                )
            B_flat = B.reshape(t.shape)
            env = np.einsum('ij,isk,jsl->kl', env, B_flat, t.conj())
        elif k == site + 1:
            A = _reshape_site_tensor(t)
            B = A
            for sp_idx, op in enumerate(factors_r):
                B = _apply_factored_op_to_axis(
                    B, op, 1, 2, 3, 4, 5, sp_idx,
                )
            B_flat = B.reshape(t.shape)
            env = np.einsum('ij,isk,jsl->kl', env, B_flat, t.conj())
        else:
            env = np.einsum('ij,isk,jsl->kl', env, t, t.conj())
    return float(np.real(env[0, 0]))


def embed_factored_to_dense(op_factors: dict[str, np.ndarray]) -> np.ndarray:
    """Materialize the dense (D_LOCAL × D_LOCAL) operator from its factors.

    This is the slow path used only for diagnostics / sub-project E. Returns
    a (65536, 65536) complex matrix. Do not call this in inner loops.
    """
    factors = _normalize_factors(op_factors)
    op = factors[0]
    for f in factors[1:]:
        op = np.kron(op, f)
    return op
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_factored_expectation.py -v`
Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/_factored_expectation.py src/qft_pcn/tests/test_logic_factored_expectation.py
git commit -m "$(cat <<'EOF'
feat(logic/_factored_expectation): per-species factored MPS expectation

Computes <psi|O|psi> for O = O_kind ⊗ O_type ⊗ O_bid ⊗ O_value ⊗ O_tobl
without materializing the dense 65536×65536 matrix. Reshapes the site
tensor into per-species axes and contracts species-by-species via einsum.

Provides factored_local_expectation, factored_two_site_expectation, and
embed_factored_to_dense (slow path, for diagnostics). Verified against
MPS.local_expectation with the corresponding dense operator on simple
factorizations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: `TypingTerm` dataclass and `TypingHamiltonian` skeleton

**Files:**
- Create: `src/qft_pcn/logic/typing_hamiltonian.py`.
- Create: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

The skeleton has the constructor, term enumeration (without per-rule operators yet), and stub methods.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
"""Tests for logic/typing_hamiltonian.py (sub-project B)."""

from __future__ import annotations

import pytest


def test_typing_term_dataclass():
    from src.qft_pcn.logic.typing_hamiltonian import TypingTerm
    t = TypingTerm(rule_id="T-Lit-Int", site=3, arity=1)
    assert t.rule_id == "T-Lit-Int"
    assert t.site == 3
    assert t.arity == 1


def test_typing_hamiltonian_constructor_takes_only_N():
    """Spec §1.1: the Hamiltonian is data-independent. No AST in ctor."""
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    H = TypingHamiltonian(N=8)
    assert H.N == 8


def test_typing_hamiltonian_enumerates_terms():
    """Spec §6.2: term enumeration is deterministic and complete.

    For N=4 the term count is:
      - 5 one-site rules × 4 sites = 20
      - T-Var: bonds (1,..,N-1) = 3 (sites 1, 2, 3)
      - T-Abs: bonds (0, .., N-2) = 3 (sites 0, 1, 2)
      - T-App-Arrow: bonds (0, .., N-2) = 3 (sites 0, 1, 2)
    Total: 20 + 3 + 3 + 3 = 29.
    """
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    H = TypingHamiltonian(N=4)
    assert len(H.terms) == 29

    rules = {t.rule_id for t in H.terms}
    assert rules == {
        "T-Lit-Int", "T-Lit-Bool", "T-Bin-Arith", "T-Bin-Cmp",
        "T-Obligation", "T-Var", "T-Abs", "T-App-Arrow",
    }


def test_typing_hamiltonian_has_5_species_in_class_attribute():
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    assert len(TypingHamiltonian.SPECIES) == 5


def test_term_not_found_raises():
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm, TermNotFound,
    )
    H = TypingHamiltonian(N=4)
    bogus = TypingTerm(rule_id="T-Bogus", site=2, arity=1)
    # Construct a state somehow — doesn't matter, raise happens before evaluation.
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    state, _ = encode(parse(r"\x:Int. x"), N=4, chi_max=32)
    with pytest.raises(TermNotFound):
        H.term_energy(state, bogus)


def test_residuals_keyed_by_rule_id_and_site():
    """Spec §6.4: residuals(state) returns dict keyed by (rule_id, site)."""
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse

    H = TypingHamiltonian(N=4)
    state, _ = encode(parse(r"\x:Int. x"), N=4, chi_max=32)
    r = H.residuals(state)
    assert isinstance(r, dict)
    for key in r:
        rule_id, site = key
        assert isinstance(rule_id, str)
        assert 0 <= site < 4
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement the skeleton**

Create `src/qft_pcn/logic/typing_hamiltonian.py`:

```python
"""Typing-rule Hamiltonian compiler (sub-project B).

See docs/superpowers/specs/2026-05-21-typing-hamiltonian-design.md.

This module builds H_typing = Σ_rules Σ_sites H_rule(site), where each term
H_rule(site) is a one-site or two-site Hermitian projector encoding a single
STLC typing rule. <ψ|H_typing|ψ⟩ = 0 exactly when the AST encoded by ψ is
well-typed; for ill-typed ASTs, the per-term energies localize violations
(consumed by sub-project D).

Design: term-by-term factored evaluation. The Hamiltonian's terms are kept
in factored (per-species) form; expectations are computed via the helpers
in _factored_expectation.py. The dense 65536×65536 operator is NEVER
materialized.

The Hamiltonian is STRUCTURAL — depends only on (a) the lattice length N,
(b) the species list, (c) the typing rules. There is NO AST data anywhere
in this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.qft_pcn.qft.mps import MPS
from .encoding import (
    SPECIES,
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    TYPE_NONE, TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
    BID_NONE, BID_0,
    VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
    TOBL_NONE,
)


# -- Rule names (matched against the spec §3 table) ---------------------------

RULE_T_LIT_INT = "T-Lit-Int"
RULE_T_LIT_BOOL = "T-Lit-Bool"
RULE_T_BIN_ARITH = "T-Bin-Arith"
RULE_T_BIN_CMP = "T-Bin-Cmp"
RULE_T_OBLIGATION = "T-Obligation"
RULE_T_VAR = "T-Var"
RULE_T_ABS = "T-Abs"
RULE_T_APP_ARROW = "T-App-Arrow"

_ONE_SITE_RULES = (
    RULE_T_LIT_INT, RULE_T_LIT_BOOL, RULE_T_BIN_ARITH,
    RULE_T_BIN_CMP, RULE_T_OBLIGATION,
)
_TWO_SITE_RULES_LEFT_ANCHOR = (RULE_T_ABS, RULE_T_APP_ARROW)
_TWO_SITE_RULES_RIGHT_ANCHOR = (RULE_T_VAR,)


# -- Exceptions ---------------------------------------------------------------


class TypingHamiltonianError(Exception):
    """Base error for TypingHamiltonian operations."""


class TermNotFound(TypingHamiltonianError):
    def __init__(self, term: "TypingTerm"):
        super().__init__(f"term {term} not in this Hamiltonian's term list")


# -- TypingTerm ---------------------------------------------------------------


@dataclass(frozen=True)
class TypingTerm:
    """A single typing-rule term, addressable by (rule_id, site)."""
    rule_id: str
    site: int
    arity: int       # 1 = one-site, 2 = two-site (bond at (site, site+1) or (site-1, site))


# -- TypingHamiltonian --------------------------------------------------------


class TypingHamiltonian:
    """Sum-of-terms typing Hamiltonian for an N-site lattice.

    Per spec §1.1, the Hamiltonian is STRUCTURAL: it depends only on N (and
    the implicit species list / basis constants from encoding.py). It has
    no AST data. The same instance evaluates correctly on any MPS produced
    by the typing-aware encoder for an AST of size <= N.
    """

    SPECIES = SPECIES   # 5-species tuple

    def __init__(self, N: int):
        self.N = int(N)
        self.terms: list[TypingTerm] = self._enumerate_terms()
        self._terms_set = frozenset(self.terms)

    def _enumerate_terms(self) -> list[TypingTerm]:
        """Deterministic enumeration per spec §6.2."""
        terms: list[TypingTerm] = []
        # One-site rules at every site.
        for rule in _ONE_SITE_RULES:
            for k in range(self.N):
                terms.append(TypingTerm(rule_id=rule, site=k, arity=1))
        # T-Var: two-site, anchored at the VAR site (which has a left bond,
        # so VAR can be at any site k >= 1). We use TypingTerm.site = k
        # (the VAR site itself, i.e. the right end of the bond).
        for k in range(1, self.N):
            terms.append(TypingTerm(rule_id=RULE_T_VAR, site=k, arity=2))
        # T-Abs: two-site, anchored at the LAM site (which has a right bond,
        # so LAM can be at any site k <= N - 2).
        for k in range(self.N - 1):
            terms.append(TypingTerm(rule_id=RULE_T_ABS, site=k, arity=2))
        # T-App-Arrow: two-site, anchored at the APP site (which has a right
        # bond pointing to fn at k+1).
        for k in range(self.N - 1):
            terms.append(TypingTerm(rule_id=RULE_T_APP_ARROW, site=k, arity=2))
        return terms

    # -- API stubs (filled in by subsequent tasks) ----------------------------

    def term_energy(self, state: MPS, term: TypingTerm) -> float:
        """⟨ψ|H_term|ψ⟩ for a single named term."""
        if term not in self._terms_set:
            raise TermNotFound(term)
        # Dispatch by rule_id.
        rule = term.rule_id
        if rule == RULE_T_LIT_INT:
            return _energy_t_lit_int(state, term.site)
        if rule == RULE_T_LIT_BOOL:
            return _energy_t_lit_bool(state, term.site)
        if rule == RULE_T_BIN_ARITH:
            return _energy_t_bin_arith(state, term.site)
        if rule == RULE_T_BIN_CMP:
            return _energy_t_bin_cmp(state, term.site)
        if rule == RULE_T_OBLIGATION:
            return _energy_t_obligation(state, term.site)
        if rule == RULE_T_VAR:
            return _energy_t_var(state, term.site)
        if rule == RULE_T_ABS:
            return _energy_t_abs(state, term.site)
        if rule == RULE_T_APP_ARROW:
            return _energy_t_app_arrow(state, term.site)
        raise TermNotFound(term)

    def total_energy(self, state: MPS) -> float:
        """Σ over all terms of term_energy."""
        return sum(self.term_energy(state, t) for t in self.terms)

    def residuals(self, state: MPS) -> dict[tuple[str, int], float]:
        """All per-term energies, keyed by (rule_id, site).

        Sub-project D consumes this. Per spec §1.5, the per-term energies
        localize typing-rule violations.
        """
        return {(t.rule_id, t.site): self.term_energy(state, t)
                for t in self.terms}


# -- Per-rule energy functions (stubs; filled in by Tasks 12-17) -------------


def _energy_t_lit_int(state: MPS, site: int) -> float:
    raise NotImplementedError("T-Lit-Int — see Task 12")


def _energy_t_lit_bool(state: MPS, site: int) -> float:
    raise NotImplementedError("T-Lit-Bool — see Task 12")


def _energy_t_bin_arith(state: MPS, site: int) -> float:
    raise NotImplementedError("T-Bin-Arith — see Task 13")


def _energy_t_bin_cmp(state: MPS, site: int) -> float:
    raise NotImplementedError("T-Bin-Cmp — see Task 13")


def _energy_t_obligation(state: MPS, site: int) -> float:
    raise NotImplementedError("T-Obligation — see Task 14")


def _energy_t_var(state: MPS, site: int) -> float:
    raise NotImplementedError("T-Var — see Task 15")


def _energy_t_abs(state: MPS, site: int) -> float:
    raise NotImplementedError("T-Abs — see Task 16")


def _energy_t_app_arrow(state: MPS, site: int) -> float:
    raise NotImplementedError("T-App-Arrow — see Task 17")
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -v`
Expected: 6 tests pass (the residuals test calls `term_energy` which will raise NotImplementedError; we need to amend the test to expect that, OR mark these tests as testing only enumeration + dispatch).

Re-edit the failing tests to skip the body-evaluating ones, or split into:

```python
def test_residuals_keyed_by_rule_id_and_site():
    # ... build state ...
    # We can only test the KEY shape here, not the values (those are NIE'd):
    H = TypingHamiltonian(N=4)
    # The whole dict-building loop raises before we can inspect keys, so we
    # check individual term lookups instead.
    expected_keys = {(t.rule_id, t.site) for t in H.terms}
    actual_keys = set()
    for t in H.terms:
        try:
            H.term_energy(state, t)
        except NotImplementedError:
            actual_keys.add((t.rule_id, t.site))
    assert actual_keys == expected_keys
```

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/typing_hamiltonian.py src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/typing_hamiltonian): TypingHamiltonian skeleton + term enumeration

Deterministically enumerates 5N one-site terms (T-Lit-Int, T-Lit-Bool,
T-Bin-Arith, T-Bin-Cmp, T-Obligation) and 3(N-1) two-site terms (T-Var,
T-Abs, T-App-Arrow). Per spec §1.1 the constructor takes only N — no AST
data flows in. Per-rule energy functions are stubs (NotImplementedError)
filled in by subsequent commits.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: T-Lit-Int and T-Lit-Bool rules

**Files:**
- Modify: `src/qft_pcn/logic/typing_hamiltonian.py` (implement `_energy_t_lit_int`, `_energy_t_lit_bool`).
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Spec §3.1, §3.2:

```
H_lit_int(k)  = P_kind=INT(k) · (I - P_type=INT(k))
H_lit_bool(k) = P_kind=BOOL(k) · (I - P_type=BOOL(k))
```

These factor naturally on kind and type registers.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def test_t_lit_int_zero_on_well_typed_int():
    """\\x:Int. 3 — site 1 is INT lit with type INT. T-Lit-Int(1) = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 3"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Lit-Int", 1, 1))
    assert abs(e) < 1e-10


def test_t_lit_int_zero_at_non_int_sites():
    """At sites that aren't INT (kind != INT), T-Lit-Int = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    for k in range(8):
        e = H.term_energy(state, TypingTerm("T-Lit-Int", k, 1))
        assert abs(e) < 1e-10, f"site {k}: T-Lit-Int unexpectedly = {e}"


def test_t_lit_int_fires_on_mismatch():
    """Manually corrupt an INT site's type to BOOL — T-Lit-Int fires."""
    import numpy as np
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    from src.qft_pcn.logic.encoding import (
        KIND_INT, KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF,
        VALUE_CUTOFF, TOBL_CUTOFF, TYPE_INT, TYPE_BOOL, BID_NONE,
        VALUE_NONE, TOBL_INT, INT_LIT_OFFSET,
    )
    state, _ = encode(parse(r"\x:Int. 3"), N=4, chi_max=32)
    # site 1 is the INT(3) literal. Its local basis state at the encoder is
    # (kind=INT=4, type=INT=1, bid=NONE=0, value=3+7=10, tobl=INT=1).
    # We swap type=INT(1) with type=BOOL(2) at site 1.
    T = state.tensors[1].copy()
    new_T = np.zeros_like(T)
    # Re-route from type=INT to type=BOOL at THIS site.
    for chi_l in range(T.shape[0]):
        for chi_r in range(T.shape[2]):
            for kind in range(KIND_CUTOFF):
                for bid_ in range(BID_CUTOFF):
                    for val in range(VALUE_CUTOFF):
                        for tobl in range(TOBL_CUTOFF):
                            old_flat = ((((kind * TYPE_CUTOFF + TYPE_INT)
                                          * BID_CUTOFF + bid_) * VALUE_CUTOFF
                                         + val) * TOBL_CUTOFF + tobl)
                            new_flat = ((((kind * TYPE_CUTOFF + TYPE_BOOL)
                                          * BID_CUTOFF + bid_) * VALUE_CUTOFF
                                         + val) * TOBL_CUTOFF + tobl)
                            new_T[chi_l, new_flat, chi_r] = T[chi_l, old_flat,
                                                              chi_r]
    state.tensors[1] = new_T
    state.normalize()
    H = TypingHamiltonian(N=4)
    e = H.term_energy(state, TypingTerm("T-Lit-Int", 1, 1))
    assert e > 0.5, f"T-Lit-Int should fire at site 1, got {e}"


def test_t_lit_bool_zero_on_well_typed_bool():
    """\\x:Int. true — site 1 is BOOL lit with type BOOL."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. true"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Lit-Bool", 1, 1))
    assert abs(e) < 1e-10
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py::test_t_lit_int_zero_on_well_typed_int -v`
Expected: `NotImplementedError`.

- [ ] **Step 3: Implement the rules**

Edit `src/qft_pcn/logic/typing_hamiltonian.py`. Add a helper for "projector onto basis index k" and implement the two rules:

```python
from ._factored_expectation import factored_local_expectation, factored_two_site_expectation
import numpy as np


def _project(d: int, idx: int) -> np.ndarray:
    """|idx⟩⟨idx| on a d-dimensional Hilbert space."""
    p = np.zeros((d, d), dtype=complex)
    p[idx, idx] = 1.0
    return p


def _project_one_minus(d: int, idx: int) -> np.ndarray:
    """I - |idx⟩⟨idx|."""
    return np.eye(d, dtype=complex) - _project(d, idx)


def _project_set(d: int, idxs: list[int]) -> np.ndarray:
    """Sum of |i⟩⟨i| for i in idxs."""
    out = np.zeros((d, d), dtype=complex)
    for i in idxs:
        out[i, i] = 1.0
    return out


def _energy_t_lit_int(state: MPS, site: int) -> float:
    """Spec §3.1: H_lit_int(k) = P_kind=INT(k) · (I - P_type=INT(k))."""
    op_factors = {
        "kind": _project(KIND_CUTOFF, KIND_INT),
        "type": _project_one_minus(TYPE_CUTOFF, TYPE_INT),
    }
    return factored_local_expectation(state, site, op_factors)


def _energy_t_lit_bool(state: MPS, site: int) -> float:
    """Spec §3.2: H_lit_bool(k) = P_kind=BOOL(k) · (I - P_type=BOOL(k))."""
    op_factors = {
        "kind": _project(KIND_CUTOFF, KIND_BOOL),
        "type": _project_one_minus(TYPE_CUTOFF, TYPE_BOOL),
    }
    return factored_local_expectation(state, site, op_factors)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "t_lit" -v`
Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/typing_hamiltonian.py src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/typing_hamiltonian): implement T-Lit-Int and T-Lit-Bool rules

One-site projectors per spec §3.1-§3.2:
  H_lit_int(k)  = P_kind=INT  · (I - P_type=INT)
  H_lit_bool(k) = P_kind=BOOL · (I - P_type=BOOL)

Both factor naturally on kind/type registers and evaluate to 0 for
well-typed literals; nonzero only at kind={INT,BOOL} sites with the wrong
type tag. Verified by surgically mutating an INT site's type register
and confirming H fires.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: T-Bin-Arith and T-Bin-Cmp rules

**Files:**
- Modify: `src/qft_pcn/logic/typing_hamiltonian.py`.
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Spec §3.3:

```
H_bin_arith(k) = P_kind=BIN · (P_value=+ + P_value=- + P_value=*) · (I - P_type=INT)
H_bin_cmp(k)   = P_kind=BIN · (P_value=< + P_value===) · (I - P_type=BOOL)
```

Factors on kind, value, and type registers.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def test_t_bin_arith_zero_for_well_typed_plus():
    """1 + 2 inside a Lam: at BIN site, type = INT, value = PLUS. Rule = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 1 + 2"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Bin-Arith", 1, 1))
    assert abs(e) < 1e-10


def test_t_bin_cmp_zero_for_well_typed_lt():
    """\\x:Int. x < 5: at BIN site, type = BOOL, value = LT. Rule = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x < 5"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Bin-Cmp", 1, 1))
    assert abs(e) < 1e-10


def test_t_bin_arith_fires_when_output_is_bool():
    """Manually corrupt a BIN(+, _, _) site's type to BOOL — T-Bin-Arith fires."""
    import numpy as np
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    from src.qft_pcn.logic.encoding import (
        TYPE_INT, TYPE_BOOL, TYPE_CUTOFF, KIND_CUTOFF, BID_CUTOFF,
        VALUE_CUTOFF, TOBL_CUTOFF,
    )
    state, _ = encode(parse(r"\x:Int. 1 + 2"), N=8, chi_max=32)
    # Site 1 is BIN(+). Swap type=INT for type=BOOL at site 1.
    T = state.tensors[1].copy()
    new_T = np.zeros_like(T)
    for chi_l in range(T.shape[0]):
        for chi_r in range(T.shape[2]):
            for kind in range(KIND_CUTOFF):
                for bid_ in range(BID_CUTOFF):
                    for val in range(VALUE_CUTOFF):
                        for tobl in range(TOBL_CUTOFF):
                            old_flat = ((((kind * TYPE_CUTOFF + TYPE_INT)
                                          * BID_CUTOFF + bid_) * VALUE_CUTOFF
                                         + val) * TOBL_CUTOFF + tobl)
                            new_flat = ((((kind * TYPE_CUTOFF + TYPE_BOOL)
                                          * BID_CUTOFF + bid_) * VALUE_CUTOFF
                                         + val) * TOBL_CUTOFF + tobl)
                            new_T[chi_l, new_flat, chi_r] = T[chi_l, old_flat,
                                                              chi_r]
    state.tensors[1] = new_T
    state.normalize()
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Bin-Arith", 1, 1))
    assert e > 0.5
```

- [ ] **Step 2: Implement the rules**

Edit `src/qft_pcn/logic/typing_hamiltonian.py`:

```python
def _energy_t_bin_arith(state: MPS, site: int) -> float:
    """Spec §3.3: H_bin_arith(k) = P_kind=BIN · (P_val=PLUS + P_val=MINUS
    + P_val=TIMES) · (I - P_type=INT)."""
    op_factors = {
        "kind":  _project(KIND_CUTOFF, KIND_BIN),
        "value": _project_set(VALUE_CUTOFF,
                              [VALUE_PLUS, VALUE_MINUS, VALUE_TIMES]),
        "type":  _project_one_minus(TYPE_CUTOFF, TYPE_INT),
    }
    return factored_local_expectation(state, site, op_factors)


def _energy_t_bin_cmp(state: MPS, site: int) -> float:
    """Spec §3.3: H_bin_cmp(k) = P_kind=BIN · (P_val=LT + P_val=EQ) ·
    (I - P_type=BOOL)."""
    op_factors = {
        "kind":  _project(KIND_CUTOFF, KIND_BIN),
        "value": _project_set(VALUE_CUTOFF, [VALUE_LT, VALUE_EQ]),
        "type":  _project_one_minus(TYPE_CUTOFF, TYPE_BOOL),
    }
    return factored_local_expectation(state, site, op_factors)
```

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "t_bin" -v`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/logic/typing_hamiltonian.py src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/typing_hamiltonian): implement T-Bin-Arith and T-Bin-Cmp

One-site projectors discriminating on the value-register op code (PLUS,
MINUS, TIMES for arith; LT, EQ for cmp) per spec §3.3. The arith variant
requires the BIN site to have type INT; the cmp variant requires type BOOL.

Operand-type constraints (operands must be Int) are handled by T-Obligation,
not by these rules.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: T-Obligation rule (the workhorse)

**Files:**
- Modify: `src/qft_pcn/logic/typing_hamiltonian.py`.
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Spec §3.6:

```
H_obligation(k) = Σ_{t ≠ T_NONE} P_tobl=t(k) · (I - P_type=t(k))
```

This is the single rule that handles T-Abs-body, T-App-arg, T-If-*, T-Bin-* operand type constraints. It factors on tobl and type registers.

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def test_t_obligation_zero_on_well_typed():
    """For well-typed programs, T-Obligation = 0 at every site."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    for src in [r"\x:Int. x", r"(\x:Int. x + 1)(2)",
                r"\f:Int->Int. \x:Int. f x"]:
        state, _ = encode(parse(src), N=8, chi_max=32)
        H = TypingHamiltonian(N=8)
        for k in range(8):
            e = H.term_energy(state, TypingTerm("T-Obligation", k, 1))
            assert abs(e) < 1e-10, f"{src}: T-Obligation@{k} = {e}"


def test_t_obligation_fires_on_ill_typed():
    """\\x:Int. (x + true): the 'true' lit is at a BIN's rhs, obligation
    TOBL_INT, actual type BOOL — T-Obligation should fire."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, meta = encode(parse(r"\x:Int. x + true"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    # Layout: LAM@0, BIN+@1, VAR_x@2, BoolLit@3, PAD...
    # Site 3 has tobl=INT (BIN's rhs obligation) and type=BOOL.
    e = H.term_energy(state, TypingTerm("T-Obligation", 3, 1))
    assert e > 0.5, f"T-Obligation@3 should fire, got {e}"
    # Other sites should be zero (or near).
    for k in [0, 1, 2, 4, 5, 6, 7]:
        e_k = H.term_energy(state, TypingTerm("T-Obligation", k, 1))
        assert abs(e_k) < 1e-10, f"T-Obligation@{k} unexpectedly = {e_k}"
```

- [ ] **Step 2: Implement**

Edit `src/qft_pcn/logic/typing_hamiltonian.py`. T-Obligation is a SUM over distinct `t ≠ T_NONE` of factorable projectors:

```python
def _energy_t_obligation(state: MPS, site: int) -> float:
    """Spec §3.6: H_obligation(k) = Σ_{t ≠ T_NONE} P_tobl=t · (I - P_type=t).

    The sum is over t in {TOBL_INT, TOBL_BOOL, TOBL_ARR_II, TOBL_ARR_IB,
    TOBL_ARR_BI, TOBL_ARR_BB, TOBL_ARR_NESTED}. For each t, the term factors
    as P_tobl=t ⊗ (I - P_type=t) on tobl and type registers.
    """
    total = 0.0
    for t in range(1, TOBL_CUTOFF):   # 1..7 (skip TOBL_NONE=0)
        op_factors = {
            "tobl": _project(TOBL_CUTOFF, t),
            "type": _project_one_minus(TYPE_CUTOFF, t),
        }
        total += factored_local_expectation(state, site, op_factors)
    return total
```

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "t_obligation" -v`
Expected: 2 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/logic/typing_hamiltonian.py src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/typing_hamiltonian): implement T-Obligation (the workhorse rule)

Per spec §3.6: a sum over t ∈ {TOBL_INT..TOBL_ARR_NESTED} of one-site
projectors P_tobl=t · (I - P_type=t). Handles T-Abs body, T-App arg,
T-If cond/then/else, T-Bin operand type constraints — all rules that
couple a child's type to its parent's expectation.

Verified zero on three well-typed demo programs and firing on an
ill-typed BIN(+, x, true).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: T-Var rule (the entanglement-coupling rule)

**Files:**
- Modify: `src/qft_pcn/logic/typing_hamiltonian.py`.
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Spec §3.4. This is the rule that exercises the bid-bond extension. At a VAR site `v`, the type must equal the param_ty of the binder carried on the channel `v` reads from. The param_ty is a real bond DOF (per Task 6).

The two-site operator on bond `(v-1, v)` factors as a sum over `(c, t)` pairs:

```
H_var(v) = P_kind=VAR(v) · Σ_t (P_bond_carries_param_ty_t(v-1, v) · (I_local(v) - P_type=t(v)))
```

where `P_bond_carries_param_ty_t(v-1, v)` projects onto two-site joint states where the bid bond between sites `v-1` and `v` carries some channel with param_ty `t`.

In factored form: the bond's bid register at site `v-1`'s OUT side and site `v`'s IN side share dimension `1 + 8|L|`. Within that, the param_ty value at the channel ridden by VAR is `t`. We project onto the subspace of all channels with param_ty = t.

The factored two-site operator on the bid register is:
- At site `v-1` (left of bond): act with `Σ_{c'} P_outgoing_channel=c'_with_t` (the "outgoing channel carries param_ty t" projector).
- At site `v` (right of bond): act with `Σ_{c} P_incoming_channel=c_with_t`.

But these are bond-side projectors, not site-local. The cleaner approach: realize T-Var as a one-site operator on the VAR site combined with the MPS's left-environment.

Specifically: at site v, the local bid register's value is the binder's depth-from-innermost (BID_0..BID_6). The bond into v carries the param_ty. The two-site coupling means the joint state `|bid_val_at_v⟩ ⊗ |left_bond_channel⟩` is constrained.

The clean formulation: project onto MPS states where, at site v, kind=VAR and the LEFT BOND'S BASIS STATE encodes a (channel, param_ty=t) AND site v's local type = t. This is a multi-tensor projection — easier to compute by ENUMERATING the bond's basis states.

**Implementation approach:** at site v, contract the MPS into the orthogonality center at v, then for each (left-bond-basis-state, local-basis-state, right-bond-basis-state) configuration with nonzero amplitude, check whether the (kind=VAR, type=t, left-bond-channel param_ty=t) constraint is violated.

In practice this means a **two-site expectation** on bond `(v-1, v)` with a per-site operator pair. The bond's local-basis-pair at sites v-1 (right end) and v (left end) is correlated through the MPS bond.

**The correct factorization (using factored_two_site_expectation):**

We need the joint operator
```
H_var(v) = P_kind=VAR(v) · Σ_t [ P_outgoing_bid_carries_param_ty_t(v-1) · (I - P_type=t(v)) ]
```

The "outgoing channel carries param_ty t" projector at site v-1 is NOT a single-site operator on v-1's local Hilbert space — it's about the BOND state. However, we can rewrite it using the equivalence: `the right bond's basis state X means site v-1's right tensor index = X`. The factored two-site expectation contracts the bond through, so we operate on (v-1, v) jointly as 2 sites + their connecting bond.

The clean way is to think site-by-site in the contracted form, but for the BID register specifically. The BID register at site v has:
- left bond index: identifies the channel from the LEFT side, carrying (channel_id, param_ty).
- local bid value: BID_k = k+1 (depth).
- right bond index: identifies the channel going right.

For a VAR site reading channel c with binder param_ty t:
- left bond is in slot `1 + 8(c-1) + t`.
- local bid is BID_k where k = depth-from-innermost.

So we want to project onto MPS configurations where:
- site v has kind=VAR.
- site v has type=t.
- site v's LEFT bond has its bid-register-slot in `{1 + 8(c-1) + t : c in [1, L]}` — but EXCLUDING t = type(v), which is the GOOD case.

Equivalently: the violating case is "left bond slot is `1 + 8(c-1) + t_binder` for some t_binder ≠ type(v)".

To express this cleanly, we sum over `t_binder ∈ TYPE_CUTOFF`:
- Constraint: left bond slot `≡ 1 + 8(c-1) + t_binder` mod 8 with `t_binder ≠ type(v)`.

The factored helper can't directly project on bond basis states. We extend the helper to accept a "bond-side projector" on the bond degree of freedom.

**Pragmatic solution:** for T-Var, use a custom factored expectation that handles the bond. We add a helper `factored_two_site_expectation_with_bond_proj` to `_factored_expectation.py`, or we work around by using only site-local factored ops and the MPS's existing two_site machinery.

Actually, the cleanest approach: T-Var's constraint can be re-expressed as a SUM OF TWO-SITE site-local ops with no bond projection needed, by enumerating which (c, t_binder) pair sits on which bond slot.

For each `t_binder ∈ [0, TYPE_CUTOFF)`:
- Site v−1's right bond and site v's left bond carry the "channel with param_ty = t_binder" slots, which are positions `{1 + 8(c-1) + t_binder : c in [1, L]}`. For a given t_binder, these are L slots, all interleaved.
- The way to project on "bond slot is one of these L positions" is to act with a bond-side projector on the BID register's bond axis.

But again, bond-side projectors aren't trivially factored. So we DO need a custom helper.

**Decision:** Add a helper `factored_bid_bond_expectation(state, site, site_local_factors, bid_bond_projector)` to `_factored_expectation.py`. The `bid_bond_projector` is a `(d_bond, d_bond)` matrix acting on the bid register's bond axis between sites `site-1` and `site` (or `site` and `site+1`).

- [ ] **Step 1: Add the bond-projector helper to `_factored_expectation.py`**

Append to `src/qft_pcn/logic/_factored_expectation.py`:

```python
def factored_left_bond_bid_expectation(
    state: MPS, site: int,
    site_op_factors: dict[str, np.ndarray],
    bid_bond_projector: np.ndarray,
) -> float:
    """⟨ψ | P_bid_bond_left(site) ⊗ O_site | ψ⟩.

    The bid_bond_projector acts on the (left-bond × left-bond) joint axis
    where the bond runs between sites (site-1, site). Its dimension equals
    state.tensors[site].shape[0] (the left-bond dim).

    The site_op_factors describe a separable per-species operator on site
    `site`'s local Hilbert space.

    This is useful for rules like T-Var that read both the bond's basis
    state (which channel/param_ty was passed in) and the site's local
    registers (kind/type).
    """
    if site <= 0:
        raise ValueError(f"site must be >= 1; got {site}")
    factors = _normalize_factors(site_op_factors)
    chi_l_at_site = state.tensors[site].shape[0]
    if bid_bond_projector.shape != (chi_l_at_site, chi_l_at_site):
        raise ValueError(
            f"bid_bond_projector shape {bid_bond_projector.shape} != "
            f"({chi_l_at_site}, {chi_l_at_site})"
        )

    env = np.ones((1, 1), dtype=complex)
    for k in range(state.N):
        t = state.tensors[k]
        if k == site:
            A = _reshape_site_tensor(t)
            B = A
            for sp_idx, op in enumerate(factors):
                B = _apply_factored_op_to_axis(
                    B, op, 1, 2, 3, 4, 5, sp_idx,
                )
            B_flat = B.reshape(t.shape)
            # Apply the bond projector to the left-bond axis (axis 0).
            B_proj = np.einsum('xy,ysk->xsk', bid_bond_projector, B_flat)
            env = np.einsum('ij,isk,jsl->kl', env, B_proj, t.conj())
        else:
            env = np.einsum('ij,isk,jsl->kl', env, t, t.conj())
    return float(np.real(env[0, 0]))
```

- [ ] **Step 2: Add a helper to build the "bond slot has param_ty != local type" projector**

In `typing_hamiltonian.py`, add (or in a new submodule `_t_var_bond.py` if you prefer cleanliness):

```python
def _bid_bond_proj_param_ty(d_bond: int, target_t: int) -> np.ndarray:
    """Projector on the bid bond (dim 1 + 8L) onto channels carrying
    param_ty = target_t.

    Bond slot layout (Task 6 of B's plan):
      slot 0:            |no_info⟩
      slot 1 + 8(i-1) + t : |channel_i, param_ty=t⟩  (i in [1, L], t in [0, 8))

    For target_t in [0, 8), the projector hits slots {1 + 8(c-1) + target_t :
    c in [1, L]}, i.e., slots target_t + 1, target_t + 9, target_t + 17, ...
    """
    proj = np.zeros((d_bond, d_bond), dtype=complex)
    # The number of channels L = (d_bond - 1) // 8 (so slot 0 is no_info and
    # the rest are 8L slots).
    if d_bond == 1:
        return proj   # no live binders; cannot project on a binder channel
    L = (d_bond - 1) // 8
    if 1 + 8 * L != d_bond:
        raise ValueError(
            f"d_bond={d_bond} is not of form 1 + 8L for any L"
        )
    for c in range(1, L + 1):
        slot = 1 + 8 * (c - 1) + target_t
        proj[slot, slot] = 1.0
    return proj
```

- [ ] **Step 3: Implement `_energy_t_var`**

Edit `typing_hamiltonian.py`:

```python
def _energy_t_var(state: MPS, site: int) -> float:
    """Spec §3.4: H_var(v) = P_kind=VAR(v) ·
        Σ_{t_binder} [ P_left_bond_carries_param_ty_{t_binder}(v) ·
                        (I - P_type=t_binder(v)) ]

    The constraint: VAR's type must equal the param_ty of the binder its
    channel reads from. The channel's param_ty rides on the left bond's
    basis state (real bond DOF, per Task 6).

    For each t_binder, the contribution is:
      P_kind=VAR(v) · P_type≠t_binder(v) · P_left_bond_in_channel_with_t_binder(v)

    Summed over t_binder, this counts: 1 if (VAR && type ≠ binder.param_ty).
    """
    from ._factored_expectation import factored_left_bond_bid_expectation

    if site == 0:
        # Site 0 has no left bond. VAR cannot live there; if it did, the
        # encoder's pre-order layout would place an APP/LAM/etc at 0 first.
        return 0.0

    d_bond = state.tensors[site].shape[0]
    if d_bond == 1:
        # No live binders crossing this bond; no Var coupling to verify.
        return 0.0

    total = 0.0
    for t_binder in range(TOBL_CUTOFF):
        # Operator at site v: P_kind=VAR · (I - P_type=t_binder).
        site_op = {
            "kind": _project(KIND_CUTOFF, KIND_VAR),
            "type": _project_one_minus(TYPE_CUTOFF, t_binder),
        }
        # Bond projector: select channels with param_ty = t_binder.
        bond_proj = _bid_bond_proj_param_ty(d_bond, t_binder)
        if np.allclose(bond_proj, 0):
            continue
        total += factored_left_bond_bid_expectation(
            state, site, site_op, bond_proj,
        )
    return total
```

- [ ] **Step 4: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def test_t_var_zero_for_well_typed_var():
    """\\x:Int. x — at the Var site (1), type=INT, binder.param_ty=INT.
    T-Var(1) = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Var", 1, 2))
    assert abs(e) < 1e-10, f"T-Var should be 0, got {e}"


def test_t_var_zero_at_non_var_sites():
    """At sites that aren't VAR, T-Var doesn't fire."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 3"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    # Site 1 is IntLit, not Var. T-Var should not fire.
    e = H.term_energy(state, TypingTerm("T-Var", 1, 2))
    assert abs(e) < 1e-10


def test_t_var_fires_on_type_binder_mismatch():
    """Manually corrupt Var's type from INT to BOOL. T-Var fires because
    the binder's param_ty (on the bond) is still INT, but local type is BOOL.
    """
    import numpy as np
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    from src.qft_pcn.logic.encoding import (
        TYPE_INT, TYPE_BOOL, TYPE_CUTOFF, KIND_CUTOFF, BID_CUTOFF,
        VALUE_CUTOFF, TOBL_CUTOFF,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=4, chi_max=32)
    T = state.tensors[1].copy()
    new_T = np.zeros_like(T)
    for chi_l in range(T.shape[0]):
        for chi_r in range(T.shape[2]):
            for kind in range(KIND_CUTOFF):
                for bid_ in range(BID_CUTOFF):
                    for val in range(VALUE_CUTOFF):
                        for tobl in range(TOBL_CUTOFF):
                            old_flat = ((((kind * TYPE_CUTOFF + TYPE_INT)
                                          * BID_CUTOFF + bid_)
                                         * VALUE_CUTOFF + val)
                                        * TOBL_CUTOFF + tobl)
                            new_flat = ((((kind * TYPE_CUTOFF + TYPE_BOOL)
                                          * BID_CUTOFF + bid_)
                                         * VALUE_CUTOFF + val)
                                        * TOBL_CUTOFF + tobl)
                            new_T[chi_l, new_flat, chi_r] = T[chi_l, old_flat,
                                                              chi_r]
    state.tensors[1] = new_T
    state.normalize()
    H = TypingHamiltonian(N=4)
    e = H.term_energy(state, TypingTerm("T-Var", 1, 2))
    assert e > 0.5, f"T-Var should fire, got {e}"
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "t_var" -v`
Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/typing_hamiltonian.py src/qft_pcn/logic/_factored_expectation.py src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/typing_hamiltonian): implement T-Var via real bid-bond DOF

T-Var is the rule that exercises B's encoder extension: the binder's
param_ty rides on the bid bond's channel as a real quantum DOF. The
Hamiltonian term sums over candidate t_binder ∈ TYPE_CUTOFF; for each,
projects onto MPS configurations where (a) site v has kind=VAR and
type != t_binder, AND (b) the left bond (channel reading) is in a slot
carrying param_ty = t_binder.

Adds factored_left_bond_bid_expectation to _factored_expectation: a
helper that applies a bond-side projector to the left bond's axis at a
chosen site, combined with site-local factored operators on the other
species.

Verified by manually corrupting a VAR site's type register and observing
T-Var fire.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: T-Abs rule

**Files:**
- Modify: `src/qft_pcn/logic/typing_hamiltonian.py`.
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Spec §3.5: at a LAM site `l`, the param_ty on the outgoing bid channel must equal the LAM's arrow-type src. T-Abs is a two-site operator on bond `(l, l+1)`.

For each flat arrow tag `a ∈ {T_ARR_II, T_ARR_IB, T_ARR_BI, T_ARR_BB}` with `src(a) = s`:

```
H_abs(l) += P_kind=LAM(l) · P_type=a(l) · (I - P_outgoing_channel_param_ty=s(l, l+1))
```

For T_ARR_NESTED, the rule passes (we cannot inspect the nested type from the flat tag).

The outgoing channel projector is the same kind of bond-side projector as T-Var, but at the RIGHT bond of site `l`.

- [ ] **Step 1: Add `factored_right_bond_bid_expectation` to `_factored_expectation.py`**

Append:

```python
def factored_right_bond_bid_expectation(
    state: MPS, site: int,
    site_op_factors: dict[str, np.ndarray],
    bid_bond_projector: np.ndarray,
) -> float:
    """Like factored_left_bond_bid_expectation but on the RIGHT bond.

    The bid_bond_projector acts on site's right-bond axis (dim equals
    state.tensors[site].shape[2]).
    """
    if site >= state.N - 1:
        raise ValueError(f"site must be < N-1; got {site}")
    factors = _normalize_factors(site_op_factors)
    chi_r_at_site = state.tensors[site].shape[2]
    if bid_bond_projector.shape != (chi_r_at_site, chi_r_at_site):
        raise ValueError(
            f"bid_bond_projector shape {bid_bond_projector.shape} != "
            f"({chi_r_at_site}, {chi_r_at_site})"
        )

    env = np.ones((1, 1), dtype=complex)
    for k in range(state.N):
        t = state.tensors[k]
        if k == site:
            A = _reshape_site_tensor(t)
            B = A
            for sp_idx, op in enumerate(factors):
                B = _apply_factored_op_to_axis(B, op, 1, 2, 3, 4, 5, sp_idx)
            B_flat = B.reshape(t.shape)
            # Apply bond projector to right-bond axis (axis 2).
            B_proj = np.einsum('isk,xk->isx', B_flat, bid_bond_projector.T)
            # Equivalent: B_proj[i, s, x] = sum_k B_flat[i, s, k] * P[x, k]
            # which is B_flat @ P^T. Be careful with the conjugation
            # convention; for a Hermitian projector, P = P^T conjugated;
            # we use P directly because projectors here are real.
            env = np.einsum('ij,isk,jsl->kl', env, B_proj, t.conj())
        else:
            env = np.einsum('ij,isk,jsl->kl', env, t, t.conj())
    return float(np.real(env[0, 0]))
```

- [ ] **Step 2: Define `_T_ABS_SRC_FOR_ARROW`**

In `typing_hamiltonian.py`:

```python
# Spec §3.5: for each flat arrow tag, the src type tag.
_ARROW_SRC = {
    TYPE_ARR_II: TYPE_INT,
    TYPE_ARR_IB: TYPE_INT,
    TYPE_ARR_BI: TYPE_BOOL,
    TYPE_ARR_BB: TYPE_BOOL,
}

# Dst for each flat arrow tag (used by T-App-Arrow, Task 17).
_ARROW_DST = {
    TYPE_ARR_II: TYPE_INT,
    TYPE_ARR_IB: TYPE_BOOL,
    TYPE_ARR_BI: TYPE_INT,
    TYPE_ARR_BB: TYPE_BOOL,
}
```

- [ ] **Step 3: Implement `_energy_t_abs`**

```python
def _energy_t_abs(state: MPS, site: int) -> float:
    """Spec §3.5: H_abs(l) = P_kind=LAM(l) · Σ_a P_type=a(l) ·
        (I - P_outgoing_channel_param_ty=src(a)(l, l+1))

    For each flat arrow tag a ∈ _ARROW_SRC, the outgoing bid channel must
    carry param_ty = src(a). T_ARR_NESTED is permissive (we cannot inspect).
    """
    from ._factored_expectation import factored_right_bond_bid_expectation

    if site >= state.N - 1:
        return 0.0
    d_bond = state.tensors[site].shape[2]
    if d_bond == 1:
        # No channels on the right bond; if kind=LAM here, the encoder
        # would have created a channel — this branch hits only at PAD or
        # mid-PAD sites that aren't LAM. T-Abs is gated on kind=LAM, so
        # safe to return 0.
        return 0.0

    total = 0.0
    for a_tag, src_tag in _ARROW_SRC.items():
        # Site operator: P_kind=LAM · P_type=a.
        site_op = {
            "kind": _project(KIND_CUTOFF, KIND_LAM),
            "type": _project(TYPE_CUTOFF, a_tag),
        }
        # Bond projector: (I - P_param_ty=src(a)) on the outgoing bond's
        # bid register. The outgoing channel carries the NEW binder's
        # param_ty, which the encoder wrote as the LAM's param_ty.
        bond_proj_correct = _bid_bond_proj_param_ty(d_bond, src_tag)
        bond_proj_violating = np.eye(d_bond, dtype=complex) - bond_proj_correct
        total += factored_right_bond_bid_expectation(
            state, site, site_op, bond_proj_violating,
        )
    return total
```

Note: the bond projector `(I - P_correct)` includes the `no_info` slot (slot 0) and all `t ≠ src(a)` slots for each channel. For a LAM site at non-zero-channel-out (the encoder DOES create a channel at LAM), the bond's slot for the LAM's new channel is `1 + 8(c-1) + lam_param_ty`. The `(I - P_correct)` projector hits everything ELSE. If the encoder is correct, `lam_param_ty` equals `src(a)` for the correct flat arrow tag `a`, so the violating projector contributes zero.

- [ ] **Step 4: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def test_t_abs_zero_for_well_typed_lam():
    """\\x:Int. x — LAM at site 0 has type T_ARR_II (src=Int).
    Outgoing bid channel carries param_ty=Int. T-Abs = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-Abs", 0, 2))
    assert abs(e) < 1e-10


def test_t_abs_zero_at_non_lam_sites():
    """If a site isn't LAM, T-Abs doesn't fire."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. 1 + 2"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    # Site 0 IS LAM. Other sites should not fire.
    for k in range(1, 7):
        e = H.term_energy(state, TypingTerm("T-Abs", k, 2))
        assert abs(e) < 1e-10, f"T-Abs@{k} should be 0 at non-LAM, got {e}"
```

- [ ] **Step 5: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "t_abs" -v`
Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/typing_hamiltonian.py src/qft_pcn/logic/_factored_expectation.py src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/typing_hamiltonian): implement T-Abs (LAM src-consistency)

For each flat arrow tag a (T_ARR_II..T_ARR_BB), if the LAM's local type is
a, the outgoing bid channel's param_ty must equal src(a). The two-site
operator on bond (l, l+1) uses the right-bond bid projector
(I - P_param_ty=src(a)) summed over all a.

T_ARR_NESTED is permissive; the encoder stores its full Ty in
nested_type_index, and the typing Hamiltonian conservatively accepts it.

Adds factored_right_bond_bid_expectation as the dual of
factored_left_bond_bid_expectation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: T-App-Arrow rule

**Files:**
- Modify: `src/qft_pcn/logic/typing_hamiltonian.py`.
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

Spec §3.7. Adjacent two-site rule on bond `(k, k+1)`: if site `k` is APP with type `y`, site `k+1` must have an arrow type whose dst is `y`.

```
H_app_arrow(k) = P_kind=APP(k) · Σ_y P_type=y(k) · (I - Σ_{a: dst(a)=y} P_type=a(k+1))
```

The allowed types at `k+1` given `k`'s APP-type `y`:

| y          | allowed type(k+1)                       |
|------------|------------------------------------------|
| T_INT      | T_ARR_II, T_ARR_BI, T_ARR_NESTED         |
| T_BOOL     | T_ARR_IB, T_ARR_BB, T_ARR_NESTED         |
| T_ARR_*    | T_ARR_NESTED                             |
| T_NONE     | (impossible — APP always has a type)     |

- [ ] **Step 1: Write the failing test**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def test_t_app_arrow_zero_for_well_typed_app():
    """(\\x:Int. x)(1): APP at site 0 has type=Int; fn at site 1 has type
    Int->Int. dst(Int->Int) = Int = APP.type. T-App-Arrow = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"(\x:Int. x)(1)"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    e = H.term_energy(state, TypingTerm("T-App-Arrow", 0, 2))
    assert abs(e) < 1e-10


def test_t_app_arrow_zero_at_non_app_sites():
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    for k in range(7):
        e = H.term_energy(state, TypingTerm("T-App-Arrow", k, 2))
        assert abs(e) < 1e-10, f"T-App-Arrow@{k}: {e}"


def test_t_app_arrow_fires_on_app_to_non_arrow():
    """Build an MPS that pretends site 0 is APP and site 1 has type=Int
    (not an arrow). T-App-Arrow should fire."""
    import numpy as np
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import (
        TypingHamiltonian, TypingTerm,
    )
    from src.qft_pcn.logic.encoding import (
        KIND_INT, KIND_APP, KIND_CUTOFF, TYPE_ARR_II, TYPE_INT, TYPE_CUTOFF,
        BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    )
    # Encode (\\x:Int. x)(1). At site 1, the LAM has type=T_ARR_II.
    state, _ = encode(parse(r"(\x:Int. x)(1)"), N=4, chi_max=32)
    # Overwrite site 1's type from T_ARR_II to T_INT.
    T = state.tensors[1].copy()
    new_T = np.zeros_like(T)
    for chi_l in range(T.shape[0]):
        for chi_r in range(T.shape[2]):
            for kind in range(KIND_CUTOFF):
                for bid_ in range(BID_CUTOFF):
                    for val in range(VALUE_CUTOFF):
                        for tobl in range(TOBL_CUTOFF):
                            old_flat = ((((kind * TYPE_CUTOFF + TYPE_ARR_II)
                                          * BID_CUTOFF + bid_)
                                         * VALUE_CUTOFF + val)
                                        * TOBL_CUTOFF + tobl)
                            new_flat = ((((kind * TYPE_CUTOFF + TYPE_INT)
                                          * BID_CUTOFF + bid_)
                                         * VALUE_CUTOFF + val)
                                        * TOBL_CUTOFF + tobl)
                            new_T[chi_l, new_flat, chi_r] = T[chi_l, old_flat,
                                                              chi_r]
    state.tensors[1] = new_T
    state.normalize()
    H = TypingHamiltonian(N=4)
    e = H.term_energy(state, TypingTerm("T-App-Arrow", 0, 2))
    assert e > 0.5
```

- [ ] **Step 2: Implement**

Edit `src/qft_pcn/logic/typing_hamiltonian.py`. Add the dst → allowed-fn-types lookup:

```python
# Spec §3.7 dst-to-allowed table.
_APP_FN_ALLOWED_BY_DST = {
    TYPE_INT:  [TYPE_ARR_II, TYPE_ARR_BI, TYPE_ARR_NESTED],
    TYPE_BOOL: [TYPE_ARR_IB, TYPE_ARR_BB, TYPE_ARR_NESTED],
    # For higher-order results, only T_ARR_NESTED is allowed at the flat-tag level.
    TYPE_ARR_II:     [TYPE_ARR_NESTED],
    TYPE_ARR_IB:     [TYPE_ARR_NESTED],
    TYPE_ARR_BI:     [TYPE_ARR_NESTED],
    TYPE_ARR_BB:     [TYPE_ARR_NESTED],
    TYPE_ARR_NESTED: [TYPE_ARR_NESTED],
    TYPE_NONE:       [],   # impossible: APP always has a non-NONE type
}


def _energy_t_app_arrow(state: MPS, site: int) -> float:
    """Spec §3.7: H_app_arrow(k) = P_kind=APP(k) · Σ_y P_type=y(k) ·
        (I - Σ_{a: dst(a)=y} P_type=a(k+1))

    Two-site adjacent operator on bond (site, site+1). Sums over all possible
    APP types y of the violating-arg contribution.
    """
    if site >= state.N - 1:
        return 0.0
    total = 0.0
    for y in range(TYPE_CUTOFF):
        allowed_at_k_plus_1 = _APP_FN_ALLOWED_BY_DST[y]
        # Site k (APP): P_kind=APP · P_type=y.
        op_left = {
            "kind": _project(KIND_CUTOFF, KIND_APP),
            "type": _project(TYPE_CUTOFF, y),
        }
        # Site k+1: I - Σ_{a: dst(a)=y} P_type=a.
        op_right = {
            "type": np.eye(TYPE_CUTOFF, dtype=complex)
                    - _project_set(TYPE_CUTOFF, allowed_at_k_plus_1),
        }
        total += factored_two_site_expectation(
            state, site, op_left, op_right,
        )
    return total
```

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -k "t_app_arrow" -v`
Expected: 3 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/logic/typing_hamiltonian.py src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/typing_hamiltonian): implement T-App-Arrow (adjacent two-site)

Per spec §3.7: at an APP site k with local type y, the adjacent site k+1
(fn root by pre-order layout) must have an arrow type whose dst is y.
For each y the contributing operator is
  P_kind=APP(k) · P_type=y(k) · (I - Σ_{dst(a)=y} P_type=a(k+1))
summed over y. T_ARR_NESTED is permissively allowed in all dst slots.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: Smoke test: total_energy on all WT and IT programs

**Files:**
- Modify: `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`.

A quick end-to-end check that the full Hamiltonian is now wired up.

- [ ] **Step 1: Write the test**

Append to `src/qft_pcn/tests/test_logic_typing_hamiltonian.py`:

```python
def test_total_energy_zero_on_p1():
    """Quick smoke: \\x:Int. x has total H = 0."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(r"\x:Int. x"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    assert abs(H.total_energy(state)) < 1e-10


def test_total_energy_nonzero_on_x_plus_true():
    """\\x:Int. x + true should have H > 0 (T-Obligation fires at the 'true')."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(r"\x:Int. x + true"), N=8, chi_max=32)
    H = TypingHamiltonian(N=8)
    assert H.total_energy(state) > 0.5


def test_residuals_sum_to_total_energy_p1():
    """Σ residuals == total_energy."""
    from src.qft_pcn.logic.encoder import encode
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
    state, _ = encode(parse(r"(\x:Int. x + 1)(2)"), N=16, chi_max=32)
    H = TypingHamiltonian(N=16)
    total = H.total_energy(state)
    residuals_sum = sum(H.residuals(state).values())
    assert abs(total - residuals_sum) < 1e-10
```

- [ ] **Step 2: Run all typing_hamiltonian tests**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_typing_hamiltonian.py -v 2>&1 | tail -30`
Expected: all tests pass (16-20 total from Tasks 11–18).

- [ ] **Step 3: Run the full logic test suite to confirm no regressions**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_logic_*.py -v 2>&1 | tail -15`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/qft_pcn/tests/test_logic_typing_hamiltonian.py
git commit -m "$(cat <<'EOF'
test(logic/typing_hamiltonian): end-to-end total_energy and residuals smoke

Three integration tests:
  - total_energy = 0 on a well-typed program
  - total_energy > 0 on an ill-typed program
  - residuals sum equals total_energy

All eight rules (T-Lit-Int, T-Lit-Bool, T-Bin-Arith, T-Bin-Cmp,
T-Obligation, T-Var, T-Abs, T-App-Arrow) are now wired up. Part 3 of the
plan covers the full acceptance test matrix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

**End of Part 2.** Continue with Part 3 (`2026-05-21-typing-hamiltonian-part3.md`): the five WT programs, five IT programs, per-rule isolation tests, classical-checker agreement, performance budget, public exports, final verification.

After Part 2 is done:
- `TypingHamiltonian(N)` is a structural Hamiltonian with 8 named rules and ~8N terms.
- `total_energy(state)` and `residuals(state) -> dict[(rule_id, site), float]` are wired.
- All 8 rules have one-site or two-site factored-expectation evaluators.
- Smoke tests confirm well-typed → 0, ill-typed → positive energy.

Part 3 hardens this with the full acceptance matrix and Sub-project D's contract.
