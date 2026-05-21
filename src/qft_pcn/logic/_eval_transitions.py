"""Off-diagonal transition couplings for the evaluation Hamiltonian
(sub-project C, Phase 3).

Each transition coupling has the form

    H_trans = -λ · ( |post⟩⟨pre| + |pre⟩⟨post| )

where |pre⟩ and |post⟩ are PRODUCT computational-basis states spanning
some contiguous window of sites — the unreduced and reduced
configurations of a redex. For Phase 3 we enumerate the rank-1 pairs
that cover:

  - If-redex (4 sites: IF, BOOL(t/f), then_lit, else_lit → INT/BOOL, PAD, PAD, PAD)
  - Arithmetic (3 sites: BIN(op), INT(a), INT(b) → INT(op(a,b)), PAD, PAD)
  - Comparison (3 sites: BIN(cmp), INT(a), INT(b) → BOOL(cmp(a,b)), PAD, PAD)

Each rank-1 outer product is represented as a TransitionTerm carrying
the absolute site index of the window's left edge, the per-site basis
indices (kind, type, bid, value, tobl) for |pre⟩ and |post⟩, and the
coupling λ. The applier in factored_evolution.py builds the small
per-site rank-1 ket-bra matrices and assembles the gate via direct-sum
across the window's interior bonds.

This module is the rank-1 ENUMERATION layer; the gate-application layer
lives in factored_evolution.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from .encoding import (
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF,
    KIND_PAD, KIND_INT, KIND_BOOL, KIND_IF, KIND_BIN,
    TYPE_NONE, TYPE_INT, TYPE_BOOL,
    BID_NONE,
    VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
    INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX,
    TOBL_NONE, TOBL_INT, TOBL_BOOL,
)


# ---- Basis-state helpers ---------------------------------------------------


SiteBasis = tuple[int, int, int, int, int]   # (kind, type, bid, value, tobl)


def _pad_site() -> SiteBasis:
    return (KIND_PAD, TYPE_NONE, BID_NONE, VALUE_NONE, TOBL_NONE)


def _int_lit_site(n: int, tobl: int = TOBL_NONE) -> SiteBasis:
    if not (INT_LIT_MIN <= n <= INT_LIT_MAX):
        raise ValueError(f"int literal {n} out of range")
    return (KIND_INT, TYPE_INT, BID_NONE, n + INT_LIT_OFFSET, tobl)


def _bool_lit_site(b: bool, tobl: int = TOBL_NONE) -> SiteBasis:
    return (KIND_BOOL, TYPE_BOOL, BID_NONE,
            VALUE_TRUE if b else VALUE_FALSE, tobl)


def _bin_site(op: int, tobl: int = TOBL_NONE) -> SiteBasis:
    # type tag is TYPE_NONE for the BIN parent itself; the result's type
    # is inferred from the op (arith → INT, cmp → BOOL).
    return (KIND_BIN, TYPE_NONE, BID_NONE, op, tobl)


def _if_site(tobl: int = TOBL_NONE) -> SiteBasis:
    return (KIND_IF, TYPE_NONE, BID_NONE, VALUE_NONE, tobl)


# ---- TransitionTerm --------------------------------------------------------


def transitions_from_program_pair(pre_src, post_src, N: int,
                                    coupling: float,
                                    rule_id: str = "T-Reduce"):
    """Build a TransitionTerm from a `(pre, post)` source-program pair.

    Encodes BOTH programs with the same N. Extracts each site's argmax
    basis index from the encoded MPS (the encoder produces deterministic
    product states so argmax is the same as the only non-zero index).
    Returns a TransitionTerm spanning ALL N sites, since both programs
    cover the same lattice.

    pre_src, post_src: either source strings (parsed via .parse) or
    already-parsed AST Node objects.
    """
    from . import encode, parse
    import numpy as np

    pre_ast = parse(pre_src) if isinstance(pre_src, str) else pre_src
    post_ast = parse(post_src) if isinstance(post_src, str) else post_src
    state_pre, _ = encode(pre_ast, N=N, chi_max=32)
    state_post, _ = encode(post_ast, N=N, chi_max=32)

    def _argmax_indices(state):
        out = []
        for k in range(state.N):
            A = state.tensors[k]
            p = (np.abs(A) ** 2).sum(axis=(0, 2))
            flat = int(np.argmax(p))
            # Decompose.
            o = flat % TOBL_CUTOFF
            flat //= TOBL_CUTOFF
            v = flat % VALUE_CUTOFF
            flat //= VALUE_CUTOFF
            b = flat % BID_CUTOFF
            flat //= BID_CUTOFF
            t = flat % TYPE_CUTOFF
            kind = flat // TYPE_CUTOFF
            out.append((kind, t, b, v, o))
        return out

    pre_sites = _argmax_indices(state_pre)
    post_sites = _argmax_indices(state_post)
    # Trim trailing pads on both sides to find the active window.
    pad = _pad_site()
    last_active = -1
    for k in range(N):
        if pre_sites[k] != pad or post_sites[k] != pad:
            last_active = k
    width = last_active + 1
    if width <= 0:
        raise ValueError(
            f"pre_src={pre_src!r}, post_src={post_src!r} encode to all-PAD states"
        )
    return TransitionTerm(
        rule_id=rule_id,
        site_left=0,
        pre=tuple(pre_sites[:width]),
        post=tuple(post_sites[:width]),
        coupling=coupling,
    )


@dataclass(frozen=True)
class TransitionTerm:
    """One rank-1 (anti-)Hermitian pair |post⟩⟨pre| + |pre⟩⟨post|.

    site_left: index of the leftmost site of the window.
    pre:  tuple of SiteBasis tuples covering window length sites.
    post: same length, the reduced configuration.
    coupling: λ in H = -λ · (|post⟩⟨pre| + |pre⟩⟨post|).
    rule_id: human-readable string for diagnostics ("T-Arith", "T-If", ...).
    """
    rule_id: str
    site_left: int
    pre: tuple[SiteBasis, ...]
    post: tuple[SiteBasis, ...]
    coupling: float

    @property
    def width(self) -> int:
        return len(self.pre)


# ---- Enumerators for each redex type ---------------------------------------


_ARITH_OPS = {
    VALUE_PLUS: lambda a, b: a + b,
    VALUE_MINUS: lambda a, b: a - b,
    VALUE_TIMES: lambda a, b: a * b,
}

_CMP_OPS = {
    VALUE_LT: lambda a, b: a < b,
    VALUE_EQ: lambda a, b: a == b,
}


def enumerate_arith_transitions(site_left: int,
                                 coupling: float) -> list[TransitionTerm]:
    """Yield rank-1 transitions for every (op, a, b) triple whose result
    fits in [INT_LIT_MIN, INT_LIT_MAX].

    Window: 3 sites starting at site_left.
      pre  = (BIN(op), INT(a), INT(b))
      post = (INT(op(a,b)), PAD, PAD)
    """
    out: list[TransitionTerm] = []
    for op_code, op_fn in _ARITH_OPS.items():
        for a in range(INT_LIT_MIN, INT_LIT_MAX + 1):
            for b in range(INT_LIT_MIN, INT_LIT_MAX + 1):
                result = op_fn(a, b)
                if not (INT_LIT_MIN <= result <= INT_LIT_MAX):
                    continue
                pre = (_bin_site(op_code), _int_lit_site(a), _int_lit_site(b))
                post = (_int_lit_site(result), _pad_site(), _pad_site())
                out.append(TransitionTerm(
                    rule_id="T-Arith", site_left=site_left,
                    pre=pre, post=post, coupling=coupling,
                ))
    return out


def enumerate_cmp_transitions(site_left: int,
                               coupling: float) -> list[TransitionTerm]:
    """Comparison transitions: BIN(<|==) INT(a) INT(b) → BOOL(result) PAD PAD."""
    out: list[TransitionTerm] = []
    for op_code, op_fn in _CMP_OPS.items():
        for a in range(INT_LIT_MIN, INT_LIT_MAX + 1):
            for b in range(INT_LIT_MIN, INT_LIT_MAX + 1):
                pre = (_bin_site(op_code), _int_lit_site(a), _int_lit_site(b))
                post = (_bool_lit_site(op_fn(a, b)),
                        _pad_site(), _pad_site())
                out.append(TransitionTerm(
                    rule_id="T-Cmp", site_left=site_left,
                    pre=pre, post=post, coupling=coupling,
                ))
    return out


def enumerate_if_transitions(site_left: int,
                              coupling: float) -> list[TransitionTerm]:
    """If-with-literal-condition + literal branches.

    Window: 4 sites.
      pre  = (IF, BOOL(t/f), then_lit, else_lit)
      post = (chosen_lit, PAD, PAD, PAD)

    Enumerate over (cond ∈ {T, F}) × (then_lit, else_lit) ∈ literals.
    Both literals can be IntLit or BoolLit; emit both kinds.
    """
    out: list[TransitionTerm] = []
    # Int branches.
    for then_n in range(INT_LIT_MIN, INT_LIT_MAX + 1):
        for else_n in range(INT_LIT_MIN, INT_LIT_MAX + 1):
            for cond in (True, False):
                chosen = _int_lit_site(then_n if cond else else_n)
                pre = (_if_site(), _bool_lit_site(cond),
                       _int_lit_site(then_n), _int_lit_site(else_n))
                post = (chosen, _pad_site(), _pad_site(), _pad_site())
                out.append(TransitionTerm(
                    rule_id="T-If", site_left=site_left,
                    pre=pre, post=post, coupling=coupling,
                ))
    # Bool branches.
    for then_b in (True, False):
        for else_b in (True, False):
            for cond in (True, False):
                chosen = _bool_lit_site(then_b if cond else else_b)
                pre = (_if_site(), _bool_lit_site(cond),
                       _bool_lit_site(then_b), _bool_lit_site(else_b))
                post = (chosen, _pad_site(), _pad_site(), _pad_site())
                out.append(TransitionTerm(
                    rule_id="T-If", site_left=site_left,
                    pre=pre, post=post, coupling=coupling,
                ))
    return out
