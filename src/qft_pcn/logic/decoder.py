"""AST decoder — recover the AST from an encoded MPS.

For a product (concrete) input, decoder is deterministic argmax.
For a hole-bearing (superposed) input, use sample() (added later).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin,
    Ty, TInt, TBool, TArrow,
)
from .encoding import (
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF, D_LOCAL,
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB,
    TYPE_ARR_NESTED, TYPE_NONE,
    BID_NONE,
    BIN_OP_FROM_VALUE,
    INT_LIT_OFFSET,
    EncodingMeta, DecodeError,
)
from src.qft_pcn.qft.mps import MPS


_FLAT_ARROW_TY_FROM_TAG = {
    TYPE_ARR_II: TArrow(src=TInt(), dst=TInt()),
    TYPE_ARR_IB: TArrow(src=TInt(), dst=TBool()),
    TYPE_ARR_BI: TArrow(src=TBool(), dst=TInt()),
    TYPE_ARR_BB: TArrow(src=TBool(), dst=TBool()),
}
_LEAF_TY_FROM_TAG = {
    TYPE_INT: TInt(),
    TYPE_BOOL: TBool(),
}


@dataclass
class DecodeResult:
    ast: Node
    residual_norm: float


def _site_marginal(state: MPS, site: int) -> np.ndarray:
    """Marginal probability over local basis at the requested site,
    computed by canonicalizing the orthogonality center there.
    """
    N = state.N
    ts = [t.copy() for t in state.tensors]
    for k in range(site + 1):
        chi_l, d, chi_r = ts[k].shape
        mat = ts[k].reshape(chi_l * d, chi_r)
        Q, R = np.linalg.qr(mat)
        ts[k] = Q.reshape(chi_l, d, Q.shape[1])
        if k + 1 < N:
            ts[k + 1] = np.einsum('rs,sdt->rdt', R, ts[k + 1], optimize='greedy')
    for k in range(N - 1, site, -1):
        chi_l, d, chi_r = ts[k].shape
        mat = ts[k].reshape(chi_l, d * chi_r)
        Q, R = np.linalg.qr(mat.conj().T)
        Q = Q.conj().T
        R = R.conj().T
        ts[k] = Q.reshape(Q.shape[0], d, chi_r)
        if k - 1 >= 0:
            ts[k - 1] = np.einsum('rds,st->rdt', ts[k - 1], R, optimize='greedy')
    A = ts[site]
    p = (np.abs(A) ** 2).sum(axis=(0, 2))
    total = p.sum()
    if total > 1e-15:
        p = p / total
    return p


def _decompose_basis_index(flat: int) -> tuple[int, int, int, int, int]:
    """Inverse of (k, t, b, v, o) -> flat index. Leftmost species slowest;
    tobl is the innermost (fastest) species.
    """
    o = flat % TOBL_CUTOFF
    flat //= TOBL_CUTOFF
    v = flat % VALUE_CUTOFF
    flat //= VALUE_CUTOFF
    b = flat % BID_CUTOFF
    flat //= BID_CUTOFF
    t = flat % TYPE_CUTOFF
    k = flat // TYPE_CUTOFF
    return (k, t, b, v, o)


def _argmax_site_basis(state: MPS, site: int
                       ) -> tuple[int, int, int, int, int, float]:
    p = _site_marginal(state, site)
    flat = int(np.argmax(p))
    p_max = float(p[flat])
    k, t, b, v, o = _decompose_basis_index(flat)
    return (k, t, b, v, o, 1.0 - p_max)


def _type_from_tag(tag: int, site: int,
                   nested_table: dict[int, Ty]) -> Ty:
    if tag in _FLAT_ARROW_TY_FROM_TAG:
        return _FLAT_ARROW_TY_FROM_TAG[tag]
    if tag in _LEAF_TY_FROM_TAG:
        return _LEAF_TY_FROM_TAG[tag]
    if tag == TYPE_ARR_NESTED:
        if site not in nested_table:
            raise DecodeError(
                f"site {site} has TYPE_ARR_NESTED but nested_type_index has no entry"
            )
        return nested_table[site]
    if tag == TYPE_NONE:
        return TInt()
    raise DecodeError(f"unknown type tag {tag} at site {site}")


def decode(state: MPS, meta: EncodingMeta) -> DecodeResult:
    """Deterministic argmax decode.

    Uses a single right-canonicalization sweep followed by a left-to-right
    walk that projects each site onto its argmax basis state and folds the
    resulting boundary vector into the next site (the same trick as
    `_sample_one_pass`). This is O(N * d * chi^2), in contrast to the
    naive per-site marginal which re-canonicalizes the chain for each
    site (O(N^2 * d * chi^2)).
    """
    N = meta.N
    ts = [t.copy() for t in state.tensors]
    # Right-canonicalize the entire chain so orthogonality center is at site 0.
    for k in range(N - 1, 0, -1):
        chi_l, d, chi_r = ts[k].shape
        mat = ts[k].reshape(chi_l, d * chi_r)
        Q, R = np.linalg.qr(mat.conj().T)
        Q = Q.conj().T
        R = R.conj().T
        ts[k] = Q.reshape(Q.shape[0], d, chi_r)
        ts[k - 1] = np.einsum('rds,st->rdt', ts[k - 1], R, optimize='greedy')
    decoded_sites: list[tuple[int, int, int, int, int]] = []
    residual_acc = 0.0
    for k in range(N):
        A = ts[k]
        p = (np.abs(A) ** 2).sum(axis=(0, 2))
        total = p.sum()
        if total > 1e-15:
            p = p / total
        flat = int(np.argmax(p))
        p_max = float(p[flat])
        residual_acc = max(residual_acc, 1.0 - p_max)
        ki, ti, bi, vi, oi = _decompose_basis_index(flat)
        decoded_sites.append((ki, ti, bi, vi, oi))
        # Project onto chosen basis state and normalize, then propagate
        # the resulting left boundary vector into site k+1.
        proj = A[:, flat, :]
        nrm = np.linalg.norm(proj)
        if nrm > 1e-15:
            proj = proj / nrm
        ts[k] = proj.reshape(A.shape[0], 1, A.shape[2])
        if k + 1 < N:
            left_vec = ts[k][:, 0, :]
            ts[k + 1] = np.einsum('lr,rds->lds', left_vec, ts[k + 1],
                                  optimize='greedy')

    pos = [0]
    binder_stack: list[Lam] = []
    name_counter = [0]

    def _fresh_name() -> str:
        n = name_counter[0]
        name_counter[0] += 1
        return f"_v{n}"

    def _parse_one() -> Node:
        if pos[0] >= meta.N:
            raise DecodeError("ran out of sites mid-parse")
        site_idx = pos[0]
        ki, ti, bi, vi, _oi = decoded_sites[site_idx]
        pos[0] += 1
        if ki == KIND_PAD:
            raise DecodeError(f"unexpected PAD at site {site_idx}")
        if ki == KIND_VAR:
            depth = bi - 1
            if depth < 0 or depth >= len(binder_stack):
                raise DecodeError(
                    f"site {site_idx}: VAR with bid={bi} (depth {depth}) "
                    f"but stack has {len(binder_stack)} binders"
                )
            target_lam = binder_stack[-1 - depth]
            return Var(name=target_lam.param)
        if ki == KIND_LAM:
            ty = _type_from_tag(ti, site_idx, meta.nested_type_index)
            param_ty = ty.src if isinstance(ty, TArrow) else TInt()
            name = _fresh_name()
            lam = Lam(param=name, param_ty=param_ty, body=Var(name=name))
            binder_stack.append(lam)
            body = _parse_one()
            binder_stack.pop()
            lam.body = body
            return lam
        if ki == KIND_APP:
            fn = _parse_one()
            arg = _parse_one()
            return App(fn=fn, arg=arg)
        if ki == KIND_INT:
            return IntLit(val=vi - INT_LIT_OFFSET)
        if ki == KIND_BOOL:
            return BoolLit(val=(vi == 1))
        if ki == KIND_IF:
            c = _parse_one(); a = _parse_one(); b = _parse_one()
            return If(cond=c, then_b=a, else_b=b)
        if ki == KIND_BIN:
            op = BIN_OP_FROM_VALUE.get(vi)
            if op is None:
                raise DecodeError(f"site {site_idx}: unknown bin op value {vi}")
            l = _parse_one(); r = _parse_one()
            return Bin(op=op, lhs=l, rhs=r)
        raise DecodeError(f"site {site_idx}: unknown kind {ki}")

    ast = _parse_one()

    while pos[0] < meta.N:
        ki, _, _, _, _ = decoded_sites[pos[0]]
        if ki != KIND_PAD:
            raise DecodeError(
                f"site {pos[0]} not PAD after AST parse (kind={ki})"
            )
        pos[0] += 1

    return DecodeResult(ast=ast, residual_norm=residual_acc)


# ---- alpha-equivalence helper ---------------------------------------------


def ast_alpha_eq(a: Node, b: Node) -> bool:
    """Structural equality of two ASTs modulo alpha-renaming."""
    return _alpha_eq(a, b, env_a={}, env_b={}, counter=[0])


def _alpha_eq(a: Node, b: Node, env_a: dict[str, int],
              env_b: dict[str, int], counter: list[int]) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, Var):
        sa = env_a.get(a.name)
        sb = env_b.get(b.name)
        if sa is None and sb is None:
            return a.name == b.name
        return sa == sb
    if isinstance(a, IntLit):
        return a.val == b.val
    if isinstance(a, BoolLit):
        return a.val == b.val
    if isinstance(a, Lam):
        if a.param_ty != b.param_ty:
            return False
        slot = counter[0]; counter[0] += 1
        ea = dict(env_a); eb = dict(env_b)
        ea[a.param] = slot; eb[b.param] = slot
        return _alpha_eq(a.body, b.body, ea, eb, counter)
    if isinstance(a, App):
        return (_alpha_eq(a.fn, b.fn, env_a, env_b, counter)
                and _alpha_eq(a.arg, b.arg, env_a, env_b, counter))
    if isinstance(a, If):
        return (_alpha_eq(a.cond, b.cond, env_a, env_b, counter)
                and _alpha_eq(a.then_b, b.then_b, env_a, env_b, counter)
                and _alpha_eq(a.else_b, b.else_b, env_a, env_b, counter))
    if isinstance(a, Bin):
        return (a.op == b.op
                and _alpha_eq(a.lhs, b.lhs, env_a, env_b, counter)
                and _alpha_eq(a.rhs, b.rhs, env_a, env_b, counter))
    return False


# ---- conditional sampling ------------------------------------------------


def sample(state: MPS, meta: EncodingMeta,
           n_samples: int = 1,
           rng: Optional[np.random.Generator] = None
           ) -> list[DecodeResult]:
    """Sample n_samples ASTs from the MPS distribution via left-to-right
    conditional measurement."""
    if rng is None:
        rng = np.random.default_rng()
    results: list[DecodeResult] = []
    for _ in range(n_samples):
        flat_indices = _sample_one_pass(state, meta, rng)
        results.append(_decode_from_indices(flat_indices, meta))
    return results


def _sample_one_pass(state: MPS, meta: EncodingMeta,
                     rng: np.random.Generator) -> list[int]:
    """Standard left-to-right MPS sampling."""
    N = meta.N
    ts = [t.copy() for t in state.tensors]
    sampled: list[int] = []
    # Right-canonicalize the entire chain so orthogonality center is at site 0.
    for k in range(N - 1, 0, -1):
        chi_l, d, chi_r = ts[k].shape
        mat = ts[k].reshape(chi_l, d * chi_r)
        Q, R = np.linalg.qr(mat.conj().T)
        Q = Q.conj().T
        R = R.conj().T
        ts[k] = Q.reshape(Q.shape[0], d, chi_r)
        ts[k - 1] = np.einsum('rds,st->rdt', ts[k - 1], R, optimize='greedy')
    for k in range(N):
        A = ts[k]
        p = (np.abs(A) ** 2).sum(axis=(0, 2))
        total = p.sum()
        if total <= 1e-15:
            s = 0
        else:
            p = p / total
            s = int(rng.choice(D_LOCAL, p=p))
        sampled.append(s)
        proj = A[:, s, :]
        norm = np.linalg.norm(proj)
        if norm > 1e-15:
            proj = proj / norm
        ts[k] = proj.reshape(A.shape[0], 1, A.shape[2])
        if k + 1 < N:
            left_vec = ts[k][:, 0, :]
            ts[k + 1] = np.einsum('lr,rds->lds', left_vec, ts[k + 1],
                                  optimize='greedy')
    return sampled


def _decode_from_indices(flat_indices: list[int],
                         meta: EncodingMeta) -> DecodeResult:
    """Build an AST from a sampled list of local-basis indices."""
    decoded_sites = [_decompose_basis_index(f) for f in flat_indices]

    pos = [0]
    binder_stack: list[Lam] = []
    name_counter = [0]

    def _fresh_name() -> str:
        n = name_counter[0]; name_counter[0] += 1
        return f"_v{n}"

    def _parse_one() -> Node:
        if pos[0] >= meta.N:
            raise DecodeError("ran out of sites mid-parse (sample)")
        site_idx = pos[0]
        ki, ti, bi, vi, _oi = decoded_sites[site_idx]
        pos[0] += 1
        if ki == KIND_PAD:
            raise DecodeError(f"unexpected PAD at site {site_idx} (sample)")
        if ki == KIND_VAR:
            depth = bi - 1
            if depth < 0 or depth >= len(binder_stack):
                raise DecodeError(
                    f"site {site_idx} (sample): VAR with bid={bi}, "
                    f"stack size {len(binder_stack)}")
            target_lam = binder_stack[-1 - depth]
            return Var(name=target_lam.param)
        if ki == KIND_LAM:
            ty = _type_from_tag(ti, site_idx, meta.nested_type_index)
            param_ty = ty.src if isinstance(ty, TArrow) else TInt()
            name = _fresh_name()
            lam = Lam(param=name, param_ty=param_ty, body=Var(name=name))
            binder_stack.append(lam)
            body = _parse_one()
            binder_stack.pop()
            lam.body = body
            return lam
        if ki == KIND_APP:
            fn = _parse_one(); arg = _parse_one()
            return App(fn=fn, arg=arg)
        if ki == KIND_INT:
            return IntLit(val=vi - INT_LIT_OFFSET)
        if ki == KIND_BOOL:
            return BoolLit(val=(vi == 1))
        if ki == KIND_IF:
            c = _parse_one(); a = _parse_one(); b = _parse_one()
            return If(cond=c, then_b=a, else_b=b)
        if ki == KIND_BIN:
            op = BIN_OP_FROM_VALUE.get(vi)
            if op is None:
                raise DecodeError(
                    f"site {site_idx} (sample): unknown bin op value {vi}")
            l = _parse_one(); r = _parse_one()
            return Bin(op=op, lhs=l, rhs=r)
        raise DecodeError(f"site {site_idx} (sample): unknown kind {ki}")

    ast = _parse_one()
    while pos[0] < meta.N:
        ki, _, _, _, _ = decoded_sites[pos[0]]
        if ki != KIND_PAD:
            raise DecodeError(
                f"site {pos[0]} (sample) not PAD after parse, kind={ki}"
            )
        pos[0] += 1
    return DecodeResult(ast=ast, residual_norm=0.0)
