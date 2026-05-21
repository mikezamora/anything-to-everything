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
    KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, D_LOCAL,
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
            ts[k + 1] = np.einsum('rs,sdt->rdt', R, ts[k + 1])
    for k in range(N - 1, site, -1):
        chi_l, d, chi_r = ts[k].shape
        mat = ts[k].reshape(chi_l, d * chi_r)
        Q, R = np.linalg.qr(mat.conj().T)
        Q = Q.conj().T
        R = R.conj().T
        ts[k] = Q.reshape(Q.shape[0], d, chi_r)
        if k - 1 >= 0:
            ts[k - 1] = np.einsum('rds,st->rdt', ts[k - 1], R)
    A = ts[site]
    p = (np.abs(A) ** 2).sum(axis=(0, 2))
    total = p.sum()
    if total > 1e-15:
        p = p / total
    return p


def _decompose_basis_index(flat: int) -> tuple[int, int, int, int]:
    v = flat % VALUE_CUTOFF
    flat //= VALUE_CUTOFF
    b = flat % BID_CUTOFF
    flat //= BID_CUTOFF
    t = flat % TYPE_CUTOFF
    k = flat // TYPE_CUTOFF
    return (k, t, b, v)


def _argmax_site_basis(state: MPS, site: int
                       ) -> tuple[int, int, int, int, float]:
    p = _site_marginal(state, site)
    flat = int(np.argmax(p))
    p_max = float(p[flat])
    k, t, b, v = _decompose_basis_index(flat)
    return (k, t, b, v, 1.0 - p_max)


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
    """Deterministic argmax decode."""
    decoded_sites: list[tuple[int, int, int, int]] = []
    residual_acc = 0.0
    for k in range(meta.N):
        ki, ti, bi, vi, residual = _argmax_site_basis(state, k)
        decoded_sites.append((ki, ti, bi, vi))
        residual_acc = max(residual_acc, residual)

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
        ki, ti, bi, vi = decoded_sites[site_idx]
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
        ki, _, _, _ = decoded_sites[pos[0]]
        if ki != KIND_PAD:
            raise DecodeError(
                f"site {pos[0]} not PAD after AST parse (kind={ki})"
            )
        pos[0] += 1

    return DecodeResult(ast=ast, residual_norm=residual_acc)
