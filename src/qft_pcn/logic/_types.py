"""Bottom-up type computation for each site of the serialized AST.

Maps a Ty to one of the eight flat tags in encoding.py, with TYPE_ARR_NESTED
as the overflow slot (the full Ty is then stored in EncodingMeta's
nested_type_index side table).

This module does NOT do type-checking — sub-project B owns that. Here we
trust the input AST is well-typed and compute the type of each subexpression.
"""

from __future__ import annotations

from typing import Optional

from .ast import (Node, Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar,
                  Ty, TInt, TBool, TArrow, TypeHole, TPi,
                  Zero, Succ, NatLit, Nil, Cons, Eq, TNat, TList, TProp,
                  Forall, Fix)
from ._serialize import NodeOccupancy
from .encoding import (
    KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL, KIND_IF, KIND_BIN,
    KIND_PAD,
    TYPE_NONE, TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
)


_FLAT_ARROW_TAGS = {
    (TYPE_INT, TYPE_INT): TYPE_ARR_II,
    (TYPE_INT, TYPE_BOOL): TYPE_ARR_IB,
    (TYPE_BOOL, TYPE_INT): TYPE_ARR_BI,
    (TYPE_BOOL, TYPE_BOOL): TYPE_ARR_BB,
}


def _resolve_typehole(ty: Ty) -> Ty:
    """Replace a TypeHole with its first candidate; identity otherwise.

    Used by the bottom-up site-type walker so the env carries a concrete
    Ty even when the source Lam/Forall/Fix has a TypeHole param_ty. The
    encoder lifts the result to a per-candidate superposition on the
    affected type leaves.
    """
    if isinstance(ty, TypeHole):
        return ty.candidates[0]
    return ty


def ty_to_tag(ty: Ty) -> tuple[int, Optional[Ty]]:
    """Map a Ty to a flat type tag.

    Returns (tag, nested_full_ty). nested_full_ty is non-None only when the
    tag is TYPE_ARR_NESTED — in which case the caller is responsible for
    storing the full Ty in the meta side table indexed by site.
    """
    if isinstance(ty, TInt):
        return (TYPE_INT, None)
    if isinstance(ty, TBool):
        return (TYPE_BOOL, None)
    if isinstance(ty, TArrow):
        src_tag, _ = ty_to_tag(ty.src)
        dst_tag, _ = ty_to_tag(ty.dst)
        if src_tag in (TYPE_INT, TYPE_BOOL) and dst_tag in (TYPE_INT, TYPE_BOOL):
            return (_FLAT_ARROW_TAGS[(src_tag, dst_tag)], None)
        return (TYPE_ARR_NESTED, ty)
    # Extended-calculus types (Nat / List / Prop) use the 16-slot type
    # register of the MERA leaf (indices 8-11); no nested side table.
    from .mera_encoding import TYPE_NAT, TYPE_LIST, TYPE_PROP
    from .ast import TNat as _TNat, TList as _TList, TProp as _TProp
    if isinstance(ty, _TNat):
        return (TYPE_NAT, None)
    if isinstance(ty, _TList):
        return (TYPE_LIST, None)
    if isinstance(ty, _TProp):
        return (TYPE_PROP, None)
    if isinstance(ty, TPi):
        # Defensive hardening: there is no dedicated "Pi" flat tag — a
        # dependent product is identified at the substrate level by the
        # root kind leaf being KIND_FORALL together with the value-leaf
        # carrying the param-type's tag (see
        # ``tn_typechecker.tn_typecheck_bundle``'s TPi branch, which
        # only calls ``ty_to_tag(expected.src)``). For any future caller
        # that hands a full ``TPi`` to this function, we return the
        # src-tag (the tag the substrate value-leaf must match). The
        # nested-Ty slot is unused: TPi never lives in the
        # ``nested_type_index`` side table (that is reserved for nested
        # ``TArrow``).
        return ty_to_tag(ty.src)
    if isinstance(ty, TypeHole):
        # Defensive: when a TypeHole leaks into ty_to_tag (e.g. as the
        # src/dst of a TArrow built from a Lam whose param_ty is a hole),
        # collapse to the first candidate's flat tag. The MERA encoder
        # treats the TypeHole as a genuine candidate-tag superposition
        # at the appropriate type leaves (see encode_mera typehole path);
        # this branch keeps the bottom-up site-type computation total
        # without erasing the encoder's superposition contract.
        return ty_to_tag(ty.candidates[0])
    raise TypeError(f"unknown Ty: {ty!r}")


def _compute_ast_type(node: Node, env: list[tuple[str, Ty]]) -> Ty:
    if isinstance(node, IntLit):
        return TInt()
    if isinstance(node, BoolLit):
        return TBool()
    if isinstance(node, Var):
        for name, t in reversed(env):
            if name == node.name:
                return t
        raise KeyError(f"unbound {node.name}")
    if isinstance(node, HoleVar):
        # A HoleVar takes the type of its (first) candidate binder. All
        # candidates are assumed type-compatible at this layer.
        if not node.candidates:
            return TInt()
        first = node.candidates[0]
        for name, t in reversed(env):
            if name == first:
                return t
        raise KeyError(f"unbound HoleVar candidate {first}")
    if isinstance(node, Lam):
        # If param_ty is a TypeHole, use its first candidate as the
        # "default" type for bottom-up computation. The encoder lifts
        # the result to a genuine candidate superposition on the type
        # leaves; this default keeps the site-type pipeline total.
        param_ty = node.param_ty
        if isinstance(param_ty, TypeHole):
            param_ty = param_ty.candidates[0]
        body_ty = _compute_ast_type(node.body,
                                    env + [(node.param, param_ty)])
        return TArrow(src=param_ty, dst=body_ty)
    if isinstance(node, App):
        fn_ty = _compute_ast_type(node.fn, env)
        if isinstance(fn_ty, TArrow):
            return fn_ty.dst
        # Ill-typed input: fall back to TInt so encoder doesn't crash.
        # Sub-project B will reject this at the Hamiltonian level.
        return TInt()
    if isinstance(node, If):
        return _compute_ast_type(node.then_b, env)
    if isinstance(node, Bin):
        if node.op in ("+", "-", "*"):
            return TInt()
        if node.op in ("<", "=="):
            return TBool()
        raise ValueError(f"unknown op {node.op}")
    # --- extended-calculus nodes ---
    if isinstance(node, (Zero, Succ, NatLit)):
        return TNat()
    if isinstance(node, Nil):
        return TList(elem=TNat())
    if isinstance(node, Cons):
        return TList(elem=_compute_ast_type(node.head, env))
    if isinstance(node, Eq):
        return TProp()
    if isinstance(node, Forall):
        return TProp()
    if isinstance(node, Fix):
        # The fixed point has the same type as the recursion variable.
        param_ty = node.param_ty
        if isinstance(param_ty, TypeHole):
            param_ty = param_ty.candidates[0]
        return param_ty
    raise TypeError(f"unknown Node: {type(node).__name__}")


def compute_site_types(root: Node,
                       sites: list[NodeOccupancy]
                       ) -> list[int]:
    """For each site, compute the flat type tag.

    Returns a list of length N. The full Ty for any TYPE_ARR_NESTED entry
    is stored on the corresponding NodeOccupancy.ty attribute so the
    encoder can populate meta.nested_type_index.
    """
    # First, walk the AST once and gather a (ast_path -> Ty) map.
    path_to_ty: dict[tuple[int, ...], Ty] = {}

    def _walk(node: Node, env: list[tuple[str, Ty]], path: tuple[int, ...]):
        ty = _compute_ast_type(node, env)
        path_to_ty[path] = ty
        if isinstance(node, Lam):
            _walk(node.body, env + [(node.param, _resolve_typehole(node.param_ty))],
                  path + (0,))
        elif isinstance(node, App):
            _walk(node.fn, env, path + (0,))
            _walk(node.arg, env, path + (1,))
        elif isinstance(node, If):
            _walk(node.cond, env, path + (0,))
            _walk(node.then_b, env, path + (1,))
            _walk(node.else_b, env, path + (2,))
        elif isinstance(node, Bin):
            _walk(node.lhs, env, path + (0,))
            _walk(node.rhs, env, path + (1,))
        elif isinstance(node, Succ):
            _walk(node.arg, env, path + (0,))
        elif isinstance(node, Cons):
            _walk(node.head, env, path + (0,))
            _walk(node.tail, env, path + (1,))
        elif isinstance(node, Eq):
            _walk(node.lhs, env, path + (0,))
            _walk(node.rhs, env, path + (1,))
        elif isinstance(node, (Forall, Fix)):
            _walk(node.body, env + [(node.param, _resolve_typehole(node.param_ty))],
                  path + (0,))

    _walk(root, [], ())

    # Now produce flat tags per site, in serialization order.
    tags: list[int] = []
    for occ in sites:
        if occ.kind == KIND_PAD:
            tags.append(TYPE_NONE)
            occ.ty = None
            continue
        ty = path_to_ty.get(occ.ast_path)
        if ty is None:
            tags.append(TYPE_NONE)
            occ.ty = None
            continue
        tag, nested = ty_to_tag(ty)
        tags.append(tag)
        # Keep the full Ty on the occupancy when nested or for the encoder
        # to use as needed.
        occ.ty = nested if nested is not None else ty
    return tags
