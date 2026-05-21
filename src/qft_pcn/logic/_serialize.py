"""Pre-order serialization of AST to a list of NodeOccupancy descriptors.

This produces the per-site descriptor list. It does NOT yet produce MPS
tensors — that's the encoder's job, using these descriptors as input.

Walks the AST in canonical order:
  Lam: visit self, then body
  App: visit self, then fn, then arg
  If: visit self, then cond, then then_b, then else_b
  Bin: visit self, then lhs, then rhs
  Var, IntLit, BoolLit: leaves (visit self only)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .ast import Node, Var, Lam, App, IntLit, BoolLit, If, Bin, Ty
from .encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN, BID_NONE,
    EncodingTooLarge, UnsupportedNode,
)
from ._resolve import resolve_binders


@dataclass
class BinderRef:
    """Attached to LAM sites; identifies the binder this Lam introduces."""
    lam_node: Lam
    lexical_depth: int   # 0 = outermost binder in the program


@dataclass
class VarRef:
    """Attached to VAR sites; identifies the resolved binder."""
    binder_site: int                 # site index of the binding LAM
    depth_from_innermost: int        # at the use site


@dataclass
class NodeOccupancy:
    """Per-site descriptor produced by the serializer.

    The descriptor records *what* the site holds; the encoder turns these
    into actual MPS tensors. binder_ref / var_ref are non-None only for
    KIND_LAM / KIND_VAR sites respectively.
    """
    kind: int
    ty: Optional[Ty] = None        # the type of the expression at this site
    int_val: Optional[int] = None  # for KIND_INT
    bool_val: Optional[bool] = None  # for KIND_BOOL
    bin_op: Optional[str] = None   # for KIND_BIN
    binder_ref: Optional[BinderRef] = None
    var_ref: Optional[VarRef] = None
    ast_path: tuple[int, ...] = ()  # path from root, for diagnostics


def count_nodes(root: Node) -> int:
    """Count AST nodes (used by callers to pre-check N)."""
    if isinstance(root, (Var, IntLit, BoolLit)):
        return 1
    if isinstance(root, Lam):
        return 1 + count_nodes(root.body)
    if isinstance(root, App):
        return 1 + count_nodes(root.fn) + count_nodes(root.arg)
    if isinstance(root, If):
        return (1 + count_nodes(root.cond) + count_nodes(root.then_b)
                + count_nodes(root.else_b))
    if isinstance(root, Bin):
        return 1 + count_nodes(root.lhs) + count_nodes(root.rhs)
    raise UnsupportedNode(node_type=type(root).__name__)


def serialize_preorder(root: Node, N: int) -> list[NodeOccupancy]:
    """Pre-order walk of root, producing N descriptors (PAD-padded).

    Resolves every Var first (via resolve_binders) so VarRef can carry the
    binder's site index. Raises EncodingTooLarge if the AST has more than N
    non-PAD nodes.
    """
    n_nodes = count_nodes(root)
    if n_nodes > N:
        raise EncodingTooLarge(n_nodes=n_nodes, N=N)

    # First pass: resolve binders to ResolvedRef objects. We need to map
    # each Lam node to its site index, which we only learn during the
    # second-pass walk. So we collect Lam->ResolvedRef during the first
    # pass (by node identity), then look up in the second pass.
    var_refs: dict[int, "ResolvedRef"] = {}   # id(Var node) -> ResolvedRef
    def _capture(var: Var, ref) -> None:
        var_refs[id(var)] = ref
    resolve_binders(root, on_var=_capture)

    # Second pass: pre-order traversal, recording site indices for Lams as
    # we go so VarRef can point back to them.
    sites: list[NodeOccupancy] = []
    lam_to_site: dict[int, int] = {}      # id(Lam) -> site index
    binder_depth_stack: list[int] = []    # tracks lexical depth on the walk

    def _emit(node: Node, ast_path: tuple[int, ...]) -> None:
        idx = len(sites)
        if isinstance(node, Var):
            ref = var_refs[id(node)]
            sites.append(NodeOccupancy(
                kind=KIND_VAR,
                var_ref=VarRef(
                    binder_site=lam_to_site[id(ref.binder)],
                    depth_from_innermost=ref.depth_from_innermost,
                ),
                ast_path=ast_path,
            ))
            return
        if isinstance(node, Lam):
            depth = len(binder_depth_stack)
            lam_to_site[id(node)] = idx
            sites.append(NodeOccupancy(
                kind=KIND_LAM,
                binder_ref=BinderRef(lam_node=node, lexical_depth=depth),
                ast_path=ast_path,
            ))
            binder_depth_stack.append(depth)
            _emit(node.body, ast_path + (0,))
            binder_depth_stack.pop()
            return
        if isinstance(node, App):
            sites.append(NodeOccupancy(kind=KIND_APP, ast_path=ast_path))
            _emit(node.fn, ast_path + (0,))
            _emit(node.arg, ast_path + (1,))
            return
        if isinstance(node, If):
            sites.append(NodeOccupancy(kind=KIND_IF, ast_path=ast_path))
            _emit(node.cond, ast_path + (0,))
            _emit(node.then_b, ast_path + (1,))
            _emit(node.else_b, ast_path + (2,))
            return
        if isinstance(node, Bin):
            sites.append(NodeOccupancy(kind=KIND_BIN, bin_op=node.op,
                                       ast_path=ast_path))
            _emit(node.lhs, ast_path + (0,))
            _emit(node.rhs, ast_path + (1,))
            return
        if isinstance(node, IntLit):
            sites.append(NodeOccupancy(kind=KIND_INT, int_val=node.val,
                                       ast_path=ast_path))
            return
        if isinstance(node, BoolLit):
            sites.append(NodeOccupancy(kind=KIND_BOOL, bool_val=node.val,
                                       ast_path=ast_path))
            return
        raise UnsupportedNode(node_type=type(node).__name__)

    _emit(root, ())

    # Pad to N with PAD descriptors.
    while len(sites) < N:
        sites.append(NodeOccupancy(kind=KIND_PAD))

    return sites
