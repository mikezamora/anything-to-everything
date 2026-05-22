"""Encoder extensions for sub-project E (synthesis).

Two responsibilities:

1. ``encode_synthesis(sketch, N, chi_max)``: encode a sketch that may
   contain ``TypeHole`` nodes. Wraps A's ``encode()`` and applies a
   FACTORED type-register projector at each TypeHole site, lifting that
   site's type register from a definite tag to an equal-amplitude
   superposition over the candidate tags. Per spec §5.3.

2. ``witness_augmented_sketch(sketch, examples)``: build the augmented
   AST including per-example witness regions (classical-copy form, per
   plan Task 8 note — spec §5.5 prefers RefVar; classical copy preserves
   the publishability claim while being implementable).

Manifesto Temptation 3: We do NOT materialize the dense
(D_LOCAL, D_LOCAL) = (65536, 65536) gate via ``embed_op``. The type
register lives at a known stride inside the flat local basis, so the
projector is applied as a slice operation on the type axis only.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.qft_pcn.logic.ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin, HoleVar,
    Ty, TInt, TBool, TArrow, TypeHole,
)
from src.qft_pcn.logic.encoding import (
    EncodingMeta, TypeHoleHandle,
    TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB,
    TYPE_ARR_NESTED, TYPE_CUTOFF,
    KIND_CUTOFF, BID_CUTOFF, VALUE_CUTOFF, TOBL_CUTOFF, D_LOCAL,
)
from src.qft_pcn.logic.encoder import encode as _base_encode


# ---- type tagging ----------------------------------------------------------


def _ty_to_tag(t: Ty) -> int:
    """Map a flat Ty to its type-register tag. Raises on deep nested arrows."""
    if isinstance(t, TInt):
        return TYPE_INT
    if isinstance(t, TBool):
        return TYPE_BOOL
    if isinstance(t, TArrow):
        if isinstance(t.src, TInt) and isinstance(t.dst, TInt):
            return TYPE_ARR_II
        if isinstance(t.src, TInt) and isinstance(t.dst, TBool):
            return TYPE_ARR_IB
        if isinstance(t.src, TBool) and isinstance(t.dst, TInt):
            return TYPE_ARR_BI
        if isinstance(t.src, TBool) and isinstance(t.dst, TBool):
            return TYPE_ARR_BB
        return TYPE_ARR_NESTED
    raise ValueError(f"cannot tag {t!r}")


# ---- TypeHole discovery in the sketch -------------------------------------


def _compute_lam_arrow_for_candidate(
    lam: Lam, candidate: Ty,
) -> int:
    """Given a Lam whose param_ty is a TypeHole, return the type-register
    tag that the Lam SITE would carry if the hole were resolved to the
    given candidate. The Lam's site type is TArrow(candidate, body_ty)
    where body_ty is computed from the existing body using the candidate
    as the param's type.

    For body types that themselves depend on the hole, we use the body's
    statically-computed type assuming `candidate` is the param type.
    """
    # Compute body's type with this candidate.
    from src.qft_pcn.logic._types import _compute_ast_type
    try:
        body_ty = _compute_ast_type(lam.body, [(lam.param, candidate)])
    except Exception:
        # Fall back: assume body has the same type as the param.
        body_ty = candidate
    return _ty_to_tag(TArrow(src=candidate, dst=body_ty))


def _find_typeholes_with_paths(
    sketch: Node,
) -> list[tuple[tuple[int, ...], TypeHole, Lam]]:
    """Walk the sketch in pre-order; return list of (ast_path, TypeHole)
    for every TypeHole found in a Lam.param_ty position.

    (TypeHole in deeper positions is out of scope per spec §2.2.)
    """
    out: list[tuple[tuple[int, ...], TypeHole, Lam]] = []

    def go(node: Node, path: tuple[int, ...]) -> None:
        if isinstance(node, Lam):
            if isinstance(node.param_ty, TypeHole):
                out.append((path, node.param_ty, node))
            go(node.body, path + (0,))
        elif isinstance(node, App):
            go(node.fn, path + (0,))
            go(node.arg, path + (1,))
        elif isinstance(node, If):
            go(node.cond, path + (0,))
            go(node.then_b, path + (1,))
            go(node.else_b, path + (2,))
        elif isinstance(node, Bin):
            go(node.lhs, path + (0,))
            go(node.rhs, path + (1,))

    go(sketch, ())
    return out


def _expand_empty_holevar_candidates(sketch: Node) -> Node:
    """Walk the AST; replace any HoleVar(candidates=[]) with a HoleVar
    whose candidates list is every in-scope binder name (spec §5.2:
    'empty = any in-scope').
    """
    def go(node: Node, env: list[str]) -> Node:
        if isinstance(node, HoleVar):
            if not node.candidates:
                # Use all in-scope binders.
                cands = list(env)
                if not cands:
                    # No binders in scope; leave as-is (will error later).
                    return node
                return HoleVar(
                    candidates=cands,
                    target_type=node.target_type,
                    name=node.name,
                )
            return node
        if isinstance(node, Lam):
            return Lam(param=node.param, param_ty=node.param_ty,
                       body=go(node.body, env + [node.param]))
        if isinstance(node, App):
            return App(fn=go(node.fn, env), arg=go(node.arg, env))
        if isinstance(node, If):
            return If(cond=go(node.cond, env),
                      then_b=go(node.then_b, env),
                      else_b=go(node.else_b, env))
        if isinstance(node, Bin):
            return Bin(op=node.op,
                       lhs=go(node.lhs, env), rhs=go(node.rhs, env))
        return node

    return go(sketch, [])


def _substitute_typeholes_with_first_candidate(sketch: Node) -> Node:
    """Return a copy of sketch with every TypeHole replaced by its first
    candidate. Used to obtain a 'definite' sketch for A's base encoder.
    """
    def fix_ty(t: Ty) -> Ty:
        if isinstance(t, TypeHole):
            return t.candidates[0]
        if isinstance(t, TArrow):
            return TArrow(src=fix_ty(t.src), dst=fix_ty(t.dst))
        return t

    def go(node: Node) -> Node:
        if isinstance(node, Lam):
            return Lam(param=node.param,
                       param_ty=fix_ty(node.param_ty),
                       body=go(node.body))
        if isinstance(node, App):
            return App(fn=go(node.fn), arg=go(node.arg))
        if isinstance(node, If):
            return If(cond=go(node.cond),
                      then_b=go(node.then_b),
                      else_b=go(node.else_b))
        if isinstance(node, Bin):
            return Bin(op=node.op, lhs=go(node.lhs), rhs=go(node.rhs))
        return node

    return go(sketch)


def _path_to_site(path: tuple[int, ...], meta: EncodingMeta) -> int:
    """Reverse lookup: site index given an AST path."""
    for site, p in meta.site_to_ast_path.items():
        if p == path:
            return site
    raise KeyError(f"AST path {path!r} not in meta.site_to_ast_path")


# ---- Factored type-register superposition application --------------------
#
# Flat basis ordering (from _tensors._basis_index):
#   flat = ((((k * T + t) * B + b) * V + v) * O + o)
# So the type-register stride is (B * V * O), and within a fixed (k, b, v, o)
# the t-index walks contiguous flat slots of stride (B*V*O).
#
# We rebuild A_out[:, flat, :] = sum_t M_type[t', t] * A[:, flat_at_t, :]
# where flat_at_t = flat with t-component swapped. This costs
# O(chi^2 * K * T^2 * B * V * O) and never materializes a (D_LOCAL, D_LOCAL)
# operator.

_T_STRIDE = BID_CUTOFF * VALUE_CUTOFF * TOBL_CUTOFF
_K_STRIDE = TYPE_CUTOFF * _T_STRIDE


def _apply_type_factored(state, site: int,
                         candidate_tags: tuple[int, ...]) -> None:
    """Apply the type-register superposition projector at `site` IN PLACE.

    Projector: |ψ⟩ = (1/√k) Σ_{c ∈ candidates} |c⟩⟨tag_0| on the type
    register; identity on every other species. The "source" tag tag_0 is
    the FIRST candidate (which is what the base encoder wrote when we
    substituted TypeHoles with their first candidate).
    """
    k = len(candidate_tags)
    amp = 1.0 / np.sqrt(k)
    src_tag = candidate_tags[0]

    A = state.tensors[site]    # (chi_l, D_LOCAL, chi_r)
    chi_l, d, chi_r = A.shape
    assert d == D_LOCAL, f"site {site} has d={d}, expected {D_LOCAL}"

    A_out = np.zeros_like(A)

    # The projector takes amplitude at type-index src_tag and DISTRIBUTES
    # it equally to every candidate tag (and zeros out non-candidate t).
    # All other (k, b, v, o) slots passthrough identity — but the type
    # register is currently in a definite state at src_tag for sites where
    # the type was computed from the TypeHole; for ALL OTHER type values
    # the projector acts as zero. To preserve the rest of the chain we
    # apply it as: for each non-src type value, leave A unchanged (it was
    # zero on this site for the substituted sketch anyway); for src_tag,
    # redistribute into the candidate slots.
    #
    # Concretely: write the action as the rank-(k) operator
    #   M = (amp) * Σ_{c} |c⟩⟨src_tag|
    # which is a partial isometry. Result:
    #   A_out[:, flat(k, c, b, v, o), :] = amp * A[:, flat(k, src_tag, b, v, o), :]
    # for c in candidates; zero elsewhere.
    #
    # That zeros out all OTHER type-register components. That's correct
    # for the definite-sketch state, where the type register at the hole
    # site was a basis vector at src_tag, so all other type slots are
    # already zero — A_out reproduces them as zero.
    for K_idx in range(KIND_CUTOFF):
        base_k = K_idx * _K_STRIDE
        for c in candidate_tags:
            flat_dst_base = base_k + c * _T_STRIDE
            flat_src_base = base_k + src_tag * _T_STRIDE
            # Copy the entire (B*V*O) block for this kind, scaled by amp.
            A_out[:, flat_dst_base:flat_dst_base + _T_STRIDE, :] += (
                amp * A[:, flat_src_base:flat_src_base + _T_STRIDE, :]
            )

    state.tensors[site] = A_out
    state.normalize()


# ---- Public encode_synthesis ----------------------------------------------


def encode_synthesis(sketch: Node, N: int = 32, chi_max: int = 32):
    """Encode a synthesis sketch (may contain TypeHole) into an MPS.

    Pipeline (spec §5.3):
        1. Find all TypeHoles in sketch (Lam.param_ty positions only).
        2. Substitute each with its first candidate to produce a definite
           sketch.
        3. Run A's encode() on the definite sketch.
        4. For each TypeHole, look up its site via meta.site_to_ast_path
           and apply the factored type-register superposition projector.
        5. Record TypeHoleHandle entries in meta.type_holes.

    Returns (state, meta). The state is normalized.

    Note: nested TypeHole inside witness ASTs is not supported (witness
    regions are concrete by construction).
    """
    expanded = _expand_empty_holevar_candidates(sketch)
    typeholes = _find_typeholes_with_paths(expanded)
    definite_sketch = _substitute_typeholes_with_first_candidate(expanded)
    state, meta = _base_encode(definite_sketch, N=N, chi_max=chi_max)

    for path, th, lam in typeholes:
        site = _path_to_site(path, meta)
        # The Lam site's type-register tag is the INDUCED ARROW tag for
        # each candidate: TArrow(candidate, body_ty(candidate)). The
        # site_to_ast_path for the lam points at this site.
        cand_tags_set: set[int] = set()
        for c in th.candidates:
            try:
                cand_tags_set.add(_compute_lam_arrow_for_candidate(lam, c))
            except Exception:
                # Fallback: treat the hole candidate directly.
                cand_tags_set.add(_ty_to_tag(c))
        cand_tags = tuple(sorted(cand_tags_set))
        if len(cand_tags) >= 2:
            _apply_type_factored(state, site, cand_tags)
        meta.type_holes[site] = TypeHoleHandle(
            hole_site=site, candidate_tags=cand_tags,
        )

    return state, meta


# ---- Witness-augmented sketch (classical copy form) -----------------------


def _wrap_in_application_chain(fn_ast: Node,
                               inputs: tuple[Node, ...]) -> Node:
    """Return App(... App(fn_ast, inputs[0]), inputs[1])..."""
    expr: Node = fn_ast
    for arg in inputs:
        expr = App(fn=expr, arg=arg)
    return expr


def witness_augmented_sketch(
    sketch: Node, examples,
) -> tuple[Node, list[tuple[int, int]]]:
    """Build a single AST that encodes ``sketch`` followed by per-example
    witness sub-ASTs (spec §5.5 RefVar-style witness encoding).

    Lattice layout (matching spec §4.2 / §5.5):

        [ sketch_region | witness_1 | witness_2 | ... | witness_K | PAD ]

    The sketch sits at the visible root; each witness is a fresh
    classical copy of the sketch wrapped in the example's application
    chain ``App(... App(sketch_copy, in_0)..., in_{n-1})``. The
    witnesses are stitched after the sketch via an If-cascade
    (``If(True, sketch, If(True, witness_1, ...))``) so the encoder's
    pre-order serializer emits each witness contiguously and the
    visible top-level value remains the sketch.

    The "RefVar form" of spec §5.5 (the witness's ``fn`` subtree
    sharing bid channels with the sketch root) is approximated by
    classical-copy because the QFT/PCN's eval-rule transitions do not
    yet include beta-reduction transitions that would in-lattice
    consume the shared ``fn``. Beta-reduction transitions are a
    follow-up; the classical-copy form here is correct (every site is
    a valid AST occupancy) and the H_examples ranking-mode evaluation
    (see ``hamiltonian.ExamplesHamiltonian``) computes the same
    witness-root value-flow expectation that a beta-reduced lattice
    would pin.

    Returns ``(augmented_ast, witness_regions)`` where
    ``witness_regions`` is a list of ``(start_site, end_site)`` pairs
    (end exclusive) identifying each witness's site range in the
    serialized lattice. When ``examples`` is empty the sketch is
    returned unchanged with ``witness_regions = []``.
    """
    examples = tuple(examples)
    if not examples:
        return sketch, []

    # Build witnesses (each: app chain over a fresh sketch copy).
    witnesses = [
        _wrap_in_application_chain(deepcopy(sketch), ex.inputs)
        for ex in examples
    ]

    # Stitch sketch + witnesses into an If-cascade (classical-copy form).
    cur: Node = witnesses[-1]
    for w in reversed(witnesses[:-1]):
        cur = If(cond=BoolLit(val=True), then_b=w, else_b=cur)
    aug = If(cond=BoolLit(val=True), then_b=sketch, else_b=cur)

    # Compute witness_regions by serializing each witness in the same
    # pre-order the encoder uses and accumulating offsets. The serialized
    # layout of the If-cascade is:
    #   [ outer-If, cond=BoolLit(True), sketch sites...,
    #     inner-If-cascade for witnesses, ]
    # We compute each witness's site count via the serializer and
    # determine its absolute start by accumulating.
    from src.qft_pcn.logic._serialize import serialize_preorder

    # Sketch's serialized footprint (excluding trailing PAD).
    def _n_nonpad_sites(node: Node) -> int:
        # Use a large N to avoid truncation; count non-PAD sites.
        try:
            occs = serialize_preorder(node, N=128)
        except Exception:
            # Fall back: assume node contributes 1 site (HoleVar/lit).
            return 1
        from src.qft_pcn.logic.encoding import KIND_PAD
        return sum(1 for o in occs if o.kind != KIND_PAD)

    sketch_size = _n_nonpad_sites(sketch)
    # Outer If contributes 1 site for the If node + 1 site for its
    # BoolLit(True) condition. Each inner-If similarly contributes 2
    # "scaffolding" sites that PRECEDE that branch's witness payload.
    # Pre-order of If is (If, cond, then, else). With then_b = sketch,
    # the outer If: [If, BoolLit, ...sketch sites..., ...else-cascade...]
    # so sketch starts at site 2.
    # The else-branch (else_b of outer If) is the inner If-cascade.
    # Each inner-If contributes [If, BoolLit, ...witness then..., ...else...]
    # so witness_i starts at: 2 + sketch_size + Σ_{j<i}(2 + size(witness_j))
    #                       + 2 (for the inner If's [If, BoolLit] header)
    # Last witness (the chain's tail else_b) sits at:
    #   2 + sketch_size + Σ_{j<K-1}(2 + size(witness_j)) + (K-1)*0
    # We compute precisely by walking the cascade structure.

    witness_regions: list[tuple[int, int]] = []
    cursor = 2 + sketch_size  # past outer If header + sketch then-branch
    # We descend the else-branch chain. Each inner If = 2 sites then the
    # then_b (a witness), then the else_b (continuing chain). The last
    # witness sits directly as else_b (no wrapping inner If).
    for i, w in enumerate(witnesses):
        if i < len(witnesses) - 1:
            # Wrapped in inner If: skip [If, BoolLit] header.
            cursor += 2
        w_size = _n_nonpad_sites(w)
        witness_regions.append((cursor, cursor + w_size))
        cursor += w_size

    return aug, witness_regions


__all__ = [
    "encode_synthesis", "witness_augmented_sketch",
    "_ty_to_tag",
]
