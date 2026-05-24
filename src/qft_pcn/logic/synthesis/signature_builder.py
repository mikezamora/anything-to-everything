"""Free-form signature ingestion for QPCN synthesis.

EXTENSIONS.md entry "free-form signature ingestion for QPCN synthesis"
(QFT_PCN_ARCHITECTURE.md §14.1 + §15).

The synthesis substrate (logic.synthesis) consumes a hole-bearing AST
sketch (HoleVar / TypeHole). For HumanEval-typed-subset and Hazel-style
inputs we receive a free-form signature string of the form::

    "name : ty1 -> ty2 -> ... -> ret"

This module turns such a signature into:

* a ``SignatureSpec`` (name + parsed ``Ty`` + heuristic ``hole_count``);
* a sketchable AST: ``\\x1:ty1. \\x2:ty2. ... HoleVar()``.

The type grammar extends the surface in ``logic.ast`` with **type
variables**: any lowercase identifier (``a``, ``b``, ``elem``) becomes a
polymorphic placeholder encoded as a ``TypeHole`` over the flat ground
candidates ``(TInt(), TBool(), TNat())``. The flat-tag basis admits
exactly the ground types that map to a single distinct tag in the
16-slot type register (TInt=1, TBool=2, TNat=8). TList is intentionally
excluded -- its ``elem`` subtype would force all ``TList(*)`` candidates
to collapse to the same TYPE_LIST=9 tag in the flat basis, defeating
the per-candidate superposition; lifting that restriction is tracked
under the HumanEval container-types EXTENSIONS entry.

The module does NOT call into the heavy substrate at import time; the
caller (``experiments/baselines/qpcn.py``) is responsible for wiring the
result into ``SynthesisProblem`` + ``synthesize``.

Note on ``chi_layer`` for polymorphic sketches: signatures with many
polymorphic type variables blow up the per-hole candidate basis
multiplicatively. The encoder's ``chi_layer`` (capped at
``max(chi_layer, 4) = 16`` by default in ``mera_encoder.py:679``) will
raise ``EncodingTooLarge`` once ``total_k = candidates ** hole_count``
exceeds that cap. The auto-sized ``chi_max`` heuristic in
``experiments/baselines/qpcn.py::_solve_synthesis_from_signature`` does
**not** lift ``chi_layer`` -- callers needing deep polymorphic synthesis
must pass both knobs explicitly via
``problem.payload["synth_knobs"] = {"chi_max": ..., "chi_layer": ...}``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.qft_pcn.logic.ast import (
    HoleVar,
    Lam,
    Node,
    TArrow,
    TBool,
    TInt,
    TList,
    TNat,
    Ty,
    TypeHole,
)


# Candidates used to ground polymorphic type variables. The encoder's
# TypeHole basis is flat ground tags (TInt / TBool / TNat / flat TArrow);
# we use the three base ground tags so polymorphic positions span the
# integer-ish, boolean, and Peano-natural worlds (the synthesis pipeline
# relaxes among them via the per-candidate type-leaf superposition).
# TList is excluded -- see module docstring + EXTENSIONS.md.
_POLY_GROUND_CANDIDATES: tuple[Ty, ...] = (TInt(), TBool(), TNat())


@dataclass(frozen=True)
class SignatureSpec:
    """The parsed signature.

    ``name``        -- the function identifier.
    ``ty``          -- the parsed ``Ty`` (may contain ``TypeHole``s for
                       polymorphic positions).
    ``hole_count``  -- heuristic = number of top-level arrows + 1
                       (one binder per argument, plus one structural body
                       hole). For non-arrow types it is 1 (just the body
                       hole).
    """
    name: str
    ty: Ty
    hole_count: int


# ---- signature parser ------------------------------------------------------


_TY_TOKEN_RE = re.compile(
    r"\s+"
    r"|(?P<arrow>->)"
    r"|(?P<lp>\()"
    r"|(?P<rp>\))"
    r"|(?P<ident>[A-Za-z_][A-Za-z_0-9]*)"
)


def _tokenize_ty(src: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    pos = 0
    while pos < len(src):
        m = _TY_TOKEN_RE.match(src, pos)
        if not m:
            raise ValueError(
                f"unexpected character in signature type at "
                f"position {pos}: {src[pos]!r}"
            )
        kind = m.lastgroup
        if kind is None:
            pos = m.end()
            continue
        out.append((kind, m.group(kind)))
        pos = m.end()
    out.append(("eof", ""))
    return out


_BUILTIN_CON_NAMES = {"Int", "Bool", "Nat", "List"}


class _TyParser:
    """Type-only parser that ALSO admits lowercase identifiers as type
    variables. Concrete constructors (``Int``, ``Bool``, ``Nat``,
    ``List``) match the ``logic.ast`` surface exactly; everything else
    in identifier position is treated as a type variable and emitted as
    a ``TypeHole`` over the ground candidate set.

    Grammar (right-assoc arrow, atomic ``List <atom>``)::

        ty      ::= atom ("->" ty)?
        atom    ::= "Int" | "Bool" | "Nat"
                  | "List" atom
                  | "(" ty ")"
                  | tyvar         # any other lowercase identifier
    """

    def __init__(self, toks: list[tuple[str, str]]):
        self.toks = toks
        self.i = 0

    def peek(self) -> tuple[str, str]:
        return self.toks[self.i]

    def eat(self, kind: str) -> tuple[str, str]:
        t = self.peek()
        if t[0] != kind:
            raise ValueError(
                f"expected token {kind!r}, got {t[0]!r}={t[1]!r}"
            )
        self.i += 1
        return t

    def parse(self) -> Ty:
        ty = self.parse_ty()
        if self.peek()[0] != "eof":
            raise ValueError(
                f"trailing tokens after type: {self.peek()!r}"
            )
        return ty

    def parse_ty(self) -> Ty:
        left = self.parse_atom()
        if self.peek()[0] == "arrow":
            self.eat("arrow")
            right = self.parse_ty()
            return TArrow(src=left, dst=right)
        return left

    def parse_atom(self) -> Ty:
        kind, value = self.peek()
        if kind == "lp":
            self.eat("lp")
            ty = self.parse_ty()
            self.eat("rp")
            return ty
        if kind == "ident":
            self.i += 1
            if value == "Int":
                return TInt()
            if value == "Bool":
                return TBool()
            if value == "Nat":
                return TNat()
            if value == "List":
                elem = self.parse_atom()
                return TList(elem=elem)
            # Lowercase / unknown identifier -> polymorphic type
            # variable. Encoded as a TypeHole over the flat ground
            # candidates (TInt, TBool).
            return TypeHole(
                candidates=_POLY_GROUND_CANDIDATES,
                name=value,
            )
        raise ValueError(
            f"expected a type atom, got {kind!r}={value!r}"
        )


def _arrow_depth(ty: Ty) -> int:
    n = 0
    cur = ty
    while isinstance(cur, TArrow):
        n += 1
        cur = cur.dst
    return n


def parse_signature_string(sig: str) -> SignatureSpec:
    """Parse ``"name : type"`` into a ``SignatureSpec``.

    ``hole_count`` is the heuristic ``(top-level arrows) + 1``: one hole
    per binder body region. For a non-arrow signature the heuristic is
    ``1`` (a single body hole).
    """
    if not isinstance(sig, str):
        raise TypeError(
            f"parse_signature_string expects str, got {type(sig).__name__}"
        )
    if ":" not in sig:
        raise ValueError(
            f"signature must contain ':' separating name and type; got "
            f"{sig!r}"
        )
    name_part, _, ty_part = sig.partition(":")
    name = name_part.strip()
    if not name:
        raise ValueError(f"empty name in signature {sig!r}")
    if not name.replace("_", "").isalnum():
        raise ValueError(f"non-identifier signature name: {name!r}")

    toks = _tokenize_ty(ty_part)
    ty = _TyParser(toks).parse()
    return SignatureSpec(
        name=name,
        ty=ty,
        hole_count=_arrow_depth(ty) + 1,
    )


# ---- signature -> sketch ---------------------------------------------------


def signature_to_sketch(spec: SignatureSpec) -> Node:
    """Build a hole-bearing AST sketch for the given signature.

    The sketch is a chain of lambdas (one per top-level arrow argument)
    with a structural ``HoleVar()`` body::

        ty = a -> b -> ret    ===>   \\_sig_x1:a. \\_sig_x2:b. HoleVar()

    Binders are prefixed with ``_sig_`` so future user-supplied binder
    names from richer signature surfaces cannot collide with the
    auto-generated argument names.

    Argument-position polymorphic ``TypeHole``s carry through to each
    ``Lam.param_ty``; the body ``HoleVar`` has an empty candidate tuple
    (the M1 "any in-scope binder" var-hole form) which the encoder is
    free to expand structurally.
    """
    if not isinstance(spec, SignatureSpec):
        raise TypeError(
            f"signature_to_sketch expects SignatureSpec, got "
            f"{type(spec).__name__}"
        )

    arg_tys: list[Ty] = []
    cur: Ty = spec.ty
    while isinstance(cur, TArrow):
        arg_tys.append(cur.src)
        cur = cur.dst

    body: Node = HoleVar()
    # Wrap in lambdas right-to-left so x1 is the outermost binder.
    for idx in range(len(arg_tys) - 1, -1, -1):
        body = Lam(
            param=f"_sig_x{idx + 1}",
            param_ty=arg_tys[idx],
            body=body,
        )
    return body
