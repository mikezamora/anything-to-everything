"""Vocabulary mapping basis names to integer indices for the canonical
kind/type/bid/value fields.

Mirrors constants from src/qft_pcn/logic/encoding.py. The DSL accepts either
integer values or these named constants in boundary entries and constraint
expression literals. Non-canonical fields (chemistry, lattice models, etc.)
accept only integer values.
"""

from __future__ import annotations

import re
from typing import Any

from src.qft_pcn.logic.encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    TYPE_NONE, TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB,
    TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
    BID_NONE, BID_0, BID_1, BID_2, BID_3, BID_4, BID_5, BID_6,
    VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ,
    INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX,
)


CANONICAL_FIELDS: tuple[str, ...] = ("kind", "type", "bid", "value")


_VOCAB: dict[str, dict[str, int]] = {
    "kind": {
        "KIND_PAD": KIND_PAD, "KIND_VAR": KIND_VAR, "KIND_LAM": KIND_LAM,
        "KIND_APP": KIND_APP, "KIND_INT": KIND_INT, "KIND_BOOL": KIND_BOOL,
        "KIND_IF":  KIND_IF,  "KIND_BIN": KIND_BIN,
        # bareword aliases
        "PAD": KIND_PAD, "VAR": KIND_VAR, "LAM": KIND_LAM, "APP": KIND_APP,
        "INT": KIND_INT, "BOOL": KIND_BOOL, "IF": KIND_IF, "BIN": KIND_BIN,
    },
    "type": {
        "T_NONE": TYPE_NONE, "T_INT": TYPE_INT, "T_BOOL": TYPE_BOOL,
        "T_ARR_II": TYPE_ARR_II, "T_ARR_IB": TYPE_ARR_IB,
        "T_ARR_BI": TYPE_ARR_BI, "T_ARR_BB": TYPE_ARR_BB,
        "T_ARR_NESTED": TYPE_ARR_NESTED,
    },
    "bid": {
        "B_NONE": BID_NONE,
        "B_0": BID_0, "B_1": BID_1, "B_2": BID_2, "B_3": BID_3,
        "B_4": BID_4, "B_5": BID_5, "B_6": BID_6,
    },
    "value": {
        "V_NONE": VALUE_NONE, "V_FALSE": VALUE_FALSE, "V_TRUE": VALUE_TRUE,
        "V_PLUS": VALUE_PLUS, "V_MINUS": VALUE_MINUS, "V_TIMES": VALUE_TIMES,
        "V_LT": VALUE_LT, "V_EQ": VALUE_EQ,
    },
}

_VALUE_OP_ALIAS: dict[str, int] = {
    "+": VALUE_PLUS, "-": VALUE_MINUS, "*": VALUE_TIMES,
    "<": VALUE_LT, "==": VALUE_EQ,
}


_V_INT_RE = re.compile(r"^V_INT\((-?\d+)\)$")


def resolve_basis(field: str, value: Any) -> int:
    """Resolve a basis name (or integer / V_INT(n)) to an integer index."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(
            f"basis value for field {field!r} must be an integer; got {value}"
        )
    if not isinstance(value, str):
        raise ValueError(
            f"basis value for field {field!r} has unsupported type {type(value).__name__}"
        )
    if field not in CANONICAL_FIELDS:
        raise ValueError(
            f"non-canonical field {field!r} only accepts integer basis values; "
            f"got {value!r}"
        )
    if field == "value":
        m = _V_INT_RE.match(value)
        if m:
            n = int(m.group(1))
            if not (INT_LIT_MIN <= n <= INT_LIT_MAX):
                raise ValueError(
                    f"V_INT({n}) out of range [{INT_LIT_MIN}, {INT_LIT_MAX}]"
                )
            return n + INT_LIT_OFFSET
        if value in _VALUE_OP_ALIAS:
            return _VALUE_OP_ALIAS[value]
    table = _VOCAB[field]
    if value in table:
        return table[value]
    raise ValueError(
        f"unknown basis name {value!r} on field {field!r}; "
        f"known: {sorted(table)[:6]}..."
    )
