"""Constants for the MERA-native logic encoder (spec §4, §8.1).

Extends — does NOT modify — the MPS-side constants in encoding.py.
encoding.py keeps KIND_CUTOFF = 8 etc. for the MPS stack; this module
adds MERA_*_CUTOFF = 16 and the extended-calculus basis indices.
"""
from __future__ import annotations

from .encoding import (
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN,
    TYPE_NONE, TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
)

# Every MERA leaf is 16-dimensional (the max species cutoff, value's 16).
MERA_LEAF_DIM = 16
MERA_KIND_CUTOFF = 16
MERA_TYPE_CUTOFF = 16

# Node-major species-leaf layout: 5 leaves per AST node.
SPECIES_ORDER = ("kind", "type", "bid", "value", "tobl")
LEAVES_PER_NODE = 5
SPECIES_LEAF_OFFSET = {name: i for i, name in enumerate(SPECIES_ORDER)}

# Extended-calculus kind indices (base kinds 0-7 from encoding.py).
KIND_ZERO = 8
KIND_SUCC = 9
KIND_NATLIT = 10
KIND_NIL = 11
KIND_CONS = 12
KIND_EQ = 13
KIND_FORALL = 14
KIND_FIX = 15

# Extended-calculus type tags (base tags 0-7 from encoding.py).
TYPE_NAT = 8
TYPE_LIST = 9
TYPE_EQ = 10
TYPE_PROP = 11
