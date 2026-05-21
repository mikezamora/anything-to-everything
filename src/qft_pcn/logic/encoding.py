"""Field-species constants, EncodingMeta, BinderHandle, and exceptions for the
AST <-> MPS encoder.

See docs/superpowers/specs/2026-05-20-ast-mps-encoder-design.md, §4, §5.1, §9.

Per-site local Hilbert space is the tensor product of four registers
(species): kind (8) x type (8) x bid (8) x value (16) = 8192. Basis ordering
within each species follows `embed_op` in qft/fock.py (leftmost species
changes slowest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.qft_pcn.qft.hamiltonian import FieldSpecies


# ---- kind register --------------------------------------------------------

KIND_PAD = 0
KIND_VAR = 1
KIND_LAM = 2
KIND_APP = 3
KIND_INT = 4
KIND_BOOL = 5
KIND_IF = 6
KIND_BIN = 7
KIND_CUTOFF = 8

KIND_NAMES = ("PAD", "VAR", "LAM", "APP", "INT", "BOOL", "IF", "BIN")


# ---- type register --------------------------------------------------------

TYPE_NONE = 0
TYPE_INT = 1
TYPE_BOOL = 2
TYPE_ARR_II = 3      # Int -> Int
TYPE_ARR_IB = 4      # Int -> Bool
TYPE_ARR_BI = 5      # Bool -> Int
TYPE_ARR_BB = 6      # Bool -> Bool
TYPE_ARR_NESTED = 7  # any higher-order arrow; full Ty stored in meta side table
TYPE_CUTOFF = 8


# ---- binder-id register ---------------------------------------------------

BID_NONE = 0
BID_0 = 1
BID_1 = 2
BID_2 = 3
BID_3 = 4
BID_4 = 5
BID_5 = 6
BID_6 = 7
BID_CUTOFF = 8

# Maximum nested binders supported (BID_0..BID_6 = 7 binders).
MAX_BINDER_DEPTH = BID_CUTOFF - 1


# ---- value register -------------------------------------------------------
# Overloaded by kind. See spec §4.4.

VALUE_NONE = 0      # default for PAD / VAR / LAM / APP / IF
VALUE_FALSE = 0     # for kind == BOOL
VALUE_TRUE = 1      # for kind == BOOL
VALUE_PLUS = 2      # for kind == BIN
VALUE_MINUS = 3
VALUE_TIMES = 4
VALUE_LT = 5
VALUE_EQ = 6
# Indices 7..15 used for int literals via offset.

INT_LIT_OFFSET = 7
INT_LIT_MIN = -7
INT_LIT_MAX = 8     # inclusive bounds [-7, 8]
VALUE_CUTOFF = 16

# Reverse maps for decoder.
BIN_OP_FROM_VALUE = {
    VALUE_PLUS: "+",
    VALUE_MINUS: "-",
    VALUE_TIMES: "*",
    VALUE_LT: "<",
    VALUE_EQ: "==",
}
BIN_VALUE_FROM_OP = {v: k for k, v in BIN_OP_FROM_VALUE.items()}


# ---- species metadata -----------------------------------------------------

SPECIES_NAMES: tuple[str, ...] = ("kind", "type", "bid", "value")
SPECIES_DIMS: tuple[int, ...] = (KIND_CUTOFF, TYPE_CUTOFF, BID_CUTOFF,
                                 VALUE_CUTOFF)
D_LOCAL: int = 1
for _d in SPECIES_DIMS:
    D_LOCAL *= _d


# FieldSpecies objects ready to drop into HamiltonianConfig (sub-projects
# B/C/E). bare_mass=0, kinetic=0 because the encoder does not require any
# dynamics on these registers — sub-projects B/C will set their own.
SPECIES: tuple[FieldSpecies, ...] = (
    FieldSpecies(name="kind",  cutoff=KIND_CUTOFF,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="type",  cutoff=TYPE_CUTOFF,  bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="bid",   cutoff=BID_CUTOFF,   bare_mass=0.0, kinetic=0.0),
    FieldSpecies(name="value", cutoff=VALUE_CUTOFF, bare_mass=0.0, kinetic=0.0),
)


# ---- binder handle --------------------------------------------------------


@dataclass(frozen=True)
class BinderHandle:
    """Uniquely identifies a binder for cross-bond bookkeeping.

    lam_site is the absolute site index of the binder's Lam node.
    depth_at_lam is the lexical depth at which it was introduced (0 = outermost).
    """
    lam_site: int
    depth_at_lam: int


# ---- encoding metadata ----------------------------------------------------


@dataclass
class EncodingMeta:
    """Side data produced by the encoder, consumed by decoder and downstream
    sub-projects B/C/D/E.
    """
    N: int
    chi_max: int
    field_dims: dict[str, int]
    species: list[FieldSpecies]
    nested_type_index: dict[int, "object"]   # site -> Ty (kept as object to
                                             # avoid circular import; encoder
                                             # writes proper Ty values)
    site_to_ast_path: dict[int, tuple[int, ...]]
    live_binders_per_bond: list[list[BinderHandle]]


# ---- exception hierarchy --------------------------------------------------


class EncodingError(Exception):
    """Base class for encoder/decoder errors."""


class EncodingTooLarge(EncodingError):
    def __init__(self, n_nodes: int, N: int):
        self.n_nodes, self.N_limit = n_nodes, N
        super().__init__(f"AST has {n_nodes} nodes but N={N}")


class TooManyBinders(EncodingError):
    def __init__(self, depth: int, cutoff: int):
        self.depth, self.cutoff = depth, cutoff
        super().__init__(
            f"scope nesting {depth} exceeds bid cutoff {cutoff}"
        )


class IntLiteralOutOfRange(EncodingError):
    def __init__(self, n: int):
        self.n = n
        super().__init__(
            f"IntLit({n}) outside [{INT_LIT_MIN}, {INT_LIT_MAX}]"
        )


class IllScopedVar(EncodingError):
    def __init__(self, name: str):
        self.name = name
        super().__init__(f"Var({name!r}) not in lexical scope")


class UnsupportedNode(EncodingError):
    def __init__(self, node_type: str):
        self.node_type = node_type
        super().__init__(f"node type {node_type} not in supported grammar")


class DecodeError(EncodingError):
    pass
