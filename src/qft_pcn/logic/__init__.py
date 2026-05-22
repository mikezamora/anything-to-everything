"""Logic layer: AST <-> MPS encoder/decoder (sub-project A of §10 roadmap).

Public API:

  Encoding/decoding:
    encode(ast, N=32, chi_max=16) -> (MPS, EncodingMeta)
    decode(state, meta) -> DecodeResult
    sample(state, meta, n_samples=1, rng=None) -> list[DecodeResult]
    ast_alpha_eq(a, b) -> bool

  AST:
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin
    HoleVar, substitute_hole
    Ty, TInt, TBool, TArrow
    parse(src) -> Node
    pretty(node) -> str

  Encoding constants and metadata:
    SPECIES, EncodingMeta, BinderHandle
    KIND_*, TYPE_*, BID_*, VALUE_* basis constants

  Errors:
    EncodingError, EncodingTooLarge, TooManyBinders,
    IntLiteralOutOfRange, IllScopedVar, UnsupportedNode, DecodeError
"""

from .ast import (
    Node, Var, Lam, App, IntLit, BoolLit, If, Bin,
    HoleVar, substitute_hole, TypeHole,
    Ty, TInt, TBool, TArrow,
    parse, pretty,
)
from .encoding import (
    SPECIES, EncodingMeta, BinderHandle,
    KIND_PAD, KIND_VAR, KIND_LAM, KIND_APP, KIND_INT, KIND_BOOL,
    KIND_IF, KIND_BIN, KIND_CUTOFF,
    TYPE_NONE, TYPE_INT, TYPE_BOOL,
    TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI, TYPE_ARR_BB, TYPE_ARR_NESTED,
    TYPE_CUTOFF,
    BID_NONE, BID_0, BID_1, BID_2, BID_3, BID_4, BID_5, BID_6, BID_CUTOFF,
    VALUE_NONE, VALUE_FALSE, VALUE_TRUE,
    VALUE_PLUS, VALUE_MINUS, VALUE_TIMES, VALUE_LT, VALUE_EQ, VALUE_CUTOFF,
    INT_LIT_OFFSET, INT_LIT_MIN, INT_LIT_MAX,
    EncodingError, EncodingTooLarge, TooManyBinders,
    IntLiteralOutOfRange, IllScopedVar, UnsupportedNode, DecodeError,
)
from .encoder import encode
from .decoder import decode, sample, DecodeResult, ast_alpha_eq
from .encoding import (
    TOBL_NONE, TOBL_INT, TOBL_BOOL,
    TOBL_ARR_II, TOBL_ARR_IB, TOBL_ARR_BI, TOBL_ARR_BB,
    TOBL_ARR_NESTED, TOBL_CUTOFF,
)
from .typing_hamiltonian import (
    TypingHamiltonian, TypingTerm,
    TypingHamiltonianError, TermNotFound,
    RULE_T_LIT_INT, RULE_T_LIT_BOOL,
    RULE_T_BIN_ARITH, RULE_T_BIN_CMP,
    RULE_T_OBLIGATION, RULE_T_VAR, RULE_T_ABS, RULE_T_APP_ARROW,
)
from .evaluation_hamiltonian import (
    EvalHamiltonian, EvalTerm,
    EvalHamiltonianError, EvalTermNotFound,
    RULE_R_BETA, RULE_R_ARITH_PRE, RULE_R_ARITH_POST,
    RULE_R_CMP_PRE, RULE_R_IF,
)
from .compose import compose_hamiltonians, ComposedHamiltonian, IncompatibleHamiltonians

# ---- Sub-project E: STLC synthesis (sub-project E) -----------------------
from .synthesis import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
    SynthesisError, SynthesisProblemError, SynthesisRuntimeError,
    synthesize,
)

# ---- Sub-project D: constraint debugger ----------------------------------
from .debugger import (
    NamedHamiltonianTerm,
    DiagnosticReport,
    RuleViolation,
    TermEvaluationError,
    diagnose,
    format_report,
    register_explanation,
    get_explanation,
    clear_explanations,
    register_stlc_seed_templates,
)

__all__ = [
    "encode", "decode", "sample", "DecodeResult", "ast_alpha_eq",
    "Node", "Var", "Lam", "App", "IntLit", "BoolLit", "If", "Bin",
    "HoleVar", "substitute_hole", "TypeHole",
    "Ty", "TInt", "TBool", "TArrow", "parse", "pretty",
    "SPECIES", "EncodingMeta", "BinderHandle",
    "KIND_PAD", "KIND_VAR", "KIND_LAM", "KIND_APP", "KIND_INT", "KIND_BOOL",
    "KIND_IF", "KIND_BIN", "KIND_CUTOFF",
    "TYPE_NONE", "TYPE_INT", "TYPE_BOOL",
    "TYPE_ARR_II", "TYPE_ARR_IB", "TYPE_ARR_BI", "TYPE_ARR_BB",
    "TYPE_ARR_NESTED", "TYPE_CUTOFF",
    "BID_NONE", "BID_0", "BID_1", "BID_2", "BID_3", "BID_4", "BID_5",
    "BID_6", "BID_CUTOFF",
    "VALUE_NONE", "VALUE_FALSE", "VALUE_TRUE", "VALUE_PLUS",
    "VALUE_MINUS", "VALUE_TIMES", "VALUE_LT", "VALUE_EQ", "VALUE_CUTOFF",
    "INT_LIT_OFFSET", "INT_LIT_MIN", "INT_LIT_MAX",
    "EncodingError", "EncodingTooLarge", "TooManyBinders",
    "IntLiteralOutOfRange", "IllScopedVar", "UnsupportedNode", "DecodeError",
    "TOBL_NONE", "TOBL_INT", "TOBL_BOOL",
    "TOBL_ARR_II", "TOBL_ARR_IB", "TOBL_ARR_BI", "TOBL_ARR_BB",
    "TOBL_ARR_NESTED", "TOBL_CUTOFF",
    "TypingHamiltonian", "TypingTerm",
    "TypingHamiltonianError", "TermNotFound",
    "RULE_T_LIT_INT", "RULE_T_LIT_BOOL",
    "RULE_T_BIN_ARITH", "RULE_T_BIN_CMP",
    "RULE_T_OBLIGATION", "RULE_T_VAR", "RULE_T_ABS", "RULE_T_APP_ARROW",
    "EvalHamiltonian", "EvalTerm",
    "EvalHamiltonianError", "EvalTermNotFound",
    "RULE_R_BETA", "RULE_R_ARITH_PRE", "RULE_R_ARITH_POST",
    "RULE_R_CMP_PRE", "RULE_R_IF",
    "compose_hamiltonians", "ComposedHamiltonian", "IncompatibleHamiltonians",
    # Sub-project D: constraint debugger
    "NamedHamiltonianTerm", "DiagnosticReport", "RuleViolation",
    "TermEvaluationError", "diagnose", "format_report",
    "register_explanation", "get_explanation", "clear_explanations",
    "register_stlc_seed_templates",
    # Sub-project E: synthesis
    "SynthesisProblem", "IOExample", "Completion", "SynthesisResult",
    "HamiltonianWeights",
    "SynthesisError", "SynthesisProblemError", "SynthesisRuntimeError",
    "synthesize",
]
