"""QPCN LLM bridge: stdio JSON-RPC + DSL runtime (spec §10.5).

Public surface:
  run_problem, diagnose_problem  - synchronous runtime API
  validate_dsl                   - validate without running
  Problem                        - DSL builder
  MockLLM, AnthropicLLM, OllamaLLM - LLM shims
  BridgeError + subclasses       - typed error hierarchy
"""

from .errors import (
    BridgeError, BadJsonError, BadSchemaError, BadReferenceError,
    BadTermError, TypeError, TermUnsupportedError, UnsupportedMethodError,
    SitesOutOfRangeError, TooLargeError, ConstraintNotAdjacentError,
    NumericFailureError, LlmUnavailableError, LlmBadOutputError,
    InternalError, ALL_CODES,
)
from .dsl.pipeline import validate_dsl, compile_dsl, CompiledDsl
from .runtime import (
    run_problem, diagnose_problem,
    RunResult, RunDiagnostic, ObservableValue,
)
from .templates import (
    Problem, FieldSpec, ConstraintSpec, ObservableSpec, SearchSpec,
    stlc_synthesis,
)
from .llm import MockLLM, AnthropicLLM, OllamaLLM


__all__ = [
    "run_problem", "diagnose_problem", "validate_dsl", "compile_dsl",
    "RunResult", "RunDiagnostic", "ObservableValue", "CompiledDsl",
    "Problem", "FieldSpec", "ConstraintSpec", "ObservableSpec",
    "SearchSpec", "stlc_synthesis",
    "MockLLM", "AnthropicLLM", "OllamaLLM",
    "BridgeError", "BadJsonError", "BadSchemaError", "BadReferenceError",
    "BadTermError", "TypeError", "TermUnsupportedError",
    "UnsupportedMethodError", "SitesOutOfRangeError", "TooLargeError",
    "ConstraintNotAdjacentError", "NumericFailureError",
    "LlmUnavailableError", "LlmBadOutputError", "InternalError",
    "ALL_CODES",
]
