"""Typed error hierarchy for the LLM bridge.

Every error has a stable string code (BRIDGE_E_*), a human-readable message,
and an optional machine-readable details dict. The LLM consumes codes and
details, not free-form prose.

See spec §10 of docs/superpowers/specs/2026-05-21-llm-bridge-design.md.
"""

from __future__ import annotations

from typing import Any


class BridgeError(Exception):
    """Base class. All bridge errors inherit; all carry a stable `code`."""
    code: str = "BRIDGE_E_INTERNAL"

    def __init__(self, message: str = "", details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(f"{self.code}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message,
                "details": self.details}


class BadJsonError(BridgeError):              code = "BRIDGE_E_BAD_JSON"
class BadSchemaError(BridgeError):            code = "BRIDGE_E_BAD_SCHEMA"
class BadReferenceError(BridgeError):         code = "BRIDGE_E_BAD_REFERENCE"
class BadTermError(BridgeError):              code = "BRIDGE_E_BAD_TERM"
class TypeError(BridgeError):                 code = "BRIDGE_E_TYPE_ERROR"  # noqa: A001
class TermUnsupportedError(BridgeError):      code = "BRIDGE_E_TERM_UNSUPPORTED"
class UnsupportedMethodError(BridgeError):    code = "BRIDGE_E_UNSUPPORTED_METHOD"
class SitesOutOfRangeError(BridgeError):      code = "BRIDGE_E_SITES_OUT_OF_RANGE"
class TooLargeError(BridgeError):             code = "BRIDGE_E_TOO_LARGE"
class ConstraintNotAdjacentError(BridgeError):code = "BRIDGE_E_CONSTRAINT_NOT_ADJACENT"
class NumericFailureError(BridgeError):       code = "BRIDGE_E_NUMERIC_FAILURE"
class LlmUnavailableError(BridgeError):       code = "BRIDGE_E_LLM_UNAVAILABLE"
class LlmBadOutputError(BridgeError):         code = "BRIDGE_E_LLM_BAD_OUTPUT"
class InternalError(BridgeError):             code = "BRIDGE_E_INTERNAL"


ALL_CODES: tuple[str, ...] = (
    BadJsonError.code, BadSchemaError.code, BadReferenceError.code,
    BadTermError.code, TypeError.code, TermUnsupportedError.code,
    UnsupportedMethodError.code, SitesOutOfRangeError.code,
    TooLargeError.code, ConstraintNotAdjacentError.code,
    NumericFailureError.code, LlmUnavailableError.code,
    LlmBadOutputError.code, InternalError.code,
)
