"""Synthesis Hamiltonians (sub-project E, spec §4).

Three constraint blocks that compose at the expectation level with B's
TypingHamiltonian and C's EvalHamiltonian via
``compose_hamiltonians(...)`` from ``logic.compose``:

  - H_examples(meta, examples, w_X):     output-pin per IOExample
  - H_target_type(meta, target_type, w_Y): root-type pin at site 0
  - H_size(meta, sketch_range, w_S):     gentle Occam penalty per non-PAD

All three are STRUCTURAL (depend on meta + problem) and one-site,
exposed as a flat .terms / .term_energy / .total_energy / .residuals
quartet so compose_hamiltonians treats them like B/C.

Manifesto Temptation 3: every term is applied through
``factored_local_expectation`` (factored species ops, max
TYPE_CUTOFF x TYPE_CUTOFF = 8x8 matrices). NO (D_LOCAL, D_LOCAL)
operator is ever materialized.

Design (per Phase 0.5 investigation):
  - These Hamiltonians participate in evolution via the COMPOSED
    expectation only — ``factored_evolve`` skips non-"R-" rule_ids by
    construction (see ``_iter_eval_terms`` in factored_evolution.py).
  - That is the documented architectural rule for QPCN: H_eval drives
    relaxation through gates+transitions; H_typing and these synthesis
    constraint Hamiltonians score the relaxed state via composed
    expectation, which is what `ranking` reads to sort completions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.logic.ast import (
    Node, Ty, TInt, TBool, TArrow, Lam, App, IntLit, BoolLit, Var,
)
from src.qft_pcn.logic.encoding import (
    EncodingMeta, KIND_PAD, KIND_CUTOFF,
    TYPE_INT, TYPE_BOOL, TYPE_ARR_II, TYPE_ARR_IB, TYPE_ARR_BI,
    TYPE_ARR_BB, TYPE_ARR_NESTED, TYPE_CUTOFF,
    KIND_INT, KIND_BOOL,
    VALUE_CUTOFF, VALUE_FALSE, VALUE_TRUE, INT_LIT_OFFSET,
)
from src.qft_pcn.logic._factored_expectation import (
    factored_local_expectation, build_envs,
)


# ---- Rule ids (intentionally NOT "R-" prefixed so factored_evolve skips) --

RULE_X_OUTPUT_PIN = "X-Output-Pin"
RULE_Y_TARGET_TYPE = "Y-Target-Type"
RULE_S_NON_PAD = "S-Non-Pad"


# ---- Term ------------------------------------------------------------------


@dataclass(frozen=True)
class SynthTerm:
    """A single synthesis-constraint term (one-site)."""
    rule_id: str
    site: int
    arity: int = 1


# ---- Small projector helpers ----------------------------------------------


def _proj(d: int, idx: int) -> np.ndarray:
    p = np.zeros((d, d), dtype=complex)
    p[idx, idx] = 1.0
    return p


def _one_minus_proj(d: int, idx: int) -> np.ndarray:
    return np.eye(d, dtype=complex) - _proj(d, idx)


# ---- Type tagging (duplicated here to avoid importing encode_ext) ---------


def _ty_to_tag(t: Ty) -> int:
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


# ---- Helpers for example-pin term construction ----------------------------


def _output_site_factors(output: Node) -> dict[str, np.ndarray]:
    """Per-species factored operator P_match for the OUTPUT-PIN penalty.

    Penalty: (I - P_match) where P_match = P_kind * P_value (factored on
    kind and value registers, identity on the rest).
    """
    if isinstance(output, IntLit):
        k_idx = KIND_INT
        v_idx = output.val + INT_LIT_OFFSET
    elif isinstance(output, BoolLit):
        k_idx = KIND_BOOL
        v_idx = VALUE_TRUE if output.val else VALUE_FALSE
    else:
        raise ValueError(
            f"IOExample.output must be IntLit or BoolLit; got {type(output).__name__}"
        )
    return {
        "kind": _proj(KIND_CUTOFF, k_idx),
        "value": _proj(VALUE_CUTOFF, v_idx),
    }


# ---- H_examples ------------------------------------------------------------


class ExamplesHamiltonian:
    """One-site output-pin terms, one per IOExample.

    For each example, the SAME root site (site 0) is penalized for not
    encoding the example's output VALUE on the kind+value registers
    (factored, no nested ops). The classical-copy witness construction
    (encode_ext.witness_augmented_sketch) places per-example witnesses
    later in the lattice, but we anchor the pin at site 0 because the
    sketch root is what the reduction dynamics must drive toward the
    pinned output — the witnesses provide structural constraints for the
    typing/eval Hamiltonians, while the output pin lives at the
    canonical root site.

    Manifesto §1.6: this is simpler than spec §4.2's witness-app-root
    boundary pin (which requires precise site-offset bookkeeping); the
    test acceptance for P1-P7 will demonstrate it works in practice.
    """

    def __init__(self, N: int, examples: tuple, weight: float = 3.0):
        self.N = int(N)
        self.examples = tuple(examples)
        self.weight = float(weight)
        # One term per example; all anchored at site 0 (the sketch root).
        self.terms: list[SynthTerm] = [
            SynthTerm(rule_id=RULE_X_OUTPUT_PIN, site=0, arity=1)
            for _ in self.examples
        ]
        self._terms_set = frozenset(self.terms)
        # Precompute the per-example factor dict.
        self._factors_per_term = [
            _output_site_factors(ex.output) for ex in self.examples
        ]

    def term_energy(self, state: MPS, term: SynthTerm, envs=None) -> float:
        if term not in self._terms_set:
            raise KeyError(term)
        idx = self.terms.index(term)
        factors = self._factors_per_term[idx]
        # Penalty: weight * <I - P_match>
        # <P_match> via factored_local_expectation.
        p_match = factored_local_expectation(state, term.site, factors)
        return self.weight * (1.0 - p_match)

    def total_energy(self, state: MPS) -> float:
        return sum(self.term_energy(state, t) for t in self.terms)

    def residuals(self, state: MPS) -> dict:
        return {(t.rule_id, t.site, idx): self.term_energy(state, t)
                for idx, t in enumerate(self.terms)}


# ---- H_target_type ---------------------------------------------------------


class TargetTypeHamiltonian:
    """One-site root-type pin (spec §4.3).

    H = w_Y * (I - |target_tag⟩⟨target_tag|) on the type register at site 0.
    """

    def __init__(self, N: int, target_type: Optional[Ty], weight: float = 2.0):
        self.N = int(N)
        self.target_type = target_type
        self.weight = float(weight)
        if target_type is None:
            self.terms: list[SynthTerm] = []
            self._factors = None
        else:
            self.terms = [SynthTerm(rule_id=RULE_Y_TARGET_TYPE, site=0,
                                    arity=1)]
            tag = _ty_to_tag(target_type)
            if tag == TYPE_ARR_NESTED:
                # Nested arrow: skip the on-lattice pin (see spec §4.3 fallback).
                self.terms = []
                self._factors = None
            else:
                self._factors = {"type": _proj(TYPE_CUTOFF, tag)}
        self._terms_set = frozenset(self.terms)

    def term_energy(self, state: MPS, term: SynthTerm, envs=None) -> float:
        if term not in self._terms_set:
            raise KeyError(term)
        p_match = factored_local_expectation(state, term.site, self._factors)
        return self.weight * (1.0 - p_match)

    def total_energy(self, state: MPS) -> float:
        return sum(self.term_energy(state, t) for t in self.terms)

    def residuals(self, state: MPS) -> dict:
        return {(t.rule_id, t.site): self.term_energy(state, t)
                for t in self.terms}


# ---- H_size ----------------------------------------------------------------


class SizeHamiltonian:
    """Gentle Occam penalty (spec §4.4).

    H = w_S * Σ_{i in sketch_range} (I - P_PAD^kind)_i

    Each non-PAD site contributes w_S. sketch_range defaults to all N
    sites; the runner narrows it to the sketch-only span when witnesses
    are involved.
    """

    def __init__(self, N: int, weight: float = 0.1,
                 sketch_range: Optional[tuple[int, int]] = None):
        self.N = int(N)
        self.weight = float(weight)
        if sketch_range is None:
            sketch_range = (0, self.N)
        self.sketch_range = sketch_range
        a, b = sketch_range
        self.terms: list[SynthTerm] = [
            SynthTerm(rule_id=RULE_S_NON_PAD, site=k, arity=1)
            for k in range(a, b)
        ]
        self._terms_set = frozenset(self.terms)
        self._factors = {"kind": _proj(KIND_CUTOFF, KIND_PAD)}

    def term_energy(self, state: MPS, term: SynthTerm, envs=None) -> float:
        if term not in self._terms_set:
            raise KeyError(term)
        p_pad = factored_local_expectation(state, term.site, self._factors)
        return self.weight * (1.0 - p_pad)

    def total_energy(self, state: MPS) -> float:
        return sum(self.term_energy(state, t) for t in self.terms)

    def residuals(self, state: MPS) -> dict:
        return {(t.rule_id, t.site): self.term_energy(state, t)
                for t in self.terms}


# ---- Public builder --------------------------------------------------------


def build_synthesis_hamiltonians(
    meta: EncodingMeta,
    problem,                   # SynthesisProblem
) -> dict[str, object]:
    """Construct the three synthesis-specific Hamiltonians as a dict
    keyed by block name. The runner composes them with B's
    TypingHamiltonian and C's EvalHamiltonian via compose_hamiltonians.

    Returns a dict with keys:
      "examples": ExamplesHamiltonian
      "target_type": TargetTypeHamiltonian
      "size": SizeHamiltonian
    """
    w = problem.weights
    return {
        "examples": ExamplesHamiltonian(
            N=meta.N, examples=problem.examples, weight=w.w_examples,
        ),
        "target_type": TargetTypeHamiltonian(
            N=meta.N, target_type=problem.target_type, weight=w.w_target_type,
        ),
        "size": SizeHamiltonian(
            N=meta.N, weight=w.w_size,
        ),
    }


__all__ = [
    "ExamplesHamiltonian", "TargetTypeHamiltonian", "SizeHamiltonian",
    "build_synthesis_hamiltonians",
    "RULE_X_OUTPUT_PIN", "RULE_Y_TARGET_TYPE", "RULE_S_NON_PAD",
    "SynthTerm",
]
