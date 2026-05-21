"""Data types for the synthesis pipeline (spec §3).

A SynthesisProblem is the input contract: a hole-bearing AST plus optional
constraints (target_type, examples). A SynthesisResult is the output:
ranked completions, diagnostics, failure_mode classification.

These are pure data classes; logic lives in `runner.py`, `hamiltonian.py`,
`encode_ext.py`, `ranking.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any

from src.qft_pcn.logic.ast import Node, Ty


@dataclass(frozen=True)
class IOExample:
    """An example-based constraint on the synthesized function.

    Applying the sketch's root expression to `inputs` (treating it as a
    curried function) must produce `output`. Inputs and output must be
    literal Nodes (IntLit / BoolLit only).
    """
    inputs: tuple
    output: Node


@dataclass(frozen=True)
class HamiltonianWeights:
    """Block weights for H_total = sum(w_block * H_block).

    Defaults from spec §4.1. Do not tune per-problem without amending the
    spec — these are intended to work across all §7 demo problems.
    """
    w_typing: float = 4.0
    w_eval: float = 2.0
    w_examples: float = 3.0
    w_target_type: float = 2.0
    w_size: float = 0.1


@dataclass(frozen=True)
class SynthesisProblem:
    """The synthesis input contract (spec §3.1).

    sketch:        AST containing one or more HoleVar / TypeHole nodes.
    target_type:   Optional pinned type for the root expression.
    examples:      Optional IO examples; each adds a witness region.
    name:          Optional label for reports.
    N, chi_max, n_samples, anneal_steps, anneal_dt: knobs.
    weights:       HamiltonianWeights instance.
    """
    sketch: Node
    target_type: Optional[Ty] = None
    examples: tuple = ()
    name: str = ""
    N: int = 32
    chi_max: int = 32
    n_samples: int = 64
    anneal_steps: int = 200
    anneal_dt: float = 0.05
    weights: HamiltonianWeights = field(default_factory=HamiltonianWeights)


@dataclass(frozen=True)
class Completion:
    """One unique completion produced by the synthesis pipeline.

    ast:               filled-in AST (no holes).
    energy:            <H_total> on the completion's product-state encoding.
    energy_breakdown:  per-block weighted energies, dict block->float.
    diagnostics:       dict from sub-project D's diagnose (may be empty).
    multiplicity:      number of samples that produced this AST.
    """
    ast: Node
    energy: float
    energy_breakdown: dict
    diagnostics: dict
    multiplicity: int


@dataclass(frozen=True)
class SynthesisResult:
    """Output of synthesize().

    completions:               sorted ascending by energy (empty iff failure).
    failure_mode:              None for success, else label per spec §6.4.
    final_state_energy:        <H> on the relaxed state pre-sampling.
    chi_observed_max:          max bond dim during evolution (descriptive).
    """
    problem: SynthesisProblem
    completions: list
    n_unique: int
    n_samples_drawn: int
    n_samples_decoded_ok: int
    final_state_energy: float
    wall_time_seconds: float
    chi_observed_max: int
    failure_mode: Optional[str]
