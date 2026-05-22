"""Synthesis problem and result data types (spec §6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ast import Node, Ty


@dataclass(frozen=True)
class IOExample:
    """Applying the synthesized expression to `inputs` must yield `output`."""
    inputs: tuple              # tuple[Node, ...] of IntLit/BoolLit/NatLit
    output: Node


@dataclass(frozen=True)
class HamiltonianWeights:
    """Block weights for the composed synthesis Hamiltonian (spec §4.1)."""
    w_T: float = 4.0           # typing
    w_E: float = 2.0           # eval
    w_X: float = 3.0           # examples
    w_Y: float = 2.0           # target_type
    w_S: float = 0.1           # size (Occam)


@dataclass(frozen=True)
class SynthesisProblem:
    """The contract: solve me (spec §6.1)."""
    sketch: Node
    target_type: Ty | None = None
    examples: tuple = ()
    name: str = ""
    n_nodes_max: int = 32
    chi_layer: int = 32
    n_samples: int = 64
    anneal_steps: int = 200
    anneal_dt: float = 0.05


@dataclass(frozen=True)
class Completion:
    ast: Node
    energy: float
    energy_breakdown: dict
    diagnostics: dict
    multiplicity: int


@dataclass(frozen=True)
class SynthesisResult:
    problem: SynthesisProblem
    completions: list
    n_unique: int
    n_samples_drawn: int
    n_samples_decoded_ok: int
    final_state_energy: float
    chi_observed_max: int
    failure_mode: str | None
