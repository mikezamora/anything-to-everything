"""The synthesize() runner (spec §3.3, §6.5).

PHASE 0.5 DESIGN INVESTIGATION RESULT (hybrid (c) with synthesis gate
extension):

  Reading `factored_evolution._iter_eval_terms` confirmed that
  `factored_trotter_step` only applies gates for rule_ids starting with
  "R-" (the eval rules). Non-"R-" terms (typing, synthesis constraints)
  participate ONLY at the composed-expectation level via
  `compose_hamiltonians.total_energy`.

  E's `EvalHamiltonian` drives relaxation via two-site projector gates
  (in `factored_trotter_step`) PLUS optional rank-1 program-pair
  transitions (in `factored_trotter_step_with_transitions`). The
  typing/synthesis Hamiltonians have non-"R-" rule ids by design so
  they are SKIPPED by factored_evolve.

  HOWEVER, for E's synthesis loop to drive a hole-superposition state
  toward the correct completion, we need the synthesis constraints
  (H_examples, H_target_type, H_size) to DAMP wrong branches during
  imag-time relaxation. The architectural rule from the spec (§6.5)
  is that ALL blocks are active during phases 2-3 of the anneal.

  Resolution adopted HERE in E's territory (does NOT modify
  factored_evolution.py per the manifesto):
    - We add a wrapper, `_apply_one_site_projector_gates`, that applies
      a single-site `exp(-dt * w * (I - P))` gate at each synthesis
      term's anchor site. P is factored across species (kind/type/value
      projectors), so the gate stays factored and never materializes a
      (D_LOCAL, D_LOCAL) operator. The form `exp(-dt*w*(I-P)) =
      e^{-dt*w}*I + (1-e^{-dt*w})*P` means we apply
      `I + α*(P-I) = I - (1 - e^{-dt*w})*(I-P)` IN PLACE on the site
      tensor — equivalently, multiply the projector-aligned component
      by 1 and the non-aligned component by e^{-dt*w}, which after
      renormalization concentrates amplitude on the projector-aligned
      branch.
    - The eval-rule gates and transitions still drive evolution via
      `factored_trotter_step` / `factored_trotter_step_with_transitions`.

This wrapper is E-local; sub-projects A/B/C are untouched.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from src.qft_pcn.qft.mps import MPS
from src.qft_pcn.logic.encoder import encode
from src.qft_pcn.logic.encoding import (
    EncodingMeta, KIND_PAD, KIND_CUTOFF, D_LOCAL,
    BID_CUTOFF, VALUE_CUTOFF, TYPE_CUTOFF, TOBL_CUTOFF,
)
from src.qft_pcn.logic.decoder import sample
from src.qft_pcn.logic.typing_hamiltonian import TypingHamiltonian
from src.qft_pcn.logic.evaluation_hamiltonian import EvalHamiltonian
from src.qft_pcn.logic.compose import compose_hamiltonians
from src.qft_pcn.logic.factored_evolution import (
    factored_trotter_step, factored_trotter_step_with_transitions,
    factored_energy,
)
from src.qft_pcn.logic._factored_expectation import (
    _apply_factors_to_site, _normalize_factors,
)
from .problem import SynthesisProblem, SynthesisResult, Completion
from ._validate import validate_problem
from .errors import SynthesisRuntimeError
from .encode_ext import encode_synthesis, witness_augmented_sketch
from .hamiltonian import (
    ExamplesHamiltonian, TargetTypeHamiltonian, SizeHamiltonian,
    build_synthesis_hamiltonians,
)
from .ranking import rank_completions, classify_failure_mode


# ---- Factored one-site projector gate -------------------------------------


def _apply_one_site_projector_gate(
    state: MPS, site: int,
    factors: dict[str, np.ndarray],
    coupling: float, dt: float, imaginary: bool = True,
) -> None:
    """Apply exp(-dt * coupling * (I - P)) at `site` IN PLACE.

    P is the factored projector specified by `factors` (per-species
    operators; species not in factors are identity). The gate stays
    factored — no D_LOCAL x D_LOCAL materialization.

    Closed form (P^2 = P):
        exp(-dt*c*(I-P)) = e^{-dt*c} * I + (1 - e^{-dt*c}) * P
    Equivalently the state transformation is
        |ψ⟩ → e^{-dt*c} * |ψ⟩ + (1 - e^{-dt*c}) * P|ψ⟩
    We renormalize after the step in the caller; the relative weights
    are what matter.
    """
    if abs(dt * coupling) < 1e-15:
        return
    if imaginary:
        scale = float(np.exp(-dt * coupling))
    else:
        scale = complex(np.exp(-1j * dt * coupling))
    norm_factors = _normalize_factors(factors)
    A = state.tensors[site]
    PA = _apply_factors_to_site(A, norm_factors)
    state.tensors[site] = scale * A + (1.0 - scale) * PA


def _apply_synthesis_gates(
    state: MPS,
    synth_blocks: dict,
    dt: float,
    imaginary: bool = True,
) -> None:
    """For each synthesis block's terms, apply the corresponding
    factored one-site (I - P) projector gate.
    """
    # NOTE: ExamplesHamiltonian gates are NOT applied here. With the
    # current classical-copy witness form (encode_ext.witness_augmented_sketch),
    # the example's output literal does NOT live at site 0 (which is the
    # sketch's LAM); pinning site 0's value register to the literal would
    # push the state away from any well-typed Lam. The ranking path still
    # uses ExamplesHamiltonian as a SCORING term against re-encoded
    # completions, where each completion's encoded root carries the
    # reduced value once the augmented program is structurally solved.
    # Driving examples via eval-rule transitions on (sketch · input ->
    # output) pairs is a follow-up; for now ranking handles example
    # constraints.
    # TargetTypeHamiltonian.
    tt_block = synth_blocks.get("target_type")
    if tt_block is not None and tt_block.terms:
        for term in tt_block.terms:
            _apply_one_site_projector_gate(
                state, term.site, tt_block._factors,
                tt_block.weight, dt, imaginary,
            )
    # SizeHamiltonian: factors = {"kind": P_PAD}.
    sz_block = synth_blocks.get("size")
    if sz_block is not None and sz_block.terms:
        for term in sz_block.terms:
            _apply_one_site_projector_gate(
                state, term.site, sz_block._factors,
                sz_block.weight, dt, imaginary,
            )


def _evolve_with_blocks(
    state: MPS,
    H_eval,
    synth_blocks: dict,
    dt: float,
    steps: int,
    chi_max: int,
) -> None:
    """Imag-time evolve under H_eval gates + synth-block one-site gates.

    Renormalizes every step. If renormalization fails (zero-norm — the
    factored gates have driven the state to zero) we stop the phase
    early and return; the caller treats this as best-effort.
    """
    for _ in range(steps):
        factored_trotter_step(
            state, H_eval, dt, imaginary=True, chi_max=chi_max,
        )
        _apply_synthesis_gates(state, synth_blocks, dt, imaginary=True)
        try:
            state.normalize()
        except ValueError:
            # Zero-norm — bail out of this phase. The runner handles the
            # state as-is for ranking; if ranking also explodes, the
            # SynthesisResult's failure_mode classifies it.
            return


# ---- Public synthesize() --------------------------------------------------


def synthesize(
    problem: SynthesisProblem,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = False,
) -> SynthesisResult:
    """Main entry point (spec §3.3).

    Pipeline:
        1. Validate the problem (§3.1 invariants).
        2. Encode the sketch with TypeHole superpositions.
        3. Build H_typing, H_eval, H_synthesis blocks.
        4. Three-phase anneal (warmup / main / fine).
        5. Sample N completions, dedupe by alpha-equivalence.
        6. Re-encode each unique AST and compute composed ⟨H⟩.
        7. Sort ascending, classify failure_mode.
    """
    if rng is None:
        rng = np.random.default_rng()
    validate_problem(problem)

    t_start = time.time()
    N = problem.N
    chi_max = problem.chi_max

    # ---- Encode ----------------------------------------------------------
    state, meta = encode_synthesis(problem.sketch, N=N, chi_max=chi_max)

    # ---- Build Hamiltonians ---------------------------------------------
    H_typing = TypingHamiltonian(N=N)
    H_eval = EvalHamiltonian(N=N)
    synth_blocks = build_synthesis_hamiltonians(meta, problem)

    # Wrap typing/eval with weighting for ranking. Synthesis Hamiltonians
    # already carry their own weight via constructor.
    class _Weighted:
        def __init__(self, H, w):
            self._H = H
            self._w = w
        def total_energy(self, state):
            return self._w * self._H.total_energy(state)

    w = problem.weights
    blocks_for_ranking = {
        "typing": _Weighted(H_typing, w.w_typing),
        "eval": _Weighted(H_eval, w.w_eval),
    }
    # Synthesis blocks (examples, target_type, size) are constructed
    # per-completion at ranking time in ranking mode (spec §5.5 value-
    # flow / AST-typer). For the "final_state_energy" we still report
    # the evolution-mode blocks computed on the relaxed state.
    blocks_for_final_state = dict(blocks_for_ranking)
    blocks_for_final_state["examples"] = synth_blocks["examples"]
    blocks_for_final_state["target_type"] = synth_blocks["target_type"]
    blocks_for_final_state["size"] = synth_blocks["size"]

    initial_energy = float(
        sum(H.total_energy(state) for H in blocks_for_final_state.values())
    )

    # ---- Three-phase anneal (spec §6.5) ----------------------------------
    main_steps = max(25, problem.anneal_steps)
    main_dt = problem.anneal_dt
    warmup_steps = max(10, main_steps // 4)
    fine_steps = max(10, main_steps // 4)
    warmup_dt = 2.0 * main_dt
    fine_dt = 0.2 * main_dt

    # Phase 1: warmup — H_eval only (no synthesis blocks yet).
    empty_blocks: dict = {}
    _evolve_with_blocks(
        state, H_eval, empty_blocks,
        dt=warmup_dt, steps=warmup_steps, chi_max=chi_max,
    )
    # Phase 2: main — all blocks active.
    _evolve_with_blocks(
        state, H_eval, synth_blocks,
        dt=main_dt, steps=main_steps, chi_max=chi_max,
    )
    # Phase 3: fine — all blocks at smaller dt.
    _evolve_with_blocks(
        state, H_eval, synth_blocks,
        dt=fine_dt, steps=fine_steps, chi_max=chi_max,
    )

    final_state_energy = float(
        sum(H.total_energy(state) for H in blocks_for_final_state.values())
    )

    # ---- Sample ---------------------------------------------------------
    asts: list = []
    try:
        decode_results = sample(state, meta, n_samples=problem.n_samples,
                                 rng=rng)
    except Exception as ex:
        wall = time.time() - t_start
        return SynthesisResult(
            problem=problem, completions=[],
            n_unique=0, n_samples_drawn=0, n_samples_decoded_ok=0,
            final_state_energy=final_state_energy,
            wall_time_seconds=wall,
            chi_observed_max=max(t.shape[0] for t in state.tensors),
            failure_mode="no_valid_completion",
        )
    for dr in decode_results:
        if getattr(dr, "ast", None) is not None:
            asts.append(dr.ast)

    # ---- Rank ----------------------------------------------------------
    completions = rank_completions(
        asts, N=N, chi_max=chi_max, hamiltonian_blocks=blocks_for_ranking,
        problem=problem,
    )

    # ---- Classify -----------------------------------------------------
    failure_mode = classify_failure_mode(
        completions, problem.weights,
    )

    wall = time.time() - t_start
    return SynthesisResult(
        problem=problem,
        completions=completions,
        n_unique=len(completions),
        n_samples_drawn=problem.n_samples,
        n_samples_decoded_ok=len(asts),
        final_state_energy=final_state_energy,
        wall_time_seconds=wall,
        chi_observed_max=max(t.shape[0] for t in state.tensors),
        failure_mode=failure_mode,
    )


__all__ = ["synthesize"]
