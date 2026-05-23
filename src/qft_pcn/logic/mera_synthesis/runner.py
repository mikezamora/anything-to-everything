"""The MERA-native synthesis runner (spec §6.2).

Pipeline:
  1. Validate the problem (§6.1 invariants).
  2. Witness-augment the sketch (structural holes become rank-k
     superposition states; examples become witness sub-trees sharing the
     sketch through the MERA tree) and encode it to ONE MERA state.
     Principle 6: NEVER loop over candidate sub-trees.
  3. Compose H_typing + H_eval + H_examples + H_target_type + H_size into
     one Hamiltonian.
  4. Three-phase factored imaginary-time evolution (M2 owns the loop):
       warmup  -- H_typing + H_eval only;
       main    -- full H, dt=problem.anneal_dt;
       fine    -- full H, dt=0.01.
  5. Sample completions via sample_mera, dedupe by alpha-equivalence.
  6. Re-encode each unique completion, rank by <H_total>, classify
     failure mode.

Principle 7: top-1 correctness is verified by the caller decoding the
AST, never by energy alone.
"""
from __future__ import annotations

import numpy as np

from .problem import (
    SynthesisProblem, SynthesisResult, Completion, HamiltonianWeights,
)
from .errors import SynthesisRuntimeError
from .encode_ext import _witness_augmented_ast
from .hamiltonian import (
    compile_mera_synthesis_hamiltonian, ComposedMeraSynthesisHamiltonian,
)
from .ranking import dedupe_by_alpha_eq, classify_failure_mode
from .._validate_synth import validate_problem


def synthesize(problem: SynthesisProblem,
               rng: np.random.Generator | None = None,
               verbose: bool = False) -> SynthesisResult:
    """Run the MERA synthesis pipeline (spec §6.2).

    Principle 6: holes are quantum superpositions -- this function NEVER
    loops over candidate sub-trees. Principle 7: ranking by <H>; the
    caller verifies the top-1 by decoding.
    """
    # Local imports to avoid a circular import: mera_encoder imports
    # mera_synthesis.encode_ext at module load, and the runner depends on
    # mera_encoder; importing the runner eagerly at package init would
    # deadlock the load. Importing the substrate here breaks the cycle.
    from ..mera_encoder import encode_mera
    from ..mera_decoder import sample_mera
    from ..mera_debugger import diagnose

    if rng is None:
        rng = np.random.default_rng()
    validate_problem(problem)
    weights = HamiltonianWeights()

    # 1. Encode the witness-augmented sketch.
    aug = _witness_augmented_ast(problem.sketch, problem.examples)
    try:
        state, meta = encode_mera(
            aug, n_nodes_max=problem.n_nodes_max,
            chi_layer=problem.chi_layer)
    except Exception as exc:
        raise SynthesisRuntimeError(f"encode failed: {exc}") from exc

    # 2. Compose H.
    H = compile_mera_synthesis_hamiltonian(meta, problem, weights)

    # 3. Three-phase imaginary-time evolution.
    state = _anneal(state, meta, H, problem, weights)

    final_energy = float(H.total_energy(state))
    if not np.isfinite(final_energy):
        raise SynthesisRuntimeError(
            f"non-finite final energy: {final_energy}")

    # 4. Sample completions; dedupe by alpha-equivalence.
    samples = sample_mera(state, meta, n_samples=problem.n_samples, rng=rng)
    groups = dedupe_by_alpha_eq(samples)
    chi_obs = getattr(state, "chi_observed_max", problem.chi_layer)
    if not groups:
        return SynthesisResult(
            problem=problem, completions=[], n_unique=0,
            n_samples_drawn=len(samples), n_samples_decoded_ok=0,
            final_state_energy=final_energy,
            chi_observed_max=chi_obs,
            failure_mode="no_valid_completion")

    # 5. Re-encode each unique completion and compute <H>.
    completions: list[Completion] = []
    for ast, mult in groups:
        try:
            c_state, c_meta = encode_mera(
                ast, n_nodes_max=problem.n_nodes_max,
                chi_layer=problem.chi_layer)
        except Exception:
            # Unencodable sample (e.g. a malformed decode); skip.
            continue
        c_H = compile_mera_synthesis_hamiltonian(c_meta, problem, weights)
        energy = float(c_H.total_energy(c_state))
        breakdown = c_H.residuals(c_state)
        try:
            report = diagnose(c_state, c_meta, list(c_H.terms))
            diag = report.to_dict()
        except Exception as exc:
            diag = {"diagnose_error": str(exc)}
        completions.append(Completion(
            ast=ast, energy=energy, energy_breakdown=breakdown,
            diagnostics=diag, multiplicity=mult))

    completions.sort(key=lambda c: (c.energy, -c.multiplicity))
    tol = 1e-3 * (weights.w_T + weights.w_E + weights.w_X + weights.w_Y)
    failure_mode = classify_failure_mode(completions, tolerance_correct=tol)

    return SynthesisResult(
        problem=problem, completions=completions,
        n_unique=len(completions),
        n_samples_drawn=len(samples),
        n_samples_decoded_ok=sum(c.multiplicity for c in completions),
        final_state_energy=final_energy,
        chi_observed_max=chi_obs,
        failure_mode=failure_mode)


def _anneal(state, meta, H, problem, weights):
    from ..mera_evolution_logic import mera_imaginary_evolve_state
    from ..mera_typing_hamiltonian import MeraTypingHamiltonian
    from ..mera_evaluation_hamiltonian import MeraEvalHamiltonian

    """Three-phase factored imaginary-time evolution (spec §6.5).

    Phase 1 (warmup) evolves under a locally-composed H_typing + H_eval
    sub-Hamiltonian -- M2 has no `typing_eval_only()` accessor on the
    composed M3 Hamiltonian, so we compose the warmup partial here. This
    keeps `hamiltonian.py`'s surface stable.
    """
    main_steps = problem.anneal_steps
    warm_steps = max(25, main_steps // 4)
    fine_steps = max(25, main_steps // 4)

    # Warmup: typing+eval only, same weights as in full H.
    h_typing = MeraTypingHamiltonian(meta)
    h_eval = MeraEvalHamiltonian(meta)
    H_partial = ComposedMeraSynthesisHamiltonian(
        weighted_sub_hams=[(h_typing, weights.w_T), (h_eval, weights.w_E)],
        extra_terms=[])

    _, state = mera_imaginary_evolve_state(
        state, H_partial, dt=0.1, steps=warm_steps,
        chi_layer=problem.chi_layer)
    _, state = mera_imaginary_evolve_state(
        state, H, dt=problem.anneal_dt, steps=main_steps,
        chi_layer=problem.chi_layer)
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.01, steps=fine_steps,
        chi_layer=problem.chi_layer)
    return state
