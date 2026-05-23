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

from dataclasses import dataclass
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from ..ast import Node
    from ..mera_encoder import MeraEncodingMeta
    from src.qft_pcn.qft.mera import MERA
    from src.qft_pcn.composition.lemma_library import LemmaLibrary


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

    # I-Task-10 blocker #5: Forall-protected leaves are frozen across all
    # three phases — universal quantification is genuine tensor-network
    # inertia, never classical iteration (§1.1 binding-as-entanglement).
    # The set is empty for encodings without any Forall, preserving prior
    # behavior bitwise (frozen_leaves default at the evolution layer).
    frozen = set(getattr(meta, "forall_protected_leaves", set()) or set())
    frozen_arg = frozen if frozen else None

    _, state = mera_imaginary_evolve_state(
        state, H_partial, dt=0.1, steps=warm_steps,
        chi_layer=problem.chi_layer, frozen_leaves=frozen_arg)
    _, state = mera_imaginary_evolve_state(
        state, H, dt=problem.anneal_dt, steps=main_steps,
        chi_layer=problem.chi_layer, frozen_leaves=frozen_arg)
    _, state = mera_imaginary_evolve_state(
        state, H, dt=0.01, steps=fine_steps,
        chi_layer=problem.chi_layer, frozen_leaves=frozen_arg)
    return state


# ---------------------------------------------------------------------------
# I-Task-10 blocker #1 — relax_program driver (spec §8.12, arch §10.8)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RelaxResult:
    """Result record of `relax_program` (I Task 10, spec §8.12).

    Fields:
      state          -- the relaxed MERA after imag-time evolution.
      meta           -- the encoding meta for `state` (the §1.2 addressing
                        data).
      hamiltonian    -- the composed H whose ground state is being
                        approached (a `ComposedMeraSynthesisHamiltonian`).
      residual       -- final `H.total_energy(state)`; the §8.12 acceptance
                        gate is `residual < eps`.
      trotter_steps  -- number of Trotter steps actually applied. May be
                        less than `max_trotter_steps` when early
                        termination on the residual gate fires.
      converged      -- True iff `residual < eps` at the time of return.
    """
    state: "MERA"
    meta: "MeraEncodingMeta"
    hamiltonian: ComposedMeraSynthesisHamiltonian
    residual: float
    trotter_steps: int
    converged: bool


def relax_program(
    ast_src,
    *,
    constraints=(),
    eps: float = 1e-3,
    dt: float = 0.05,
    max_trotter_steps: int = 200,
    chi_layer: int = 16,
    n_nodes_max: int = 32,
    frozen_leaves: set[int] | None = None,
    lemma_library: "LemmaLibrary | None" = None,
) -> RelaxResult:
    """Encode `ast_src`, compose H = H_typing + H_eval (+ promoted lemma
    constraints), and run imag-time evolution to residual < eps or until
    `max_trotter_steps` is consumed (spec §8.12, arch §10.8).

    Distinct from `synthesize(SynthesisProblem)`: no sketch holes, no
    witness augmentation, no completion ranking — this is the
    program-relaxation driver used by I Task 10's two-stage acceptance
    demo.

    Parameters
    ----------
    ast_src:
        Either an `ast.Node` (used directly) or a `str` (parsed via
        `logic.ast.parse`).
    constraints:
        Iterable of DSL constraint dicts. Presently only
        `{"kind": "use_lemma", ...}` is honored. Each is compiled via the
        Promoter (mode='init_clamp'), and the union of clamped leaves is
        added to `frozen_leaves`. `lemma_library` is required when
        `constraints` is non-empty.
    eps:
        Residual gate. Evolution stops as soon as `H.total_energy(state)`
        drops below `eps`. Use `eps=0.0` to force full-budget consumption.
    dt, max_trotter_steps, chi_layer, n_nodes_max:
        Imag-time evolution + encoder hyperparameters.
    frozen_leaves:
        Optional caller-supplied set of host leaf indices to freeze for
        the full duration of evolution. Merged with the union of
        leaves clamped by `use_lemma` constraints (§1.5 operator-algebraic
        promotion). When non-empty (and the Forall-protected set on the
        meta is also non-empty), both are unioned in.
    """
    # Local imports mirror `synthesize`'s rationale: mera_encoder imports
    # mera_synthesis.encode_ext at module load, so the substrate is loaded
    # at call time to break the cycle.
    from ..ast import Node as _Node, parse as _parse
    from ..mera_encoder import encode_mera
    from ..mera_typing_hamiltonian import MeraTypingHamiltonian
    from ..mera_evaluation_hamiltonian import MeraEvalHamiltonian
    from ..mera_evolution_logic import mera_imaginary_evolve_state

    ast = _parse(ast_src) if isinstance(ast_src, str) else ast_src
    if not isinstance(ast, _Node):
        raise TypeError(
            "relax_program expected an ast.Node or str source; got "
            f"{type(ast).__name__}")

    state, meta = encode_mera(ast, n_nodes_max=n_nodes_max,
                              chi_layer=chi_layer)

    # Compose H_typing + H_eval with unit weights — relaxation is judged
    # by residual, not by ranking against alternatives.
    h_typing = MeraTypingHamiltonian(meta)
    h_eval = MeraEvalHamiltonian(meta)
    H = ComposedMeraSynthesisHamiltonian(
        weighted_sub_hams=[(h_typing, 1.0), (h_eval, 1.0)],
        extra_terms=[])

    # Build the frozen set:
    #   * caller's `frozen_leaves` (e.g. arbitrary external clamp);
    #   * leaves clamped by `use_lemma` constraints (§1.5 promotion);
    #   * meta.forall_protected_leaves (§1.1 universal-binder protection,
    #     blocker #5; defaults to the empty set on encodings without any
    #     Forall, so prior behavior is preserved bitwise).
    frozen: set[int] = set(frozen_leaves or ())
    frozen |= set(
        getattr(meta, "forall_protected_leaves", set()) or set())

    constraints_list = list(constraints) if constraints else []
    if constraints_list:
        if lemma_library is None:
            raise ValueError(
                "constraints non-empty but lemma_library is None")
        from src.qft_pcn.composition.promoter import Promoter
        promoter = Promoter(lemma_library, mode="init_clamp")
        for c in constraints_list:
            kind = c.get("kind")
            if kind != "use_lemma":
                raise NotImplementedError(
                    f"unsupported constraint kind: {kind!r}")
            promoted = promoter.compile_constraint(c)
            clamped = promoter.apply_init_clamp(
                state, meta, promoted, strength=1.0)
            frozen |= clamped

    frozen_arg = frozen if frozen else None

    # Imag-time evolution with chunked early termination. Chunking is a
    # pure control-flow wrapper around `mera_imaginary_evolve_state`; the
    # inner driver owns the physics (§7.5). Between chunks we check the
    # residual gate and return as soon as it is below eps.
    chunk = max(1, min(10, max_trotter_steps))
    steps_taken = 0
    residual = float(H.total_energy(state))
    while steps_taken < max_trotter_steps and residual > eps:
        remaining = max_trotter_steps - steps_taken
        this_chunk = min(chunk, remaining)
        _, state = mera_imaginary_evolve_state(
            state, H, dt=dt, steps=this_chunk, chi_layer=chi_layer,
            frozen_leaves=frozen_arg)
        steps_taken += this_chunk
        residual = float(H.total_energy(state))

    converged = bool(residual < eps)
    return RelaxResult(
        state=state, meta=meta, hamiltonian=H,
        residual=residual, trotter_steps=steps_taken,
        converged=converged)
