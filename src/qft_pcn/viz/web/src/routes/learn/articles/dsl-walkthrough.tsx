// src/qft_pcn/viz/web/src/routes/learn/articles/dsl-walkthrough.tsx
/**
 * §5.1 DSL Walkthrough — the DSL schema (fields/hamiltonian/observables/
 * run), dsl_to_runspec mapping, and the LLM round-trip for natural-
 * language problem specification.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'dsl-walkthrough',
  title: 'DSL: from human-readable spec to a running QPCN',
  sectionPath: ['§5 DSL', '5.1 DSL Walkthrough'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The DSL is the human-readable interface to the QPCN. It is a small YAML/JSON schema with four top-level keys — `fields`, `hamiltonian`, `observables`, `run` — that together specify everything `run_problem` needs to construct a substrate and execute a run. Every preset shipped with the visualiser is a DSL document; every custom problem the user authors is a DSL document; and the LLM round-trip (described later) lets you go from a natural-language description to a DSL document and back.',
        'The four top-level keys map directly onto substrate-builder calls. `fields` lists the PCN fields that live on the manifold (their names, types, grid resolutions, initial conditions). `hamiltonian` lists the terms of the QPCN Hamiltonian (each term has a name, an operator decomposition, and a learnable coefficient). `observables` lists the operators whose expectations get read out each frame (their names and operator decompositions). `run` gives the run-config: frame count, Trotter steps per frame, coupling constants, and any seed for reproducibility.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Concretely, a minimal DSL looks like: `fields: [{name: phi, grid: [16, 16], init: gaussian}]; hamiltonian: [{name: kinetic, op: ddx(phi)^2, coeff: 0.5}, {name: mass, op: phi^2, coeff: 0.1}]; observables: [{name: phi_mean, op: phi}, {name: phi_var, op: phi^2 - <phi>^2}]; run: {frames: 200, trotter_steps: 4, kappa_R: 0.05, seed: 42}`. That nine-line spec is enough to drive a complete QPCN run.',
        'The DSL parser is intentionally permissive about syntax. Both YAML and JSON inputs are accepted; the parser does a single pass to normalise into the canonical `RunSpec` Python dataclass. Operator expressions in `op` fields are parsed by a small expression grammar (kinetic terms via `ddx`, polynomial terms via `^`, multi-field products via juxtaposition) that compiles to a list of `(coefficient, operator-string)` tuples — exactly the shape the QPCN Hamiltonian builder expects.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'hamiltonian-decomp',
      caption: 'The Hamiltonian decomposition that DSL `hamiltonian` entries compile to — one term per entry, each with a learnable coefficient.',
    },
    {
      kind: 'prose',
      body: [
        'The dsl_to_runspec mapping is straightforward but non-trivial: `fields` becomes the manifold grid spec and the PCN layer-zero state; `hamiltonian` becomes a list of `(c_k, H_k)` pairs that the QPCN Hamiltonian builder assembles; `observables` becomes the operator list whose expectations the snapshot extractor reads each frame; `run` becomes the per-frame loop config. The mapping is deterministic and well-typed — every field has an explicit Python dataclass on the receiving side.',
        'Validation happens at parse time. Missing required fields produce errors before the substrate is built; ambiguous operator expressions produce warnings; over-large grids or chi values produce confirmations (asking the user to confirm before starting an expensive run). This pre-flight check is the only reason it is safe to wire the DSL up to an LLM round-trip — the parser catches the vast majority of LLM hallucinations before they hit the substrate.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'free-energy-functional',
      caption: 'F is computed from the field-and-hamiltonian assembly the DSL produces — the DSL is the entry point to the entire learning objective.',
    },
    {
      kind: 'prose',
      body: [
        'The LLM round-trip is the natural-language interface. The user types something like "show me a relaxing scalar field with a quartic self-interaction"; the viz wraps this in a prompt template that asks the LLM to emit a DSL document; the LLM responds with YAML; the parser validates it; if it passes, the run starts. If the LLM emits invalid DSL, the parser\'s error message is fed back into the LLM in a retry loop (up to a configurable retry budget). Successful round-trips are cached so identical natural-language queries do not re-hit the LLM.',
        'The §5.1 panel exposes both halves of the round-trip: a natural-language input box at the top, the emitted DSL in the middle (user-editable), and a "start run" button at the bottom. Editing the DSL between runs lets you A/B compare variations — change a coefficient, re-run, compare the resulting traces in the timeline panel. This is the workflow the visualiser is optimised for.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'param-update',
      caption: 'Once the DSL\'s hamiltonian terms have learnable coefficients, this is the update rule that adjusts them — same rule as §4.3, sourced via DSL.',
    },
    {
      kind: 'workedExample',
      example: {
        title: 'A 2-field DSL through to the first frame\'s snapshot',
        setup:
          'Author a minimal 2-field DSL (one scalar phi, one scalar chi with a Yukawa coupling), trace it through dsl_to_runspec, and identify what the first frame\'s snapshot contains.',
        steps: [
          {
            description:
              'DSL: `fields: [{name: phi, grid: [8, 8], init: zero}, {name: chi, grid: [8, 8], init: gaussian}]; hamiltonian: [{name: kin_phi, op: ddx(phi)^2, coeff: 0.5}, {name: kin_chi, op: ddx(chi)^2, coeff: 0.5}, {name: yukawa, op: phi chi^2, coeff: 0.1}]; observables: [{name: phi_mean, op: phi}, {name: chi_mean, op: chi}]; run: {frames: 50, trotter_steps: 2, kappa_R: 0.01, seed: 1}`.',
            result: 'DSL parsed into a 4-key dictionary; 2 fields, 3 Hamiltonian terms, 2 observables, run-config set.',
            equationId: 'free-energy-functional',
          },
          {
            description:
              'dsl_to_runspec maps: fields -> RunSpec.field_specs = [(name=phi, grid=(8,8), init=zero), (name=chi, grid=(8,8), init=gaussian)]; hamiltonian -> RunSpec.ham_terms = [(c=0.5, op=kin_phi), (c=0.5, op=kin_chi), (c=0.1, op=yukawa)]; observables -> RunSpec.obs = [phi_mean, chi_mean]; run -> RunSpec.run_cfg.',
            result: 'Typed RunSpec produced with all four substructures populated.',
            equationId: 'hamiltonian-decomp',
          },
          {
            description:
              '_build_qpcn(spec) reads ham_terms and assembles H = 0.5*H_kin_phi + 0.5*H_kin_chi + 0.1*H_yukawa as a sum of MPO terms. The MPS belief is initialised as the product state |phi=zero, chi=gaussian>. The QPCN is now live.',
            result: 'QPCN built: H = 0.5*kin_phi + 0.5*kin_chi + 0.1*yukawa; |psi_0> = |zero, gaussian>.',
            equationId: 'param-update',
          },
          {
            description:
              'Frame 0 runs the per-frame loop once. PCN layers initialise from field_specs (phi=zero, chi=gaussian). Stress-energy is zero on frame 0 (no errors yet); manifold stays flat. QPCN takes 2 Trotter steps of imag-time; MPS relaxes slightly toward the ground state of H. Operator expectations phi_mean and chi_mean are read out (phi_mean ~ 0, chi_mean ~ gaussian-mean).',
            result: 'Frame 0 substrate state: PCN as initialised, manifold flat, MPS one Trotter step in.',
          },
          {
            description:
              'Snapshot extractor packs: PCN beliefs (phi, chi fields), manifold (flat, with trotter_steps=2 metadata field — frame 0 only), QPCN summaries (<phi_mean>=0, <chi_mean>=0.5, entanglement entropy, bond profile), Hamiltonian params (c_kin_phi=0.5, c_kin_chi=0.5, c_yukawa=0.1), free energy F = <H>, frame metadata. The JSON payload is shipped to the viz.',
            result: 'Frame-0 snapshot shipped; viz renders the first frame; trotter_steps=2 noted for the run.',
          },
        ],
        takeaway:
          'A nine-line DSL describing a 2-field Yukawa system became a fully-typed RunSpec, became a live substrate, became a frame-0 snapshot, became a viz rendering — without the user writing any Python. The DSL is the most efficient entry point into the QPCN; the LLM round-trip raises that further to natural language. The §5.1 panel is where users author and iterate on DSL documents; editing a DSL and re-running is the canonical "A/B compare" workflow.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'param-update',
      expect: [
        'Editing a DSL between runs lets you A/B compare — the timeline panel can overlay multiple completed RunResults.',
        'LLM-emitted DSL almost always parses on first try when the prompt template is well-formed.',
        'Parser errors are surfaced inline in the DSL editor with line numbers.',
        'Run-config changes (frames, kappa_R) take effect on the next run; substrate changes (fields, hamiltonian) require a fresh substrate build.',
      ],
      pathologies: [
        { signal: 'LLM keeps emitting invalid DSL', cause: 'Prompt template missing schema examples. Update the few-shot exemplars in the template.' },
        { signal: 'DSL parses but the run produces no observables', cause: 'Observables list is empty or operator expressions evaluated to identity. Recheck the observables block.' },
        { signal: 'Run starts but immediately stalls', cause: 'kappa_R too large for the chosen field amplitudes; coupling overshoots from frame 0. Reduce kappa_R in the run block.' },
        { signal: 'Identical DSL produces different results across runs', cause: 'Missing seed in the run block. Add `seed: <int>` for reproducibility.' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §11.5 (DSL)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/viz/dsl.py', href: '../../dsl.py' },
  ],
};
