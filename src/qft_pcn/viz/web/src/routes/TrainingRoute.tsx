// src/qft_pcn/viz/web/src/routes/TrainingRoute.tsx
/**
 * Training route — a single long-form article walking through how the
 * whole QPCN trains across one run, annotated with the actual frame
 * numbers from the `qpcn.quarter-density-target` preset.
 *
 * Renders via the same LearnArticle component so all section kinds
 * (prose / equation / workedExample / trainingDynamics / callout) work.
 */

import type { ArticleSpec } from '../lib/article-types';
import { LearnArticle } from './learn/LearnArticle';

const trainingArticle: ArticleSpec = {
  id: 'training-end-to-end',
  title: 'Training the QPCN end-to-end',
  sectionPath: ['§T Training'],
  sections: [
    {
      kind: 'prose',
      body: [
        'This article walks through one run of the QPCN end-to-end. We follow what happens between the moment a DSL spec arrives and the moment the run\'s last frame ships its observables back out — annotated with the actual numbers from the `qpcn.quarter-density-target` preset (single species A, target ⟨n_0⟩ = 0.25).',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**§1 The objective.** The whole system minimises variational free energy. PCN does it on the classical fields; QFT does it on the MPS\'s parameters; the manifold metric is the joint substrate that lets them share the objective.',
      ],
    },
    { kind: 'equation', equationId: 'free-energy-functional' },
    {
      kind: 'prose',
      body: [
        '**§2 Substrate construction from the DSL.** `dsl_to_runspec` reads the DSL\'s `fields`, `hamiltonian.terms`, and `observables`, projects them into a `RunSpec.params` flat dict, and the substrate builders (`_build_qpcn`, `_build_network`, etc.) construct the actual Python objects. Frame 0 captures the initial state.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**§3 The per-frame loop.** Each frame does all of the following in sequence:',
        '  1. The PCN network steps: errors propagate up, predictions down, beliefs settle.',
        '  2. The error field\'s stress-energy sources the metric: `h_μν += κ_R T_μν[E]`.',
        '  3. The new metric reshapes belief diffusion via Laplace-Beltrami.',
        '  4. The QPCN evolves: one Trotter step of imag-time on the MPS.',
        '  5. Operator expectations are read out; the prediction error against targets descends the Hamiltonian parameters.',
        '  6. Snapshot extractors collect every layer\'s public state; one Frame ships.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'Step (4) — the MPS relaxes toward the current Hamiltonian\'s ground state.',
    },
    {
      kind: 'equation',
      equationId: 'param-update',
      caption: 'Step (5) — the Hamiltonian\'s learnable parameters chase observation targets.',
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'free-energy-functional',
      expect: [
        'PCN\'s total F decreases monotonically if the prior is well-matched.',
        'QPCN\'s ⟨H⟩ decreases monotonically under imag-time.',
        'Mean |R| concentrates at error spikes, then plateaus, then decays.',
        'Pred-errors shrink as Hamiltonian params adapt.',
      ],
      pathologies: [
        { signal: 'F oscillates', cause: 'PCN learning rate or κ_R too large' },
        { signal: '⟨H⟩ stalls above the ground state', cause: 'χ_max too small (entanglement clipped)' },
        { signal: 'Mean |R| diverges', cause: 'Numerical instability in the metric update; reduce κ_R' },
        { signal: 'Pred-errors freeze nonzero', cause: 'Substrate parameters at a local minimum unrelated to the target' },
      ],
    },
    {
      kind: 'prose',
      body: [
        '**§4 Failure modes.** Three patterns recur: stalling (everything freezes at nonzero error — usually a learning-rate mismatch), mode collapse (the QPCN finds a trivial ground state with all observables = 0 — the target should regularise against this), and gauge-fixing drift (the MPS canonical form decays under repeated truncation — `mps.normalize()` between steps helps).',
      ],
    },
    {
      kind: 'callout',
      severity: 'note',
      body: 'Each per-panel article (§2-§4 of Learn) drills into its panel\'s individual dynamics. This article is the system-level glue.',
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3', href: '../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/qft/qpcn.py', href: '../../src/qft_pcn/qft/qpcn.py' },
    { label: 'src/qft_pcn/network.py', href: '../../src/qft_pcn/network.py' },
  ],
};

export function TrainingRoute() {
  return (
    <div className="training-route">
      <main className="training-route-body">
        <LearnArticle article={trainingArticle} />
      </main>
    </div>
  );
}
