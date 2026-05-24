// src/qft_pcn/viz/web/src/routes/learn/articles/orientation.tsx
/**
 * §0 Orientation — what the QPCN is and how to navigate the Learn outline.
 * Prose-first; no equations required.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'orientation',
  title: 'Orientation — what the QPCN is',
  sectionPath: ['§0 Orientation'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The QPCN (Quantum Predictive Coding Network) fuses two computational substrates that grew up far apart. Predictive coding (PCN) is a model from theoretical neuroscience that frames perception as the minimisation of prediction error along a hierarchy of generative models — every layer predicts the layer below it, the residual flows back up, and the system relaxes when predictions match. Quantum field theory (QFT) is the framework physics uses to describe interacting fields on a continuous background; in our setting we use a tensor-network discretisation (matrix product states, MERA) that you can think of as a structured, low-rank way to store a wavefunction over many sites.',
        'The thesis of this project: a learning system whose substrate is a genuine QFT will outperform classical deep networks on data that has algebraic, conservation-law, or compositional structure — chemistry, physics, formal logic, programming, structured causal inference. The QPCN is the engineering instantiation of that thesis: a Python codebase where a PCN hierarchy and a tensor-network QFT live on a shared Riemannian manifold and exchange information through that geometry.',
        'If you have written modern deep nets, treat this as "what if the latent space were not a flat vector space but a quantum state living on a curved manifold whose curvature is sourced by the prediction error?" That is the whole pitch in one sentence; the rest of the Learn route unpacks each clause.',
      ],
    },
    {
      kind: 'callout',
      severity: 'note',
      body: 'No physics background assumed. Foundations §1.1 – §1.4 teach the math from a software-engineer baseline; then §2 builds the QFT side, §3 builds the PCN side, and §4 explains how they fuse.',
    },
    {
      kind: 'prose',
      body: [
        'This visualizer is a window into a small running instance of the QPCN. Each panel shows one substrate layer — a manifold patch, the MPS bond structure, a Hamiltonian term graph, PCN field activations — and updates per frame as the run progresses. Most panels embed an Explainer pane on the right; the same content is mirrored, in longer form, in the Learn route articles you are reading.',
        'A useful mental model: the panels show you live state, the Explainer tabs show you "what is happening on this frame and why," and the Learn articles give you the textbook so the panels make sense in the first place. The three views agree on terminology and on a small palette of role colours (input, learnable parameter, constant, output, state, observable) — once you have those colours internalised, every equation you see in the viz tells you at a glance which symbols are knobs, which are knobs you turn, and which are what comes out.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'How to read the Learn route. The chapters are ordered as a textbook, not as a reference. Foundations (§1) covers the math vocabulary you need before either substrate makes sense: ndarrays, Hilbert space, variational inference, Riemannian geometry. The QFT side (§2) then builds tensor networks (MPS, MERA), the Hamiltonian, and variational quantum circuits on top of that vocabulary. The PCN side (§3) does the same for fields, dynamics, and the multifield extension. Fusion (§4) is where it all becomes one system — the dynamic manifold, the bidirectional coupling, the logic-evaluation Hamiltonian. The DSL chapter (§5) covers how natural-language problems are translated into the QPCN\'s execution format.',
        'You can also read it as a reference: skip to the per-panel article that matches whatever panel you are staring at, and follow its prerequisites links back as far as you need. Every article cites its substrate source file so you can drop into the Python and verify what the prose claims.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'A note on rigor. We are deliberately writing this for software engineers who can compute, not for physicists. That means we use ndarray-and-loop intuition liberally; we will say "a tensor is a multi-dim array with named legs" before we say "an element of the tensor product of vector spaces." Where a piece of math has a genuine subtlety that bites you in the code (gauge ambiguity in canonical-form MPS, numerical instability of explicit Christoffel symbols, the difference between F and -ELBO sign conventions), the relevant article calls it out in a callout. Everything else is an honest, hand-computable presentation that should let you read the codebase and modify it without needing to consult external references.',
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §1', href: '../../QFT_PCN_ARCHITECTURE.md' },
  ],
};
