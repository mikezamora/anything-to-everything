// src/qft_pcn/viz/web/src/routes/learn/articles/foundations-variational-fe.tsx
/**
 * §1.3 Foundations — variational inference, KL, free energy, and the
 * predictive-coding Gaussian special case.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'foundations-variational-fe',
  title: 'Variational inference and free energy',
  sectionPath: ['§1 Foundations', '1.3 Variational & Free Energy'],
  sections: [
    {
      kind: 'prose',
      body: [
        'Predictive coding is one specific way of doing variational inference, and variational inference is one specific way of doing approximate Bayesian inference. So before we can read the PCN-side articles, we need a clean engineer-level model of all three. Start from the inference problem: there is some hidden cause `z` (the brightness of a wall behind a shadow, the next token in a sentence, the latent state of a physical system) and there is an observation `x`. Bayes\' rule says the posterior is `p(z|x) = p(x|z) p(z) / p(x)`. The denominator `p(x) = integral of p(x|z) p(z) dz` is generally an intractable integral over a high-dimensional latent space; we want to avoid computing it.',
        'Variational inference\'s answer: pick a tractable family of distributions `q(z; phi)` parametrised by `phi` (Gaussian, mean-field, normalising-flow, whatever) and minimise some divergence between `q(z; phi)` and the unknown true posterior `p(z|x)`. The clever part is using a divergence — Kullback-Leibler, `KL(q || p)` — that, after algebra, decomposes into terms you actually can evaluate even though `p(z|x)` is intractable.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The algebra goes like this. By definition `KL(q || p(.|x)) = E_q[log q(z) - log p(z|x)] = E_q[log q(z) - log p(x, z) + log p(x)]`. Pull the `log p(x)` (a constant in `z`) out of the expectation: `KL = -ELBO + log p(x)`, where `ELBO = E_q[log p(x, z) - log q(z)]` is the **evidence lower bound**. Rearranging: `log p(x) = ELBO + KL(q || p(.|x))`. Since `KL >= 0`, `log p(x) >= ELBO` — maximising the ELBO with respect to `phi` simultaneously tightens the bound and minimises the KL to the true posterior, without ever needing to compute `p(z|x)` directly.',
        'In the **free-energy** convention used in predictive coding, we flip the sign: `F = -ELBO = KL(q || p(.|x)) - log p(x)`. Minimising `F` is the same as maximising ELBO. The reason to prefer `F` is partly historical (Friston\'s neuroscience literature uses it) and partly because "minimise an energy" feels right for a dynamical system that descends a gradient. In the QPCN both PCN and the QFT side minimise free energy; the only thing that differs is what the distributions and the energy functional are.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Predictive coding is the special case where every distribution involved is Gaussian, and the variational family `q` factorises across layers. Concretely: the belief at layer `l` is a Gaussian with mean `Phi_l` (the predicted value of the layer-below activations) and precision `Pi_l` (one over the variance, treated as a learnable parameter). The "observation" at the bottom is data; the "prior" at layer `l+1` predicts `Phi_l` via a top-down generative model. The KL between these Gaussians, after dropping constants, becomes the very simple sum `F = sum over layers of (1/2) * Pi_l * E_l^2 - (1/2) * log Pi_l`, where `E_l` is the prediction error at layer `l` (the difference between the layer-`l+1` prediction and the actual `Phi_l`).',
        'That is why predictive coding code looks like "compute errors top-down, send errors back up, do gradient descent on `Phi_l` and `Pi_l`": the whole training loop is literally `dPhi/dt = -dF/dPhi`, `dPi/dt = -dF/dPi`. No backprop, no replay buffer, no exotic optimiser — descend `F`, get inference + learning together. The QPCN inherits this structure verbatim; the QFT side just adds a tensor-network state whose own parameters also descend `F` (via imaginary-time evolution and observation-target gradients).',
      ],
    },
    {
      kind: 'prose',
      body: [
        'A subtle point for the curious. In standard variational inference, "data" is fixed and you optimise `q` to match a posterior. In PCN, the analog of `q` is the running belief field, and observations stream in continuously — there is no batch boundary, no separate "training" vs "inference" phase. Inference *is* the dynamics of `Phi` descending `F`; learning *is* the slower dynamics of `Pi` and the prior-predictive parameters descending `F`. The same gradient drives both, on different time scales. This unification is the part of PCN that justifies the "biological plausibility" claim: a single local rule explains both perception and learning.',
        'When we get to §4.2 (Fusion: PCN coupling) we will see how the QFT side joins this same `F`: its observable expectations become predicted observations at the bottom of the PCN hierarchy, and its Hamiltonian parameters descend the same total `F` along with everything else. One objective, many substrates.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'free-energy-functional',
      caption: 'The continuum form: a manifold integral of precision-weighted squared error minus the log-precision regulariser. The next worked example computes a discrete one-site instance by hand.',
    },
    {
      kind: 'equation',
      equationId: 'coupling-descent',
      caption: 'Once free energy is your objective, every learnable parameter — including the cross-field Yukawa couplings — descends the same F.',
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Compute F for a 1-site Gaussian',
        setup:
          'Take a single PCN site with belief Phi = 1.0, a top-down prediction = 0.4 (so prediction error E = Phi - prediction = 0.6), and precision Pi = 2.0. Use the per-site free-energy contribution F_site = 0.5 * Pi * E^2 - 0.5 * log(Pi) and compute F_site numerically.',
        steps: [
          {
            description:
              'Compute the squared error term: 0.5 * Pi * E^2 = 0.5 * 2.0 * (0.6)^2 = 0.5 * 2.0 * 0.36 = 0.36.',
            result: 'Squared-error term = 0.36.',
          },
          {
            description:
              'Compute the log-precision regulariser: 0.5 * log(Pi) = 0.5 * log(2.0) ≈ 0.5 * 0.6931 ≈ 0.3466.',
            result: 'Log-precision term ≈ 0.3466.',
          },
          {
            description:
              'Combine: F_site = 0.36 - 0.3466 ≈ 0.0134.',
            result: 'F_site ≈ 0.0134.',
            equationId: 'free-energy-functional',
          },
          {
            description:
              'Now check the gradient direction by perturbing Pi. dF/dPi = 0.5 * E^2 - 0.5 / Pi = 0.5 * 0.36 - 0.5 / 2 = 0.18 - 0.25 = -0.07. Negative gradient means F decreases as Pi grows — the optimiser will raise precision when the error is small relative to the current precision\'s implied variance.',
            result: 'dF/dPi ≈ -0.07; Pi will be increased by gradient descent on F.',
          },
          {
            description:
              'And the gradient with respect to Phi (through E = Phi - prediction): dF/dPhi = Pi * E = 2.0 * 0.6 = 1.2. So Phi descends toward the prediction, with a force proportional to the precision-weighted error. That is exactly the predictive-coding update rule.',
            result: 'dF/dPhi = 1.2; Phi will be pulled toward the top-down prediction.',
          },
        ],
        takeaway:
          'Free energy gives you both the inference update (move Phi to reduce precision-weighted error) and the learning update (move Pi up when the model is confident, down when it is not), from one gradient. Every PCN article in §3 is unpacking one part of this same minimisation.',
      },
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' },
  ],
};
