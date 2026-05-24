// src/qft_pcn/viz/web/src/routes/learn/articles/pcn-dynamics.tsx
/**
 * §3.2 PCN side — Dynamics: total free energy as system-level objective,
 * per-layer contributions, and how learning rate sets descent speed.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'pcn-dynamics',
  title: 'Dynamics: free energy and descent',
  sectionPath: ['§3 PCN side', '3.2 Dynamics'],
  prerequisites: ['pcn-fields', 'foundations-variational-fe'],
  sections: [
    {
      kind: 'prose',
      body: [
        'Given the per-site fields `(Phi, E, Pi)` from §3.1, we now ask: what does the whole network actually minimise? The answer is a single scalar, the total free energy `F`, summed over every site of every layer. Per site, the contribution is the precision-weighted squared error minus a log-precision regulariser; the layer total is the integral over the manifold; the network total is the sum over layers. There is one number, `F`, and *every* learnable thing in the network — the beliefs `Phi`, the precisions `Pi`, the parameters of every top-down generative map — descends gradients of that same `F`.',
        'This is the single most important architectural fact about a PCN, and it is what justifies the "one objective, many substrates" framing of the QPCN. Inference is not separate from learning; the bottom layer\'s observation matching is not separate from the upper layers\' prior matching; the precisions and the means and the weights are not on different schedules in any deep sense. They differ only in *which derivative of F* they follow, and in the step size attached to that derivative.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Concretely, write the per-layer contribution as `F_l = integral_M [ 0.5 * Pi_l * E_l^2 - 0.5 * log(Pi_l) ] dV_g`, where the integral is over the manifold with volume element `dV_g = sqrt(det(g)) d^2x`. The total free energy is `F = sum over l of F_l`, plus, in the QPCN, a coupling term `F_QFT` carrying the QFT-side observation-target loss. The classical PCN trains with the QFT-side term absent; the QPCN turns it on and lets it shape upper-layer beliefs via the same gradient machinery.',
        'Each layer\'s contribution is *local in `(Phi_l, Phi_{l+1}, Pi_l)`* (plus the parameters of the down-map from `l+1` to `l`). So `dF/dPhi_l` only depends on `E_l` (own-layer term) and `E_{l-1}` propagated upward through the layer-`l` down-map\'s Jacobian. That locality is what makes PCN updates cheap and (philosophically) biologically plausible — no global error pipe, no per-parameter chain rule across the whole stack.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The dynamics. Define a step size `eta_Phi` for the belief updates and `eta_Pi` for the precision updates (typically `eta_Pi << eta_Phi`, so precisions move on a slower scale than beliefs). The continuous-time descent is `dPhi_l/dt = -eta_Phi * dF/dPhi_l` and `dPi_l/dt = -eta_Pi * dF/dPi_l`. In code (`network.py`), this is a discrete Euler step: read the current `Phi`, compute the gradient by running predictions down and errors up, then update `Phi <- Phi - eta_Phi * grad`. Repeat. The "training loop" and the "inference loop" are literally the same loop.',
        'The crucial parameter is `eta_Phi` — the inference learning rate. Too small and the network is slow to settle and slow to react to new observations. Too large and the discrete Euler steps overshoot the bottom of the quadratic and oscillate. Because `F` is locally quadratic in `Phi` (the error term is `0.5 * Pi * E^2`), the stability boundary is roughly `eta_Phi * max(Pi) < 2`. In practice, with `Pi` on the order of 1 to 10, `eta_Phi` lives somewhere in `0.01` to `0.1`. A site with unusually high precision can be the one that destabilises the whole network — a useful diagnostic when oscillation appears.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '"Well-posed observations" is a technical condition that matters a lot in practice: it means the observations driving layer 0 are consistent with *some* setting of the upper-layer beliefs reachable through the generative maps. If yes, `F` decays monotonically (modulo step-size noise) and converges to a fixed point that is the network\'s inference. If no — for example, if you suddenly feed a 16×16 dataset to a network whose generative map can only produce 8×8-resolvable structures — `F` will descend to a *positive* asymptote, the residual error reflecting what the model fundamentally cannot represent. That residual is a meaningful diagnostic, not a bug.',
        'A second pathology: if observations are *internally* inconsistent across the layer-0 field — for example, two regions demand mutually contradictory upper-layer causes — `F` may not strictly monotonically decay; instead it can oscillate as the inference flicks between two attractors. In that case you do not have a step-size problem; you have an *under-determined inference problem*, and the fix is either to add information (more observations, a tighter prior) or to widen the generative map\'s expressive capacity at the relevant layer.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Where do `Pi` updates come from? Take the derivative of `F_l` with respect to `Pi_l`: `dF_l/dPi_l = 0.5 * E_l^2 - 0.5 / Pi_l`. Setting this to zero (the fixed-point condition for precision) gives `Pi_l = 1 / E_l^2` — the natural-parameter form of the Gaussian, the precision equals the inverse of the empirical squared error. That is the limit the slow `Pi` dynamics is descending toward: precision tracks the inverse of the running squared error, which is the variational-Gaussian posterior\'s estimate of the noise floor at that site.',
        'In code this is implemented as a slow exponential moving average of `1 / max(E^2, epsilon)`, which is mathematically equivalent to a small-step gradient descent on `F` with a learning rate proportional to `Pi^2` (the rescaling from `dPi` to `d(log Pi)`). The `epsilon` floor prevents `Pi` from blowing up at sites where the model is essentially perfect and the error happens to be machine-zero. Without it, perfect sites would dominate the loss landscape and freeze the rest of the network.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Why "free energy" and not just "loss"? The reason is partly historical and partly load-bearing. Historically: this is Friston\'s terminology, imported from variational Bayes for biological-plausibility narratives. Load-bearing: `F` is *not* the same shape as a deep-learning loss — it is `KL(q || posterior) - log(evidence)`, which decomposes (after the Gaussian collapse) into a per-layer sum of "complexity" (the KL between adjacent-layer beliefs) and "accuracy" (the data-fit term at the bottom). Minimising `F` simultaneously fits the data and keeps the upper-layer posteriors close to their priors — automatic regularisation, no separate weight-decay term. Every layer gets the regularisation for free.',
        'In the QPCN context, `F` is the *only* objective. The QFT side adds an observation-target term, the manifold adds a curvature-coupling term, the multi-field layer adds a Yukawa cross-term — but they all sit inside the same `F`, all parameters descend the same total, and the convergence proof (such as it is) is one Lyapunov argument on `F`. Articles in §4 (Fusion) will show this explicitly; here in §3.2 we set up the classical-only special case so the cross-field generalisation in §3.3 and the QFT coupling in §4 land cleanly.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'free-energy-functional',
      caption: 'Total F is a sum over layers of the per-layer free-energy density integrated over the manifold. The classical PCN has only these terms; the QPCN adds a QFT-coupling term that lives in the same F.',
    },
    {
      kind: 'equation',
      equationId: 'coupling-descent',
      caption: 'Every learnable parameter — Phi, Pi, generative-map weights, and (in §3.3) the Yukawa couplings g_ij — descends gradients of the same F. The descent equation is the same shape for all of them; only the gradient and the step size differ.',
    },
    {
      kind: 'equation',
      equationId: 'laplace-beltrami',
      caption: 'When the metric is non-flat, the inference gradient on Phi picks up a Laplace–Beltrami smoothing term. In the classical PCN with g = identity this reduces to a plain Laplacian; in the QPCN, with a metric sourced by stress-energy, it is the load-bearing piece that couples curvature to belief dynamics.',
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Compute F for a 2-site Gaussian PCN',
        setup:
          'Take a single-layer PCN with two sites. The beliefs are Phi = (1.0, 0.5). The top-down predictions (treated as constants — there is no layer above in this toy) are prediction = (0.4, 0.6). The precisions are Pi = (2.0, 3.0). Use the per-site free energy F_site = 0.5 * Pi * E^2 - 0.5 * log(Pi) and compute the total F = F_site_1 + F_site_2.',
        steps: [
          {
            description: 'Compute the errors: E_1 = 1.0 - 0.4 = 0.6, E_2 = 0.5 - 0.6 = -0.1.',
            result: 'E = (0.6, -0.1).',
          },
          {
            description: 'Compute the squared-error terms: 0.5 * 2.0 * 0.6^2 = 0.36 at site 1, and 0.5 * 3.0 * 0.01 = 0.015 at site 2.',
            result: 'squared-error contribution = 0.36 + 0.015 = 0.375.',
          },
          {
            description: 'Compute the log-precision regulariser: 0.5 * log(2.0) ≈ 0.3466, and 0.5 * log(3.0) ≈ 0.5493. Total ≈ 0.8959.',
            result: 'log-precision contribution ≈ 0.8959.',
          },
          {
            description: 'Combine: F = 0.375 - 0.8959 ≈ -0.5209.',
            result: 'F ≈ -0.5209.',
            equationId: 'free-energy-functional',
          },
          {
            description: 'Now take one inference step. dF/dPhi_1 = Pi_1 * E_1 = 2.0 * 0.6 = 1.2. dF/dPhi_2 = 3.0 * (-0.1) = -0.3. With eta_Phi = 0.1, Phi <- Phi - 0.1 * grad gives Phi_1 = 1.0 - 0.12 = 0.88 and Phi_2 = 0.5 - (-0.03) = 0.53.',
            result: 'After one step: Phi ≈ (0.88, 0.53), already closer to the predictions (0.4, 0.6).',
          },
          {
            description: 'Recompute F at the new Phi. New errors: E = (0.48, -0.07). New F = 0.5 * 2 * 0.2304 + 0.5 * 3 * 0.0049 - 0.8959 ≈ 0.2304 + 0.00735 - 0.8959 ≈ -0.6582. F decreased by ≈ 0.137, as required.',
            result: 'F ≈ -0.6582 (decreased monotonically, as expected for a well-posed step).',
          },
        ],
        takeaway:
          'For well-posed observations, F decreases on every step of size below the stability boundary. The decrease can be measured: it is a Lyapunov certificate that the network is actually doing what the math says. When F stops decreasing or starts oscillating, that is the dynamics telling you something is wrong with either the step size or the well-posedness of the inference problem.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'coupling-descent',
      expect: [
        'F decreases monotonically (up to discretisation noise) for step sizes below the stability bound eta_Phi * max(Pi) < 2.',
        'Phi converges to a fixed point where every Pi * E balances the upward-message contribution from below.',
        'Pi tracks the inverse of the running squared error on a slower timescale (eta_Pi << eta_Phi), so confident sites get louder over training.',
        'When observations change abruptly, Phi tracks the change within a few steps; Pi readjusts over many more.',
        'F asymptotes at a positive value if the generative map cannot represent the observations exactly — that residual is the network\'s irreducible model mismatch, not a numerical artefact.',
      ],
      pathologies: [
        { signal: 'F oscillates between two values per step', cause: 'eta_Phi too large — likely a high-Pi site is past the stability bound; reduce eta_Phi or clip Pi' },
        { signal: 'F asymptotes well above the noise floor', cause: 'generative-map capacity too small to represent observations; widen the down-map or add a layer' },
        { signal: 'F starts to increase mid-training', cause: 'Pi was raised too quickly while Phi had not yet settled; reduce eta_Pi or warm-up Phi first' },
        { signal: 'F is NaN', cause: 'Pi underflow to zero made 1/Pi diverge; raise the epsilon floor on Pi or initialise Pi away from 0' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3.1 (Predictive coding and variational free energy)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'QFT_PCN_ARCHITECTURE.md §4.4 (QFTPCNNetwork — hierarchical stack)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/network.py (QFTPCNNetwork.step: descent on F)', href: '../../src/qft_pcn/network.py' },
    { label: 'src/qft_pcn/layer.py (QFTPCNLayer.update_precision: Pi dynamics)', href: '../../src/qft_pcn/layer.py' },
  ],
};
