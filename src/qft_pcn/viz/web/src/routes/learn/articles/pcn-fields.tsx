// src/qft_pcn/viz/web/src/routes/learn/articles/pcn-fields.tsx
/**
 * §3.1 PCN side — Fields: belief Phi, error E, precision Pi per layer; top-down
 * predictions and bottom-up errors; the predictive-coding hierarchy.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'pcn-fields',
  title: 'Fields: belief, error, and precision per layer',
  sectionPath: ['§3 PCN side', '3.1 Fields'],
  prerequisites: ['foundations-variational-fe', 'foundations-riemannian'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The classical PCN layer carries three scalar fields per site: a belief field `Phi`, an error field `E`, and a precision field `Pi`. Every site of the manifold has all three. The belief `Phi_l` at layer `l` is what the network currently *thinks* the layer-below activations are; the error `E_l` is the mismatch between the top-down prediction generated from layer `l+1` and the actual `Phi_l`; the precision `Pi_l` is the inverse-variance the network assigns to that error. These three fields, defined per layer of a hierarchical stack, are the entire state of a classical predictive-coding network — there is no separate weight tensor, no activations buffer, no hidden parameters. Everything the network knows lives in `(Phi, E, Pi)` at every site of every layer.',
        'It helps to think of `Phi` as the noun and `E` as the verb. `Phi` describes what the system currently believes; `E` describes how that belief is wrong, in a direction the system can use to update. `Pi` is the volume knob on the correction: a site with high precision says "this error is real, listen to it"; a site with low precision says "this error is noise, ignore it". The whole hierarchy operates by sending predictions down and errors up; precision modulates how loudly each layer is heard.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The hierarchy itself is just a stack of layers `0, 1, ..., L-1`. Layer `0` sits at the sensory floor — its `Phi_0` is meant to track an observation `x` arriving from the outside world (or, in QPCN, from the QFT expectation `<O>` projected to the manifold). Layer `l+1` predicts `Phi_l` via a top-down generative map `g_{l+1 -> l}(Phi_{l+1})`. The error at layer `l` is then `E_l = Phi_l - g_{l+1 -> l}(Phi_{l+1})`. There is no error at the topmost layer in the same sense — the top layer has only a prior. Likewise, the bottom layer\'s error is special: it compares the belief to the observation, not to a layer below.',
        'Top-down predictions and bottom-up errors travel along the same connections, in opposite directions. This is the load-bearing structural claim of predictive coding: a single, local message-passing rule per layer captures both inference (refine `Phi_l` to reduce `E_l`) and learning (refine the generative map\'s parameters, and `Pi`, to reduce `E` on average). Nothing in the architecture needs a backprop chain; nothing needs a global error signal; the *only* thing crossing layer boundaries is `Phi` going down and `E` going up.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'In the substrate (`src/qft_pcn/layer.py`), each `QFTPCNLayer` owns three NumPy arrays of shape `(H, W)` — one for `Phi`, one for `E`, one for `Pi`. The layer also owns the *parameters* of its outgoing top-down map (in the simplest case a small set of convolution-like weights). The bottom-up error message is computed inside `forward()` as `E = Phi - prediction_from_above`. The precision is updated on a slower schedule via `update_precision()`, which raises `Pi` when squared errors are small relative to its current implied variance and lowers it otherwise. All three updates are gradient descent on the same total free energy `F`; they only differ in which derivative they take.',
        'The `QFTPCNNetwork` (`src/qft_pcn/network.py`) wires layers together top-down: `network.step()` first walks from the top layer to the bottom propagating predictions, then walks back up propagating errors, then takes a small gradient step on every `Phi_l`. After enough steps the belief field settles into a state where `Phi_l` matches the top-down prediction up to the precision-weighted noise floor. That settled state is the network\'s inference about the world; the parameters that survive after many such settlings are what the network has *learned*.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'A useful mental image for precision: each site is a tiny weighing scale. The belief field rests on the scale; the error is the displacement from the prediction; the precision is the stiffness of the spring under the scale. Stiff springs (high `Pi`) snap the belief tightly toward the prediction with small error; weak springs (low `Pi`) let the belief wander even when the error is large. Importantly, the springs are *learnable*: if a site\'s errors are consistently small, the optimiser raises `Pi` (the data is reliable here); if a site\'s errors are consistently large, the optimiser lowers `Pi` (the data is noisy or the model is wrong here). This is how PCNs learn what to trust.',
        'Three caveats are worth saying out loud. First, "precision" in PCN is **per-site, per-layer**, not a single scalar — different parts of the visual field, different layers in the hierarchy, all have their own `Pi`. Second, the generative map `g_{l+1 -> l}` is **directional**: predictions flow down, errors flow up. There is no symmetric weight matrix. Third, `Phi_l` is not the same kind of thing as a deep-learning activation — it is a probabilistic *belief* about the layer-below state, and the dynamics that settle it are quite literally Bayesian inference over a Gaussian family.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'How is this different from a feedforward neural network? In a deep net, layer `l` *computes* layer `l+1` via a forward pass; activations move only upward; gradients move only downward; the two passes are different operations on different objects. In a PCN, layer `l+1` *predicts* layer `l`; the same channel carries predictions down and errors up; the two passes are symmetric in machinery (same arrays, same metric, just different directions of message-passing). The settled `Phi` *is* the inference; there is no separate output head.',
        'How is this different from a Boltzmann machine or an EBM? Those minimise a global energy over the entire joint configuration. PCN minimises a free energy that *factorises across layers* — the per-layer contributions add. That factorisation is what makes PCN updates local (each layer only needs `Phi_{l-1}, Phi_l, Phi_{l+1}, E_l, Pi_l`), and it is also what gives the "biological plausibility" handwave: a real cortex could implement these updates with local circuitry.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'A last connection back to §1.3 (Variational & Free Energy): the three fields `Phi`, `E`, `Pi` are not arbitrary state variables — they are precisely the parameters of the variational Gaussian posterior at each layer. `Phi_l` is the mean, `Pi_l` is the precision (inverse variance), and `E_l` is the residual under the current generative model. The reason PCN code looks so clean is that the algebra of variational inference for Gaussians collapses to exactly these three quantities; everything else (KL terms, log-partition functions, ELBO bookkeeping) reduces to either a quadratic in `E` weighted by `Pi`, or a log-precision regulariser, and the gradients of both have one-line closed forms.',
        'That is why the next article (§3.2 Dynamics) can talk about the *total* free energy as just a sum of per-layer contributions, and why §3.3 (Multi-field) can generalise to multiple species by simply broadcasting `(Phi, E, Pi)` across a species index without touching the layer-to-layer message structure. The fields are the load-bearing primitive; the dynamics and the cross-species coupling are layered on top.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'free-energy-functional',
      caption: 'The continuum free-energy functional is a layer-local quadratic in the error E_l, weighted by the precision Pi_l, minus a log-precision regulariser. Every PCN-side update is a gradient of this F with respect to one of (Phi, Pi, generative-map parameters).',
    },
    {
      kind: 'equation',
      equationId: 'laplace-beltrami',
      caption: 'When the manifold has nontrivial geometry (which the QPCN couples to via §4), per-site error gradients pick up a Laplace–Beltrami term that smooths Phi against curvature. The bare flat-space PCN is the special case g = identity, but the substrate already uses the metric-aware form in layer.py.',
    },
    {
      kind: 'equation',
      equationId: 'metric-perturbation',
      caption: 'The metric perturbation h_munu sourced by stress-energy is what carries error information into the geometry. In §3.1 we treat g as fixed, but the same Phi/E/Pi sit on top of a metric that, in the fusion layer, will be sourced by these very errors.',
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Compute the layer-0 error for a 2-layer network',
        setup:
          'Take a tiny PCN with two layers, each a single site. Layer 1 has belief Phi_1 = 0.8. The top-down generative map from layer 1 to layer 0 is the affine prediction g(Phi_1) = 0.5 * Phi_1 + 0.1. Layer 0 has belief Phi_0 = 0.7 and precision Pi_0 = 4.0. Compute the layer-0 error E_0, the precision-weighted error, and the gradient force on Phi_0.',
        steps: [
          {
            description: 'Compute the top-down prediction at layer 0: prediction = 0.5 * Phi_1 + 0.1 = 0.5 * 0.8 + 0.1 = 0.5.',
            result: 'prediction_0 = 0.5.',
          },
          {
            description: 'Compute the error: E_0 = Phi_0 - prediction_0 = 0.7 - 0.5 = 0.2.',
            result: 'E_0 = 0.2.',
          },
          {
            description: 'Compute the precision-weighted error (this is what the layer-0 contribution to F sees): Pi_0 * E_0^2 = 4.0 * (0.2)^2 = 4.0 * 0.04 = 0.16. Half of that — 0.08 — is the squared-error term in F_layer-0.',
            result: 'Pi_0 * E_0^2 = 0.16; layer-0 F contribution from squared-error = 0.08.',
            equationId: 'free-energy-functional',
          },
          {
            description: 'Compute the gradient force on Phi_0 (the inference update): dF/dPhi_0 = Pi_0 * E_0 = 4.0 * 0.2 = 0.8. With a small step size eta, Phi_0 will be pulled down toward the prediction 0.5 by an amount eta * 0.8 per step.',
            result: 'dF/dPhi_0 = 0.8; Phi_0 descends toward 0.5.',
          },
          {
            description: 'Now think about the bottom-up message. The same E_0 = 0.2 is also what layer 1 will *see* as its bottom-up signal when it next updates Phi_1. So a positive E_0 means layer 1 will be told "your prediction was too low" and the gradient on Phi_1 will push it upward (through the generative map\'s Jacobian, here 0.5).',
            result: 'Bottom-up message to layer 1: dF/dPhi_1 picks up -0.5 * Pi_0 * E_0 = -0.4 from this error.',
          },
        ],
        takeaway:
          'The layer-0 error E_0 has two jobs in one number: it pulls Phi_0 down toward the prediction (inference), and it pushes Phi_1 upward through the top-down map\'s Jacobian (it is the bottom-up message). One quantity, two flows of information — that is the whole content of the PCN message-passing rule.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'free-energy-functional',
      expect: [
        'On each step, predictions are computed top-down (layer L-1 down to layer 0), then errors are computed bottom-up (layer 0 up to layer L-1).',
        'Phi at every layer descends Pi * E in the direction that reduces its own error contribution and the contribution it makes to the layer above.',
        'Without observations driving layer 0, the hierarchy settles into a fixed point where every E is at the noise floor implied by its Pi.',
        'With observations on layer 0, the settled Phi at upper layers is the network\'s inference about the latent causes of the observation.',
        'Pi updates on a slower schedule: high-confidence sites grow Pi, low-confidence sites shrink it. This is how the network learns what to trust.',
      ],
      pathologies: [
        { signal: 'Phi oscillates from step to step', cause: 'inference step size eta_Phi too large; reduce it or add damping' },
        { signal: 'errors grow unboundedly at one layer', cause: 'generative-map parameters diverging; check learning rate on the down-map and clip if needed' },
        { signal: 'Pi collapses to zero everywhere', cause: 'log-precision regulariser disabled or weighted too low; restore it or anneal more slowly' },
        { signal: 'top layer\'s Phi never moves', cause: 'no prior gradient — top layer has no error of its own; expected behaviour for a flat prior, fix by adding a top-layer prior term if learning is required there' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3.1 (Predictive coding and variational free energy)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'QFT_PCN_ARCHITECTURE.md §4.2 (Fields — representation, error, precision)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/layer.py (QFTPCNLayer: Phi/E/Pi arrays + forward/update)', href: '../../src/qft_pcn/layer.py' },
    { label: 'src/qft_pcn/network.py (QFTPCNNetwork.step: top-down predictions, bottom-up errors)', href: '../../src/qft_pcn/network.py' },
  ],
};
