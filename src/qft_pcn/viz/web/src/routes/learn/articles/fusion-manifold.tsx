// src/qft_pcn/viz/web/src/routes/learn/articles/fusion-manifold.tsx
/**
 * §4.1 Fusion — Dynamic Riemannian manifold sourced by PCN error
 * stress-energy. Establishes how κ_R controls the bidirectional coupling
 * between classical errors and the geometric substrate.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'fusion-manifold',
  title: 'Fusion: a dynamic manifold sourced by error stress-energy',
  sectionPath: ['§4 QPCN Fusion', '4.1 Manifold'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The §1 Foundations chapter on Riemannian geometry described the metric `g_{mu nu} = eta_{mu nu} + h_{mu nu}` as a static substrate on which other things live. The Fusion-side picture takes that substrate and makes it move. The perturbation field `h_{mu nu}(x, t)` is not constant — it is sourced, in every frame, by the PCN error field `E`. The coupling constant `kappa_R` controls how strongly errors deform the geometry: at `kappa_R = 0` the manifold stays flat forever and the QPCN behaves like a vanilla PCN coexisting with an unrelated QFT; as `kappa_R` grows, errors carve curvature into the substrate and that curvature reshapes how subsequent beliefs diffuse.',
        'This article is the bridge between the foundations chapters and the §4.2 coupling chapter. It establishes one specific story: how a localised PCN error spike turns into a localised metric perturbation, and from there into a localised Ricci-scalar bump that subsequent beliefs flow around. Everything later in the Fusion section builds on this picture.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Mechanically, the manifold lives as a `(Nx, Ny, 2, 2)` ndarray of metric components in `manifold.py`. Each frame, the network produces an error field `E(x)` — also a grid quantity. We construct a stress-energy tensor `T_{mu nu}[E]` from `E` in the analogue of `T_{mu nu} = partial_mu phi * partial_nu phi - (1/2) g_{mu nu} (partial phi)^2` for a scalar field, but with `phi -> E`. This gives a symmetric `2x2`-per-point tensor that is large where `E` is large or rapidly varying, and zero where `E` is flat. The update rule is then `h_{mu nu} += kappa_R * T_{mu nu}[E]`, applied each frame.',
        'The reason this is the "right" coupling shape, rather than (say) `h_{mu nu} += kappa_R * E * eta_{mu nu}`, is that stress-energy is a tensor with the right index structure to feed directly into a metric perturbation — it is the way Einstein\'s equations source geometry in actual general relativity. We are not literally doing GR (no `c`, no `8 pi G`, no covariance constraints), but we borrow its index structure as a principled discipline.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'metric-perturbation',
      caption: 'The metric split — `eta` is the flat reference, `h` is the PCN-sourced perturbation that accumulates over frames.',
    },
    {
      kind: 'prose',
      body: [
        'Once `h` is updated, the rest of the Riemannian machinery from §1.4 runs unchanged: finite-differenced Christoffels give the Ricci tensor and scalar, and the Laplace–Beltrami operator on the new metric reshapes belief flow on subsequent frames. The §4.1 panel visualises three things at once: the error field `E` (the source), the Ricci scalar `R` (the response), and the metric-perturbation magnitude `|h|` (the accumulated history). When `kappa_R` is small, all three drift quietly; when it grows, the Ricci heatmap tracks the error heatmap with a lag.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'ricci-scalar',
      caption: 'The Ricci scalar at the centre of the hot-spot is the headline number — it is what the panel\'s mean-curvature trace plots over time.',
    },
    {
      kind: 'prose',
      body: [
        'Two practical knobs change the character of the dynamics. **`kappa_R` (the coupling constant):** small values give a quiet, mostly-flat manifold with mild curvature ripples; large values cause hot-spots to dominate. There is a critical regime around `kappa_R ~ 0.1` for the default presets where the manifold starts forming persistent curved regions rather than just transient bumps. **`diffusion timestep`:** how aggressively the Laplace–Beltrami operator smears the field each frame. Together with `kappa_R`, this sets whether curvature concentrates (high coupling, low diffusion) or smears out (low coupling, high diffusion).',
        'The §4.1 panel\'s default preset deliberately picks parameters in the "interesting" regime: you should see localised curvature appear where the PCN happens to have prediction errors, then diffuse outward as the metric smooths under its own Laplace–Beltrami dynamics. Try setting `kappa_R = 0` from the controls — the Ricci panel will go quiet immediately, confirming the source is genuinely the error field.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'laplace-beltrami',
      caption: 'After the metric updates, subsequent beliefs diffuse under this operator on the new geometry — closing the loop between errors and information flow.',
    },
    {
      kind: 'prose',
      body: [
        '**Why a dynamic metric at all.** The architectural case for a learnable geometry, made in §1 of `QFT_PCN_ARCHITECTURE.md`, is that *belief geometry is the right primitive*. A classical PCN already has an implicit geometry — the Fisher metric on its variational posterior, the precision-weighting that decides which prediction errors matter — but it is buried in scalar variance hyperparameters and never made explicit. By promoting this geometry to a first-class object `g_{mu nu}(x, t)` that lives on the same grid as the beliefs themselves, two things become possible. First, structurally distinct regions of the input domain can develop structurally distinct precision profiles without needing separate hyperparameters per region: the metric *is* the precision profile, indexed by location. Second, the dynamics of that metric — how it deforms under error, how it relaxes under diffusion — become a directly observable, directly tunable part of training, rather than a side-effect of optimiser tricks.',
        'This is the load-bearing departure from §1.1. The metric is not a backdrop for computation; it is *what learns*. Errors curve it; curvature reshapes how subsequent errors propagate; the architecture as a whole is just this loop running indefinitely. Every other Fusion-section construction (the operator coupling in §4.2, the MPS belief state in §4.3, the relaxation dynamics in §4.5) presupposes that the geometry has this status. Drop dynamic geometry and you have two independent systems sharing a grid; keep it and you have a single coupled system whose state lives in the geometry.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**The kappa_R coupling constant in depth.** `kappa_R` is the single scalar that sets the strength with which PCN errors deform the substrate. At `kappa_R = 0` the manifold is decoupled — it stays flat forever and the QPCN is just a PCN and a QFT running in adjacent windows. As `kappa_R` grows, the regime changes qualitatively in three stages. (i) **Weak coupling** (`kappa_R << 0.01`): curvature ripples appear briefly where errors spike, but diffusion erases them within a frame or two. The metric is essentially a noisy version of flat space. (ii) **Critical coupling** (`kappa_R ~ 0.01–0.1` for default presets): curvature persists in regions of repeated error, forming stable wells that subsequent beliefs flow into. This is the interesting regime — it is where the manifold genuinely accumulates structure from data. (iii) **Strong coupling** (`kappa_R > 0.1`): the metric perturbation magnitude can grow faster than diffusion can drain it, and the linearisation `g = eta + h` breaks down.',
        'A useful intuition: `kappa_R` controls how much *memory* the geometry has for the error history. Small `kappa_R` = forgetful manifold (only sees the current frame). Large `kappa_R` = sticky manifold (carries error history for many frames). The §4.2 panel\'s parameter-update rule depends on this memory — if the manifold cannot remember an error long enough for the QFT to respond to it, the bidirectional loop never closes. Tuning `kappa_R` is therefore the primary architectural knob for setting the timescale of the coupled dynamics.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'A Gaussian error hot-spot on a 4x4 patch',
        setup:
          'Take a 4x4 grid. Initialise h = 0 everywhere (manifold starts flat). Drop a Gaussian error E(x, y) = exp(-((x - 2)^2 + (y - 2)^2) / sigma^2) with sigma = 1.0 and amplitude 1.0 — so E peaks at 1.0 at the centre cell (2, 2) and is ~0.37 in the four nearest neighbours. Use kappa_R = 0.1 and compute the central h_xx contribution after one frame.',
        steps: [
          {
            description:
              'At the centre (2, 2) the Gaussian peaks. Approximating partial_x E and partial_y E by central differences: E(3, 2) - E(1, 2) = 0.37 - 0.37 = 0, dividing by 2*dx gives partial_x E ~ 0 at the peak — as expected, since the Gaussian is symmetric. Same for partial_y E.',
            result: 'At the exact peak, gradient contributions vanish.',
          },
          {
            description:
              'Move one cell off-peak to (2, 3) (one step in +y). Now partial_x E(2, 3) = (E(3, 3) - E(1, 3))/(2 dx) is still zero by symmetry, but partial_y E(2, 3) = (E(2, 4) - E(2, 2))/(2 dx) ~ (0.018 - 1.0)/2 = -0.49 (taking dx = 1 for the grid). So at (2, 3) we have partial_y E ~ -0.49 and partial_x E = 0.',
            result: 'Off-peak gradient: partial_y E ~ -0.49 at (2, 3).',
          },
          {
            description:
              'The stress-energy tensor T_{mu nu} = partial_mu E * partial_nu E - (1/2) eta_{mu nu} (partial E)^2 at (2, 3): (partial E)^2 = 0 + (-0.49)^2 = 0.24. T_xx = 0 - 0.5 * 0.24 = -0.12. T_yy = 0.24 - 0.5 * 0.24 = +0.12. T_xy = 0 * -0.49 = 0.',
            result: 'T_xx(2, 3) ~ -0.12, T_yy(2, 3) ~ +0.12.',
            equationId: 'metric-perturbation',
          },
          {
            description:
              'The metric update is h_{mu nu} += kappa_R * T_{mu nu}. With kappa_R = 0.1: h_xx(2, 3) gets -0.012, h_yy(2, 3) gets +0.012, h_xy stays 0. The metric at (2, 3) is now g = eta + h = [[0.988, 0], [0, 1.012]] — slightly compressed along x, slightly stretched along y, exactly as you would expect from a y-aligned gradient of the field.',
            result: 'After one frame: g(2, 3) = [[0.988, 0], [0, 1.012]], perturbed by ~1% — anisotropic, aligned with the error gradient.',
            equationId: 'ricci-scalar',
          },
          {
            description:
              'On subsequent frames, the Ricci scalar at this point becomes nonzero because the metric is no longer constant across the patch. The Laplace–Beltrami operator on the deformed metric now diffuses fields preferentially across the contracted direction (x), encouraging information flow perpendicular to the original error gradient — i.e. away from the hot-spot rather than along it.',
            result: 'R(2, 3) becomes nonzero on frame 2; belief diffusion respects the new geometry.',
            equationId: 'laplace-beltrami',
          },
        ],
        takeaway:
          'A single Gaussian error spike of amplitude 1.0, at kappa_R = 0.1, produces a ~1% anisotropic metric perturbation in one frame. Realistic runs accumulate this over hundreds of frames, with the diffusion term continuously smoothing — the steady state is curvature concentrated near persistent error sources, diffusing slowly into the surrounding flat region. The §4.1 panel shows exactly this concentration-then-diffusion pattern.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'metric-perturbation',
      expect: [
        'Curvature concentrates at error spikes within a few frames.',
        'Mean |R| (curvature averaged across the grid) rises sharply, then plateaus as diffusion balances source.',
        'Hot-spots smear outward over O(10) frames at default diffusion timestep.',
        'At kappa_R = 0, the Ricci panel stays exactly zero; the source is verifiably the error field.',
      ],
      pathologies: [
        { signal: 'Mean |R| diverges', cause: 'kappa_R too large; metric perturbation outpaces diffusion. Reduce kappa_R by 10x.' },
        { signal: 'Curvature spreads uniformly with no hot-spots', cause: 'Diffusion timestep too large; the metric is over-smoothed. Reduce diffusion dt.' },
        { signal: 'Manifold stays flat despite visible errors', cause: 'kappa_R = 0 or stress-energy tensor not being applied. Check the bridge wiring.' },
        { signal: 'Curvature persists after errors vanish', cause: 'Expected — h is accumulative; reset h = 0 between runs if you want a fresh manifold.' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3.1 (Riemannian substrate)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/manifold.py', href: '../../../manifold.py' },
  ],
};
