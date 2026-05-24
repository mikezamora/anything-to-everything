// src/qft_pcn/viz/web/src/routes/learn/articles/foundations-riemannian.tsx
/**
 * §1.4 Foundations — Riemannian geometry: metric, curvature, Ricci scalar,
 * Laplace-Beltrami, presented from a software-engineer baseline.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'foundations-riemannian',
  title: 'Riemannian geometry: metric, curvature, Ricci',
  sectionPath: ['§1 Foundations', '1.4 Riemannian Geometry'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The QPCN\'s manifold is the geometric substrate on which both the PCN fields and the QFT operator expectations live. To read any Fusion-side article you need a working model of three Riemannian concepts: the **metric**, the **curvature**, and the **Laplace–Beltrami operator**. None of these are deep — they are concrete numerical objects you can compute on a grid — and we will keep them concrete.',
        'Start with the metric. A metric `g_{mu nu}(x)` is, at each point `x` of the manifold, a small symmetric positive-definite matrix that tells you how to compute distances and inner products of tangent vectors at that point. In two dimensions (our default), it is a `2x2` matrix per grid point. If `g` is the identity at every point, you have flat Euclidean space and lengths are the usual `sqrt(dx^2 + dy^2)`. If `g` varies, the local "ruler" stretches and rotates from point to point: the same coordinate displacement `(dx, dy)` corresponds to different physical lengths in different regions. In code, the metric is just an `ndarray` of shape `(Nx, Ny, 2, 2)`.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The QPCN parameterises the metric as a flat background plus a learnable perturbation: `g = eta + h`, with `eta` the constant identity-like reference metric and `h` a learnable field of small `2x2` symmetric corrections. The reason for the split is numerical: it keeps the metric well-conditioned for small `h`, and it lets the learning rule act on the perturbation rather than the full metric, which is cleaner.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'metric-perturbation',
      caption: 'The metric decomposition used throughout the QPCN — flat reference plus learnable perturbation field.',
    },
    {
      kind: 'prose',
      body: [
        'Curvature is the obstruction to "everything looks flat locally" being globally true. The intuitive picture: pick a small loop in the manifold, parallel-transport a tangent vector around it, compare to its starting orientation. On a flat space (a plane, an infinite cylinder) it comes back unchanged. On a sphere, it returns rotated by an angle proportional to the enclosed area. The Riemann curvature tensor `R^a_{bcd}` is the bookkeeping device that records this rotation for arbitrary loop orientations and starting vectors. It is built from the metric by a deterministic, somewhat ugly formula involving partial derivatives of `g` and inverse-metric contractions — in our code it is computed by `manifold.py` as a finite-difference stencil and we never write it out by hand.',
        'The **Ricci tensor** `R_{mu nu}` is a contracted Riemann tensor (sum over two of its indices); it measures how the volume of a small ball deviates from the Euclidean value. The **Ricci scalar** `R` is the Ricci tensor contracted with the inverse metric: one number per point. A flat region has `R = 0`. A sphere has positive `R`. A saddle has negative `R`. For our purposes the Ricci scalar is the single most-displayed curvature quantity — it gives a coordinate-invariant "how curved is it here" heatmap.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'ricci-scalar',
      caption: 'The Ricci scalar — one real number per grid point, sign tells you whether the patch is sphere-like (+), flat (0), or saddle-like (−).',
    },
    {
      kind: 'prose',
      body: [
        'Finally, the **Laplace–Beltrami operator** `Delta_g` is the natural generalisation of the flat Laplacian to a curved metric. On flat space it is just `d^2/dx^2 + d^2/dy^2`; on a curved manifold it picks up `sqrt(|g|)` weight factors so that the resulting diffusion respects the geometry. Concretely, if you diffuse a scalar field `Phi` according to `dPhi/dt = Delta_g Phi`, the field spreads faster in directions where the metric is "smaller" (closer points in physical length) and slower in directions where it is "larger." This is the mechanism by which manifold curvature reshapes belief flow in the QPCN: a region of high error sources curvature, the curvature deforms `Delta_g`, and Phi propagates preferentially along the deformed geodesics.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'laplace-beltrami',
      caption: 'Belief diffusion via Laplace–Beltrami: geometry literally sets the flow of information across the manifold.',
    },
    {
      kind: 'prose',
      body: [
        'A practical note about how this lives in the code. We never store the full Riemann tensor; we compute the Ricci tensor and scalar directly from the metric via finite differences. The metric perturbation `h` is initialised to zero (so we start exactly flat), and it is updated by descending a coupling-weighted contribution from the PCN error stress-energy — `h_{mu nu} += kappa_R * T_{mu nu}[E]`. When `kappa_R = 0`, the manifold stays flat and the QPCN behaves like a vanilla PCN plus a fixed-background QFT. As `kappa_R` increases, the manifold starts curving in response to errors, and the §4.1 panel will show you the resulting hot-spots of `R`. This single coupling constant is the most informative knob to turn when first exploring the visualizer.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Confirm R = 0 for a 2x2 flat patch',
        setup:
          'Take a tiny manifold consisting of 2x2 grid points, with the metric set to the 2x2 identity matrix at every point. Compute the Ricci scalar at the centre.',
        steps: [
          {
            description:
              'The metric at every grid point is g = [[1, 0], [0, 1]]. The inverse is also g^{-1} = [[1, 0], [0, 1]]. Every entry of g is constant across the grid, so all partial derivatives of g_{ij} with respect to x and y are zero.',
            result: 'All d g_{ij} / dx^k = 0 across the patch.',
          },
          {
            description:
              'The Christoffel symbols Gamma^a_{bc} are built from these partial derivatives via Gamma = 0.5 * g^{ad} * (d_b g_{cd} + d_c g_{bd} - d_d g_{bc}). Every term in the parentheses is zero, so every Gamma^a_{bc} = 0.',
            result: 'All Christoffel symbols vanish.',
            equationId: 'metric-perturbation',
          },
          {
            description:
              'The Riemann tensor R^a_{bcd} is built from Christoffel symbols and their derivatives: R = d Gamma - d Gamma + Gamma Gamma - Gamma Gamma. All Gammas are zero and their derivatives are zero, so every component of the Riemann tensor is zero.',
            result: 'R^a_{bcd} = 0 identically.',
          },
          {
            description:
              'The Ricci tensor R_{bd} = sum_a R^a_{bad} is a contraction of zero, hence zero. The Ricci scalar R = g^{bd} R_{bd} = sum_{bd} g^{bd} * 0 = 0.',
            result: 'R = 0 at the centre (and at every point) — the patch is flat, as expected.',
            equationId: 'ricci-scalar',
          },
          {
            description:
              'Sanity check the Laplace–Beltrami operator on this flat patch. With g = I, sqrt(|g|) = 1 and g^{mu nu} = delta^{mu nu}, so Delta_g Phi reduces to the flat Laplacian d^2 Phi / dx^2 + d^2 Phi / dy^2 — exactly what you would get without ever having mentioned a metric.',
            result: 'Delta_g reduces to the standard flat Laplacian, confirming the formalism is consistent.',
            equationId: 'laplace-beltrami',
          },
        ],
        takeaway:
          'A flat manifold has zero curvature and a standard Laplacian; this is the limit the QPCN starts from at kappa_R = 0. Everything interesting in §4.1 happens when the PCN error sources non-trivial h_{mu nu} and the metric stops being flat — but the same machinery (finite-differenced Christoffels, contracted Riemann, scalar Ricci) computes R(x) at every point on the grid.',
      },
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3.1', href: '../../QFT_PCN_ARCHITECTURE.md' },
  ],
};
