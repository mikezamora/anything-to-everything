// src/qft_pcn/viz/web/src/routes/learn/articles/pcn-multifield.tsx
/**
 * §3.3 PCN side — Multi-field: multiple species sharing one manifold,
 * Yukawa coupling g_ij, and coupling descent.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'pcn-multifield',
  title: 'Multi-field PCN: species, Yukawa coupling, coupling descent',
  sectionPath: ['§3 PCN side', '3.3 Multi-field'],
  prerequisites: ['pcn-fields', 'pcn-dynamics'],
  sections: [
    {
      kind: 'prose',
      body: [
        'A single-field PCN carries one `(Phi, E, Pi)` triple per layer. A multi-field PCN carries one such triple *per species* — for example, a `Phi_a` for the colour channel, a `Phi_b` for the depth channel, and so on. All species share the same manifold (the same grid, the same metric, the same hierarchy of layers); they differ only in which physical quantity their belief field is *about*. This factorisation is the predictive-coding analogue of having multiple matter fields on a common spacetime: same geometry, different field content.',
        'The minimum sensible multi-field setup is two species, and the substrate (`src/qft_pcn/multifield.py`) supports an arbitrary number. Each species `a` has its own generative maps `g_a` between layers, its own precisions `Pi_a`, and its own bottom-layer observations (if any). On its own, each species runs the §3.2 dynamics independently. The interesting part — the part that makes "multi-field" worth talking about at all — is the **coupling** that lets species inform each other.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The coupling is *Yukawa-style*: a bilinear cross-field term added to the free energy of the form `F_coupling = sum over (a, b, l) of g_{ab}^{(l)} * <Phi_a^{(l)}, Phi_b^{(l)}>_g`, where `<.,.>_g` is the metric inner product over the manifold and `g_{ab}^{(l)}` is a layer-resolved cross-species coupling matrix. The diagonal `g_{aa}` is conventionally absorbed into the single-species terms; the off-diagonals carry all the cross-field information. The name "Yukawa" is borrowed from particle physics, where a Yukawa coupling is a bilinear `bar(psi) phi psi` term in the Lagrangian — same structural shape, very different content.',
        'Bilinear is the right minimum-complexity choice for two reasons. First, it gives a *symmetric* coupling: if species `a` informs `b`, then `b` informs `a` with the same coupling constant, which is what conservation arguments demand. Second, the gradient of a bilinear is *linear*, so the cross-field message added to `dPhi_a/dt` is just `g_{ab} * Phi_b`, summed over `b`. That linearity keeps the dynamics tractable and the implementation a one-line broadcast over the species index.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Here is the substrate-level picture. `MultiFieldNetwork.step()` (in `src/qft_pcn/multifield.py`) does a normal PCN step for every species independently — top-down predictions, bottom-up errors, gradient on `Phi`. Then, before applying the gradient, it adds the cross-field message: for each layer, for each species `a`, the additional contribution `delta_Phi_a += eta_Phi * sum_b g_{ab} * Phi_b`. (The sign is chosen so that *positive* `g_{ab}` makes the two species align; *negative* `g_{ab}` makes them anti-align.) The result is a per-species update that respects single-species dynamics but is nudged toward cross-species coherence by the coupling.',
        'The coupling matrix `g_{ab}` is itself learnable, by descent on the *same* total `F`. Take the derivative: `dF / dg_{ab} = <Phi_a, Phi_b>_g`. This is just the metric inner product of the two species\' belief fields. When the two species\' fields are *correlated* (the inner product is large positive), the gradient is positive, the descent step on `g_{ab}` is negative — wait, that needs care. The sign convention in `multifield.py` is that `g_{ab}` *enters* `F_coupling` with a minus sign so that positive coupling *lowers* `F` when the fields are positively correlated. With that convention, `dF/dg_{ab} = -<Phi_a, Phi_b>_g`, and the descent rule `g_{ab} <- g_{ab} - eta_g * dF/dg_{ab}` *grows* `g_{ab}` when the fields are correlated. That is the textbook Hebbian intuition: "fields that fire together wire together", emerging directly from gradient descent on free energy.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'A worked intuition. Suppose species `a` is "edge intensity" and species `b` is "depth discontinuity" in a vision setting. Empirically these correlate strongly — depth discontinuities almost always coincide with edges. After enough training the coupling `g_{ab}` grows large, and the multi-field network exploits this: when species `a` sees a strong edge but species `b` is uncertain about depth, the cross-field message from `g_{ab} * Phi_a` will *push `Phi_b` toward a discontinuity* even though species `b`\'s own observations were ambiguous. The two species help each other infer.',
        'Conversely, if you put two genuinely uncorrelated species in the same multi-field network, their cross-field inner products will average to zero, the descent on `g_{ab}` will fluctuate around zero, and over time `g_{ab}` will shrink (since the descent picks up *any* anti-correlation as a negative gradient and the noise floor will drive the magnitude down). The network learns *which* species are worth coupling, not just by how much.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Some subtle points about the substrate. First, the layer-resolved coupling `g_{ab}^{(l)}` is in principle independent at each layer — the network could discover that two species are correlated at the bottom (raw-pixel level) but uncorrelated at the top (semantic level), or vice versa. In practice the visualiser shows a single coupling matrix per layer, and the EQUATIONS entry `multifield-yukawa` writes it without the layer index for readability. Second, the coupling matrix is *not* required to be symmetric in the implementation, but the symmetric part is the only piece that contributes to `F`; the antisymmetric part is gauge.',
        'Third, the cross-field message uses the *metric* inner product, not a flat Euclidean one — this is what makes the multi-field PCN behave correctly when sitting on top of the manifold whose geometry is sourced by the QFT-side stress-energy (the §4.1 fusion). On a flat manifold the inner product is the ordinary sum-of-products; on a curved manifold it picks up `sqrt(det(g))` weighting and the message is bent by curvature. This is the simplest non-trivial place where the §2 (Riemannian) and §3 (PCN) material actually have to talk to each other.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'Why bother with multi-field at all, given that you can always concatenate species into a single bigger field? Two reasons. First, the per-species precisions `Pi_a` let you say "this species is noisy here, that one is reliable" *independently* at each site, which a single concatenated field cannot express without a separate per-channel-per-site precision tensor that ends up looking exactly like the multi-field formulation anyway. Second, the cross-species coupling `g_{ab}` is *explicit and inspectable* — you can read off the learned coupling matrix and see which species the network thinks belong together. That interpretability is load-bearing for the visualiser, which shows the coupling matrix as a heat-map in §3.3 of the viz.',
        'Third, and most important for the QPCN as a whole, multi-field is how multiple physical observables (charge density, momentum, spin) all live on the same QFT substrate in §4.3. The Yukawa coupling here in the classical PCN is the *direct analogue* of the multi-species fermion-boson coupling in the QFT-side Hamiltonian (the cross-species hopping/interaction terms). Same algebraic structure, same gradient shape, same descent rule — the §3.3 ↔ §4.3 mapping is what makes the QPCN a coherent architecture instead of two independent stacks bolted together.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'multifield-yukawa',
      caption: 'The Yukawa cross-species coupling in the free energy. The diagonal entries are absorbed into the per-species F; the off-diagonals couple distinct species via a bilinear term in their belief fields.',
    },
    {
      kind: 'equation',
      equationId: 'coupling-descent',
      caption: 'g_{ab} descends gradients of the same total F as Phi and Pi do. The gradient is the metric inner product of the two species\' fields, giving the Hebbian "correlated fields grow coupling" rule directly.',
    },
    {
      kind: 'equation',
      equationId: 'free-energy-functional',
      caption: 'The multi-field total F = sum_a F_species_a + F_coupling. Per-species terms are exactly the §3.2 single-species F applied to each species independently; F_coupling is the only new piece.',
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Cross-field message for 2 species at one site',
        setup:
          'Take a multi-field PCN with two species, a and b, one layer, one site. The beliefs are Phi_a = 0.8 and Phi_b = -0.3. The coupling is g_{ab} = 0.5 (and g_{ba} = 0.5 by symmetry). With eta_Phi = 0.1, compute the cross-field contribution to dPhi_a/dt and dPhi_b/dt, and then take one step.',
        steps: [
          {
            description: 'The cross-field message added to dPhi_a/dt is delta_a = sum_b g_{ab} * Phi_b. With only species b contributing: delta_a = g_{ab} * Phi_b = 0.5 * (-0.3) = -0.15.',
            result: 'delta_a = -0.15 (Phi_a is pushed downward by the negative Phi_b).',
            equationId: 'multifield-yukawa',
          },
          {
            description: 'Similarly delta_b = g_{ba} * Phi_a = 0.5 * 0.8 = 0.4. Phi_b is pushed upward by the positive Phi_a.',
            result: 'delta_b = 0.4 (Phi_b is pushed upward toward zero from -0.3).',
          },
          {
            description: 'Take one inference step with eta_Phi = 0.1 (ignoring the single-species term for clarity — assume the species are already at their single-species fixed points). New Phi_a = 0.8 + 0.1 * (-0.15) = 0.785. New Phi_b = -0.3 + 0.1 * 0.4 = -0.26.',
            result: 'Phi_a ≈ 0.785, Phi_b ≈ -0.26 — both moved toward each other, as expected for positive coupling.',
          },
          {
            description: 'Now check the gradient on the coupling itself. dF/dg_{ab} = -<Phi_a, Phi_b> = -0.8 * (-0.3) = 0.24. With eta_g = 0.01, the descent step is g_{ab} <- g_{ab} - 0.01 * 0.24 = 0.5 - 0.0024 = 0.4976.',
            result: 'g_{ab} ≈ 0.4976 (decreased slightly because the fields are anti-correlated at this snapshot).',
            equationId: 'coupling-descent',
          },
          {
            description: 'Sanity-check the sign. The fields have *opposite* signs (0.8 vs -0.3), so they are anti-correlated, so the Hebbian rule says coupling should *shrink* — and indeed g_{ab} went from 0.5 to 0.4976. If both fields had the same sign, the inner product would be positive, dF/dg would be negative, and g_{ab} would have grown.',
            result: 'Sign check passes: anti-correlated fields ⇒ coupling shrinks; correlated fields ⇒ coupling grows.',
          },
        ],
        takeaway:
          'The cross-field message g * Phi_b is a one-line addition to dPhi_a/dt; the coupling update is a one-line addition to the descent loop. Two lines of code give you full multi-field dynamics with learnable cross-species coupling. The fact that the same gradient does both inference (move Phi) and coupling-learning (move g) is the same "one F" invariant from §3.2 holding up under the multi-field generalisation.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'multifield-yukawa',
      expect: [
        'Per-species dynamics behave as in §3.2 — Phi descends per-species F to a fixed point modulo the cross-field perturbation.',
        'Cross-field messages g_{ab} * Phi_b push correlated species toward each other and anti-correlated species apart.',
        'Coupling matrix g_{ab} grows for species pairs whose fields are persistently correlated; shrinks for uncorrelated or anti-correlated pairs.',
        'On a manifold with curvature, the cross-field message is metric-weighted: high-volume regions contribute proportionally more to the coupling gradient.',
        'When a species has effectively zero belief field (Phi ≈ 0 everywhere), its couplings to other species cannot grow — the coupling needs *something* to align.',
      ],
      pathologies: [
        { signal: 'g_{ab} grows without bound', cause: 'eta_g too large or no regularisation on g; add a small g^2 penalty or clip the coupling magnitude' },
        { signal: 'Phi_a and Phi_b lock into a degenerate co-mode and ignore observations', cause: 'coupling has overwhelmed the per-species data terms; reduce g or raise per-species Pi' },
        { signal: 'g_{ab} oscillates sign every few steps', cause: 'eta_g too large relative to the timescale of Phi convergence; reduce eta_g so coupling moves slower than beliefs' },
        { signal: 'coupling matrix becomes asymmetric', cause: 'a sign-convention bug or a non-symmetric initialisation — the symmetric part of g is the only physical content; symmetrise periodically' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §3.3 (Multi-field generalisation)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'QFT_PCN_ARCHITECTURE.md §4.5 (MultiFieldNetwork — coupled field types on one manifold)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/multifield.py (MultiFieldNetwork.step: cross-field messages + coupling descent)', href: '../../src/qft_pcn/multifield.py' },
    { label: 'src/qft_pcn/manifold.py (Manifold2D: metric inner product used for cross-field bilinear)', href: '../../src/qft_pcn/manifold.py' },
  ],
};
