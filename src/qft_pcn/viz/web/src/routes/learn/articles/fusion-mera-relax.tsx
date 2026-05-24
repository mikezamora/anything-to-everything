// src/qft_pcn/viz/web/src/routes/learn/articles/fusion-mera-relax.tsx
/**
 * §4.5 Fusion — MERA-Relax: the §10.10 induction-theorem demo with
 * Forall-protected leaves. The headline test of the logic substrate.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'fusion-mera-relax',
  title: 'Fusion: MERA-Relax — the induction-theorem demo',
  sectionPath: ['§4 QPCN Fusion', '4.5 MERA-Relax'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The MERA-Relax panel runs the §10.10 architecture demo: relax the term `forall x : Nat. Eq (x + Zero) x` to its proof. This is the smallest non-trivial inductive theorem (commutativity of `Zero` for `+`), and it tests every piece of the logic-as-Hamiltonian substrate at once — the typing rules, the reduction rules, the universal-quantifier handling, and the MERA-style tensor-network coarse-graining that makes the relaxation tractable on multi-bit naturals.',
        'The headline mechanism is **Forall-protection**. When the term has a universally-quantified subterm `forall x. P(x)`, the bound-variable position `x` is marked as a *protected leaf*: the MPS tensors at that position are constrained to encode the "any natural" superposition, and the reduction dynamics are forbidden from collapsing them to a concrete value. The proof has to go through for all `x` simultaneously — the dynamics cannot cheat by instantiating `x = Zero` and walking the easy path.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The relaxation proceeds in stages. **Stage 1: R-AddZero** fires across the entire `x + Zero` subtree. Because `x` is a protected superposition over all naturals, the rule applies pointwise in the superposition: each component-of-x sees a `+ Zero` and reduces to itself. After stage 1, the term has reduced to `forall x : Nat. Eq x x`. **Stage 2: R-Eq-Refl** then fires on `Eq x x` — reflexivity of equality applied to a syntactically identical pair gives `True`. After stage 2, the term has reduced to `forall x : Nat. True`, which is the proof term.',
        'The §4.5 panel exposes both stages with separate per-rule residual traces. You should see R-AddZero\'s residual drop first (the leftmost subtree gets resolved), then R-Eq-Refl\'s residual drop (once the syntactic equality is exposed). The total H_eval drops in two visible plateaus rather than one smooth descent — the signature of the staged reduction.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'hamiltonian-decomp',
      caption: 'The evaluation Hamiltonian for this run — one term per rule (R-AddZero, R-Eq-Refl, typing rules); the protected leaves enter as boundary conditions on the MPS.',
    },
    {
      kind: 'prose',
      body: [
        'The MERA part of the name refers to the multi-scale entanglement renormalisation ansatz used to encode the multi-bit naturals. A Nat is represented as a binary expansion to some max-bit-width B; the MERA layers coarse-grain bits into scales, with isometries and disentanglers between layers. This is why the relaxation is tractable: bond entanglement between coarse-grained bits stays bounded even when B is large, so the MPS does not blow up. The §1.2 article on tensor networks introduced the MERA structure; this panel is the first place it shows up in earnest.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'entanglement-entropy',
      caption: 'The entanglement entropy across the protected-leaf cut — should stay near log_2(2^B) = B, the entropy of the uniform superposition.',
    },
    {
      kind: 'prose',
      body: [
        'The protected-leaves invariant is the single property to watch most carefully. The MPS tensors at protected positions should be **bitwise stable** across the entire run: their amplitudes should not change. Any drift indicates the dynamics are leaking into the protected sector, which would corrupt the universal quantifier. The panel exposes this as a "protected-leaf delta" trace that should stay at machine zero (~1e-15). If it climbs, the run is invalid.',
        'Imag-time evolution on H_eval is performed with the protected-leaf positions held fixed: at each Trotter step, after applying the gate, the protected-leaf tensors are restored from their initial values. This is a projection rather than a soft constraint — it guarantees exact protection, at the cost of slightly slower convergence on rules whose patterns straddle the protected boundary.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'Imag-time evolution restricted to the non-protected subspace — protected leaves stay at their initial values exactly.',
    },
    {
      kind: 'prose',
      body: [
        '**The §10.10 demo\'s significance.** This is the first end-to-end inductive theorem the QPCN can prove using nothing but the substrate itself — no external interpreter, no hand-coded proof-search heuristic, no manual case split. `forall x : Nat. Eq (x + Zero) x` is small as theorems go, but it is the smallest one whose proof requires a *universal* claim about an infinite domain (every natural number), not merely a closed-form rewrite. Producing the proof from the bare evaluation Hamiltonian — protected leaves plus the standard rule set — closes the loop opened by the §10 logic-as-Hamiltonian construction: it demonstrates that the same physical relaxation that performs reduction can also perform *deduction*, on an inductively defined type.',
        'The significance for the broader architecture is twofold. First, it constitutes the smallest existence proof that the QPCN is computationally adequate for *proof* and not merely for *evaluation* — these are different competences, and many evaluation-by-relaxation systems can do the former without the latter. Second, it gives a concrete benchmark for everything that follows: if the §4.5 panel can drive this term to `Forall(x, Nat, True)` with the protected-leaf delta at machine zero, then more elaborate inductive proofs (associativity, distributivity, the §10 standard library) are differences of degree rather than kind. The demo is the floor of the architecture\'s logical reach, not its ceiling.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**Why protected leaves matter for universal quantification.** The §1.1 architecture-soul directive is that variable binding *is* bond entanglement — never a classical name-lookup table. Protected leaves are the operational consequence of this directive when a binder is universal. If `x` is a free name in a closed term, classical evaluators substitute a value for it; the QPCN entangles its leaf positions and lets the rules act on the entangled superposition. For `forall x. P(x)`, the bound positions hold a uniform superposition over the entire type (here `Nat` truncated to bit-width B), and the bond entanglement encodes the constraint *every occurrence of `x` carries the same value across the superposition*. Without that entanglement, two occurrences of `x` would be independent, and the dynamics could collapse them to different values — `forall x. Eq x x` would become `forall x, y. Eq x y`, an entirely different (and false) claim.',
        'Protection enforces this by freezing the protected tensors against the dynamics. The rule projectors are still allowed to *match patterns containing* the protected positions (otherwise R-Eq-Refl could never fire on `Eq x x`), but they are not allowed to *modify* those positions. The combination — entangled superposition + bitwise freeze — is the QPCN\'s mechanism for honest universal quantification. The result is that any rule that succeeds on a protected term has succeeded on *every* instantiation of the bound variable simultaneously, which is exactly what universal quantification means. The "protected-leaf delta" trace in the panel is therefore not a diagnostic of the simulator\'s numerical health; it is a diagnostic of whether the proof is honest. A nonzero delta means the system has secretly weakened the claim, and the resulting normal form is not a proof of the original term.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Identify protected leaves for forall x : Nat. Eq (x + Zero) x',
        setup:
          'Take the term `forall x : Nat. Eq (x + Zero) x`. Encode it as an AST, identify which leaf positions are bound by the forall, and write down the protected-leaf set explicitly.',
        steps: [
          {
            description:
              'Parse the AST. Top-level: Forall(binder = "x", type = Nat, body = Eq(...)). Body: Eq(lhs = Add(Var("x"), Zero), rhs = Var("x")). The leaves are: Var("x") (in the +Zero subtree), Zero (literal), Var("x") (the rhs).',
            result: 'Leaf positions: Var("x")@lhs.left, Zero@lhs.right, Var("x")@rhs.',
          },
          {
            description:
              'The forall binds the name "x". Every leaf whose value is Var("x") is a *bound-variable occurrence* of the forall, and therefore must be protected. Zero is a literal, not bound — it is not protected. So the protected positions are exactly the two Var("x") leaves: lhs.left and rhs.',
            result: 'Protected leaf set = {lhs.left, rhs} — the two occurrences of Var("x"). Zero is unprotected.',
            equationId: 'entanglement-entropy',
          },
          {
            description:
              'In the MPS encoding, each Var("x") leaf is represented as a uniform superposition over all naturals up to the bit-width B: |x_uniform> = (1/sqrt(2^B)) * sum_{n=0..2^B-1} |n>. The two protected positions hold this state; they are also entangled together (same x), which is the bond-entanglement encoding of variable binding from §1.1.',
            result: 'Each protected leaf holds |x_uniform>; the two are bond-entangled, encoding name equality.',
          },
          {
            description:
              'Apply Stage 1: R-AddZero. The rule pattern matches Add(_, Zero) and rewrites to _. Because lhs.left is protected, the dynamics apply the rule *pointwise in the superposition*: each |n> component of Var("x") sees an Add(n, Zero) and reduces to n. After the rule, lhs becomes Var("x") (the protected superposition, unchanged), and the lhs.right Zero leaf is consumed.',
            result: 'After Stage 1: term = Eq(Var("x"), Var("x")); protected leaves bitwise-identical to start.',
            equationId: 'hamiltonian-decomp',
          },
          {
            description:
              'Apply Stage 2: R-Eq-Refl. Pattern matches Eq(a, a) for syntactically identical a. The two Var("x") leaves are entangled and (by §1.1) represent the *same* x — they are syntactically identical in the protected encoding. The rule fires, rewriting Eq(Var("x"), Var("x")) to True. The protected leaves are released (no more occurrences) and the term is now Forall(x, Nat, True), the proof term.',
            result: 'After Stage 2: term = Forall(x, Nat, True). Protected-leaf check throughout: delta = 0 at every frame.',
            equationId: 'imag-time-evolution',
          },
        ],
        takeaway:
          'The two Var("x") leaves are the protected leaf set for this theorem; they remain bitwise-identical throughout the entire two-stage reduction, even though the rules around them rewrite the AST significantly. This bitwise protection is exactly what makes the result an honest universal proof rather than a check at a single instantiation. The §4.5 panel\'s "protected-leaf delta" trace should be at machine zero for the entire run; if it climbs, the run is invalid and should be re-examined.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'imag-time-evolution',
      expect: [
        'Protected-leaf delta stays at machine zero (~1e-15) for the entire run.',
        'R-AddZero residual decays in Stage 1; R-Eq-Refl residual decays in Stage 2.',
        'Total H_eval shows two clear plateaus, one per stage.',
        'Bond entanglement across the protected cut stays near log_2(2^B).',
      ],
      pathologies: [
        { signal: 'Protected-leaf delta climbs above ~1e-10', cause: 'Projection is failing — dynamics leaking into the protected sector. Audit the protection mask.' },
        { signal: 'R-Eq-Refl fires before R-AddZero', cause: 'Rule scheduling is wrong; Eq-Refl is matching the unrelaxed lhs. Check rule priority or projector commutation.' },
        { signal: 'Bond entanglement collapses to ~0', cause: 'Protected superposition has been instantiated to a single n. The forall is no longer universal — invalid proof.' },
        { signal: 'Total H_eval plateaus above zero after Stage 2', cause: 'A typing rule is still firing — possibly Forall-handling is missing. Re-examine the Hamiltonian.' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §10.10 (induction-theorem demo)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/qft/logic/mera_evaluation_hamiltonian.py', href: '../../../qft/logic/mera_evaluation_hamiltonian.py' },
  ],
};
