// src/qft_pcn/viz/web/src/routes/learn/articles/fusion-logic.tsx
/**
 * §4.4 Fusion — Logic on the QPCN. The evaluation Hamiltonian: each
 * reduction rule contributes a term whose ground state is the well-typed,
 * fully-reduced program. This is how computation becomes physics.
 */

import type { ArticleSpec } from '../../../lib/article-types';

export const article: ArticleSpec = {
  id: 'fusion-logic',
  title: 'Fusion: logic as a Hamiltonian — evaluation by relaxation',
  sectionPath: ['§4 QPCN Fusion', '4.4 Logic'],
  sections: [
    {
      kind: 'prose',
      body: [
        'The Logic panel reframes program evaluation as a physical relaxation. Instead of running a tree-walking interpreter that reduces an AST step by step, we build a Hamiltonian whose ground state **is** the fully-reduced, well-typed term, and let the QPCN\'s imag-time evolution find that ground state. Each typing rule and each reduction rule contributes one local term to the Hamiltonian; the ground-state energy is zero iff the term is well-typed and fully reduced; any non-zero energy localises the rule violation.',
        'This is not a syntactic curiosity — it is the load-bearing observation that lets the QPCN do logic at all. Bond entanglement is the variable-binding mechanism (from the §1.1 architecture-soul note); reduction-rule terms in H are the dynamics. Putting them together gives a substrate where executing a program is the same operation as relaxing an MPS toward its ground state. The §1.1 directive — variable binding equals bond entanglement, never classical lookup — is what justifies treating reduction as physical evolution.',
      ],
    },
    {
      kind: 'prose',
      body: [
        'The decomposition is `H_eval = sum_R c_R * H_R`, summed over reduction rules R. Each `H_R` is a local projector (or sum of local projectors) onto term configurations that violate R. For example, `R-AddZero` says `n + Zero = n`; the Hamiltonian term `H_{R-AddZero}` is a projector that is 1 on AST configurations of the form `Add(n, Zero)` (where the +Zero pattern is unreduced) and 0 elsewhere. Once the reduction has happened — the `Add(n, Zero)` node has been rewritten to `n` — the projector evaluates to 0 and the term contributes no energy.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'hamiltonian-decomp',
      caption: 'The evaluation Hamiltonian — one term per reduction rule, learnable coefficients let us weight them (default: equal).',
    },
    {
      kind: 'prose',
      body: [
        'The ground state of `H_eval` is the term in normal form: all reduction rules are satisfied (their projectors return zero) and the program has been fully evaluated. Imag-time evolution from any well-formed initial term will steer the MPS toward this ground state, performing the reductions in whatever order the diffusion + Trotter dynamics happen to favour. There is no fixed evaluation order; the system finds *an* order that gets the energy down.',
        'A practical wrinkle: the projector for one rule may overlap (non-commute) with another rule\'s projector when the patterns share nodes. This is fine; it just means the dynamics interleave their reductions rather than doing one rule to completion before the next. The §4.4 panel\'s per-term residual plot shows exactly this interleaving — you see multiple residuals decay in parallel rather than serially.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'imag-time-evolution',
      caption: 'Imag-time evolution on H_eval — the same machinery from §4.3 — performs the reductions automatically.',
    },
    {
      kind: 'prose',
      body: [
        'There is also a typing-rule subsystem. Each typing rule (e.g. `T-AddNat: if e1 : Nat and e2 : Nat then Add(e1, e2) : Nat`) contributes a projector that is nonzero on ill-typed configurations. A program that does not type-check has a nonzero ground-state energy under the typing part of H — the imag-time dynamics will struggle to relax such a term, which is the QPCN\'s analogue of a "compile error". The panel exposes this: ill-typed inputs show a residual that never decays below a threshold.',
        'This is also where soundness of the encoding becomes visible. If the Hamiltonian is wrong (a rule\'s projector has the wrong pattern), some valid normal-form programs will have nonzero energy, and you\'ll see them stuck. If it is correct, every type-correct, fully-reduced term sits at exactly zero. The architecture document\'s §10 walks through this construction rule by rule; the panel lets you check the result interactively.',
      ],
    },
    {
      kind: 'equation',
      equationId: 'param-update',
      caption: 'The rule-coefficient update — letting some rules carry more weight can speed up convergence on programs that need them.',
    },
    {
      kind: 'prose',
      body: [
        '**Programs as Hamiltonians.** The construction in this article has a clean type-theoretic reading. Under Curry–Howard, a well-typed term is a proof; under the QPCN, a well-typed term is *the ground state of an evaluation Hamiltonian*. The two readings match: a configuration that violates a typing rule is a non-proof and carries energy from the typing projectors; a configuration that violates a reduction rule is an unfinished proof and carries energy from the reduction projectors; the ground state is the unique configuration that violates neither, i.e. a fully-reduced, well-typed term — a proof in normal form. Computation, in this picture, is not a sequence of state transitions performed by an external interpreter; it is the natural relaxation of a physical system toward its lowest-energy configuration.',
        'This reframing matters because it turns "running a program" into the same kind of activity as "finding the ground state of a Hamiltonian" — a problem the QFT-side machinery already knows how to do. The §4.3 imag-time evolution, the §4.5 MERA relaxation, the §4.1 manifold all become directly applicable to evaluation. There is no special "interpreter subsystem" — the substrate that does inference is the substrate that does computation. The cost is that the Hamiltonian has to be carefully constructed (every typing and reduction rule must contribute exactly the right projector); the benefit is that everything we learn about relaxing physical systems transfers immediately to evaluating logical ones.',
      ],
    },
    {
      kind: 'prose',
      body: [
        '**Why imag-time relaxation equals beta-reduction.** Conventional beta-reduction is a discrete rewrite: spot a redex `(lambda x. body) arg`, substitute `arg` for `x` in `body`, repeat. The QPCN replaces this with a continuous-time relaxation. The beta-rule contributes a projector `P_beta` that is nonzero on configurations containing an un-reduced redex; the imag-time step `exp(-tau H_eval)` suppresses exactly those configurations and amplifies the substituted form. As `tau` grows, the unreduced amplitude shrinks exponentially, and the state converges to the reduced configuration — the same final state classical beta-reduction would produce, reached as a fixed point rather than a sequence of edits.',
        'The dynamical-systems view has three practical consequences. (i) **Evaluation order is emergent, not prescribed.** Multiple redexes in the same term are reduced in parallel, weighted by how strongly each appears in the current amplitude; the system finds *an* order that drives the energy down, with no fixed left-to-right or innermost-first policy. (ii) **Non-terminating reductions show as energy plateaus, not as infinite loops.** A term that classically diverges (e.g. `Omega = (lambda x. x x)(lambda x. x x)`) produces a Hamiltonian whose ground state lies outside the bond-dimension envelope of the MPS; the energy plateaus above zero, signalling non-termination without hanging the interpreter. (iii) **Confluence becomes a spectral statement.** Two reduction strategies that reach the same normal form correspond to two paths in the imag-time flow that converge on the same ground state; if they converge on different ground states, the rule system is non-confluent and the Hamiltonian has degenerate minima — which the panel exposes as a residual that the dynamics cannot drive uniformly to zero.',
      ],
    },
    {
      kind: 'workedExample',
      example: {
        title: 'Residual for R-AddZero on NatLit(5) + Zero',
        setup:
          'Take the program term `Add(NatLit(5), Zero)`. The R-AddZero rule says `n + Zero -> n` for any n. Write the residual energy from the R-AddZero Hamiltonian term and verify it is non-zero on this input and zero after reduction.',
        steps: [
          {
            description:
              'The R-AddZero projector P_{AddZero} acts on a sub-tree pattern matching Add(_, Zero) where the second child is exactly the Zero constructor. Concretely, P_{AddZero} = |Add><Add| (root) tensor I (left child) tensor |Zero><Zero| (right child). It is a projector — it returns the basis component that matches the pattern, zero otherwise.',
            result: 'P_{AddZero} = |Add><Add| (x) I (x) |Zero><Zero|.',
            equationId: 'hamiltonian-decomp',
          },
          {
            description:
              'Apply to the input term |Add, NatLit(5), Zero>: the root matches |Add>, left child matches anything (the I), right child matches |Zero>. So <psi|P_{AddZero}|psi> = 1.0. The contribution to H_eval at this node is c_R * 1 = c_R (with c_R = 1 by default).',
            result: 'Residual energy from R-AddZero on the input: 1.0 (with default c_R = 1).',
          },
          {
            description:
              'Apply imag-time evolution. The R-AddZero reduction operator U_{R-AddZero} sends Add(n, Zero) to n — it rewrites the tree. After one (or a few) Trotter steps, the state becomes mostly the rewritten term |NatLit(5)>.',
            result: 'After reduction: state is approximately |NatLit(5)>.',
            equationId: 'imag-time-evolution',
          },
          {
            description:
              'Now apply P_{AddZero} to |NatLit(5)>: the root no longer matches |Add>, so the projector returns zero. <psi_new|P_{AddZero}|psi_new> = 0. The residual for this rule is now zero.',
            result: 'After reduction: R-AddZero residual = 0.',
          },
          {
            description:
              'Sanity check the other rule residuals. NatLit(5) is a value — no further reduction applies — so every other reduction-rule projector also returns zero on it. Total energy = 0; the program is in normal form. The typing residual: NatLit(5) : Nat is well-typed, so the typing projector also returns zero. All residuals zero confirms successful evaluation.',
            result: 'Total H_eval = 0 on |NatLit(5)>; the program has been fully evaluated.',
            equationId: 'param-update',
          },
        ],
        takeaway:
          'A single reduction rule\'s residual went from 1.0 to 0 as the imag-time dynamics rewrote Add(NatLit(5), Zero) into NatLit(5). This is the smallest possible demonstration of evaluation-by-relaxation. The §4.4 panel runs this on much larger terms (the standard library, induction proofs, the §10.10 demo); you should see exactly this pattern of per-rule residuals dropping to zero one by one (or in interleaved parallel) as the program reduces.',
      },
    },
    {
      kind: 'trainingDynamics',
      updateRuleId: 'imag-time-evolution',
      expect: [
        'Per-rule residuals drop to zero in succession (or in interleaved parallel for non-commuting rules).',
        'Total H_eval decreases monotonically under imag-time, hitting zero on fully-reducible programs.',
        'Typing-rule residuals stay near zero throughout (well-typed input).',
        'Per-rule coefficient c_R, if learned, drifts mildly to favour rules on the critical path.',
      ],
      pathologies: [
        { signal: 'Total H_eval plateaus above zero', cause: 'Program is not fully reducible under current rule set (missing a rule, or genuinely stuck term). Add the rule or fix the input.' },
        { signal: 'Typing residual stays high', cause: 'Input is ill-typed; the QPCN cannot type-check it. Examine the AST.' },
        { signal: 'Residuals oscillate', cause: 'Two reduction rules form a loop (e.g. unconditional rewrite back-and-forth). Audit rule confluence.' },
        { signal: 'Some residuals never decay', cause: 'Their projector pattern never matches the current term — likely a rule whose pattern is too narrow. Check pattern indices.' },
      ],
    },
  ],
  citations: [
    { label: 'QFT_PCN_ARCHITECTURE.md §10 (Logic-as-Hamiltonian)', href: '../../QFT_PCN_ARCHITECTURE.md' },
    { label: 'src/qft_pcn/qft/logic/evaluation_hamiltonian.py', href: '../../../qft/logic/evaluation_hamiltonian.py' },
  ],
};
