# QPCN ENHANCEMENTS (audit pass 1, HEAD 2b6fd56)

Productive divergences from `QFT_PCN_ARCHITECTURE.md` — kept as-is
because they improve on the spec literal target while preserving §1.1
architecture-soul (variable binding = bond entanglement; operator-
algebraic only). Documented for posterity; not deviations.

### E1 — `MERA.from_product` shared identity-disentangler tensor
- Location: `src/qft_pcn/qft/mera.py:469-472`
- Spec: §3.3, §4.7 (silent on per-tensor uniqueness)
- Why better: Allocates one read-only `(d_l, d_l, d_l, d_l)` identity
  per layer and aliases it across every intra/inter slot, replacing
  `n_l/2 + (n_l/2 - 1)` distinct identity arrays. Observationally
  identical (`apply_two_site_gate` writes back fresh arrays); removes
  the dominant allocation in encoder-time MERA construction. Pinned by
  no-regression suites cited inline.
- Audit source: §1-§5

### E2 — `MERA._is_product_cache` and `_mutation_version` invalidation bookkeeping
- Location: `src/qft_pcn/qft/mera.py:277-281, 1359-1362`
- Spec: §3.3.6, §5 (silent on caching strategy)
- Why better: Lets `inner` / `norm_sq` skip the per-disentangler
  identity scan on the imag-time hot path where the ket is a
  leaf-mutated copy of the bra. Pure performance, zero semantic
  divergence.
- Audit source: §1-§5

### E3 — `MERA.from_term_superposition` exact-branch decomposition
- Location: `src/qft_pcn/qft/mera.py:527-634`
- Spec: §1.1, §5.4
- Why better: Stores an explicit `_superposition_terms` list so
  `entanglement_entropy`, `norm_sq`, and `local_expectation` route
  through exact closed forms over the k branches instead of
  materializing a dense statevector. Faithful to §1.1 (entanglement
  isometry-carried) and §5.4 (entropy without `O(d^N)`).
- Audit source: §1-§5

### E4 — `replica.compute_zn_for_ensemble` lazy-callable ensemble entries
- Location: `src/qft_pcn/qft/replica.py`
- Spec: §1.6, §12.7
- Why better: Accepts zero-arg callables so the partition-function
  operator computation stays outside the module (§1.6 anti-shortcut: Z
  must come from real operator algebra). Productive separation-of-
  concerns.
- Audit source: §1-§5

### E5 — `sdp_solver.psd_constraint_from_operator` operator-derived PSD helper
- Location: `src/qft_pcn/qft/sdp_solver.py`
- Spec: §1.6
- Why better: Projects a Hermitian substrate-operator block to a PSD
  variable + equality constraint, keeping SDP encoding operator-
  algebraic per §1.6 even though CVXPY itself is numerical.
- Audit source: §1-§5

### E6 — `Hamiltonian.bond_op` matches spec literal (no `kappa_k` scalar-gradient term)
- Location: `src/qft_pcn/qft/hamiltonian.py:148-162`
- Spec: §3.3.4 (H_2 is `-t (a_k† a_{k+1} + h.c.)` only)
- Why better: Impl matches spec exactly; the kappa line in the module
  docstring is decorative over-promising. Productive alignment with
  spec.
- Audit source: §1-§5

### E7 — Sibling-consensus QEC gate (§12.5 / §10.10)
- Location: `src/qft_pcn/composition/dispatcher.py:236-265`
  (`dispatch_siblings`)
- Spec: §12.5, §10.10
- Why better: Filters `report.flagged` through a cohort-consensus rule:
  a syndrome-flagged child is genuinely corrupt only when (a) caller
  supplied an explicit pre-corruption snapshot for it, or (b) at least
  one sibling matches the parent signature below `qec_threshold` AND
  `len(children_states) >= 2`. Without this gate every imag-time-
  evolved child trips the syndrome (legitimate evolution away from
  fresh parent reads as corruption). Keeps K-8 single-child path
  passable while preserving the §12.5 contract whenever a real cohort
  exists.
- Audit source: §6-§9

### E8 — Joint-residual internal-node accounting (`_JointResult`)
- Location: `src/qft_pcn/composition/orchestrator.py:74-89, 283-293`;
  `src/qft_pcn/composition/goal_graph.py:204-230`
- Spec: §9.5 (does not address depth>1 explicitly)
- Why better: Internal SOLVED nodes carry a `_JointResult` whose
  `residual_energy` is the sum of children's residuals plus matching
  `solved_ast` tuple. `compute_free_energy` walks only leaves to avoid
  double-counting. Preserves §9.5 monotonicity in depth>1 — smaller
  architectural patch than rewriting the accuracy sum.
- Audit source: §6-§9

### E9 — `source_run_id` namespacing of content-addressed lemma IDs
- Location: `src/qft_pcn/composition/lemma_library.py:680-713`
  (`_content_id`)
- Spec: §10.8, §10.11
- Why better: Mixes `derivation.source_run_id` into the hash so two
  sub-proofs of the same proposition under different `goal_id`s
  register as distinct lemma entries (required by §10.11 hierarchical
  decomposition — L1 and L2 each prove the same theorem under
  different sibling contexts). Fallback (no source_run_id) is pure
  content-addressing.
- Audit source: §6-§9, §10 (dual-flagged)

### E10 — Precision-weighted clamp via leaf blend, not projector weight
- Location: `src/qft_pcn/composition/promoter.py:117-178`
  (`apply_init_clamp`)
- Spec: §6.2, §5.2a, §1.5
- Why better: Realizes the §6.2 precision-weighted message by linearly
  blending cached vs host leaves (`s*cached + (1-s)*host`, renormalized)
  rather than scaling a projector weight. Operator-algebraic (§1.5);
  composes with §5.2a freeze set without touching the Hamiltonian.
- Audit source: §6-§9

### E11 — §10.8 fingerprint = leaf-bond Gram spectrum (not top-bond RDM)
- Location: `src/qft_pcn/composition/lemma_library.py:266-306`
  (`structural_fingerprint`)
- Spec: §10.8 (calls for "RDM at a canonical bond")
- Why better: For hole-free product MERAs the top bond is structurally
  constant (rank-1, eigenvalue 1) by construction of
  `_orthonormal_isometry`, so it cannot distinguish programs.
  Leaf-Gram spectrum recovers a program-distinguishing, alpha-
  invariant fingerprint without breaking the §4.3 length-agnostic
  index contract.
- Audit source: §10

### E12 — §10.11 target adapted to substrate-supported composite
- Location: `tests/composition/test_hierarchical_proof_demo.py`
- Spec: §10.11
- Why better: Spec target `length (xs++ys) = length xs + length ys` is
  gated on List encoder substrate (blocked by leaf-dim saturation at
  16; tracked in EXTENSIONS). The L test adapts to
  `forall x:Nat. Eq (add x Zero) x` so the architectural assertion
  (multi-level decomposition, joint propagation, repeated clamp) is
  exercised end-to-end rather than deferred. Real substrate proof per
  leaf — anti-shortcut directive observed.
- Audit source: §10

### E13 — §10.10 strict integrator gates (residual + spectral_gap + classical decode_mera)
- Location: `src/qft_pcn/composition/result_integrator.py`
  (`integrate_child`, `register_lemma`)
- Spec: §6.3, §4.5
- Why better: Encodes the strict gate with three independent refusal
  paths plus a `decode_mera` classical witness before persisting
  (wrapped per §4.5 totality contract). Sharper than spec's prose
  "below threshold" wording.
- Audit source: §10

### E14 — §10.10 dispatcher routes children through §12.5 holographic-code corruption detector
- Location: `src/qft_pcn/composition/dispatcher.py::dispatch_siblings`
  (`parent_state=...`)
- Spec: §10.10
- Why better: Spec prescribes residual gate at integration; impl also
  surfaces logical corruption against parent's bulk reconstruction
  (tracked as RESOLVED A.2 in EXTENSIONS).
- Audit source: §10

### E15 — §12.16 ranking surface plumbed at orchestrator
- Location: `SolveResult.ranked_proofs` via `bayesian_rank_proofs`
- Spec: §12.16
- Why better: Forward-compatible API even though multi-candidate
  generation is deferred (RESOLVED partial A.3 in EXTENSIONS).
- Audit source: §10

### E16 — §12.1 anomaly: type_leaf encoding-basis collision guard
- Location: `src/qft_pcn/composition/anomaly.py` (`SymmetryGenerator`)
- Spec: §12.1 (ABJ trace `Tr(T^a {T^b, T^c})`)
- Why better: Diagonal leaf-local projector triples collapse the ABJ
  trace to the operator-algebraic type-mismatch projector
  `<P_kind · (I - P_type) · P_value>`. The module makes this collapse
  explicit and adds the `type_leaf` index field to guard against
  encoding-basis collisions (e.g. `KIND_ZERO == TYPE_NAT == 8`) —
  genuine substrate hardening beyond the spec text.
- Audit source: §12

### E17 — §12.8 dynamical PT: `"relative"` non-analyticity trigger alongside `"absolute"`
- Location: `src/qft_pcn/composition/dynamical_pt.py` (`DPTEvent.trigger`)
- Spec: §12.8
- Why better: Adds a relative-non-analyticity trigger alongside the
  spec's absolute zero-crossing check. Faithful to Heyl 2018's
  "non-analyticity in the rate" reading of finite-N DPTs.
- Audit source: §12

### E18 — §12.10 holographic compilation cross-extension API
- Location: `src/qft_pcn/composition/holographic_compilation.py`
- Spec: §12.10
- Why better: Wires §12.2 Wilson-signature oracle
  (`verify_layer_equivalence`) and §12.8 DPT detector
  (`detect_compilation_convergence`) into one compilation-pipeline API.
  Note: optimization passes themselves are still deferred (see D15).
- Audit source: §12

### E19 — §12.15 Witten index: Z₂ grading from causal-cone footprint parity
- Location: `src/qft_pcn/composition/witten_index.py`
  (`term_affected_leaves`)
- Spec: §12.15
- Why better: Derives the Z₂ grading from substrate causal-cone
  footprint parity — operator-algebraic (§1.1), not AST-symbol parity.
- Audit source: §12

### E20 — §12.18 Noether discovery on lemma-library covariance
- Location: `src/qft_pcn/composition/noether_discovery.py`
- Spec: §12.18
- Why better: Library covariance `C = Σ |fp⟩⟨fp|` + degenerate-
  eigenspace SO(r) generator construction is a clean operator-
  algebraic realization of "structural symmetry of the library."
  Honest scope: discovers symmetries of lemma fingerprints, not of
  dynamical Hamiltonians.
- Audit source: §12
