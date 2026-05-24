# Audit Pass 1 — §10 sub-projects — HEAD 2b6fd56

Scope: §10.8 lemma library (I), §10.9 wake-sleep (J), §10.10 cross-level
(K-8), §10.11 hierarchical demo (L), plus I-10 substrate (R-AddZero /
R-Eq-Refl / forall_protected_leaves).

## ENHANCEMENTS (productive divergence)

- **§10.8 fingerprint = leaf-bond Gram spectrum, not top-bond RDM**
  (`composition/lemma_library.py::structural_fingerprint`, lines 266-306).
  Spec calls for "RDM at a canonical bond"; for hole-free product MERAs
  the top bond is structurally constant (rank-1, eigenvalue 1) by
  construction of `_orthonormal_isometry`, so it cannot distinguish
  programs. The leaf-Gram spectrum recovers a program-distinguishing,
  alpha-invariant fingerprint without breaking the §4.3 length-agnostic
  index contract. Docstring is explicit. This is a real architectural
  refinement of an otherwise vacuous spec read.

- **§10.8 content-addressing extended with `source_run_id` namespace**
  (`lemma_library._content_id`, lines 680-713). Pure content-addressing
  collapsed two L1/L2 sub-proofs of the same proposition to one entry;
  namespacing the hash by `derivation.source_run_id` makes hierarchical
  decomposition (§10.11) yield two distinct lemma entries while
  preserving dedup for genuine re-saves (source_run_id=None). Documented
  in EXTENSIONS.md (architectural note around line 564).

- **§10.11 target adapted to substrate-supported composite.** The spec
  target `length (xs++ys) = length xs + length ys` is gated on List
  encoder substrate (EXTENSIONS.md "List arithmetic in encoder substrate"
  — blocked by leaf-dim saturation at 16). The L test
  (`test_hierarchical_proof_demo.py`) adapts to `forall x:Nat. Eq (add x
  Zero) x` so the architectural assertion (multi-level decomposition,
  joint propagation, repeated clamp) is exercised end-to-end rather
  than deferred. Anti-shortcut directive is observed: it is a real
  substrate proof per leaf, not a stub, and the gating is documented at
  the top of the test file. ENHANCEMENT, not a deviation, because the
  spec gates the literal target on a documented EXTENSIONS entry and the
  architectural acceptance (composition mechanism) is preserved.

- **§10.10 strict integrator gates (residual + spectral_gap + classical
  cross-check via `decode_mera`).** `result_integrator.integrate_child`
  encodes the spec §6.3 strict gate explicitly with three independent
  refusal paths; `register_lemma` also runs `decode_mera` as a
  classical witness before persisting. The decode call is wrapped in
  try/except per the spec §4.5 totality contract. This is sharper than
  the spec's prose "below threshold" wording.

- **§10.10 dispatcher routes children through §12.5 holographic-code
  corruption detector** (`dispatcher.dispatch_siblings(...,
  parent_state=...)`). Spec §10.10 prescribes residual gate at
  integration; the implementation additionally surfaces logical
  corruption against the parent's bulk reconstruction, tracked in
  EXTENSIONS as RESOLVED A.2.

- **§12.16 ranking surface plumbed at orchestrator** (`SolveResult.
  ranked_proofs` via `bayesian_rank_proofs`), tracked as RESOLVED
  (partial) A.3 in EXTENSIONS. Multi-candidate generation is deferred,
  the API is forward-compatible.

## DEVIATIONS (must fix)

- **§10.9 CONSOLIDATE step is incomplete: the new primitive subsumes
  the *cached lemma* rather than the spec's "re-derive the lemma using
  the new primitives".** `composition/wake_sleep.py::_consolidate`
  (lines 120-165): when one cluster member's mined sub-MERA matches a
  newly-promoted primitive's canonical density, the entire cached lemma
  `sid` is added to `stale_dynamic` and pruned via `library.prune`. The
  spec §10.9 pseudocode (architecture lines 877-882) says:
  "FOR each |Ψ_i⟩ in library: IF can_be_expressed_using_new_primitives:
  replace with shorter solution" — i.e. re-derive a *shorter* MPS that
  uses the new primitive, then `prune_redundant_lemmas`. The current
  code does neither: it does not synthesise a replacement (no
  `library.replace(...)` call), it just deletes the parent lemma when
  a single sub-piece matches. This silently destroys lemmas whose
  larger structure has not been replaced. NOT tracked in EXTENSIONS.md
  (no consolidate/re-derive entry).

- **`use_log` field overloaded with two distinct senses
  (provenance vs. subsumed-source-ids) by design comment, but the
  semantic conflation propagates into the adapter's tier inference.**
  `lemma_library_adapter.LemmaLibraryAdapter` docstring (lines 90-94)
  states the tier_of fallback infers "core" from a "non-empty use_log";
  meanwhile `wake_sleep._consolidate` appends subsumed-sid markers to
  `prim.provenance.use_log`, and `LemmaLibraryAdapter.replace` appends
  `"replace:{old_id}"` markers there too. The adapter's `_tiers` dict
  IS populated explicitly (all entries are written as `"dynamic"`), so
  the use_log-based heuristic in the docstring is dead code, but the
  inconsistency means any future reader who follows the docstring will
  misclassify subsumed/replaced primitives as "core" — a §3.3
  core-immunity violation in a single touch. The dual-sense field is
  flagged as TODO in `wake_sleep.py` lines 150-156 but not in
  EXTENSIONS.md. Minor — refactor or remove the docstring fallback.

- **`composition/lemma_library._validate_decoded` is a hard tautology
  returning `(True, "no-checker-available")`** (lines 649-671). This is
  the §4.5 step-2 validation pass; the residual gate carries the load.
  The docstring is explicit and the anti-shortcut comment is correct
  (an inline Python AST typechecker would be a §1.6 violation), but
  there is no EXTENSIONS.md entry tracking the missing tensor-network-
  side typechecker as a deferred extension. Per the no-placeholders
  directive, every "not yet implemented" surface should be logged.
  Add an EXTENSIONS.md entry for "Tensor-network-side AST validator
  for `register_lemma` step 2" so the gap is visible. Minor doc gap.

- **`orchestrator._frontier_priority` is a structural fan-out proxy,
  not the spec §5.3 expected-ΔF estimator.** Documented in the
  docstring (lines 91-101); flagged as "a follow-on per spec §5.3".
  NOT tracked in EXTENSIONS.md. Either log it as a deferred extension
  or upgrade. Documentation-only gap; behaviour is principled.

- **K-8 `test_orchestrator_solves_inductive_theorem_end_to_end` and
  L `test_hierarchical_proof_solves_top_theorem` opt out of the §9.7
  dense-tensor memory ceiling via an autouse fixture shadow.** The
  opt-out is principled (real `encode_mera` of the §10.10 composite
  needs 16-dim leaves) and documented at the top of each file. Already
  partially tracked in EXTENSIONS (RESOLVED at 369075e for the cap
  raise, but the file-level shadow remains). NOT a deviation — flagged
  for transparency only.
