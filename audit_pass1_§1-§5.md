# Audit Pass 1 — §1-§5 — HEAD 2b6fd56

Scope: `QFT_PCN_ARCHITECTURE.md` §1 Vision/Intent, §2 Architectural
Overview, §3 Mathematical Foundations, §4 Components As Built, §5
Tests/Verification. Implementation files audited:
- `src/qft_pcn/qft/` (fock, mps, hamiltonian, evolution, mera,
  mera_evolution, mpo, replica, sdp_solver, qpcn)
- `src/qft_pcn/logic/` (encoder, decoder, mera_encoder,
  mera_evaluation_hamiltonian, mera_typing_hamiltonian,
  evaluation_hamiltonian, typing_hamiltonian, _mera_layout)

Known items in `EXTENSIONS.md` (List arithmetic, two-MPS overlap,
multi-cycle GNVW, complex-block Hermiticity, decoder Gap E/F, etc.)
are NOT re-catalogued.

## ENHANCEMENTS (productive divergence)

- **`MERA.from_product` shared identity-disentangler tensor**
  (`src/qft_pcn/qft/mera.py:469-472`) — allocates a single read-only
  `(d_l, d_l, d_l, d_l)` identity per layer and aliases it across
  every intra/inter disentangler slot, instead of allocating
  `n_l/2 + (n_l/2 - 1)` distinct identity arrays. Spec §3.3 / §4.7 say
  nothing about per-tensor uniqueness; the alias is observationally
  identical (`apply_two_site_gate` writes back fresh arrays) and
  removes the dominant allocation in encoder-time MERA construction.
  Soundness pinned by the no-regression suites cited in the inline
  comment.

- **`MERA._is_product_cache` and `_mutation_version` invalidation
  bookkeeping** (`src/qft_pcn/qft/mera.py:277-281, 1359-1362`) —
  spec §3.3.6/§5 expectations don't dictate any caching strategy.
  The cache lets `inner`/`norm_sq` skip the per-disentangler
  identity scan on the imag-time hot path (where the ket is a
  leaf-mutated copy of the bra). Productive — pure performance,
  zero semantic divergence.

- **`MERA.from_term_superposition` exact-branch decomposition**
  (`src/qft_pcn/qft/mera.py:527-634`) — stores the explicit
  `_superposition_terms` list so `entanglement_entropy`,
  `norm_sq`, and `local_expectation` route through exact closed
  forms over the k branches instead of materializing a dense
  statevector. Faithful to §1.1 (entanglement is isometry-carried)
  and §5.4 (entropy without `O(d^N)` materialization).

- **`replica.compute_zn_for_ensemble` lazy-callable ensemble entries**
  (`src/qft_pcn/qft/replica.py`) — accepts zero-arg callables so the
  partition-function operator computation stays outside the module
  (§1.6 anti-shortcut: `Z` must come from real operator algebra).
  Documented in module docstring + tests. Spec §12.7 doesn't require
  laziness; productive separation-of-concerns.

- **`sdp_solver.psd_constraint_from_operator` operator-derived PSD
  helper** (`src/qft_pcn/qft/sdp_solver.py`) — projects a Hermitian
  substrate-operator block to a PSD variable + equality constraint,
  keeping the SDP encoding operator-algebraic per §1.6 even though
  CVXPY itself is numerical. Spec only mandates "SDP solver"; this
  shape preserves the substrate handoff.

- **`Hamiltonian.bond_op` omits the docstring-advertised `kappa_k
  (phi_k - phi_{k+1})^2` scalar-gradient term**
  (`src/qft_pcn/qft/hamiltonian.py:148-162`) — the spec §3.3.4 H_2 is
  `-t (a_k† a_{k+1} + h.c.)` only; the impl matches the spec exactly
  and the kappa line in the module docstring is decorative. Productive
  alignment with the spec (the docstring is over-promising; the code
  is right).

## DEVIATIONS (must fix)

- **`mera_typing_hamiltonian._stub` is dead code**
  (`src/qft_pcn/logic/mera_typing_hamiltonian.py:212-213`). Function
  is defined but never referenced. Per `memory/no-placeholders.md`,
  TODO/stub identifiers in tree are forbidden; this one slipped in
  with the §5 typing-rule scaffolding. **Fix scope: small** — delete
  the function (and any leftover import if applicable). Verify with
  `grep -r _stub src/qft_pcn/`.

- **`MERA._ascend_one_layer` ignores inter-pair disentanglers without
  guarding the non-product/non-unitary post-evolution case**
  (`src/qft_pcn/qft/mera.py:883-918`). The docstring asserts the
  simplification is "exact on product/vacuum MERAs because inter-pair
  disentanglers are identity," but `apply_two_site_gate` writes back
  an SVD-truncated, possibly non-unitary, possibly non-identity
  `recon4` into `inter_disentanglers[0][j]` (lines 1329-1357). After
  any imaginary-time Trotter step on an odd bond, `local_expectation`
  / `_ascend_one_layer` silently drops the inter-pair disentangler
  from the causal-cone contraction, biasing the result. Spec §5.4
  requires the expectation to honour the FULL ascending superoperator
  (Vidal 2008 §III.5: disentangler + isometry; inter-pair is part of
  the bond's causal cone for an adjacent-leaf op once the gate has
  fired). This is NOT an EXTENSIONS item — the EXTENSIONS register
  tracks gate-then-measure as "covered by Task 19 vs `materialize`,"
  but `_materialize` raises `NotImplementedError` for any higher-layer
  non-identity (mera.py:1265, 1272), so the cross-check is unreachable
  and the bias is unobserved. **Fix scope: medium** — either (a)
  extend `_ascend_one_layer` to fold in the relevant adjacent
  `inter_disentanglers[ell][j_inter]` (or `[j_inter-1]`) for the
  layer-0 ascent, or (b) hard-guard `local_expectation` /
  `two_site_expectation` to refuse / route to a different code path
  when any `inter_disentanglers[0]` slot is non-identity, restoring
  the documented invariant. Option (a) is principled (§1.2 causal
  cone honoured); (b) is the cheap honest-fail unblock.

(Two findings total. Neither overlaps an EXTENSIONS.md entry.)
