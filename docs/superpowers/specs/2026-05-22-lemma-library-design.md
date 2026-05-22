# Spec: Lemma Library and Promotion (migration sub-project I)

**Document type**: Implementation specification.
**Date**: 2026-05-22.
**Branch**: `claude/qft-pcn-hybrid-architecture-ihCIR`.
**Realizes**: `QFT_PCN_ARCHITECTURE.md` §10.8 (Lemma library and promotion — hierarchical composition, mechanism 1). Conceptual grounding: §11.1–11.3 (why hierarchical composition; the Wilson RG picture), Theorem 13.3 / 13.3.1 (lemma composition is exact on disjoint subsystems; consistent on shared variables).
**Status**: Approved design; ready for implementation plan.

---

## 0. How to read this spec

This specifies sub-project **I**: the **lemma library** (persistent, addressable storage of solved MERA sub-problems) and **promotion** (the `use_lemma` DSL constraint compiler that clamps a cached sub-state instead of re-deriving it).

A "solved sub-problem" in this architecture is a MERA ground state with low residual energy under some constraint Hamiltonian. By Curry-Howard the ground state is a *proof object*: it inhabits the proposition the Hamiltonian encodes. Sub-project I makes that proof object **persistent and reusable** — a future QPCN run clamps the cached sub-state rather than re-relaxing it.

Everything here is the contract. If something needed during implementation is missing, **stop and ask** — do not guess.

No time, duration, or effort estimates appear in this document. Sequencing and dependency order are specified; calendar/effort are not. Standing project directive.

This spec follows the structure of `docs/superpowers/specs/2026-05-22-mera-native-encoder-design.md` (M1): driving principles, scope, design sections with concrete APIs, error model, file layout, numbered acceptance criteria, contract to downstream, glossary.

---

## 1. Driving principles (non-negotiable — the anti-shortcut manifesto)

Inherited project-wide (see `2026-05-22-mera-migration-design.md` §1 and M1 §1) and specialized here. Every subagent prompt that touches this sub-project MUST carry this list verbatim, paired with the shortcut it forbids.

1. **No time, duration, or effort estimates.** Critical directive violation otherwise. Sequencing and dependency order only.

2. **Variable binding is genuine entanglement, never a classical lookup** (architecture §8.1, M1 §1.2). A lemma that binds variables shared with its host program carries that binding *through the MERA tree tensors* on the connecting path. The library never reduces a binding to a Python `dict[use, binder]`. When a clamped lemma shares a bound variable with the host, the shared-variable structure is enforced by the host Hamiltonian's typing constraints (Theorem 13.3.1), not by a side-table.
   - *Forbidden shortcut*: storing a lemma's binder/use correspondence as a serialized dict and "re-linking" it on load.
   - *Principled alternative*: the lemma's MERA tensors carry the entanglement; promotion clamps those tensors onto the host leaves and lets `H_coupling` (§5.4) verify consistency via residual energy.

3. **No dense operator at scale.** A lemma occupies `5·m` MERA leaves (m AST nodes), each leaf 16-dimensional. The promotion projector `−W|Ψ_L⟩⟨Ψ_L|` is **never materialized** as a `16^(5m)`-dimensional matrix. It is applied factored per species-leaf, exactly as M1's `mera_window_expectation_factored` and M2's Hamiltonian terms are factored.
   - *Forbidden shortcut*: building `|Ψ_L⟩⟨Ψ_L|` densely "because the lemma is small."
   - *Principled alternative*: the projector is represented by the lemma's own MERA tensor list; clamping is a tensor-network operation on the host MERA's causal cone over the lemma's leaf window.

4. **`optimize='greedy'` on every `numpy.einsum` / `tensordot` contraction path.** No exceptions.

5. **Lemma promotion is operator-algebraic, not syntactic.** Promotion clamps a *cached sub-state* (or, equivalently, adds a strong projector to the Hamiltonian); it does NOT substitute cached AST source text into the host program and re-derive. The composition is **referential** — "by Lemma L" — exactly as a mathematician cites a lemma without re-proving it (architecture §10.8 "Principles"). A `use_lemma` constraint resolves to a tensor-network clamp on the host MERA. It never resolves to a string/AST splice followed by a fresh relaxation of the spliced region.
   - *Forbidden shortcut*: `use_lemma` expands to "decode lemma L to an AST, paste it at the named hole, re-run the typing Hamiltonian over the pasted region."
   - *Principled alternative*: `use_lemma` compiles to either (a) initialization of the host MERA's lemma-window leaves to the cached lemma tensors, or (b) a `−W|Ψ_L⟩⟨Ψ_L|` projector term added to the host Hamiltonian. The cached *quantum state* is the proof object reused; the AST is never re-derived for the lemma region.

6. **Append-only library; exactness by construction.** The library only grows (architecture §10.8 risks table, §11.2). Registration is gated by a validation pass. By Theorem 13.3, clamping a sub-MERA to a cached lemma ground state is *mathematically identical* to having found that ground state in the current run — there is no correctness drift. The acceptance test (§8) is the operational proof of this.

---

## 2. Scope

### 2.1 In scope

- `src/qft_pcn/composition/lemma_library.py` — the `Lemma` record, the `LemmaLibrary` store: `.npz` serialization with bond-dimension compression, three-tier indexing (proposition type / structural fingerprint / derivation cost), and validated registration.
- `src/qft_pcn/composition/promoter.py` — the `use_lemma` DSL constraint compiler: compiles `{"kind": "use_lemma", "lemma_id": L, "leaves": [...]}` into either a leaf-clamp at MERA initialization or a `−W|Ψ_L⟩⟨Ψ_L|` projector Hamiltonian term.
- `src/qft_pcn/composition/__init__.py` — public surface.
- `src/qft_pcn/composition/tests/test_lemma_library.py` — the §8 acceptance suite, including the two-stage `x+0=x` → `(x+0)+0=x` demo.

### 2.2 Out of scope (later Phase-D pieces / own specs)

- Abstraction discovery / the wake-sleep cycle (architecture §10.9 — sub-project J). I supplies J the storage and indexing substrate; J supplies the clustering and promotion of *patterns* (as opposed to whole solved sub-problems).
- Cross-level message passing between QPCN runs (architecture §10.10 — sub-project K). I exposes the registration and lookup API K orchestrates.
- Lemma rot / dependency-graph invalidation across library revisions (architecture §10.8 risks table). I records the dependency edges (`derivation_metadata.lemma_deps`); acting on them is deferred.
- Periodic clustering + pruning of near-duplicate lemmas (architecture §10.8 risks table). The fingerprint index (§4.3) makes near-duplicate detection cheap; the pruning policy itself is deferred to J.
- Distributed / multi-process library access. The store is single-process, file-backed.

### 2.3 Will not do

- Materialize a lemma's projector or state as a dense `16^(5m)` tensor (§1.3).
- Implement `use_lemma` as an AST splice + re-derivation (§1.5).
- Modify M1's `mera_encoder.py` / `mera_decoder.py` / `_mera_window.py`, M2's Hamiltonians, or F's `qft/mera.py`. I builds strictly on top. A genuinely missing primitive is fixed in its home module with that module's tests still green; otherwise I builds a wrapper.
- Register an unvalidated state (§5).

---

## 3. The lemma as a proof object

### 3.1 What a lemma is

A lemma is a solved sub-problem: a MERA state `|Ψ_L⟩` over `n_leaves_L = 5·m` leaves (m AST nodes) plus PAD, that achieved residual energy `< ε_register` under a constraint Hamiltonian `H_L`. By Curry-Howard `|Ψ_L⟩` inhabits the proposition `H_L` encodes; that proposition is the lemma's **type**.

Three facts make reuse rigorous (architecture §10.8 "Principles"):

1. **Lemmas have type signatures.** Lookup is type-directed (§4.2).
2. **Lemmas have provenance.** Each records `H_L`'s identity, the energy gap to the first excited state (a certainty proxy), and any auxiliary assumptions clamped during derivation (§3.3).
3. **Lemmas compose under Curry-Howard.** Disjoint-subsystem composition is exact (Theorem 13.3); shared-variable composition is detected as positive residual energy at the coupling sites (Theorem 13.3.1).

### 3.2 The `Lemma` record

```python
# src/qft_pcn/composition/lemma_library.py

@dataclass(frozen=True)
class DerivationMetadata:
    hamiltonian_id: str            # stable hash of the H_L constraint set
    residual_energy: float         # <Ψ_L|H_L|Ψ_L> achieved at registration
    energy_gap: float              # E_1 - E_0 estimate; certainty proxy
    trotter_steps: int             # steps the original derivation took
    assumptions: tuple[str, ...]   # ids of lemmas/axioms clamped during derivation
    lemma_deps: tuple[str, ...]    # lemma_ids this lemma was built on (dependency edges)
    conditional: bool              # True iff assumptions contains an unverified entry
    source_run_id: str             # the QPCN run that produced it

@dataclass(frozen=True)
class Lemma:
    lemma_id: str                  # content hash of (tensors, proposition_type)
    proposition_type: str          # canonical type-signature string (§4.2)
    mera_tensors: MeraTensorBundle # disentanglers + isometries + leaf vectors of |Ψ_L>
    encoding_meta: MeraEncodingMeta# M1 meta: leaf layout, binder_leaves, etc.
    derivation: DerivationMetadata
    fingerprint: np.ndarray        # structural fingerprint vector (§4.3)
```

`MeraTensorBundle` is a thin numpy-array container (the disentangler/isometry/leaf arrays of an M1 `MERA`, plus the tree shape) that round-trips losslessly through `.npz`. It is NOT a live `MERA` object; `LemmaLibrary.materialize(lemma)` rebuilds a `MERA` from it on demand.

### 3.3 Conditional lemmas

A lemma derived while other lemmas/axioms were clamped (`assumptions` non-empty) and at least one of those is itself `conditional` or unverified is marked `conditional = True`. Conditional lemmas are stored and indexed but the promoter (§5) refuses to clamp a conditional lemma unless the host run explicitly opts in (`allow_conditional=True`). This is the architecture §10.8 "lemma drift" mitigation: a lemma proved under assumptions is only sound where those assumptions hold.

---

## 4. Storage and indexing (`lemma_library.py`)

### 4.1 Serialization with bond-dimension compression

Each lemma serializes to one `.npz` file under the library root:

```
<library_root>/
├── manifest.json                 # lemma_id -> {proposition_type, file, indices}
└── lemmas/
    └── <lemma_id>.npz             # MeraTensorBundle + encoding_meta + derivation + fingerprint
```

Before serialization the lemma's MERA tensors are **bond-compressed** to the minimum bond dimension that preserves the ground-state energy to within `ε_compress` (default `1e-9`): an SVD truncation sweep over the tree, dropping singular values below the threshold that would change `⟨Ψ_L|H_L|Ψ_L⟩` by more than `ε_compress`. Compression is *information-preserving for the lemma's purpose* — the architecture §10.8 "Storage layer" requirement that "bond dimensions [are] compressed to the minimum that preserves the lemma's ground-state energy."

```python
class LemmaLibrary:
    def __init__(self, root: str | Path, eps_compress: float = 1e-9): ...
    def save(self, lemma: Lemma) -> None: ...          # writes .npz + updates manifest
    def load(self, lemma_id: str) -> Lemma: ...
    def materialize(self, lemma_id: str) -> MERA: ...  # rebuild a live MERA
    def all_ids(self) -> list[str]: ...
```

`save` never overwrites an existing `lemma_id` (append-only, §1.6); a re-`save` of an identical content hash is a no-op, a hash collision with differing content raises `LemmaHashCollision`.

### 4.2 Primary index — by proposition type

`proposition_type` is a canonical string computed at registration by running M2's typing Hamiltonian over the decoded AST and reading the inferred top-level type signature (e.g. `"forall x:Nat. Eq (add x Zero) x"`). The primary index is `dict[proposition_type, list[lemma_id]]`.

```python
    def find_by_type(self, proposition_type: str) -> list[Lemma]: ...
```

Type strings are normalized (alpha-equivalence of bound names, canonical argument order) so that lookup is robust to cosmetic differences.

### 4.3 Secondary index — by structural fingerprint

The fingerprint is a fixed-length real vector derived from the **reduced density matrix at a canonical bond** of `|Ψ_L⟩` (architecture §10.8 "Indexing layer": "structural fingerprint of the lemma's reduced density matrix at a canonical bond"). The canonical bond is the top isometry's bond (the coarsest cut). The fingerprint is the sorted eigenvalue spectrum of that reduced density matrix, padded/truncated to `FINGERPRINT_DIM = 32`. Similarity is `trace_distance`-style L1 distance between fingerprints.

```python
    def find_similar(self, query_fingerprint: np.ndarray,
                     max_distance: float = 0.1) -> list[tuple[Lemma, float]]: ...
```

This answers "is there a lemma roughly shaped like this?" without a type match — the entry point sub-project J's clustering will build on.

### 4.4 Tertiary index — by derivation cost

`dict` ordered by `derivation.trotter_steps`. When `find_by_type` returns multiple lemmas proving the same proposition, the promoter (§5) prefers the one cheapest to *apply*; cheapest-to-apply is approximated by smallest `n_leaves_L` (fewer leaves to clamp), with `trotter_steps` as the tie-breaker.

```python
    def cheapest_for_type(self, proposition_type: str) -> Lemma | None: ...
```

### 4.5 Registration

```python
@dataclass(frozen=True)
class RegistrationResult:
    accepted: bool
    lemma_id: str | None
    reason: str                    # "ok" | "residual_too_high" | "validation_failed:<detail>"

def register_lemma(library: LemmaLibrary,
                   state: MERA,
                   meta: MeraEncodingMeta,
                   hamiltonian: object,            # an M2 MeraTypingHamiltonian / MeraEvalHamiltonian
                   derivation: DerivationMetadata,
                   eps_register: float = 1e-8) -> RegistrationResult: ...
```

Registration pipeline (architecture §10.8 "Registration"):

1. **Residual gate.** If `derivation.residual_energy >= eps_register`, reject with `"residual_too_high"`. Log as a near-miss (write to `<root>/near_misses.log`) but do not register.
2. **Validation pass.** Decode `state` to an AST via M1's `decode_mera`. Classically type-check the decoded AST against the calculus's typing rules (reuse `src/qft_pcn/logic/synthesis/_validate.py`'s checker). If the structure is a non-proof object, run the constraint-verifier instead. On failure, reject with `"validation_failed:<detail>"` and log the near-miss.
3. **Type computation.** Compute `proposition_type` (§4.2) from the validated AST.
4. **Fingerprint.** Compute the canonical-bond fingerprint (§4.3).
5. **Compress + persist.** Bond-compress (§4.1), assemble the `Lemma`, `library.save`.
6. Return `RegistrationResult(accepted=True, lemma_id=..., reason="ok")`.

Validation-before-registration is the architecture §10.8 "cache pollution" mitigation: a bad solution never enters the library.

---

## 5. Promotion — the `use_lemma` constraint compiler (`promoter.py`)

### 5.1 The DSL constraint

A new DSL constraint kind, consumed by the host Hamiltonian compiler:

```python
{"kind": "use_lemma",
 "lemma_id": "<lemma_id>",
 "leaves": [i0, i1, ...],         # absolute host MERA leaf indices the lemma occupies
 "weight": W,                      # projector strength, default LEMMA_PROJECTOR_WEIGHT
 "allow_conditional": False}
```

`leaves` is the contiguous `5·m`-leaf window of the host MERA where the lemma is being cited. Its length MUST equal the lemma's `n_leaves_L`; a mismatch raises `LemmaLeafCountMismatch`.

### 5.2 Two compilation targets — both operator-algebraic

The promoter compiles a `use_lemma` constraint into one of two equivalent forms (architecture §10.8 "Promotion to Hamiltonian primitive"). Both are referential clamps of the cached state; neither is a syntactic splice (§1.5).

**(a) Initialization clamp.** At host-MERA construction, the lemma-window leaves and the disentanglers/isometries inside their causal cone are *set* to the cached lemma's tensors (re-indexed from lemma-local leaves to the host's `leaves` window). Imaginary-time relaxation then proceeds on the rest of the tree with the lemma window held fixed (clamped — excluded from the variational update).

**(b) Projector term.** A Hamiltonian term `−W·|Ψ_L⟩⟨Ψ_L|` is added, restricted to the lemma window. During imag-time relaxation this term pulls the window's sub-MERA into `|Ψ_L⟩`. The projector is represented and applied **factored** via the lemma's own MERA tensor list over the window's causal cone — `|Ψ_L⟩⟨Ψ_L|` is never densified (§1.3). `W` is large relative to the host's other term weights (default `LEMMA_PROJECTOR_WEIGHT = 1e3`) so the clamp dominates.

The implementation plan picks one as production and uses the other as the cross-check (mirroring M1's analytic-vs-gate choice). Form (a) is the natural production choice — it is exactly the architecture §10.8 "clamps the specified sites to that state during initialization"; form (b) is the cross-check, and is also the form sub-project K's cross-run message passing will reuse.

```python
@dataclass(frozen=True)
class PromotedLemma:
    lemma_id: str
    host_leaves: tuple[int, ...]
    mode: str                      # "init_clamp" | "projector"
    weight: float

class Promoter:
    def __init__(self, library: LemmaLibrary,
                 mode: str = "init_clamp"): ...

    def compile_constraint(self, constraint: dict) -> PromotedLemma: ...
        # validates the constraint, loads the lemma, checks leaf count and
        # conditional flag, returns the PromotedLemma descriptor.

    def apply_init_clamp(self, host: MERA, meta: MeraEncodingMeta,
                         promoted: PromotedLemma) -> set[int]:
        # writes the lemma tensors into the host MERA's lemma window;
        # returns the set of host tensor ids that must be frozen during relaxation.

    def projector_energy(self, host: MERA, meta: MeraEncodingMeta,
                         promoted: PromotedLemma) -> float:
        # <host| (-W |Ψ_L><Ψ_L|) |host>, evaluated factored over the causal cone.
        # This is the Hamiltonian term the projector mode contributes.
```

### 5.3 Re-indexing the lemma onto the host window

The lemma was derived with its own leaf layout (`encoding_meta`). The host cites it at `leaves`. The promoter re-indexes lemma-local leaf `j` to host leaf `leaves[j]`, asserting species agreement: `lemma.encoding_meta.species_of_leaf[j] == host_meta.species_of_leaf[leaves[j]]` for all `j`. A species mismatch raises `LemmaSpeciesMismatch` — citing a lemma at a window whose node-major species pattern does not align is a structural error, not a silently-tolerated case.

### 5.4 Shared variables and composition correctness

When the cited lemma binds a variable also bound in the host (binding-as-entanglement, §1.2), the host Hamiltonian already carries a `bid`-coupling typing term across those leaves (M2). The promoter does NOT add its own coupling logic. By Theorem 13.3.1:

- If the shared-variable structure is consistent, the joint ground-state energy equals the sum of the parts and residual stays zero.
- If two clamped lemmas (or a lemma and the host) demand incompatible `bid`/`type` structure, the host Hamiltonian's coupling term reports **positive residual energy at the coupling leaves** — the operational signal of "Lemma A and Lemma B are inconsistent" (architecture §10.8 "Composition correctness", §13.3.1).

The promoter exposes `composition_residual(host, meta, promoted_list, hamiltonian)` which sums the host Hamiltonian's coupling-term energies over all leaves shared between any two promoted lemmas (or a lemma and a host constraint). A caller refuses a composition whose `composition_residual` exceeds `ε_register`.

### 5.5 Disjoint composition is exact

When two cited lemmas occupy disjoint leaf windows with no shared `bid`, `apply_init_clamp` for both, then relax: by Theorem 13.3 the result is the exact tensor product and residual is zero with no relaxation needed in the clamped windows. The acceptance test §8 exercises exactly this for the nested `(x+0)+0` case.

---

## 6. Error model

New exception family in `src/qft_pcn/composition/errors.py`, all subclassing `CompositionError`:

- `LemmaHashCollision` — `save` of a content hash that already maps to different content.
- `LemmaNotFound` — `load` / `materialize` / `compile_constraint` of an unknown `lemma_id`.
- `LemmaLeafCountMismatch` — `use_lemma` `leaves` length ≠ lemma `n_leaves_L`.
- `LemmaSpeciesMismatch` — re-indexing onto a host window with a misaligned species pattern (§5.3).
- `ConditionalLemmaRefused` — `compile_constraint` of a `conditional` lemma without `allow_conditional=True`.
- `LemmaValidationError` — internal: registration validation pass raised; surfaced via `RegistrationResult.reason`, not thrown to the caller.
- `CompressionError` — bond compression could not reach `eps_compress` without exceeding it (degenerate / pathological tensor).

Registration is *total*: it always returns a `RegistrationResult`, never throws for a bad candidate state (a bad candidate is a `reason`, not an exception). The promoter raises for *structural* misuse (unknown id, leaf-count/species mismatch, refused conditional) — these are caller bugs, not data conditions.

---

## 7. File layout

```
src/qft_pcn/composition/
├── __init__.py                # public surface: LemmaLibrary, Lemma, register_lemma,
│                              #   Promoter, PromotedLemma, the use_lemma constraint kind
├── lemma_library.py           # Lemma, DerivationMetadata, MeraTensorBundle,
│                              #   LemmaLibrary (save/load/materialize, 3 indices),
│                              #   register_lemma, RegistrationResult
├── promoter.py                # Promoter, PromotedLemma, composition_residual
├── errors.py                  # CompositionError + the §6 family
└── tests/
    ├── __init__.py
    └── test_lemma_library.py  # the §8 acceptance suite
```

All files are new. `mera_encoder.py`, `mera_decoder.py`, `_mera_window.py`, M2's Hamiltonians, F's `qft/mera.py`, and the MPS logic stack are NOT modified (§2.3).

---

## 8. Acceptance criteria

I is complete when all of the following pass, each backed by fresh `pytest` output (§1 verification directive):

1. **Round-trip storage.** A lemma `save`d then `load`ed is bit-identical in `proposition_type`, `derivation`, and `fingerprint`; the `materialize`d MERA has `|inner(original, materialized)|² > 1 − 1e-10`.

2. **Bond compression is lossless for purpose.** After compression, `⟨Ψ_L|H_L|Ψ_L⟩` differs from the pre-compression value by `< eps_compress`; the compressed `.npz` is no larger than the uncompressed one.

3. **Three-tier indexing.** `find_by_type` returns exactly the lemmas of a queried proposition type; `find_similar` ranks a near-duplicate lemma above an unrelated one by fingerprint distance; `cheapest_for_type` returns the lowest-`n_leaves_L` lemma when several prove the same proposition.

4. **Registration gates.** A state with residual `≥ eps_register` is rejected with `reason == "residual_too_high"` and logged as a near-miss, not registered. A state that decodes to an ill-typed AST is rejected with `reason` starting `"validation_failed"`. A valid low-residual state is accepted and appears in all three indices.

5. **Append-only.** Re-`save` of an identical lemma is a no-op; `save` of a colliding hash with different content raises `LemmaHashCollision`.

6. **Promotion is operator-algebraic.** `apply_init_clamp` writes the lemma tensors into the host window and returns the frozen tensor-id set; the host's lemma-window leaves decode to the lemma's AST. The test asserts the promoter performs **no AST splice and no re-derivation** of the lemma region (verified structurally: the lemma window's tensors are byte-equal to the cached lemma's after clamping, before any relaxation step).

7. **Projector / init-clamp equivalence.** For a fixed lemma and host, `mode="projector"` relaxation and `mode="init_clamp"` relaxation converge to host states with `|inner|² > 1 − 1e-8` — the two compilation targets of §5.2 are equivalent.

8. **No dense projector.** No test materializes `|Ψ_L⟩⟨Ψ_L|` or any operator larger than `16²` densely; the factored causal-cone path is used. Verified by the `conftest.py` memory ceiling.

9. **Disjoint composition is exact.** Two lemmas clamped at disjoint windows yield a host state whose residual is `< 1e-10` with zero relaxation steps in the clamped windows (Theorem 13.3).

10. **Inconsistent composition is detected.** Two lemmas demanding incompatible `bid`/`type` structure at a shared leaf yield `composition_residual > eps_register` (Theorem 13.3.1).

11. **Conditional lemmas are refused by default.** `compile_constraint` of a `conditional` lemma raises `ConditionalLemmaRefused` unless `allow_conditional=True`.

12. **Two-stage demo (the architecture §10.8 acceptance test).**
    - **Run 1**: encode and relax `∀x. x + 0 = x` (the extended-calculus `Forall`/`Eq`/`add`/`Zero` encoding from M1) to a MERA ground state with residual `< eps_register`; `register_lemma` accepts it; call the registered lemma `L_addzero`.
    - **Run 2**: encode `∀x. (x + 0) + 0 = x` with **two** `use_lemma` constraints citing `L_addzero` at the inner `x + 0` window and the outer `(x+0) + 0` window; compile via the `Promoter`; relax.
    - **Assert**: Run 2 converges to residual `< eps_register`, and Run 2's Trotter-step count to convergence is **at most half** Run 1's step count for the same theorem re-derived from axioms (no `use_lemma`). The "substantially fewer Trotter steps" of architecture §10.8 is pinned to the `≤ 0.5×` ratio.

13. **No regression.** F's MERA tests, M1's encoder tests, and the MPS logic stack's tests all still pass.

14. Every completion claim is backed by fresh `pytest` output.

---

## 9. Contract I exposes to J (abstraction discovery) and K (cross-level message passing)

- **To J (§10.9):** `LemmaLibrary.find_similar` + the fingerprint index are J's entry point for clustering reduced density matrices. `Lemma.derivation.lemma_deps` gives J the dependency edges it needs when it replaces a primitive with a composite. J registers its discovered abstractions through the same `register_lemma` path; an abstraction is just a lemma whose `derivation.assumptions` cites the cluster members.
- **To K (§10.10):** `register_lemma` and `Promoter.compile_constraint` are the two API ends K orchestrates — a child QPCN run hands its solved state to `register_lemma`; a parent run's goal graph emits `use_lemma` constraints that K routes through the `Promoter`. The `projector` compilation mode (§5.2b) is the form K's cross-run message passing reuses, because it lets a parent add a lemma clamp to an already-running relaxation.
- The `use_lemma` constraint kind and the `Lemma` / `RegistrationResult` / `PromotedLemma` dataclasses are the stable public surface; J and K import them from `src/qft_pcn/composition/__init__.py`.

---

## 10. Glossary (I-local)

- **Lemma** — a solved sub-problem: a MERA ground state with residual `< ε_register`, cached as a proof object (Curry-Howard).
- **Proof object** — the MERA ground state itself; it *inhabits* the proposition its Hamiltonian encodes.
- **Promotion** — turning a cached lemma into a host-Hamiltonian primitive via the `use_lemma` constraint.
- **Init clamp** — promotion mode (a): write the lemma tensors into the host MERA window and freeze them during relaxation.
- **Projector term** — promotion mode (b): add `−W|Ψ_L⟩⟨Ψ_L|` to the host Hamiltonian, applied factored.
- **Referential composition** — citing a lemma by reusing its cached *state*, never by re-deriving its AST (the §1.5 principle).
- **Structural fingerprint** — the sorted eigenvalue spectrum of the reduced density matrix at the lemma's canonical (top) bond; the secondary-index key.
- **Conditional lemma** — a lemma proved under unverified assumptions; refused by the promoter unless explicitly allowed.
- **Composition residual** — the host Hamiltonian's coupling-term energy summed over leaves shared between cited lemmas; positive ⇒ inconsistent composition.
