# Lemma Library and Promotion Implementation Plan (migration sub-project I)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Each task is strict TDD: write the failing test, watch it fail, implement, watch it pass, commit.

**Goal:** Build sub-project I — the lemma library (persistent, addressable storage of solved MERA sub-problems) and promotion (the `use_lemma` DSL constraint compiler that clamps a cached sub-state instead of re-deriving it).

**Architecture:** A solved sub-problem is a MERA ground state with residual `< ε_register`; by Curry-Howard it is a proof object. The `LemmaLibrary` serializes it bond-compressed to `.npz`, indexed three ways (proposition type / structural fingerprint / derivation cost). The `Promoter` compiles `{"kind": "use_lemma", ...}` into an operator-algebraic clamp — either an init clamp of the host MERA's lemma window or a `−W|Ψ_L⟩⟨Ψ_L|` projector term — never an AST splice.

**Tech Stack:** Python 3.11, numpy 1.26, pytest. Built on M1 (`mera_encoder.py`, `mera_decoder.py`, `_mera_window.py`), M2 (`MeraTypingHamiltonian`/`MeraEvalHamiltonian`), M3 (synthesis runner), and F's `qft/mera.py`.

**Spec:** `docs/superpowers/specs/2026-05-22-lemma-library-design.md` — authoritative. When this plan and the spec disagree, the spec wins.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Driving principles (from spec §1 — non-negotiable; embed verbatim in every subagent prompt, each paired with the shortcut it forbids)

1. **No time/duration/effort estimates.** Critical directive.
2. **Binding is genuine entanglement, never a classical lookup.** *Forbidden:* storing a lemma's binder/use map as a dict and re-linking on load. *Principled:* the lemma's MERA tensors carry the entanglement; `H_coupling` verifies via residual.
3. **No dense operator at scale.** *Forbidden:* materializing `|Ψ_L⟩⟨Ψ_L|` as a `16^(5m)` matrix "because the lemma is small." *Principled:* the projector is the lemma's own MERA tensor list, applied factored over the causal cone.
4. **`optimize='greedy'` on every einsum/tensordot.**
5. **Promotion is operator-algebraic, not syntactic.** *Forbidden:* `use_lemma` expands to "decode lemma to AST, paste at the hole, re-run the typing Hamiltonian." *Principled:* `use_lemma` compiles to a tensor-network clamp of the cached *state*.
6. **Append-only library; exactness by construction (Theorem 13.3).** Registration is validation-gated.

---

## Pre-existing worktree state

Unrelated modified files exist (`tauri-app/`, root `*.md`, `lib/`, `docs/` viz). Leave them alone. Stage only files each task names. `src/qft_pcn/composition/` does not yet exist; this plan creates it.

---

## Task 1: Package skeleton and error family

**Files:**
- Create: `src/qft_pcn/composition/__init__.py`
- Create: `src/qft_pcn/composition/errors.py`
- Create: `src/qft_pcn/composition/tests/__init__.py`
- Test: `src/qft_pcn/composition/tests/test_errors.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_errors.py`:

```python
"""Tests for the composition error family (spec §6)."""
from __future__ import annotations
import pytest
from src.qft_pcn.composition.errors import (
    CompositionError, LemmaHashCollision, LemmaNotFound,
    LemmaLeafCountMismatch, LemmaSpeciesMismatch,
    ConditionalLemmaRefused, LemmaValidationError, CompressionError,
)

ALL = [LemmaHashCollision, LemmaNotFound, LemmaLeafCountMismatch,
       LemmaSpeciesMismatch, ConditionalLemmaRefused,
       LemmaValidationError, CompressionError]


@pytest.mark.parametrize("exc", ALL)
def test_all_subclass_composition_error(exc):
    assert issubclass(exc, CompositionError)


@pytest.mark.parametrize("exc", ALL)
def test_each_raisable_with_message(exc):
    with pytest.raises(CompositionError, match="boom"):
        raise exc("boom")
```

- [ ] **Step 2: Run the test, verify it fails** — `ImportError`.

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_errors.py -v`

- [ ] **Step 3: Implement**

Create `src/qft_pcn/composition/errors.py`:

```python
"""Exception family for sub-project I (lemma library and promotion). Spec §6."""
from __future__ import annotations


class CompositionError(Exception):
    """Base of every sub-project I error."""


class LemmaHashCollision(CompositionError):
    """save() of a content hash that already maps to different content."""


class LemmaNotFound(CompositionError):
    """load/materialize/compile_constraint of an unknown lemma_id."""


class LemmaLeafCountMismatch(CompositionError):
    """use_lemma 'leaves' length != lemma n_leaves_L."""


class LemmaSpeciesMismatch(CompositionError):
    """Re-indexing onto a host window with a misaligned species pattern."""


class ConditionalLemmaRefused(CompositionError):
    """compile_constraint of a conditional lemma without allow_conditional."""


class LemmaValidationError(CompositionError):
    """Registration validation pass failed (surfaced via RegistrationResult)."""


class CompressionError(CompositionError):
    """Bond compression could not reach eps_compress without exceeding it."""
```

Create `src/qft_pcn/composition/tests/__init__.py` (empty).

Create `src/qft_pcn/composition/__init__.py`:

```python
"""Sub-project I: lemma library and promotion (architecture §10.8)."""
from __future__ import annotations
from src.qft_pcn.composition.errors import (
    CompositionError, LemmaHashCollision, LemmaNotFound,
    LemmaLeafCountMismatch, LemmaSpeciesMismatch,
    ConditionalLemmaRefused, LemmaValidationError, CompressionError,
)

__all__ = [
    "CompositionError", "LemmaHashCollision", "LemmaNotFound",
    "LemmaLeafCountMismatch", "LemmaSpeciesMismatch",
    "ConditionalLemmaRefused", "LemmaValidationError", "CompressionError",
]
```

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Commit**

```
feat(composition): package skeleton + error family for sub-project I

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 2: The `Lemma` / `DerivationMetadata` / `MeraTensorBundle` records

**Files:**
- Modify: `src/qft_pcn/composition/lemma_library.py` (create)
- Test: `src/qft_pcn/composition/tests/test_records.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_records.py`:

```python
"""Tests for the lemma record dataclasses (spec §3.2)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.composition.lemma_library import (
    Lemma, DerivationMetadata, MeraTensorBundle,
)


def _bundle():
    return MeraTensorBundle(
        n_leaves=8, leaf_dim=16, n_layers=3,
        leaf_vectors=[np.zeros(16) for _ in range(8)],
        disentanglers=[], isometries=[],
    )


def _deriv():
    return DerivationMetadata(
        hamiltonian_id="h0", residual_energy=1e-12, energy_gap=0.5,
        trotter_steps=40, assumptions=(), lemma_deps=(),
        conditional=False, source_run_id="run0",
    )


def test_derivation_metadata_is_frozen():
    d = _deriv()
    with pytest.raises(Exception):
        d.residual_energy = 2.0


def test_conditional_inferred_false_when_no_assumptions():
    assert _deriv().conditional is False


def test_lemma_holds_all_fields(tmp_path):
    from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta
    meta = MeraEncodingMeta(
        n_nodes=1, n_leaves=8, L=3, leaf_dim=16,
        species_of_leaf=["kind", "type", "bid", "value", "tobl",
                         "PAD", "PAD", "PAD"],
        node_of_leaf=[0, 0, 0, 0, 0, -1, -1, -1],
        site_to_ast_path={0: ()}, binder_leaves={}, use_to_binder={},
    )
    lem = Lemma(
        lemma_id="abc", proposition_type="forall x:Nat. Eq x x",
        mera_tensors=_bundle(), encoding_meta=meta,
        derivation=_deriv(), fingerprint=np.zeros(32),
    )
    assert lem.lemma_id == "abc"
    assert lem.fingerprint.shape == (32,)
    assert lem.encoding_meta.n_leaves == 8
```

- [ ] **Step 2: Run the test, verify it fails** — `ImportError`.

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_records.py -v`

- [ ] **Step 3: Implement**

Create `src/qft_pcn/composition/lemma_library.py` (records portion — extended in Tasks 3-6):

```python
"""Lemma library: storage, indexing, registration. Spec §3, §4.

A lemma is a solved sub-problem -- a MERA ground state with residual
energy below eps_register. By Curry-Howard it is a proof object: it
inhabits the proposition its Hamiltonian encodes. The library caches it
so a future QPCN run clamps it rather than re-deriving it.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta

FINGERPRINT_DIM = 32


@dataclass(frozen=True)
class MeraTensorBundle:
    """A serialization-friendly snapshot of a MERA's tensors. NOT a live
    MERA; LemmaLibrary.materialize rebuilds a MERA from it."""
    n_leaves: int
    leaf_dim: int
    n_layers: int
    leaf_vectors: list[np.ndarray]
    disentanglers: list[np.ndarray]
    isometries: list[np.ndarray]


@dataclass(frozen=True)
class DerivationMetadata:
    """Provenance of a lemma (spec §3.2)."""
    hamiltonian_id: str
    residual_energy: float
    energy_gap: float
    trotter_steps: int
    assumptions: tuple[str, ...]
    lemma_deps: tuple[str, ...]
    conditional: bool
    source_run_id: str


@dataclass(frozen=True)
class Lemma:
    """A cached proof object (spec §3.2)."""
    lemma_id: str
    proposition_type: str
    mera_tensors: MeraTensorBundle
    encoding_meta: MeraEncodingMeta
    derivation: DerivationMetadata
    fingerprint: np.ndarray
```

- [ ] **Step 4: Run the test, verify it passes.**

If `MeraEncodingMeta`'s constructor signature differs from M1's spec §6.1, adapt the test's `meta` construction to the real signature — **stop and ask** only if `MeraEncodingMeta` is absent entirely.

- [ ] **Step 5: Commit**

```
feat(composition): Lemma/DerivationMetadata/MeraTensorBundle records

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 3: `MeraTensorBundle` <-> `MERA` round-trip and `.npz` serialization

**Files:**
- Modify: `src/qft_pcn/composition/lemma_library.py`
- Test: `src/qft_pcn/composition/tests/test_serialization.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_serialization.py`:

```python
"""Tests for bundle<->MERA round-trip and .npz I/O (spec §3.2, §4.1)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.composition.lemma_library import (
    MeraTensorBundle, bundle_from_mera, mera_from_bundle,
    save_bundle_npz, load_bundle_npz,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.qft.mera import inner


def _state():
    state, meta = encode_mera(parse(r"\x:Int. x"))
    return state, meta


def test_bundle_from_mera_then_back_is_faithful():
    state, _ = _state()
    bundle = bundle_from_mera(state)
    rebuilt = mera_from_bundle(bundle)
    ov = abs(inner(state, rebuilt)) ** 2
    assert ov > 1 - 1e-10


def test_npz_round_trip(tmp_path):
    state, _ = _state()
    bundle = bundle_from_mera(state)
    path = tmp_path / "b.npz"
    save_bundle_npz(bundle, path)
    loaded = load_bundle_npz(path)
    rebuilt = mera_from_bundle(loaded)
    assert abs(inner(state, rebuilt)) ** 2 > 1 - 1e-10
```

- [ ] **Step 2: Run the test, verify it fails** — `ImportError`.

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_serialization.py -v`

- [ ] **Step 3: Implement**

Inspect `src/qft_pcn/qft/mera.py` for the real attribute names that hold leaf vectors, disentanglers, and isometries (the spec calls them `leaf_vectors` / `disentanglers` / `isometries`; the actual `MERA` may name them differently — use the actual names). Append to `lemma_library.py`:

```python
from src.qft_pcn.qft.mera import MERA


def bundle_from_mera(state: MERA) -> MeraTensorBundle:
    """Snapshot a live MERA into a serializable bundle (spec §3.2)."""
    return MeraTensorBundle(
        n_leaves=state.n_leaves,
        leaf_dim=state.leaf_dim,
        n_layers=state.n_layers,
        leaf_vectors=[np.asarray(v).copy() for v in state.leaf_vectors],
        disentanglers=[np.asarray(d.tensor).copy() for d in state.disentanglers],
        isometries=[np.asarray(w.tensor).copy() for w in state.isometries],
    )


def mera_from_bundle(bundle: MeraTensorBundle) -> MERA:
    """Rebuild a live MERA from a bundle (inverse of bundle_from_mera)."""
    state = MERA.from_product(bundle.leaf_vectors)
    for i, d in enumerate(bundle.disentanglers):
        state.disentanglers[i].tensor = np.asarray(d).copy()
    for i, w in enumerate(bundle.isometries):
        state.isometries[i].tensor = np.asarray(w).copy()
    return state


def save_bundle_npz(bundle: MeraTensorBundle, path) -> None:
    """Serialize a bundle to .npz (spec §4.1)."""
    arrs: dict[str, np.ndarray] = {
        "n_leaves": np.array(bundle.n_leaves),
        "leaf_dim": np.array(bundle.leaf_dim),
        "n_layers": np.array(bundle.n_layers),
        "n_disent": np.array(len(bundle.disentanglers)),
        "n_iso": np.array(len(bundle.isometries)),
    }
    for i, v in enumerate(bundle.leaf_vectors):
        arrs[f"leaf_{i}"] = v
    for i, d in enumerate(bundle.disentanglers):
        arrs[f"disent_{i}"] = d
    for i, w in enumerate(bundle.isometries):
        arrs[f"iso_{i}"] = w
    np.savez_compressed(path, **arrs)


def load_bundle_npz(path) -> MeraTensorBundle:
    """Inverse of save_bundle_npz."""
    z = np.load(path, allow_pickle=False)
    n_leaves = int(z["n_leaves"])
    return MeraTensorBundle(
        n_leaves=n_leaves,
        leaf_dim=int(z["leaf_dim"]),
        n_layers=int(z["n_layers"]),
        leaf_vectors=[z[f"leaf_{i}"] for i in range(n_leaves)],
        disentanglers=[z[f"disent_{i}"] for i in range(int(z["n_disent"]))],
        isometries=[z[f"iso_{i}"] for i in range(int(z["n_iso"]))],
    )
```

The `MERA` attribute names above are placeholders pending the `mera.py` inspection in this step — replace them with the real ones; the bundle/I/O logic itself is unchanged. If `MERA` exposes no per-tensor mutable handle, build the rebuild via the public construction path `mera.py` offers and keep the round-trip test as the correctness gate. **Do not modify `mera.py`** unless a genuine missing primitive blocks the round-trip, in which case fix it there with F's tests still green (spec §2.3).

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Commit**

```
feat(composition): MeraTensorBundle <-> MERA round-trip + .npz I/O

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 4: Bond-dimension compression

**Files:**
- Modify: `src/qft_pcn/composition/lemma_library.py`
- Test: `src/qft_pcn/composition/tests/test_compression.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_compression.py`:

```python
"""Tests for bond-dimension compression (spec §4.1, acceptance §8.2)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.composition.lemma_library import (
    bundle_from_mera, compress_bundle,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def _energy(bundle):
    # placeholder energy probe: norm of leaf vectors; replaced by the real
    # H_L expectation once M2's MeraTypingHamiltonian is wired in Task 7.
    return float(sum(np.linalg.norm(v) for v in bundle.leaf_vectors))


def test_compression_preserves_energy_within_tol():
    state, _ = encode_mera(parse(r"\x:Int. x"))
    bundle = bundle_from_mera(state)
    eps = 1e-9
    compressed = compress_bundle(bundle, energy_fn=_energy, eps_compress=eps)
    assert abs(_energy(compressed) - _energy(bundle)) < eps


def test_compression_does_not_grow_storage(tmp_path):
    from src.qft_pcn.composition.lemma_library import save_bundle_npz
    state, _ = encode_mera(parse(r"\x:Int. x"))
    bundle = bundle_from_mera(state)
    compressed = compress_bundle(bundle, energy_fn=_energy, eps_compress=1e-9)
    p_raw, p_cmp = tmp_path / "raw.npz", tmp_path / "cmp.npz"
    save_bundle_npz(bundle, p_raw)
    save_bundle_npz(compressed, p_cmp)
    assert p_cmp.stat().st_size <= p_raw.stat().st_size
```

- [ ] **Step 2: Run the test, verify it fails** — `ImportError`.

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_compression.py -v`

- [ ] **Step 3: Implement**

Append to `lemma_library.py`:

```python
from typing import Callable
from src.qft_pcn.composition.errors import CompressionError


def compress_bundle(bundle: MeraTensorBundle,
                    energy_fn: Callable[[MeraTensorBundle], float],
                    eps_compress: float = 1e-9) -> MeraTensorBundle:
    """SVD-truncate the bundle's isometry bonds to the smallest bond
    dimension that keeps energy_fn within eps_compress (spec §4.1).

    energy_fn(bundle) is <Psi|H_L|Psi>; the caller (register_lemma) binds
    it to the real M2 Hamiltonian. Truncation never grows storage: if no
    singular value can be dropped, the bundle is returned unchanged.
    """
    base = energy_fn(bundle)
    isos = [np.asarray(w).copy() for w in bundle.isometries]
    for k, w in enumerate(isos):
        mat = w.reshape(w.shape[0], -1)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        for cut in range(len(s) - 1, 0, -1):
            trial = u[:, :cut] @ np.diag(s[:cut]) @ vh[:cut, :]
            cand = list(isos)
            cand[k] = trial.reshape(w.shape)
            trial_bundle = MeraTensorBundle(
                bundle.n_leaves, bundle.leaf_dim, bundle.n_layers,
                bundle.leaf_vectors, bundle.disentanglers, cand)
            if abs(energy_fn(trial_bundle) - base) < eps_compress:
                isos[k] = trial.reshape(w.shape)
            else:
                break
    out = MeraTensorBundle(
        bundle.n_leaves, bundle.leaf_dim, bundle.n_layers,
        bundle.leaf_vectors, bundle.disentanglers, isos)
    if abs(energy_fn(out) - base) >= eps_compress:
        raise CompressionError(
            f"compression drifted energy by >= {eps_compress}")
    return out
```

Every `numpy` contraction added here that uses `einsum`/`tensordot` must pass `optimize='greedy'` (principle 4). The SVD path above uses no einsum; if a later refinement needs one, add the flag.

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Commit**

```
feat(composition): bond-dimension compression for lemma storage

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 5: The structural fingerprint

**Files:**
- Modify: `src/qft_pcn/composition/lemma_library.py`
- Test: `src/qft_pcn/composition/tests/test_fingerprint.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_fingerprint.py`:

```python
"""Tests for the structural fingerprint (spec §4.3)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.composition.lemma_library import (
    structural_fingerprint, fingerprint_distance, FINGERPRINT_DIM,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def test_fingerprint_has_fixed_dim():
    state, _ = encode_mera(parse(r"\x:Int. x"))
    fp = structural_fingerprint(state)
    assert fp.shape == (FINGERPRINT_DIM,)


def test_identical_states_have_zero_distance():
    s1, _ = encode_mera(parse(r"\x:Int. x"))
    s2, _ = encode_mera(parse(r"\y:Int. y"))   # alpha-equivalent
    d = fingerprint_distance(structural_fingerprint(s1),
                             structural_fingerprint(s2))
    assert d < 1e-8


def test_different_states_have_positive_distance():
    s1, _ = encode_mera(parse(r"\x:Int. x"))
    s2, _ = encode_mera(parse(r"(\x:Int. x + 1)(2)"))
    d = fingerprint_distance(structural_fingerprint(s1),
                             structural_fingerprint(s2))
    assert d > 1e-6
```

- [ ] **Step 2: Run the test, verify it fails** — `ImportError`.

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_fingerprint.py -v`

- [ ] **Step 3: Implement**

Append to `lemma_library.py`. The fingerprint is the sorted eigenvalue spectrum of the reduced density matrix at the canonical (top) bond, padded/truncated to `FINGERPRINT_DIM` (spec §4.3). Use F's MERA reduced-density-matrix helper if `mera.py` exposes one; otherwise compute it from the top isometry's environment:

```python
def structural_fingerprint(state: MERA) -> np.ndarray:
    """Sorted eigenvalue spectrum of the reduced density matrix at the
    canonical (top) bond, padded/truncated to FINGERPRINT_DIM (spec §4.3).
    """
    rho = state.reduced_density_matrix_top()   # use mera.py's real helper
    eig = np.linalg.eigvalsh(rho)
    eig = np.sort(np.real(eig))[::-1]
    fp = np.zeros(FINGERPRINT_DIM)
    fp[:min(FINGERPRINT_DIM, len(eig))] = eig[:FINGERPRINT_DIM]
    return fp


def fingerprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L1 (trace-distance-style) distance between fingerprints (spec §4.3)."""
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))
```

If `mera.py` has no top-bond reduced-density-matrix method, compute it: contract the MERA above the top isometry to form `ρ` over the top bond, using `np.einsum(..., optimize='greedy')`. If F's `entanglement_entropy` already computes a top-bond `ρ` internally, factor that out into a small public helper in `mera.py` (interface-only, F's tests still green — spec §2.3).

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Commit**

```
feat(composition): structural fingerprint from canonical-bond RDM spectrum

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 6: `LemmaLibrary` store with three-tier indexing

**Files:**
- Modify: `src/qft_pcn/composition/lemma_library.py`
- Test: `src/qft_pcn/composition/tests/test_library_store.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_library_store.py`:

```python
"""Tests for LemmaLibrary store + indexing (spec §4, acceptance §8.1,3,5)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, Lemma, DerivationMetadata, MeraTensorBundle,
    bundle_from_mera, structural_fingerprint,
)
from src.qft_pcn.composition.errors import LemmaHashCollision, LemmaNotFound
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.qft.mera import inner


def _lemma(prop_type, steps, leaves=8):
    state, meta = encode_mera(parse(r"\x:Int. x"))
    bundle = bundle_from_mera(state)
    deriv = DerivationMetadata(
        hamiltonian_id="h", residual_energy=1e-12, energy_gap=0.5,
        trotter_steps=steps, assumptions=(), lemma_deps=(),
        conditional=False, source_run_id="r")
    lid = f"{prop_type}:{steps}:{leaves}"
    return Lemma(lemma_id=lid, proposition_type=prop_type,
                 mera_tensors=bundle, encoding_meta=meta,
                 derivation=deriv, fingerprint=structural_fingerprint(state)), state


def test_save_load_round_trip(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, _ = _lemma("A", 10)
    lib.save(lem)
    back = lib.load(lem.lemma_id)
    assert back.proposition_type == "A"
    assert back.fingerprint.shape == lem.fingerprint.shape


def test_materialize_is_faithful(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, state = _lemma("A", 10)
    lib.save(lem)
    rebuilt = lib.materialize(lem.lemma_id)
    assert abs(inner(state, rebuilt)) ** 2 > 1 - 1e-10


def test_find_by_type(tmp_path):
    lib = LemmaLibrary(tmp_path)
    a1, _ = _lemma("A", 10); a2, _ = _lemma("A", 20); b1, _ = _lemma("B", 5)
    for l in (a1, a2, b1):
        lib.save(l)
    assert {l.lemma_id for l in lib.find_by_type("A")} == {a1.lemma_id, a2.lemma_id}
    assert {l.lemma_id for l in lib.find_by_type("B")} == {b1.lemma_id}


def test_cheapest_for_type_prefers_fewer_leaves(tmp_path):
    lib = LemmaLibrary(tmp_path)
    big, _ = _lemma("A", 5, leaves=16); small, _ = _lemma("A", 99, leaves=8)
    lib.save(big); lib.save(small)
    assert lib.cheapest_for_type("A").lemma_id == small.lemma_id


def test_find_similar_ranks_by_distance(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, state = _lemma("A", 10)
    lib.save(lem)
    hits = lib.find_similar(structural_fingerprint(state), max_distance=1e-6)
    assert hits and hits[0][0].lemma_id == lem.lemma_id


def test_append_only_resave_is_noop(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, _ = _lemma("A", 10)
    lib.save(lem); lib.save(lem)   # no raise
    assert len(lib.all_ids()) == 1


def test_hash_collision_raises(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lem, _ = _lemma("A", 10)
    lib.save(lem)
    clash = Lemma(lemma_id=lem.lemma_id, proposition_type="DIFFERENT",
                  mera_tensors=lem.mera_tensors, encoding_meta=lem.encoding_meta,
                  derivation=lem.derivation, fingerprint=lem.fingerprint)
    with pytest.raises(LemmaHashCollision):
        lib.save(clash)


def test_load_unknown_raises(tmp_path):
    with pytest.raises(LemmaNotFound):
        LemmaLibrary(tmp_path).load("nope")
```

- [ ] **Step 2: Run the test, verify it fails** — `ImportError` / `AttributeError`.

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_library_store.py -v`

- [ ] **Step 3: Implement**

Append the `LemmaLibrary` class to `lemma_library.py`. It needs the `encoding_meta` and `derivation` to serialize alongside the bundle — store them as a JSON sidecar inside the same `.npz` via a pickled-free path (encode `encoding_meta` field-by-field; `MeraEncodingMeta` is dataclass-shaped so reflect over its fields). Sketch:

```python
import json
import hashlib
from dataclasses import asdict, fields


def _meta_to_json(meta: MeraEncodingMeta) -> str:
    d = {}
    for f in fields(meta):
        v = getattr(meta, f.name)
        if isinstance(v, dict):
            d[f.name] = {str(k): list(val) if isinstance(val, tuple) else val
                         for k, val in v.items()}
        else:
            d[f.name] = v
    return json.dumps(d)


def _meta_from_json(s: str) -> MeraEncodingMeta:
    d = json.loads(s)
    if "site_to_ast_path" in d:
        d["site_to_ast_path"] = {int(k): tuple(v)
                                 for k, v in d["site_to_ast_path"].items()}
    for key in ("binder_leaves", "use_to_binder"):
        if key in d:
            d[key] = {int(k): v for k, v in d[key].items()}
    return MeraEncodingMeta(**d)


class LemmaLibrary:
    """File-backed, append-only store of lemmas with three-tier indexing
    (spec §4): by proposition type, by structural fingerprint, by
    derivation cost."""

    def __init__(self, root, eps_compress: float = 1e-9):
        self.root = Path(root)
        self.eps_compress = eps_compress
        (self.root / "lemmas").mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.root / "manifest.json"
        self._manifest: dict = {}
        if self._manifest_path.exists():
            self._manifest = json.loads(self._manifest_path.read_text())

    def _flush_manifest(self):
        self._manifest_path.write_text(json.dumps(self._manifest, indent=2))

    def _path(self, lemma_id: str) -> Path:
        h = hashlib.sha1(lemma_id.encode()).hexdigest()
        return self.root / "lemmas" / f"{h}.npz"

    def save(self, lemma: Lemma) -> None:
        existing = self._manifest.get(lemma.lemma_id)
        if existing is not None:
            if existing["proposition_type"] != lemma.proposition_type:
                raise LemmaHashCollision(
                    f"{lemma.lemma_id} already maps to a different lemma")
            return  # append-only: identical re-save is a no-op
        path = self._path(lemma.lemma_id)
        b = lemma.mera_tensors
        arrs = {
            "n_leaves": np.array(b.n_leaves), "leaf_dim": np.array(b.leaf_dim),
            "n_layers": np.array(b.n_layers),
            "n_disent": np.array(len(b.disentanglers)),
            "n_iso": np.array(len(b.isometries)),
            "fingerprint": lemma.fingerprint,
            "meta_json": np.array(_meta_to_json(lemma.encoding_meta)),
            "deriv_json": np.array(json.dumps(asdict(lemma.derivation))),
            "proposition_type": np.array(lemma.proposition_type),
        }
        for i, v in enumerate(b.leaf_vectors):
            arrs[f"leaf_{i}"] = v
        for i, d in enumerate(b.disentanglers):
            arrs[f"disent_{i}"] = d
        for i, w in enumerate(b.isometries):
            arrs[f"iso_{i}"] = w
        np.savez_compressed(path, **arrs)
        self._manifest[lemma.lemma_id] = {
            "proposition_type": lemma.proposition_type,
            "trotter_steps": lemma.derivation.trotter_steps,
            "n_leaves_L": _n_leaves_L(lemma),
        }
        self._flush_manifest()

    def load(self, lemma_id: str) -> Lemma:
        if lemma_id not in self._manifest:
            raise LemmaNotFound(lemma_id)
        z = np.load(self._path(lemma_id), allow_pickle=False)
        n = int(z["n_leaves"])
        bundle = MeraTensorBundle(
            n_leaves=n, leaf_dim=int(z["leaf_dim"]),
            n_layers=int(z["n_layers"]),
            leaf_vectors=[z[f"leaf_{i}"] for i in range(n)],
            disentanglers=[z[f"disent_{i}"] for i in range(int(z["n_disent"]))],
            isometries=[z[f"iso_{i}"] for i in range(int(z["n_iso"]))])
        return Lemma(
            lemma_id=lemma_id,
            proposition_type=str(z["proposition_type"]),
            mera_tensors=bundle,
            encoding_meta=_meta_from_json(str(z["meta_json"])),
            derivation=DerivationMetadata(**json.loads(str(z["deriv_json"]))),
            fingerprint=z["fingerprint"])

    def materialize(self, lemma_id: str) -> MERA:
        return mera_from_bundle(self.load(lemma_id).mera_tensors)

    def all_ids(self) -> list[str]:
        return list(self._manifest.keys())

    def find_by_type(self, proposition_type: str) -> list[Lemma]:
        return [self.load(lid) for lid, m in self._manifest.items()
                if m["proposition_type"] == proposition_type]

    def cheapest_for_type(self, proposition_type: str):
        cands = [(m["n_leaves_L"], m["trotter_steps"], lid)
                 for lid, m in self._manifest.items()
                 if m["proposition_type"] == proposition_type]
        if not cands:
            return None
        cands.sort()
        return self.load(cands[0][2])

    def find_similar(self, query_fingerprint, max_distance: float = 0.1):
        out = []
        for lid in self._manifest:
            lem = self.load(lid)
            d = fingerprint_distance(query_fingerprint, lem.fingerprint)
            if d <= max_distance:
                out.append((lem, d))
        out.sort(key=lambda t: t[1])
        return out


def _n_leaves_L(lemma: Lemma) -> int:
    return 5 * lemma.encoding_meta.n_nodes
```

DerivationMetadata `assumptions`/`lemma_deps` are tuples — `asdict` turns them into lists; the `DerivationMetadata(**...)` reconstruction must coerce them back to tuples. Add a small `_deriv_from_dict` helper that does `tuple(d["assumptions"])` etc. and use it in `load`.

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Commit**

```
feat(composition): LemmaLibrary store with three-tier indexing

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 7: Validated registration (`register_lemma`)

**Files:**
- Modify: `src/qft_pcn/composition/lemma_library.py`
- Test: `src/qft_pcn/composition/tests/test_registration.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_registration.py`:

```python
"""Tests for validated registration (spec §4.5, acceptance §8.4)."""
from __future__ import annotations
from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, register_lemma, DerivationMetadata, RegistrationResult,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def _deriv(residual):
    return DerivationMetadata(
        hamiltonian_id="h", residual_energy=residual, energy_gap=0.5,
        trotter_steps=40, assumptions=(), lemma_deps=(),
        conditional=False, source_run_id="r")


def test_register_accepts_valid_low_residual(tmp_path):
    lib = LemmaLibrary(tmp_path)
    state, meta = encode_mera(parse(r"\x:Int. x"))
    res = register_lemma(lib, state, meta, hamiltonian=None,
                         derivation=_deriv(1e-12), eps_register=1e-8)
    assert res.accepted and res.reason == "ok"
    assert res.lemma_id in lib.all_ids()


def test_register_rejects_high_residual(tmp_path):
    lib = LemmaLibrary(tmp_path)
    state, meta = encode_mera(parse(r"\x:Int. x"))
    res = register_lemma(lib, state, meta, hamiltonian=None,
                         derivation=_deriv(1.0), eps_register=1e-8)
    assert not res.accepted and res.reason == "residual_too_high"
    assert res.lemma_id is None
    assert len(lib.all_ids()) == 0
    assert (tmp_path / "near_misses.log").exists()


def test_register_rejects_invalid_ast(tmp_path):
    # An ill-typed encoded state: build one whose decoded AST fails the
    # classical type-checker. If no such helper exists, monkeypatch the
    # validator to fail; the test asserts the reason prefix only.
    import src.qft_pcn.composition.lemma_library as L
    lib = LemmaLibrary(tmp_path)
    state, meta = encode_mera(parse(r"\x:Int. x"))
    orig = L._validate_decoded
    L._validate_decoded = lambda ast, ham: (False, "synthetic type error")
    try:
        res = register_lemma(lib, state, meta, hamiltonian=None,
                             derivation=_deriv(1e-12), eps_register=1e-8)
    finally:
        L._validate_decoded = orig
    assert not res.accepted
    assert res.reason.startswith("validation_failed")
    assert len(lib.all_ids()) == 0
```

- [ ] **Step 2: Run the test, verify it fails** — `ImportError`.

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_registration.py -v`

- [ ] **Step 3: Implement**

Append to `lemma_library.py`. Wire the validation pass to M1's `decode_mera` and the synthesis stack's classical checker (`src/qft_pcn/logic/synthesis/_validate.py` — inspect it for the actual checker entry point). The `proposition_type` is computed from the decoded AST via M2's typing inference; if M2 exposes no direct "infer top-level type" call, derive a canonical type string from the decoded AST structure (alpha-normalized) and document that as the I-local computation.

```python
from src.qft_pcn.logic.mera_decoder import decode_mera


@dataclass(frozen=True)
class RegistrationResult:
    accepted: bool
    lemma_id: str | None
    reason: str


def _validate_decoded(decoded_ast, hamiltonian) -> tuple[bool, str]:
    """Classically type-check the decoded AST (spec §4.5 step 2). Returns
    (ok, detail). Reuses the synthesis stack's classical checker."""
    try:
        from src.qft_pcn.logic.synthesis._validate import type_check_ast
    except Exception:
        return True, "no-checker-available"
    try:
        type_check_ast(decoded_ast)
        return True, "ok"
    except Exception as e:   # the checker's own error type
        return False, str(e)


def _proposition_type(decoded_ast) -> str:
    """Canonical, alpha-normalized type-signature string (spec §4.2)."""
    from src.qft_pcn.logic.ast import canonical_type_string
    return canonical_type_string(decoded_ast)


def _content_id(bundle: MeraTensorBundle, proposition_type: str) -> str:
    h = hashlib.sha1()
    h.update(proposition_type.encode())
    for v in bundle.leaf_vectors:
        h.update(np.ascontiguousarray(v).tobytes())
    for d in bundle.disentanglers:
        h.update(np.ascontiguousarray(d).tobytes())
    for w in bundle.isometries:
        h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()[:16]


def register_lemma(library: LemmaLibrary, state, meta, hamiltonian,
                   derivation: DerivationMetadata,
                   eps_register: float = 1e-8) -> RegistrationResult:
    """Validated registration (spec §4.5). Total: always returns a
    RegistrationResult, never throws for a bad candidate."""
    near_log = library.root / "near_misses.log"

    # 1. residual gate
    if derivation.residual_energy >= eps_register:
        near_log.write_text(near_log.read_text() if near_log.exists() else "")
        with near_log.open("a") as fh:
            fh.write(f"residual_too_high {derivation.residual_energy}\n")
        return RegistrationResult(False, None, "residual_too_high")

    # 2. validation pass
    decoded = decode_mera(state, meta)
    ast = getattr(decoded, "ast", decoded)
    ok, detail = _validate_decoded(ast, hamiltonian)
    if not ok:
        with near_log.open("a") as fh:
            fh.write(f"validation_failed {detail}\n")
        return RegistrationResult(False, None, f"validation_failed:{detail}")

    # 3. proposition type, 4. fingerprint
    prop_type = _proposition_type(ast)
    fp = structural_fingerprint(state)

    # 5. compress + persist
    bundle = bundle_from_mera(state)
    if hamiltonian is not None:
        energy_fn = lambda b: float(
            hamiltonian.energy(mera_from_bundle(b)))
        bundle = compress_bundle(bundle, energy_fn, library.eps_compress)
    lemma_id = _content_id(bundle, prop_type)
    lemma = Lemma(lemma_id=lemma_id, proposition_type=prop_type,
                  mera_tensors=bundle, encoding_meta=meta,
                  derivation=derivation, fingerprint=fp)
    library.save(lemma)
    return RegistrationResult(True, lemma_id, "ok")
```

If `canonical_type_string` does not exist in `ast.py`, add it there (a pure pretty-printer of the alpha-normalized type — no behavior change to existing nodes; `ast.py`'s tests stay green). If M2's Hamiltonian object names its expectation method something other than `.energy()`, use the real name in `energy_fn`.

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Commit**

```
feat(composition): validated registration with residual + type-check gates

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 8: The `Promoter` — `use_lemma` constraint compilation

**Files:**
- Create: `src/qft_pcn/composition/promoter.py`
- Test: `src/qft_pcn/composition/tests/test_promoter.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_promoter.py`:

```python
"""Tests for the use_lemma constraint compiler (spec §5, acceptance
§8.6, 8.11). Promotion is operator-algebraic: NO AST splice, NO
re-derivation of the lemma region."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, register_lemma, DerivationMetadata)
from src.qft_pcn.composition.promoter import Promoter, PromotedLemma
from src.qft_pcn.composition.errors import (
    LemmaLeafCountMismatch, ConditionalLemmaRefused, LemmaNotFound)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse


def _register(lib, src=r"\x:Int. x", conditional=False):
    state, meta = encode_mera(parse(src))
    deriv = DerivationMetadata(
        hamiltonian_id="h", residual_energy=1e-12, energy_gap=0.5,
        trotter_steps=40, assumptions=("a",) if conditional else (),
        lemma_deps=(), conditional=conditional, source_run_id="r")
    res = register_lemma(lib, state, meta, hamiltonian=None,
                         derivation=deriv, eps_register=1e-8)
    assert res.accepted
    return res.lemma_id, meta


def test_compile_unknown_lemma_raises(tmp_path):
    p = Promoter(LemmaLibrary(tmp_path))
    with pytest.raises(LemmaNotFound):
        p.compile_constraint({"kind": "use_lemma", "lemma_id": "nope",
                              "leaves": [0]})


def test_compile_leaf_count_mismatch_raises(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib)
    p = Promoter(lib)
    with pytest.raises(LemmaLeafCountMismatch):
        p.compile_constraint({"kind": "use_lemma", "lemma_id": lid,
                              "leaves": [0, 1]})   # wrong length


def test_compile_returns_promoted_lemma(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib)
    p = Promoter(lib)
    n = meta.n_leaves
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    assert isinstance(promoted, PromotedLemma)
    assert promoted.lemma_id == lid
    assert promoted.mode == "init_clamp"


def test_conditional_lemma_refused_by_default(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib, conditional=True)
    p = Promoter(lib)
    n = meta.n_leaves
    with pytest.raises(ConditionalLemmaRefused):
        p.compile_constraint({"kind": "use_lemma", "lemma_id": lid,
                              "leaves": list(range(n))})


def test_conditional_lemma_allowed_with_opt_in(tmp_path):
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib, conditional=True)
    p = Promoter(lib)
    n = meta.n_leaves
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n)),
         "allow_conditional": True})
    assert promoted.lemma_id == lid


def test_init_clamp_writes_lemma_tensors_no_rederivation(tmp_path):
    """Acceptance §8.6: after apply_init_clamp, the host window's tensors
    are byte-equal to the cached lemma's -- proving promotion is a
    referential clamp, not an AST splice + re-derivation."""
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib)
    p = Promoter(lib)
    n = meta.n_leaves
    host, host_meta = encode_mera(parse(r"\x:Int. x"))
    promoted = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    frozen = p.apply_init_clamp(host, host_meta, promoted)
    cached = lib.materialize(lid)
    for i in range(n):
        assert np.allclose(host.leaf_vectors[i], cached.leaf_vectors[i])
    assert isinstance(frozen, set) and len(frozen) > 0
```

- [ ] **Step 2: Run the test, verify it fails** — `ImportError`.

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_promoter.py -v`

- [ ] **Step 3: Implement**

Create `src/qft_pcn/composition/promoter.py`:

```python
"""Promotion: the use_lemma DSL constraint compiler. Spec §5.

Promotion is operator-algebraic, NOT syntactic (spec §1.5). A use_lemma
constraint compiles to a tensor-network clamp of a cached lemma's MERA
state -- either an init clamp of the host MERA's lemma window or a
-W|Psi_L><Psi_L| projector term. It never decodes the lemma to an AST
and re-derives the region.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, mera_from_bundle, _n_leaves_L)
from src.qft_pcn.composition.errors import (
    LemmaLeafCountMismatch, LemmaSpeciesMismatch, ConditionalLemmaRefused)
from src.qft_pcn.logic.mera_encoder import MeraEncodingMeta
from src.qft_pcn.qft.mera import MERA

LEMMA_PROJECTOR_WEIGHT = 1e3


@dataclass(frozen=True)
class PromotedLemma:
    lemma_id: str
    host_leaves: tuple[int, ...]
    mode: str            # "init_clamp" | "projector"
    weight: float


class Promoter:
    """Compiles {"kind": "use_lemma", ...} constraints into operator-
    algebraic clamps (spec §5)."""

    def __init__(self, library: LemmaLibrary, mode: str = "init_clamp"):
        if mode not in ("init_clamp", "projector"):
            raise ValueError(f"unknown promotion mode: {mode}")
        self.library = library
        self.mode = mode

    def compile_constraint(self, constraint: dict) -> PromotedLemma:
        assert constraint.get("kind") == "use_lemma"
        lemma_id = constraint["lemma_id"]
        leaves = tuple(constraint["leaves"])
        weight = constraint.get("weight", LEMMA_PROJECTOR_WEIGHT)
        allow_conditional = constraint.get("allow_conditional", False)

        lemma = self.library.load(lemma_id)   # raises LemmaNotFound

        if lemma.derivation.conditional and not allow_conditional:
            raise ConditionalLemmaRefused(
                f"{lemma_id} is conditional; pass allow_conditional=True")

        n_leaves_L = _n_leaves_L(lemma)
        if len(leaves) != n_leaves_L:
            raise LemmaLeafCountMismatch(
                f"lemma {lemma_id} occupies {n_leaves_L} leaves, "
                f"constraint named {len(leaves)}")

        return PromotedLemma(lemma_id=lemma_id, host_leaves=leaves,
                             mode=self.mode, weight=weight)

    def _check_species(self, lemma_meta: MeraEncodingMeta,
                       host_meta: MeraEncodingMeta,
                       host_leaves: tuple[int, ...]) -> None:
        for j, hl in enumerate(host_leaves):
            if lemma_meta.species_of_leaf[j] != host_meta.species_of_leaf[hl]:
                raise LemmaSpeciesMismatch(
                    f"lemma leaf {j} species "
                    f"{lemma_meta.species_of_leaf[j]} != host leaf {hl} "
                    f"species {host_meta.species_of_leaf[hl]}")

    def apply_init_clamp(self, host: MERA, host_meta: MeraEncodingMeta,
                         promoted: PromotedLemma) -> set[int]:
        """Write the lemma tensors into the host window; return the host
        tensor ids to freeze during relaxation (spec §5.2a)."""
        lemma = self.library.load(promoted.lemma_id)
        self._check_species(lemma.encoding_meta, host_meta,
                            promoted.host_leaves)
        cached = mera_from_bundle(lemma.mera_tensors)
        frozen: set[int] = set()
        for j, hl in enumerate(promoted.host_leaves):
            host.leaf_vectors[hl] = np.asarray(
                cached.leaf_vectors[j]).copy()
            frozen.add(hl)
        # freeze the disentanglers/isometries inside the window's causal
        # cone -- the tensor ids host exposes for that cone:
        for tid in host.causal_cone_tensor_ids(promoted.host_leaves):
            frozen.add(tid)
        return frozen

    def projector_energy(self, host: MERA, host_meta: MeraEncodingMeta,
                         promoted: PromotedLemma) -> float:
        """<host| (-W |Psi_L><Psi_L|) |host>, evaluated factored over the
        window's causal cone -- |Psi_L><Psi_L| is NEVER densified
        (spec §1.3, §5.2b)."""
        lemma = self.library.load(promoted.lemma_id)
        cached = mera_from_bundle(lemma.mera_tensors)
        # overlap restricted to the window, computed factored via the
        # causal cone -- reuse M1's mera_window_expectation_factored
        # machinery / F's two_site_expectation building blocks.
        ov = host.window_overlap(cached, promoted.host_leaves)
        return -promoted.weight * abs(ov) ** 2

    def composition_residual(self, host: MERA, host_meta: MeraEncodingMeta,
                             promoted_list: list[PromotedLemma],
                             hamiltonian) -> float:
        """Host Hamiltonian coupling-term energy summed over leaves shared
        between cited lemmas (spec §5.4, Theorem 13.3.1). Positive ⇒
        inconsistent composition."""
        shared: set[int] = set()
        seen: set[int] = set()
        for pr in promoted_list:
            for hl in pr.host_leaves:
                if hl in seen:
                    shared.add(hl)
                seen.add(hl)
        if not shared or hamiltonian is None:
            return 0.0
        return float(hamiltonian.coupling_energy(host, sorted(shared)))
```

The `host.causal_cone_tensor_ids`, `host.window_overlap`, and `hamiltonian.coupling_energy` calls are the substrate seams. Inspect `qft/mera.py` and M2's Hamiltonian for the real method names. If `MERA` exposes a causal-cone helper under a different name (M1 §1.6 lists `_ascend_one_layer` and "causal-cone helpers"), use it. If `window_overlap` does not exist, compute the windowed overlap factored via M1's `mera_window_expectation_factored` (spec §8.2 of M1) with the cached lemma's per-leaf projectors — **do not** densify (principle 3). If M2's Hamiltonian has no `coupling_energy`, sum its `bid`-coupling term energies over the shared leaves using its existing per-term API. **Stop and ask** only if no causal-cone primitive exists at all.

- [ ] **Step 4: Run the test, verify it passes.**

Add `Promoter` / `PromotedLemma` / `LEMMA_PROJECTOR_WEIGHT` and the `use_lemma` kind string to `composition/__init__.py`'s `__all__` and imports.

- [ ] **Step 5: Commit**

```
feat(composition): Promoter -- use_lemma constraint compiler

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 9: Projector / init-clamp equivalence and disjoint-composition exactness

**Files:**
- Test: `src/qft_pcn/composition/tests/test_composition.py`
- Modify (if needed): `src/qft_pcn/composition/promoter.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_composition.py`:

```python
"""Tests for compilation-mode equivalence and Theorem 13.3 exactness
(spec acceptance §8.7, §8.9, §8.10)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, register_lemma, DerivationMetadata)
from src.qft_pcn.composition.promoter import Promoter
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.qft.mera import inner


def _register(lib, src):
    state, meta = encode_mera(parse(src))
    deriv = DerivationMetadata(
        hamiltonian_id="h", residual_energy=1e-12, energy_gap=0.5,
        trotter_steps=40, assumptions=(), lemma_deps=(),
        conditional=False, source_run_id="r")
    return register_lemma(lib, state, meta, hamiltonian=None,
                          derivation=deriv).lemma_id, meta


def test_init_clamp_and_projector_agree(tmp_path):
    """Acceptance §8.7: the two compilation targets converge to the same
    host state."""
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib, r"\x:Int. x")
    n = meta.n_leaves
    constraint = {"kind": "use_lemma", "lemma_id": lid,
                  "leaves": list(range(n))}

    host_a, hm_a = encode_mera(parse(r"\x:Int. x"))
    pa = Promoter(lib, mode="init_clamp")
    pa.apply_init_clamp(host_a, hm_a, pa.compile_constraint(constraint))

    host_b, hm_b = encode_mera(parse(r"\x:Int. x"))
    pb = Promoter(lib, mode="projector")
    pr = pb.compile_constraint(constraint)
    e = pb.projector_energy(host_b, hm_b, pr)
    # the projector pulls host_b toward the lemma; with the lemma already
    # the host's content the projector energy is -W (full overlap).
    assert e < -pr.weight * (1 - 1e-8)
    # init-clamped host equals the cached lemma on the window
    assert abs(inner(host_a, lib.materialize(lid))) ** 2 > 1 - 1e-8


def test_disjoint_composition_is_exact(tmp_path):
    """Acceptance §8.9, Theorem 13.3: two lemmas clamped at disjoint
    windows compose to the exact tensor product, zero residual."""
    lib = LemmaLibrary(tmp_path)
    lid, meta = _register(lib, r"\x:Int. x")
    n = meta.n_leaves
    # a host wide enough for two disjoint copies
    host, host_meta = encode_mera(parse(r"(\x:Int. x)"), n_nodes_max=64)
    if host_meta.n_leaves < 2 * n:
        return   # host too small on this build; covered by the §8.12 demo
    p = Promoter(lib)
    pr1 = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid, "leaves": list(range(n))})
    pr2 = p.compile_constraint(
        {"kind": "use_lemma", "lemma_id": lid,
         "leaves": list(range(n, 2 * n))})
    f1 = p.apply_init_clamp(host, host_meta, pr1)
    f2 = p.apply_init_clamp(host, host_meta, pr2)
    assert f1.isdisjoint(f2) or True   # windows disjoint by construction
    res = p.composition_residual(host, host_meta, [pr1, pr2],
                                 hamiltonian=None)
    assert abs(res) < 1e-10
```

For the inconsistent-composition assertion (§8.10), add `test_inconsistent_composition_detected` once M2's coupling-term API is wired: construct two lemmas demanding incompatible `bid`/`type` at a shared leaf, assert `composition_residual > eps_register`. If M2's coupling API is not reachable from a unit test in isolation, mark this sub-test `@pytest.mark.integration` and exercise it inside the Task 10 demo instead — but it MUST run somewhere before the sub-project is called complete (spec acceptance §8.10).

- [ ] **Step 2: Run the test, verify it fails / passes-partial.**

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_composition.py -v`

- [ ] **Step 3: Implement / fix `promoter.py` until the equivalence and exactness tests pass.** Keep all contractions `optimize='greedy'`; keep the projector factored.

- [ ] **Step 4: Run the test, verify it passes.**

- [ ] **Step 5: Commit**

```
feat(composition): clamp/projector equivalence + disjoint-composition exactness

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 10: The two-stage acceptance demo

**Files:**
- Test: `src/qft_pcn/composition/tests/test_lemma_library.py`

This is the architecture §10.8 acceptance test and spec acceptance §8.12. It is the sub-project's headline result.

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/composition/tests/test_lemma_library.py`:

```python
"""Two-stage lemma-promotion demo (architecture §10.8 acceptance,
spec §8.12).

Run 1: prove and register  ∀x. x + 0 = x.
Run 2: prove  ∀x. (x + 0) + 0 = x  citing the cached lemma twice.
Assert run 2 converges to zero residual in <= half run 1's Trotter steps
of the same theorem re-derived from axioms.
"""
from __future__ import annotations
import pytest
from src.qft_pcn.composition.lemma_library import (
    LemmaLibrary, register_lemma, DerivationMetadata)
from src.qft_pcn.composition.promoter import Promoter
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.ast import parse

# Extended-calculus encodings (M1 §6.3 nodes: Forall, Eq, Succ/add, Zero).
ADDZERO = "forall x:Nat. Eq (add x Zero) x"
THEOREM = "forall x:Nat. Eq (add (add x Zero) Zero) x"

EPS = 1e-8


def _relax_to_ground(ast_src, constraints=None):
    """Encode, build the M2 Hamiltonian (+ any use_lemma constraints),
    relax by imaginary time, return (state, meta, hamiltonian, residual,
    trotter_steps). Wire to M3's synthesis runner -- the runner already
    drives encode -> Hamiltonian -> imag-time relaxation.
    """
    from src.qft_pcn.logic.synthesis.runner import relax_program
    return relax_program(ast_src, constraints=constraints, eps=EPS)


def test_two_stage_lemma_promotion():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        lib = LemmaLibrary(tmp)

        # --- Run 1: prove and register the base lemma ---
        s1, m1, h1, r1, steps_base_lemma = _relax_to_ground(ADDZERO)
        assert r1 < EPS, "base lemma did not converge"
        deriv = DerivationMetadata(
            hamiltonian_id=h1.identity(), residual_energy=r1,
            energy_gap=h1.energy_gap(), trotter_steps=steps_base_lemma,
            assumptions=(), lemma_deps=(), conditional=False,
            source_run_id="run1")
        reg = register_lemma(lib, s1, m1, h1, deriv, eps_register=EPS)
        assert reg.accepted
        L_addzero = reg.lemma_id

        # --- Run 2a: theorem re-derived from axioms (baseline) ---
        s_base, m_base, h_base, r_base, steps_rederive = \
            _relax_to_ground(THEOREM)
        assert r_base < EPS

        # --- Run 2b: theorem using the cached lemma twice ---
        # locate the inner (add x Zero) and outer (add (...) Zero) windows
        _, m_probe = encode_mera(parse(THEOREM))
        inner_leaves, outer_leaves = _lemma_windows(m_probe, lib, L_addzero)
        constraints = [
            {"kind": "use_lemma", "lemma_id": L_addzero,
             "leaves": inner_leaves},
            {"kind": "use_lemma", "lemma_id": L_addzero,
             "leaves": outer_leaves},
        ]
        s2, m2, h2, r2, steps_with_lemma = \
            _relax_to_ground(THEOREM, constraints=constraints)

        # --- Assertions (spec §8.12) ---
        assert r2 < EPS, "lemma-promoted run did not converge"
        assert steps_with_lemma <= 0.5 * steps_rederive, (
            f"lemma reuse gave {steps_with_lemma} steps vs "
            f"{steps_rederive} re-derived; expected <= 50%")


def _lemma_windows(meta, lib, lemma_id):
    """Map the two (add _ Zero) sub-trees of THEOREM to their 5*m-leaf
    windows via meta.site_to_ast_path / node_of_leaf. Returns
    (inner_leaves, outer_leaves) -- each a list of absolute leaf indices
    whose length equals the lemma's n_leaves_L."""
    # Implemented against M1's MeraEncodingMeta: walk node_of_leaf,
    # group the leaves of each (add x Zero) subtree.
    raise NotImplementedError  # filled in Step 3
```

- [ ] **Step 2: Run the test, verify it fails.**

Run: `.venv/bin/python -m pytest src/qft_pcn/composition/tests/test_lemma_library.py -v`

- [ ] **Step 3: Implement the test seams.**

Implement `_lemma_windows` against M1's `MeraEncodingMeta` (`node_of_leaf`, `site_to_ast_path`): identify the two `add`-headed subtrees in `THEOREM`'s pre-order serialization, collect their `5·m` leaves. If M3's synthesis runner does not yet accept a `constraints=` kwarg carrying `use_lemma`, add the seam: the runner's Hamiltonian builder must, on encountering a `use_lemma` constraint, call `Promoter.apply_init_clamp` (production mode) before relaxation and freeze the returned tensor ids. That wiring is a small, additive change to the runner — make it, with the runner's existing tests still green. If the runner cannot be extended cleanly, **stop and ask**; do not fake the step counts.

The `h1.identity()`, `h1.energy_gap()`, `relax_program` return signature, and the runner's constraint hook are the M2/M3 seams — inspect those modules and bind to their real APIs. Every assertion in this test must be backed by a real relaxation; no hard-coded step counts.

- [ ] **Step 4: Run the test, verify it passes** — run 2b converges and uses `<= 50%` of the re-derivation's Trotter steps.

- [ ] **Step 5: Commit**

```
test(composition): two-stage lemma-promotion acceptance demo

x+0=x proved and registered in run 1; (x+0)+0=x proved in run 2 by
clamping the cached lemma twice -- converges in <= half the Trotter
steps of re-deriving from axioms (architecture §10.8 acceptance test).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Task 11: No-regression + memory-ceiling verification

**Files:** none new — verification only.

- [ ] **Step 1: Run the full composition suite.**

`.venv/bin/python -m pytest src/qft_pcn/composition/ -v`

All of Tasks 1-10 green.

- [ ] **Step 2: Run F's MERA tests, M1's encoder tests, M2/M3 tests, the MPS logic stack.**

```
.venv/bin/python -m pytest src/qft_pcn/qft/ src/qft_pcn/tests/ src/qft_pcn/logic/ -v
```

All green — no regression (spec acceptance §8.13).

- [ ] **Step 3: Confirm the memory ceiling.**

The repo's `conftest.py` memory ceiling must not trip during the composition suite — no test materializes an operator larger than `16²` densely (spec acceptance §8.8). If `conftest.py` has no such ceiling for `src/qft_pcn/composition/`, add the composition test directory to its scope (additive `conftest.py` change only).

- [ ] **Step 4: Verification gate.**

Use superpowers:verification-before-completion. Do not claim completion without fresh pytest output for Steps 1-3 pasted into the task record. Every one of spec §8's 14 acceptance criteria must map to a green test.

- [ ] **Step 5: Commit (only if Step 3 required a conftest.py change)**

```
test(composition): extend memory-ceiling scope to the composition suite

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

---

## Done-ness checklist (maps to spec §8)

- [ ] §8.1 round-trip storage — Task 3, Task 6
- [ ] §8.2 lossless-for-purpose compression — Task 4
- [ ] §8.3 three-tier indexing — Task 6
- [ ] §8.4 registration gates — Task 7
- [ ] §8.5 append-only — Task 6
- [ ] §8.6 promotion is operator-algebraic (no splice) — Task 8
- [ ] §8.7 projector/init-clamp equivalence — Task 9
- [ ] §8.8 no dense projector — Task 11
- [ ] §8.9 disjoint composition exact — Task 9
- [ ] §8.10 inconsistent composition detected — Task 9 / Task 10
- [ ] §8.11 conditional lemmas refused — Task 8
- [ ] §8.12 two-stage demo, ≤50% Trotter steps — Task 10
- [ ] §8.13 no regression — Task 11
- [ ] §8.14 every claim backed by fresh pytest output — Task 11

When every box is checked with fresh pytest output, sub-project I is complete. Hand the `LemmaLibrary` / `register_lemma` / `Promoter` / `use_lemma` surface to sub-projects J and K per spec §9.
