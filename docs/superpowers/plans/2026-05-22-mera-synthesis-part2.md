# MERA-Native Debugger + Synthesis Implementation Plan — Part 2 of 2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the synthesis problem types, the synthesis Hamiltonian, the `synthesize()` runner, the publishable P1–P8 demo, and the full acceptance suite — on top of Part 1's debugger and structural-superposition encoder.

**Driving principles:** see Part 1's preamble — all ten, embedded in every subagent prompt. The two most in play in Part 2: **principle 6** (holes are quantum superpositions; the runner never loops over candidates) and **principle 7** (rank by ⟨H⟩, verify by decoding the AST).

**Spec:** `docs/superpowers/specs/2026-05-22-mera-synthesis-design.md` — authoritative.

**Test runner:** `.venv/bin/python -m pytest <path> -v`.

---

## Pre-existing worktree state

Same as Part 1: unrelated modified files exist; leave them alone, stage only files each task names.

---

## Task 6: Synthesis problem types + errors

**Files:**
- Create: `src/qft_pcn/logic/mera_synthesis/problem.py`
- Create: `src/qft_pcn/logic/mera_synthesis/errors.py`
- Test: `src/qft_pcn/tests/test_mera_synthesis_problem.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_synthesis_problem.py`:

```python
"""Tests for synthesis problem data types (spec §6.1)."""
from __future__ import annotations
import pytest
from src.qft_pcn.logic.ast import Lam, TInt, Var, HoleVar
from src.qft_pcn.logic.mera_synthesis.problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)
from src.qft_pcn.logic.mera_synthesis.errors import (
    SynthesisProblemError, SynthesisRuntimeError,
)


def test_io_example_holds_inputs_and_output():
    from src.qft_pcn.logic.ast import IntLit
    ex = IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3))
    assert ex.inputs[0].val == 3


def test_problem_defaults():
    hole = HoleVar(candidates=("x",))
    p = SynthesisProblem(sketch=Lam(param="x", param_ty=TInt(), body=hole))
    assert p.chi_layer == 32
    assert p.n_samples == 64
    assert p.target_type is None


def test_hamiltonian_weights_defaults():
    w = HamiltonianWeights()
    assert (w.w_T, w.w_E, w.w_X, w.w_Y, w.w_S) == (4.0, 2.0, 3.0, 2.0, 0.1)


def test_errors_are_distinct_classes():
    assert issubclass(SynthesisProblemError, Exception)
    assert issubclass(SynthesisRuntimeError, Exception)
    assert SynthesisProblemError is not SynthesisRuntimeError
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_problem.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `errors.py` and `problem.py`**

Create `src/qft_pcn/logic/mera_synthesis/errors.py`:

```python
"""Synthesis error model (spec §10)."""
from __future__ import annotations


class SynthesisError(Exception):
    """Base class for MERA synthesis runner errors."""


class SynthesisProblemError(SynthesisError):
    """Malformed SynthesisProblem (spec §6.1 invariants violated)."""


class SynthesisRuntimeError(SynthesisError):
    """Evolution diverged, NaN energies, or zero decodable samples.
    A problem with no good completion is NOT raised -- it is reported
    via SynthesisResult.failure_mode (spec §1.10)."""
```

Create `src/qft_pcn/logic/mera_synthesis/problem.py`:

```python
"""Synthesis problem and result data types (spec §6.1)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..ast import Node, Ty


@dataclass(frozen=True)
class IOExample:
    """Applying the synthesized expression to `inputs` must yield `output`."""
    inputs: tuple              # tuple[Node, ...] of IntLit/BoolLit/NatLit
    output: Node


@dataclass(frozen=True)
class HamiltonianWeights:
    """Block weights for the composed synthesis Hamiltonian (spec §4.1)."""
    w_T: float = 4.0           # typing
    w_E: float = 2.0           # eval
    w_X: float = 3.0           # examples
    w_Y: float = 2.0           # target_type
    w_S: float = 0.1           # size (Occam)


@dataclass(frozen=True)
class SynthesisProblem:
    """The contract: solve me (spec §6.1)."""
    sketch: Node
    target_type: Ty | None = None
    examples: tuple = ()
    name: str = ""
    n_nodes_max: int = 32
    chi_layer: int = 32
    n_samples: int = 64
    anneal_steps: int = 200
    anneal_dt: float = 0.05


@dataclass(frozen=True)
class Completion:
    ast: Node
    energy: float
    energy_breakdown: dict
    diagnostics: dict
    multiplicity: int


@dataclass(frozen=True)
class SynthesisResult:
    problem: SynthesisProblem
    completions: list
    n_unique: int
    n_samples_drawn: int
    n_samples_decoded_ok: int
    final_state_energy: float
    chi_observed_max: int
    failure_mode: str | None
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_problem.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_synthesis/problem.py \
        src/qft_pcn/logic/mera_synthesis/errors.py \
        src/qft_pcn/tests/test_mera_synthesis_problem.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_synthesis/problem): synthesis problem + result types

SynthesisProblem, IOExample, Completion, SynthesisResult,
HamiltonianWeights, and the SynthesisProblemError / SynthesisRuntimeError
error model. chi_layer defaults to 32 for hole superpositions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: The synthesis Hamiltonian — `H_examples` / `H_target_type` / `H_size`

**Files:**
- Modify: `src/qft_pcn/logic/mera_synthesis/encode_ext.py` (add `_witness_augmented_ast`, `RefVar`)
- Create: `src/qft_pcn/logic/mera_synthesis/hamiltonian.py`
- Test: `src/qft_pcn/tests/test_mera_synthesis_hamiltonian.py`

Read M2's Hamiltonian API first: `compose_mera_hamiltonians`, the `NamedMeraTerm` contract (M2 exposes `.terms`/`.term_energy`/`.residuals`/`.total_energy`), and `mera_window_expectation_factored` (M1). The synthesis blocks are factored per-leaf operators — **no dense `16**k`** (principle 3).

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_synthesis_hamiltonian.py`:

```python
"""Tests for the synthesis Hamiltonian blocks (spec §4.2-§4.5, §9.5)."""
from __future__ import annotations
import numpy as np
from src.qft_pcn.logic.ast import Lam, TInt, Var, IntLit, HoleVar
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_synthesis.problem import (
    SynthesisProblem, IOExample, HamiltonianWeights,
)
from src.qft_pcn.logic.mera_synthesis.hamiltonian import (
    build_size_terms, build_target_type_terms, build_example_terms,
    compile_mera_synthesis_hamiltonian,
)
from src.qft_pcn.logic.mera_debugger import NamedMeraTerm


def _concrete():
    return Lam(param="x", param_ty=TInt(), body=Var(name="x"))


def test_size_terms_are_named_mera_terms():
    state, meta = encode_mera(_concrete())
    terms = build_size_terms(meta, w_S=0.1)
    assert terms, "no size terms built"
    for t in terms:
        assert isinstance(t, NamedMeraTerm)
        assert t.rule_class == "S-Size"


def test_size_penalty_counts_non_pad_nodes():
    state, meta = encode_mera(_concrete())
    terms = build_size_terms(meta, w_S=0.1)
    total = sum(t.expectation(state) for t in terms)
    # 2 non-PAD nodes -> 2 * 0.1 = 0.2 (PAD projector complement).
    assert abs(total - 0.2) < 1e-6


def test_target_type_term_zero_when_type_matches():
    from src.qft_pcn.logic.ast import TArrow
    state, meta = encode_mera(_concrete())
    tt = TArrow(src=TInt(), dst=TInt())
    terms = build_target_type_terms(meta, tt, w_Y=2.0)
    total = sum(t.expectation(state) for t in terms)
    assert abs(total) < 1e-6, total


def test_target_type_term_nonzero_when_type_mismatches():
    from src.qft_pcn.logic.ast import TBool
    state, meta = encode_mera(_concrete())
    terms = build_target_type_terms(meta, TBool(), w_Y=2.0)
    total = sum(t.expectation(state) for t in terms)
    assert total > 1e-6, total


def test_compose_yields_total_energy():
    hole = HoleVar(candidates=("x",))
    problem = SynthesisProblem(
        sketch=Lam(param="x", param_ty=TInt(), body=hole),
        examples=(IOExample(inputs=(IntLit(val=3),), output=IntLit(val=3)),),
    )
    # encode the witness-augmented sketch (Task 7 builds it).
    from src.qft_pcn.logic.mera_synthesis.encode_ext import (
        _witness_augmented_ast,
    )
    aug = _witness_augmented_ast(problem.sketch, problem.examples)
    state, meta = encode_mera(aug, chi_layer=32)
    H = compile_mera_synthesis_hamiltonian(meta, problem, HamiltonianWeights())
    e = H.total_energy(state)
    assert isinstance(e, float)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_hamiltonian.py -v`
Expected: ImportError.

- [ ] **Step 3: Add `_witness_augmented_ast` + `RefVar` to `encode_ext.py`**

Append to `src/qft_pcn/logic/mera_synthesis/encode_ext.py`:

```python
from ..ast import App, Lam   # add to existing imports as needed


@dataclass
class RefVar(Node):
    """Encoder-internal node: references a previously-encoded sub-tree.

    Carries KIND_VAR; its bid leaf is entangled with the referenced
    node's bid leaf (the witness shares its function sub-tree with the
    sketch through the tree -- spec §4.2, not a classical copy).
    `target` is the AST node index of the referenced sub-tree root.
    """
    target: int = 0


def _witness_augmented_ast(sketch: Node, examples: tuple) -> Node:
    """Return an AST where `sketch` sits at the head and each example
    appends a (App ... (App (RefVar root) in_0) ... in_{n-1}) witness
    sub-tree (spec §4.2). The witness's RefVar shares the sketch root's
    function sub-tree through the encoder's binder-channel mechanism --
    NOT by duplicating the encoding (principle: §1.2, witnesses share).

    The augmented AST is a synthetic container node the encoder walks;
    its exact container form depends on M1's serialize_preorder -- if M1
    has no multi-root container, introduce a minimal internal `Bundle`
    node (KIND used only by the encoder) that serialize_preorder visits
    children-in-order. Read _serialize.py; reuse, do not fork.
    """
    ...
```

The container question (how multiple roots — the sketch plus K witness sub-trees — sit on one MERA tree) is resolved by reading M1's `_serialize.py`. If M1 already serializes a forest, reuse it; if not, add a minimal internal `Bundle` node. Pin the choice in `encode_ext.py`; the `test_compose_yields_total_energy` test is the oracle.

- [ ] **Step 4: Implement `hamiltonian.py`**

Create `src/qft_pcn/logic/mera_synthesis/hamiltonian.py`:

```python
"""Synthesis Hamiltonian: H_examples + H_target_type + H_size, composed
with M2's H_typing and H_eval (spec §4).

Every term is a factored per-leaf operator measured via
mera_window_expectation_factored -- no dense 16**k operator (principle 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mera_encoder import MeraEncodingMeta
from ..mera_encoding import (
    MERA_LEAF_DIM, KIND_PAD, SPECIES_LEAF_OFFSET,
)
from .._mera_window import mera_window_expectation_factored
from ..ast import Ty


def _pad_complement_op() -> np.ndarray:
    """I - |KIND_PAD><KIND_PAD| on a kind leaf."""
    op = np.eye(MERA_LEAF_DIM, dtype=complex)
    op[KIND_PAD, KIND_PAD] = 0.0
    return op


def _basis_complement_op(index: int) -> np.ndarray:
    """I - |index><index|."""
    op = np.eye(MERA_LEAF_DIM, dtype=complex)
    op[index, index] = 0.0
    return op


@dataclass
class _FactoredTerm:
    """A NamedMeraTerm whose expectation is one factored per-leaf op."""
    name: str
    rule_class: str
    node: int
    leaves: tuple
    weight: float
    leaf_ops: dict          # {absolute_leaf: (16,16) op}

    def expectation(self, state) -> float:
        val = mera_window_expectation_factored(state, self.leaf_ops)
        e = float(np.real(val)) * self.weight
        if abs(np.imag(val)) > 1e-10:
            raise ValueError(
                f"{self.name}: non-Hermitian term, Im={np.imag(val)}")
        return e


def build_size_terms(meta: MeraEncodingMeta, w_S: float) -> list:
    """One S-Size term per sketch node: w_S * <I - P_PAD> on its kind leaf
    (spec §4.4). Witness nodes excluded."""
    terms = []
    witness = set()
    for r in getattr(meta, "witness_node_ranges", []):
        witness.update(r)
    for node in range(meta.n_nodes):
        if node in witness:
            continue
        kind_leaf = 5 * node + SPECIES_LEAF_OFFSET["kind"]
        terms.append(_FactoredTerm(
            name=f"S-Size@node_{node}", rule_class="S-Size", node=node,
            leaves=(kind_leaf,), weight=w_S,
            leaf_ops={kind_leaf: _pad_complement_op()}))
    return terms


def build_target_type_terms(meta: MeraEncodingMeta, target_type: Ty | None,
                            w_Y: float) -> list:
    """One T-Target term pinning node 0's type leaf (spec §4.3)."""
    if target_type is None:
        return []
    from ..mera_encoder import type_tag_of   # reuse M1's type->tag map
    tag = type_tag_of(target_type)
    type_leaf = 5 * 0 + SPECIES_LEAF_OFFSET["type"]
    return [_FactoredTerm(
        name="T-Target@node_0", rule_class="T-Target", node=0,
        leaves=(type_leaf,), weight=w_Y,
        leaf_ops={type_leaf: _basis_complement_op(tag)})]


def build_example_terms(meta: MeraEncodingMeta, examples: tuple,
                        w_X: float) -> list:
    """One X-Example term per example: a boundary pin on the witness
    root's value leaf, w_X * <I - |output><output|> (spec §4.2)."""
    terms = []
    for i, ex in enumerate(examples):
        wit_root = _witness_root_node(meta, i)   # from meta.witness_node_ranges
        val_leaf = 5 * wit_root + SPECIES_LEAF_OFFSET["value"]
        out_idx = _value_index_of(ex.output)     # reuse M1's value encoding
        terms.append(_FactoredTerm(
            name=f"X-Example@node_{wit_root}", rule_class="X-Example",
            node=wit_root, leaves=(val_leaf,), weight=w_X,
            leaf_ops={val_leaf: _basis_complement_op(out_idx)}))
    return terms


def compile_mera_synthesis_hamiltonian(meta, problem, weights):
    """Compose H_typing (M2) + H_eval (M2) + H_examples + H_target_type +
    H_size via M2's compose_mera_hamiltonians. Returns the composed object
    exposing .terms / .term_energy / .residuals / .total_energy."""
    from ..mera_typing_hamiltonian import MeraTypingHamiltonian   # M2
    from ..mera_eval_hamiltonian import MeraEvalHamiltonian       # M2
    from ..mera_hamiltonian import compose_mera_hamiltonians      # M2

    h_typing = MeraTypingHamiltonian(meta)
    h_eval = MeraEvalHamiltonian(meta)
    synth_terms = (
        build_example_terms(meta, problem.examples, weights.w_X)
        + build_target_type_terms(meta, problem.target_type, weights.w_Y)
        + build_size_terms(meta, weights.w_S)
    )
    return compose_mera_hamiltonians(
        (h_typing, weights.w_T), (h_eval, weights.w_E),
        extra_terms=synth_terms)
```

The M2 import paths (`mera_typing_hamiltonian`, `mera_eval_hamiltonian`, `mera_hamiltonian`, `compose_mera_hamiltonians`) and the M1 helpers (`type_tag_of`, `_value_index_of`, `_witness_root_node`) are placeholders — **read M2's actual module names and M1's actual helper names before implementing**, and use the real ones. M2's spec is being written in parallel; if a needed M2 entry point is genuinely absent, escalate (principle 9) — do not stub it.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_hamiltonian.py -v --timeout=60`
Expected: 5 passed. If M2 is not yet importable, the `test_compose_yields_total_energy` test is `pytest.skip`-gated on M2 — add the skip guard and proceed; the other 4 tests (size/target_type, M3-local) must pass unconditionally.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/mera_synthesis/encode_ext.py \
        src/qft_pcn/logic/mera_synthesis/hamiltonian.py \
        src/qft_pcn/tests/test_mera_synthesis_hamiltonian.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_synthesis/hamiltonian): synthesis Hamiltonian blocks

H_examples (witness-root value-leaf boundary pin), H_target_type (root
type-leaf pin), H_size (per-node kind-leaf PAD penalty), each a list of
factored NamedMeraTerm measured via mera_window_expectation_factored --
no dense 16**k. compile_mera_synthesis_hamiltonian composes them with
M2's H_typing and H_eval. Adds _witness_augmented_ast / RefVar for the
example-witness regions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Ranking — dedupe, rerank, classify failure_mode

**Files:**
- Create: `src/qft_pcn/logic/mera_synthesis/ranking.py`
- Test: `src/qft_pcn/tests/test_mera_synthesis_ranking.py`

- [ ] **Step 1: Write the failing test**

Create `src/qft_pcn/tests/test_mera_synthesis_ranking.py`:

```python
"""Tests for synthesis ranking (spec §6.3, §6.4)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import Lam, TInt, Var
from src.qft_pcn.logic.mera_decoder import DecodeResult
from src.qft_pcn.logic.mera_synthesis.ranking import (
    dedupe_by_alpha_eq, classify_failure_mode,
)
from src.qft_pcn.logic.mera_synthesis.problem import Completion


def _id():
    return Lam(param="x", param_ty=TInt(), body=Var(name="x"))


def _id_renamed():
    return Lam(param="y", param_ty=TInt(), body=Var(name="y"))


def test_dedupe_groups_alpha_equivalent_samples():
    samples = [DecodeResult(ast=_id(), residual_norm=0.0),
               DecodeResult(ast=_id_renamed(), residual_norm=0.0),
               DecodeResult(ast=_id(), residual_norm=0.0)]
    groups = dedupe_by_alpha_eq(samples)
    assert len(groups) == 1
    assert groups[0][1] == 3            # multiplicity


def test_dedupe_drops_high_residual_samples():
    samples = [DecodeResult(ast=_id(), residual_norm=0.5)]
    assert dedupe_by_alpha_eq(samples) == []


def test_classify_failure_mode_none_on_low_energy():
    c = Completion(ast=_id(), energy=1e-5, energy_breakdown={},
                   diagnostics={}, multiplicity=1)
    assert classify_failure_mode([c], tolerance_correct=1e-2) is None


def test_classify_failure_mode_no_completion():
    assert classify_failure_mode([], tolerance_correct=1e-2) \
        == "no_valid_completion"


def test_classify_failure_mode_ambiguous():
    a = Completion(ast=_id(), energy=1e-5, energy_breakdown={},
                   diagnostics={}, multiplicity=1)
    b = Completion(ast=_id_renamed(), energy=1e-5 + 1e-7,
                   energy_breakdown={}, diagnostics={}, multiplicity=1)
    assert classify_failure_mode([a, b], tolerance_correct=1e-2,
                                 tolerance_ambiguous=1e-3) == "ambiguous_top1"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_ranking.py -v`
Expected: ImportError.

- [ ] **Step 3: Implement `ranking.py`**

Read M1's `mera_decoder.py` for `ast_alpha_eq` and `DecodeResult`. Create `src/qft_pcn/logic/mera_synthesis/ranking.py`:

```python
"""Synthesis ranking: dedupe, rerank, classify failure mode (spec §6.3)."""
from __future__ import annotations

from ..mera_decoder import ast_alpha_eq


def dedupe_by_alpha_eq(samples, residual_max: float = 1e-3):
    """Group decoded samples by alpha-equivalence (spec §6.3).

    Drops samples whose decode residual exceeds `residual_max`. Returns a
    list of (representative_ast, multiplicity) for each unique group.
    """
    groups = []   # list of [ast, count]
    for s in samples:
        if s.residual_norm > residual_max:
            continue
        for g in groups:
            if ast_alpha_eq(g[0], s.ast):
                g[1] += 1
                break
        else:
            groups.append([s.ast, 1])
    return [(g[0], g[1]) for g in groups]


def classify_failure_mode(completions, tolerance_correct: float,
                          tolerance_ambiguous: float = 1e-3) -> str | None:
    """Classify a sorted-ascending completion list (spec §6.4)."""
    if not completions:
        return "no_valid_completion"
    top = completions[0]
    if len(completions) >= 2:
        gap = completions[1].energy - top.energy
        if gap < tolerance_ambiguous and top.energy < tolerance_correct:
            return "ambiguous_top1"
    if top.energy < tolerance_correct:
        return None
    return "imag_time_did_not_converge"
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_ranking.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/qft_pcn/logic/mera_synthesis/ranking.py \
        src/qft_pcn/tests/test_mera_synthesis_ranking.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_synthesis/ranking): dedupe + failure-mode classification

dedupe_by_alpha_eq groups decoded samples by alpha-equivalence and drops
high-residual samples; classify_failure_mode labels a sorted completion
list (None | no_valid_completion | ambiguous_top1 |
imag_time_did_not_converge).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: The `synthesize()` runner + public re-exports

**Files:**
- Create: `src/qft_pcn/logic/mera_synthesis/runner.py`
- Modify: `src/qft_pcn/logic/mera_synthesis/__init__.py`
- Modify: `src/qft_pcn/logic/__init__.py` (additive re-exports)
- Test: `src/qft_pcn/tests/test_mera_synthesis_runner.py`

- [ ] **Step 1: Write the failing test (P1–P8)**

Create `src/qft_pcn/tests/test_mera_synthesis_runner.py`. Build the eight problems exactly as spec §7 specifies, as ASTs (the parser may not accept structural-hole syntax — build with constructors). Example skeleton (fill in all eight):

```python
"""Synthesis runner acceptance: P1-P8 (spec §7, §9.6)."""
from __future__ import annotations
import numpy as np
import pytest
from src.qft_pcn.logic.ast import (
    Lam, TInt, TBool, TArrow, Var, App, Bin, If, IntLit, BoolLit,
    HoleVar, TypeHole, Eq,
)
from src.qft_pcn.logic.mera_synthesis import synthesize, SynthesisProblem
from src.qft_pcn.logic.mera_synthesis.problem import IOExample
from src.qft_pcn.logic.mera_decoder import ast_alpha_eq


def _P1():
    return SynthesisProblem(
        name="P1",
        sketch=Lam(param="x", param_ty=TInt(),
                   body=HoleVar(candidates=("x",))))

def _P3():
    hole = HoleVar(candidates=(
        Var(name="x"),
        App(fn=Var(name="f"), arg=Var(name="x")),
        App(fn=Var(name="f"),
            arg=App(fn=Var(name="f"), arg=Var(name="x"))),
    ))
    sketch = Lam(param="f", param_ty=TArrow(src=TInt(), dst=TInt()),
                 body=Lam(param="x", param_ty=TInt(), body=hole))
    succ = Lam(param="n", param_ty=TInt(),
               body=Bin(op="+", lhs=Var(name="n"), rhs=IntLit(val=1)))
    return SynthesisProblem(
        name="P3", sketch=sketch,
        examples=(IOExample(inputs=(succ, IntLit(val=2)),
                            output=IntLit(val=3)),))

# ... _P2, _P4, _P5, _P6, _P7, _P8 per spec §7 ...

_EXPECTED = {
    "P1": Lam(param="x", param_ty=TInt(), body=Var(name="x")),
    "P3": App(fn=Var(name="f"), arg=Var(name="x")),     # the body
    # ... all eight ...
}


@pytest.mark.parametrize("builder", [_P1, _P3])   # extend to all 8
def test_problem_runs_without_raising(builder):
    res = synthesize(builder(), rng=np.random.default_rng(0))
    assert res is not None


@pytest.mark.parametrize("name,builder", [("P1", _P1), ("P3", _P3)])  # all 7
def test_top1_is_correct(name, builder):
    """Spec §9.6.2: P1,P2,P3,P4,P5,P6,P7 -> correct top-1, verified by
    decoding the AST (principle 7), not energy alone."""
    res = synthesize(builder(), rng=np.random.default_rng(0))
    assert res.failure_mode is None, f"{name}: {res.failure_mode}"
    assert res.completions, f"{name}: no completions"
    top = res.completions[0].ast
    # compare the synthesized body against the expected program.
    assert ast_alpha_eq(top, _EXPECTED[name]) \
        or ast_alpha_eq(_body(top), _EXPECTED[name]), \
        f"{name}: top-1 {top!r} != expected"


def test_P8_is_refused():
    """Spec §9.6.4: the deliberately unsolvable problem is refused."""
    res = synthesize(_P8(), rng=np.random.default_rng(0))
    assert res.failure_mode is not None


def test_energy_gap_to_second_completion():
    """Spec §9.6.3: top-2 strictly higher, gap >= 0.5 * w_T = 2.0."""
    res = synthesize(_P3(), rng=np.random.default_rng(0))
    if len(res.completions) >= 2:
        gap = res.completions[1].energy - res.completions[0].energy
        assert gap >= 2.0 - 1e-6, f"small gap {gap}"
```

`_body(top)` is a small helper that descends through leading `Lam`s to the synthesized body — the structural-hole completions fill a body, so compare bodies. Write all eight problem builders and the full `_EXPECTED` map per spec §7.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_runner.py -v --timeout=300`
Expected: ImportError on `synthesize`.

- [ ] **Step 3: Implement `runner.py`**

Create `src/qft_pcn/logic/mera_synthesis/runner.py`:

```python
"""The MERA-native synthesis runner (spec §6.2)."""
from __future__ import annotations

import numpy as np

from ..mera_encoder import encode_mera
from ..mera_decoder import sample_mera
from ..mera_debugger import diagnose
from .problem import (
    SynthesisProblem, SynthesisResult, Completion, HamiltonianWeights,
)
from .errors import SynthesisProblemError, SynthesisRuntimeError
from .encode_ext import _witness_augmented_ast
from .hamiltonian import compile_mera_synthesis_hamiltonian
from .ranking import dedupe_by_alpha_eq, classify_failure_mode
from .._validate_synth import validate_problem   # see note


def synthesize(problem: SynthesisProblem,
               rng: np.random.Generator | None = None,
               verbose: bool = False) -> SynthesisResult:
    """Run the MERA synthesis pipeline (spec §6.2).

    Principle 6: holes are quantum superpositions -- this function NEVER
    loops over candidate sub-trees. The candidates exist only as branch
    directions inside the one MERA state encode_mera builds.
    Principle 7: the top-1 is ranked by <H> and the caller verifies it by
    decoding -- the runner does not declare correctness on energy alone.
    """
    rng = rng or np.random.default_rng()
    validate_problem(problem)               # raises SynthesisProblemError
    weights = HamiltonianWeights()

    # 1. Encode the witness-augmented sketch. Structural holes become
    #    rank-k superposition states (Part 1, Task 4); TypeHoles become
    #    type-register superpositions. ONE encoding, ONE state.
    aug = _witness_augmented_ast(problem.sketch, problem.examples)
    try:
        state, meta = encode_mera(aug, n_nodes_max=problem.n_nodes_max,
                                  chi_layer=problem.chi_layer)
    except Exception as exc:                # encoder exceptions are fatal
        raise SynthesisRuntimeError(f"encode failed: {exc}") from exc

    # 2. Compose H_typing + H_eval + H_examples + H_target_type + H_size.
    H = compile_mera_synthesis_hamiltonian(meta, problem, weights)

    # 3. Three-phase factored imaginary-time evolution (M2 owns the loop).
    state = _anneal(state, meta, H, problem)

    final_energy = float(H.total_energy(state))
    if not np.isfinite(final_energy):
        raise SynthesisRuntimeError("non-finite final energy")

    # 4. Sample completions; dedupe by alpha-equivalence.
    samples = sample_mera(state, meta, n_samples=problem.n_samples, rng=rng)
    groups = dedupe_by_alpha_eq(samples)
    if not groups:
        return SynthesisResult(
            problem=problem, completions=[], n_unique=0,
            n_samples_drawn=len(samples), n_samples_decoded_ok=0,
            final_state_energy=final_energy,
            chi_observed_max=getattr(state, "chi_observed_max",
                                     problem.chi_layer),
            failure_mode="no_valid_completion")

    # 5. Rank: re-encode each unique completion, compute <H_total>.
    completions = []
    for ast, mult in groups:
        c_state, c_meta = encode_mera(
            ast, n_nodes_max=problem.n_nodes_max,
            chi_layer=problem.chi_layer)
        c_H = compile_mera_synthesis_hamiltonian(c_meta, problem, weights)
        energy = float(c_H.total_energy(c_state))
        breakdown = c_H.residuals(c_state)          # per-block dict (M2)
        report = diagnose(c_state, c_meta, list(c_H.terms))
        completions.append(Completion(
            ast=ast, energy=energy, energy_breakdown=breakdown,
            diagnostics=report.to_dict(), multiplicity=mult))

    completions.sort(key=lambda c: (c.energy, -c.multiplicity))
    tol = 1e-3 * (weights.w_T + weights.w_E + weights.w_X + weights.w_Y)
    failure_mode = classify_failure_mode(completions, tolerance_correct=tol)

    return SynthesisResult(
        problem=problem, completions=completions, n_unique=len(completions),
        n_samples_drawn=len(samples),
        n_samples_decoded_ok=sum(c.multiplicity for c in completions),
        final_state_energy=final_energy,
        chi_observed_max=getattr(state, "chi_observed_max",
                                 problem.chi_layer),
        failure_mode=failure_mode)


def _anneal(state, meta, H, problem):
    """Three-phase factored imaginary-time evolution (spec §6.5).

    Delegates each phase to M2's factored imaginary-time evolution
    primitive. Read M2's API for the exact entry point (mera evolve);
    phase 1 evolves under H_typing + H_eval only, phases 2-3 under full H.
    Every step normalizes; bond dim capped at problem.chi_layer.
    """
    from ..mera_evolution import mera_imag_time_evolve   # M2
    main = problem.anneal_steps
    warm = max(25, main // 4)
    fine = max(25, main // 4)
    H_partial = H.typing_eval_only()      # M2 sub-Hamiltonian accessor
    state = mera_imag_time_evolve(state, H_partial, steps=warm, dt=0.1,
                                  chi_max=problem.chi_layer)
    state = mera_imag_time_evolve(state, H, steps=main,
                                  dt=problem.anneal_dt,
                                  chi_max=problem.chi_layer)
    state = mera_imag_time_evolve(state, H, steps=fine, dt=0.01,
                                  chi_max=problem.chi_layer)
    return state
```

The M2 entry points (`mera_imag_time_evolve`, `H.typing_eval_only()`, `H.residuals`, `H.terms`, `H.total_energy`) are placeholders — **read M2's actual API and use the real names**. M2's contract (per the spec) is that the composed Hamiltonian mirrors B/C: `.terms`, `.term_energy`, `.residuals`, `.total_energy`. If `typing_eval_only` has no M2 equivalent, build the warmup sub-Hamiltonian by composing only `h_typing` and `h_eval` in `compile_mera_synthesis_hamiltonian` and returning it as a second value, or expose a `weights`-zeroed variant. Pin the choice; escalate if M2 genuinely cannot supply a warmup Hamiltonian (principle 9).

Also create `src/qft_pcn/logic/_validate_synth.py` with `validate_problem(problem)` enforcing the §6.1 invariants (at least one hole; structural candidates well-scoped; type-hole candidates flat-tag; non-hole Vars scoped) — raise `SynthesisProblemError` on violation. Keep it small and pure.

- [ ] **Step 4: Fill `mera_synthesis/__init__.py` re-exports**

```python
"""MERA-native STLC synthesis (migration sub-project M3)."""
from .problem import (
    SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights,
)
from .runner import synthesize
from .errors import (
    SynthesisError, SynthesisProblemError, SynthesisRuntimeError,
)

__all__ = [
    "SynthesisProblem", "IOExample", "Completion", "SynthesisResult",
    "HamiltonianWeights", "synthesize", "SynthesisError",
    "SynthesisProblemError", "SynthesisRuntimeError",
]
```

Add to `src/qft_pcn/logic/__init__.py` (additive — M1's exports stay):

```python
from .mera_debugger import (
    NamedMeraTerm, DiagnosticReport, RuleViolation, TermEvaluationError,
    diagnose, format_report, register_explanation, get_explanation,
    clear_explanations,
)
from .mera_synthesis import (
    synthesize, SynthesisProblem, IOExample, Completion, SynthesisResult,
    HamiltonianWeights, SynthesisProblemError, SynthesisRuntimeError,
)
```

- [ ] **Step 5: Run the runner acceptance suite**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_runner.py -v --timeout=600`

This is the milestone gate. **7 of 8 of P1–P8 must produce a correct top-1, with P3/P4/P6/P7 (the multi-node completions) among them.** If a multi-node problem fails:
- Use `superpowers:systematic-debugging`. Root-cause: is the structural superposition encoded correctly (Part 1 Task 4 marker test), or is the Hamiltonian not selecting the right branch?
- Inspect the energy breakdown per completion — `diagnose()` output names the stuck term.
- If P3/P4/P6/P7 fail because evolution does not collapse onto the right branch, the fault is in `H_examples`/`H_eval` weight or the anneal schedule (spec §4.1, §6.5) — tune within the spec's documented latitude (double the main phase before escalating, spec §6.5 / §12). Do NOT add a classical enumerator (principle 6).
- A genuine encoder bug is fixed in Part 1's Task 4 code, with a regression test.

Expected: 7/8 correct top-1; P8 refused; energy-gap test passes.

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/mera_synthesis/runner.py \
        src/qft_pcn/logic/mera_synthesis/__init__.py \
        src/qft_pcn/logic/_validate_synth.py \
        src/qft_pcn/logic/__init__.py \
        src/qft_pcn/tests/test_mera_synthesis_runner.py
git commit -m "$(cat <<'EOF'
feat(logic/mera_synthesis/runner): synthesize() on the MERA substrate

The MERA synthesis pipeline: encode the witness-augmented sketch (holes
become rank-k superposition states -- ONE encoding, never a candidate
loop), compose H_typing+H_eval+H_examples+H_target_type+H_size,
three-phase factored imaginary-time evolution, sample_mera, dedupe by
alpha-equivalence, rank by re-encoded <H>. 7/8 of P1-P8 produce a correct
top-1; P3/P4/P6/P7 multi-node completions are now reachable.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: The publishable demo + final acceptance

**Files:**
- Create: `src/qft_pcn/logic/demo_mera_stlc_synthesis.py`
- Test: `src/qft_pcn/tests/test_mera_synthesis_demo.py`

- [ ] **Step 1: Write the demo smoke test**

Create `src/qft_pcn/tests/test_mera_synthesis_demo.py`:

```python
"""Smoke test for the MERA STLC synthesis demo (spec §9.7)."""
from __future__ import annotations
import subprocess
import sys


def test_demo_runs_end_to_end():
    proc = subprocess.run(
        [sys.executable, "-m",
         "src.qft_pcn.logic.demo_mera_stlc_synthesis"],
        capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    for name in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"):
        assert name in out, f"{name} missing from demo output"
    assert "expected" in out.lower()
    assert "energy" in out.lower()
    assert "failure_mode" in out.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_demo.py -v`
Expected: failure — demo module does not exist.

- [ ] **Step 3: Implement `demo_mera_stlc_synthesis.py`**

Create `src/qft_pcn/logic/demo_mera_stlc_synthesis.py`. It builds the eight P1–P8 `SynthesisProblem`s (reuse the builders from `test_mera_synthesis_runner.py` — import them or factor a shared `_problems.py`), runs `synthesize()` on each, and prints a structured block per problem: name, expected top-1, actual top-1, energy, `energy_breakdown`, multiplicity, `n_unique`, `n_samples`, `failure_mode`. For P8 it prints the residual `H_typing` breakdown from `diagnose()`. Honest reporting (principle 10): on a wrong top-1 print `top1 INCORRECT` and the energy distribution; never silently hide.

```python
"""Publishable demo: MERA-native STLC synthesis on 8 problems P1-P8.

Realizes QFT_PCN_ARCHITECTURE.md §10.7 on the MERA substrate. P3/P4/P6/P7
are the multi-node-completion problems that the structural superposition
(spec §5) makes reachable -- the half the MPS-era synthesis could not
solve. Run: python -m src.qft_pcn.logic.demo_mera_stlc_synthesis
"""
from __future__ import annotations

import numpy as np

from .mera_synthesis import synthesize
from .mera_decoder import ast_alpha_eq
# import the P1-P8 builders + _EXPECTED (shared with the runner test)


def _print_problem(name, problem, expected):
    res = synthesize(problem, rng=np.random.default_rng(0))
    print(f"=== {name} ===")
    print(f"  expected top-1 : {expected!r}")
    if res.completions:
        top = res.completions[0]
        ok = ast_alpha_eq(top.ast, expected)
        print(f"  actual top-1   : {top.ast!r}  "
              f"[{'CORRECT' if ok else 'INCORRECT'}]")
        print(f"  energy         : {top.energy:.6g}")
        print(f"  energy_breakdown: {top.energy_breakdown}")
        print(f"  multiplicity   : {top.multiplicity}")
    else:
        print("  actual top-1   : <none>")
    print(f"  n_unique       : {res.n_unique}")
    print(f"  n_samples      : {res.n_samples_drawn}")
    print(f"  failure_mode   : {res.failure_mode}")
    if name == "P8" and res.completions:
        print("  P8 residual H_typing breakdown:")
        print(f"    {res.completions[0].diagnostics}")
    print()


def main() -> None:
    # for name, (problem, expected) in PROBLEMS.items():
    #     _print_problem(name, problem, expected)
    ...


if __name__ == "__main__":
    main()
```

Factor the eight problem builders and `_EXPECTED` into a shared `src/qft_pcn/logic/mera_synthesis/_problems.py` so the demo and `test_mera_synthesis_runner.py` import the same definitions — do not duplicate them.

- [ ] **Step 4: Run the demo smoke test**

Run: `.venv/bin/python -m pytest src/qft_pcn/tests/test_mera_synthesis_demo.py -v --timeout=900`
Expected: 1 passed.

- [ ] **Step 5: Full M3 acceptance + regression sweep**

Run the whole M3 suite and confirm no regression:

```bash
.venv/bin/python -m pytest \
  src/qft_pcn/tests/test_mera_structural_ast.py \
  src/qft_pcn/tests/test_mera_debugger.py \
  src/qft_pcn/tests/test_mera_structural_layout.py \
  src/qft_pcn/tests/test_mera_structural_holes.py \
  src/qft_pcn/tests/test_mera_structural_decode.py \
  src/qft_pcn/tests/test_mera_synthesis_problem.py \
  src/qft_pcn/tests/test_mera_synthesis_hamiltonian.py \
  src/qft_pcn/tests/test_mera_synthesis_ranking.py \
  src/qft_pcn/tests/test_mera_synthesis_runner.py \
  src/qft_pcn/tests/test_mera_synthesis_demo.py \
  -v --timeout=900 2>&1 | tail -25
```

Then the regression sweep (spec §9.8):

```bash
.venv/bin/python -m pytest src/qft_pcn/tests/ --timeout=900 -q 2>&1 | tail -15
```

Expected: all M3 tests green; all M1, M2, F, and MPS-stack tests still green. If anything regresses, root-cause with `superpowers:systematic-debugging` before claiming completion (`superpowers:verification-before-completion` — evidence before assertions).

- [ ] **Step 6: Commit**

```bash
git add src/qft_pcn/logic/demo_mera_stlc_synthesis.py \
        src/qft_pcn/logic/mera_synthesis/_problems.py \
        src/qft_pcn/tests/test_mera_synthesis_demo.py
git commit -m "$(cat <<'EOF'
feat(logic/demo_mera_stlc_synthesis): the publishable P1-P8 demo

Runs the 8 STLC synthesis problems on the MERA substrate and prints a
structured block per problem (expected/actual top-1, energy breakdown,
multiplicity, failure_mode). P3/P4/P6/P7 -- the multi-node completions
the MPS-era synthesis could not reach -- now solve via structural
superposition. P8 is correctly refused. This is the §10.7 milestone.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Acceptance checklist (spec §11)

M3 is complete when, with fresh pytest output as evidence (`superpowers:verification-before-completion`):

- [ ] Debugger suite (`test_mera_debugger.py`) passes; end-to-end test passes or skips cleanly.
- [ ] Structural-hole encoding (`test_mera_structural_holes.py`, `test_mera_structural_decode.py`) passes.
- [ ] The structural-superposition marker (`test_structural_hole_is_not_a_product_state`, `test_structural_marker_exceeds_value_only_superposition`) passes — proving principles 2 and 6: candidate sub-trees are a genuine MERA superposition with measurable entanglement entropy.
- [ ] Synthesis Hamiltonian blocks (`test_mera_synthesis_hamiltonian.py`) pass with the factored-window contract — no dense `16**k`.
- [ ] Runner suite (`test_mera_synthesis_runner.py`): 7 of 8 of P1–P8 produce a correct top-1; P8 refused; P3/P4/P6/P7 among the passing.
- [ ] Demo smoke test (`test_mera_synthesis_demo.py`) passes.
- [ ] No regression in M1, M2, F, or the MPS stack.
- [ ] `from qft_pcn.logic import synthesize, diagnose` works.

---

**End of Part 2.** With both parts implemented, M3 retargets sub-projects D and E onto MERA and delivers the structural superposition over candidate AST sub-trees — the extension that lifts the §10.7 publishable milestone past the MPS-era 4/8 wall. Multi-node completions (`f x`, `if x<5 then x else x+1`, `f (g x)`, `x < y`) become reachable as genuine quantum superpositions, never classical enumerations.
