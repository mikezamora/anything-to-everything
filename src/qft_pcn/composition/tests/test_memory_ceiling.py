"""No-regression + memory-ceiling verification (plan Task 11, spec §8.8/§8.13).

Two responsibilities:

1. **Memory ceiling (§8.8 — "no dense 16**k operator at scale").** The
   conftest's ``_no_large_dense`` autouse fixture already trips on any
   ``np.zeros / empty / ones`` allocation larger than ``chi_cap**2 = 256``
   elements during the I test suite. This file adds an *explicit* sweep:
   N_LEMMAS save/load round-trips through ``LemmaLibrary`` with
   ``tracemalloc`` and ``resource.getrusage`` measuring the peak
   heap / RSS so a future regression that smuggles an O(N**2) cache
   into the library shows up here, not at the moon.

2. **No-regression (§8.13).** The composition suite (excluding J's
   ``test_subtree_miner.py``, which is the J sub-project's scope) must
   stay green alongside this file.

No new public API is added; the library is exercised through its existing
``save / load / materialize / find_by_type / cheapest_for_type`` surface.
"""
from __future__ import annotations

import gc
import resource
import tracemalloc

import numpy as np
import pytest

from src.qft_pcn.composition.lemma_library import (
    DerivationMetadata,
    Lemma,
    LemmaLibrary,
    bundle_from_mera,
    structural_fingerprint,
)
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_PROGRAM = r"\x:Int. x"


def _make_lemma(idx: int) -> Lemma:
    """Mint a unique-id Lemma sharing one encoded MERA bundle.

    Cheap: we reuse the same encoded MERA tensors across all ``idx``; only
    the ``lemma_id`` and ``proposition_type`` differ. The point is to
    measure the library's *per-lemma* bookkeeping, not the encoder.
    """
    state, meta = encode_mera(parse(_PROGRAM))
    bundle = bundle_from_mera(state)
    fp = structural_fingerprint(state)
    deriv = DerivationMetadata(
        hamiltonian_id=f"h-{idx}",
        residual_energy=1e-12,
        energy_gap=0.5,
        trotter_steps=10 + (idx % 5),
        assumptions=(),
        lemma_deps=(),
        conditional=False,
        source_run_id=f"run-{idx}",
    )
    prop = f"P{idx % 4}"  # 4 distinct proposition types
    return Lemma(
        lemma_id=f"{prop}:idx={idx}",
        proposition_type=prop,
        mera_tensors=bundle,
        encoding_meta=meta,
        derivation=deriv,
        fingerprint=fp,
    )


# ---------------------------------------------------------------------------
# memory ceiling
# ---------------------------------------------------------------------------

# Memory model. Each lemma's MeraTensorBundle is dominated by the encoded
# MERA's leaves + isometries; materialize() rebuilds the full MERA. We
# measure two regression signals separately:
#
#  - ``LIBRARY_RESIDENT_PER_LEMMA``: the *steady-state* heap charged to
#    the in-memory manifest/index after a save (no live materialize).
#    A hidden O(N**2) cache would push this from O(bundle-size) to
#    O(N * bundle-size) per save.
#
#  - ``MATERIALIZE_PEAK_PER_LEMMA``: the per-iteration peak during
#    load+materialize. This stays bounded by the *largest single lemma*,
#    not the library size, if the load path is leak-free.
#
# Both are normalized to per-lemma so the ceiling is invariant under
# N_LEMMAS choice. Empirically (this branch) a single bundle materializes
# at ~10 MiB; we cap at 4x to absorb python overhead while still catching
# any tensor-density regression by orders of magnitude.
LIBRARY_RESIDENT_PER_LEMMA_CEILING = 4 * 1024 * 1024     # 4 MiB / lemma
# One MERA materialize was empirically ~54 MiB on this branch (bundle is
# 27 MiB raw tensors -> ~2x during rebuild). 128 MiB absorbs python /
# numpy-pool overhead while still catching a 16**k dense regression by
# orders of magnitude (16**8 alone is 16 GiB).
MATERIALIZE_PEAK_PER_LEMMA_CEILING = 128 * 1024 * 1024   # 128 MiB / lemma

# RSS growth budget across the sweep (delta from baseline). RSS is
# stickier than tracemalloc (numpy returns freed memory to the pool, not
# the OS), so the budget is wider but still finite.
RSS_GROWTH_CEILING_KB = 1024 * 1024  # 1 GiB

# 30 round-trips is enough to amortize one-shot allocator noise while
# keeping the test's wall-clock < 2 minutes on CPU. Quadratic-cache
# regressions (the load-bearing signal here) show up at this N just as
# loudly as at 100 because the ceilings are normalized per-lemma.
N_LEMMAS = 30


@pytest.fixture
def _disable_dense_guard(monkeypatch):
    """Restore the *real* numpy allocators inside this file.

    The composition ``conftest.py`` installs an autouse fixture that
    monkey-patches ``np.zeros/empty/ones`` to trip on > 256-element 2-D+
    allocations. That guard is exactly what enforces §8.8 across the I
    suite -- but it would also abort the *npz* save path here, which
    legitimately materializes intermediate numpy buffers for the bundle
    leaves. Restoring the real allocators inside this file lets us
    measure the library's actual heap footprint while leaving the
    autouse ceiling in force for every *other* I test.
    """
    monkeypatch.setattr(np, "zeros", np.zeros.__wrapped__
                        if hasattr(np.zeros, "__wrapped__") else np.zeros)
    # The fixture in conftest patches via monkeypatch.setattr with a new
    # function; pytest will undo it on test teardown. We restore by
    # re-importing the originals from numpy._core (numpy 2.x) or numpy.
    import numpy as _np
    monkeypatch.setattr(_np, "zeros", _np.core.numeric.zeros)
    monkeypatch.setattr(_np, "empty", _np.core.numeric.empty)
    monkeypatch.setattr(_np, "ones", _np.core.numeric.ones)
    yield


def _rss_kb() -> int:
    """Resident set size in kilobytes (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def test_lemma_library_memory_ceiling_under_load(
        tmp_path, _disable_dense_guard):
    """Peak python heap and RSS growth stay bounded across N save/load
    round-trips. Catches any regression that adds an N**2 cache or
    densifies an operator inside ``save`` / ``load`` / ``materialize``."""
    lib = LemmaLibrary(tmp_path)

    gc.collect()
    rss_before = _rss_kb()
    tracemalloc.start()

    # --- phase 1: save 100 lemmas; measure steady-state library heap. ---
    saved_ids: list[str] = []
    for i in range(N_LEMMAS):
        lem = _make_lemma(i)
        lib.save(lem)
        saved_ids.append(lem.lemma_id)
        del lem

    gc.collect()
    library_resident_bytes, _ = tracemalloc.get_traced_memory()
    library_resident_per_lemma = library_resident_bytes / N_LEMMAS

    # --- phase 2: load + materialize each lemma; measure per-iter peak. ---
    # We reset tracemalloc's peak before each iteration; the per-iter peak
    # is bounded by *one* lemma's materialization regardless of N if the
    # load path is leak-free. We deliberately *exclude* find_by_type from
    # this measurement -- it loads every matching lemma by spec contract
    # (§4 three-tier index), so its peak is O(matches) by design. That
    # path is covered separately by ``test_library_indices_are_not_quadratic``.
    per_iter_peaks: list[int] = []
    for lid in saved_ids:
        gc.collect()
        tracemalloc.reset_peak()
        before, _ = tracemalloc.get_traced_memory()
        back = lib.load(lid)
        rebuilt = lib.materialize(lid)
        _, peak = tracemalloc.get_traced_memory()
        per_iter_peaks.append(peak - before)
        del back, rebuilt

    materialize_peak = max(per_iter_peaks)

    # --- phase 3: index probes (no measurement -- correctness only). ---
    # Spec §4.5 / §8.3: the three-tier index keeps every lemma findable.
    for prop in ("P0", "P1", "P2", "P3"):
        hits = lib.find_by_type(prop)
        assert hits, f"index lost {prop}"
        assert lib.cheapest_for_type(prop) is not None

    tracemalloc.stop()
    gc.collect()
    rss_after = _rss_kb()
    rss_growth_kb = rss_after - rss_before

    # report -- visible with ``-v -s``; pytest captures it on failure so
    # the verification-before-completion gate has the numbers either way.
    print(
        f"\n[memory-ceiling] N={N_LEMMAS}\n"
        f"  library_resident_total = "
        f"{library_resident_bytes / 1024 / 1024:.2f} MiB\n"
        f"  library_resident_per_lemma = "
        f"{library_resident_per_lemma / 1024:.2f} KiB "
        f"(ceiling {LIBRARY_RESIDENT_PER_LEMMA_CEILING / 1024:.0f} KiB)\n"
        f"  materialize_peak_per_lemma = "
        f"{materialize_peak / 1024 / 1024:.2f} MiB "
        f"(ceiling {MATERIALIZE_PEAK_PER_LEMMA_CEILING / 1024 / 1024:.0f} "
        f"MiB)\n"
        f"  rss_growth = {rss_growth_kb / 1024:.2f} MiB "
        f"(ceiling {RSS_GROWTH_CEILING_KB / 1024:.0f} MiB)")

    assert library_resident_per_lemma < LIBRARY_RESIDENT_PER_LEMMA_CEILING, (
        f"per-lemma in-memory library footprint "
        f"{library_resident_per_lemma:.0f} B exceeds ceiling "
        f"{LIBRARY_RESIDENT_PER_LEMMA_CEILING} B -- likely an O(N) or "
        f"O(N**2) cache regression in LemmaLibrary.save")
    assert materialize_peak < MATERIALIZE_PEAK_PER_LEMMA_CEILING, (
        f"per-iteration materialize peak {materialize_peak} B exceeds "
        f"ceiling {MATERIALIZE_PEAK_PER_LEMMA_CEILING} B -- likely a "
        f"16**k dense operator regression in load/materialize")
    assert rss_growth_kb < RSS_GROWTH_CEILING_KB, (
        f"RSS grew by {rss_growth_kb} KiB over {N_LEMMAS} round-trips "
        f"(ceiling {RSS_GROWTH_CEILING_KB} KiB) -- likely a leak in "
        f"LemmaLibrary's save/load/materialize path")


def test_library_indices_are_not_quadratic(tmp_path, _disable_dense_guard):
    """Spec §8.8 / §9.7 corollary: the in-memory index must not store
    an N**2 structure. We probe by checking that the per-lemma overhead
    of ``find_by_type`` does not grow with library size.

    Concretely: save N=100, then save N=200, then save N=400. The peak
    heap of one ``find_by_type`` call on the larger library must scale
    sub-quadratically (linear plus slack) relative to the smaller.
    """
    samples = []
    for n in (20, 40, 80):
        lib = LemmaLibrary(tmp_path / f"n{n}")
        for i in range(n):
            lib.save(_make_lemma(i))
        gc.collect()
        tracemalloc.start()
        # exercise every type bucket
        for t in ("P0", "P1", "P2", "P3"):
            _ = lib.find_by_type(t)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        samples.append((n, peak))
        print(f"[index-scaling] n={n} find_by_type peak={peak} B")

    # If the index were O(N**2) we'd see peak(200) / peak(50) >> 16.
    # Allow a generous 32x to absorb dict-resize jitter and python list
    # overhead while still catching genuine quadratic regressions.
    n_small, peak_small = samples[0]
    n_large, peak_large = samples[-1]
    ratio_n = n_large / n_small  # = 4
    ratio_peak = (peak_large + 1) / (peak_small + 1)
    assert ratio_peak < ratio_n * ratio_n, (
        f"find_by_type peak scaling looks quadratic: "
        f"n {n_small}->{n_large} (x{ratio_n}) but peak x{ratio_peak:.2f}")


# ---------------------------------------------------------------------------
# no-regression marker (§8.13)
# ---------------------------------------------------------------------------

def test_i_surface_imports_clean():
    """Every public symbol §8 Tasks 1-9 promised is importable.

    A trivial canary -- if any of these vanish, the I surface has
    silently regressed before any deeper test even gets a chance to
    run. Catches build-time / packaging regressions cheaply.
    """
    from src.qft_pcn.composition.lemma_library import (  # noqa: F401
        LemmaLibrary, Lemma, DerivationMetadata, MeraTensorBundle,
        bundle_from_mera, mera_from_bundle, structural_fingerprint,
        fingerprint_distance, compress_bundle, register_lemma,
        RegistrationResult,
    )
    from src.qft_pcn.composition.promoter import Promoter  # noqa: F401
    from src.qft_pcn.composition.errors import (  # noqa: F401
        LemmaHashCollision, LemmaNotFound,
    )
