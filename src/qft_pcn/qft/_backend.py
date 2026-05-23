"""GPU/CPU backend toggle for MERA hot-path contractions.

Selects between NumPy (CPU) and CuPy + cuTensorNet (GPU) for the
multi-tensor contractions on MERA's causal-cone ascent. Selection is
controlled by the ``QPCN_BACKEND`` env var:

    QPCN_BACKEND=cpu    -> force NumPy (default fallback if GPU missing)
    QPCN_BACKEND=gpu    -> force CuPy; raise if unavailable
    QPCN_BACKEND=auto   -> use GPU when available, else CPU (default)

Hot paths routed through this module (file:function):
    - src/qft_pcn/logic/mera_decoder.py:_ascend_one_layer_batched
      (the 4-/5-tensor batched ascent through L-1 layers)
    - src/qft_pcn/qft/mera.py:_cross_layer1 / _cross_ascend / inner
      (the 4-tensor double-network cross ascent used by inner)
    - src/qft_pcn/qft/mera.py:_ascend_one_layer / local_expectation
      (single-leaf ascent used by Hamiltonian term_energy)

For 2-tensor einsums the cuTensorNet path-finder overhead exceeds the
contraction itself; we fall back to ``xp.einsum`` (where ``xp`` is
``cupy`` on GPU, ``numpy`` on CPU). cuTensorNet is engaged only for the
3+ tensor contractions, with an optimized path cached per
(equation, shape signature) so repeated calls reuse the contraction
tree.

Hard constraint: the CPU path is fully functional. Setting
``QPCN_BACKEND=cpu`` makes this module a no-op: ``contract`` becomes
``np.einsum`` and ``to_device``/``to_host`` are identity functions.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np

# CuPy + cuTensorNet are imported lazily: importing cupy unconditionally
# pre-allocates a sizeable host-side pinned-memory pool which trips WSL
# allocations in CPU-mode tests that build dense (4096, 4096) matrices.
# We only pay the import cost when GPU is actually selected.
_cp: Any = None
_ctn: Any = None
_GPU_IMPORT_OK: bool | None = None
_GPU_IMPORT_ERROR: Exception | None = None


def _try_import_gpu() -> bool:
    global _cp, _ctn, _GPU_IMPORT_OK, _GPU_IMPORT_ERROR
    if _GPU_IMPORT_OK is not None:
        return _GPU_IMPORT_OK
    try:
        import cupy as cp  # type: ignore
        from cuquantum import cutensornet as ctn  # type: ignore
        _cp = cp
        _ctn = ctn
        _GPU_IMPORT_OK = True
    except Exception as e:
        _GPU_IMPORT_OK = False
        _GPU_IMPORT_ERROR = e
    return _GPU_IMPORT_OK


def _resolve_backend() -> str:
    """Return 'cpu' or 'gpu' after honoring QPCN_BACKEND."""
    requested = os.environ.get("QPCN_BACKEND", "auto").strip().lower()
    if requested not in ("cpu", "gpu", "auto"):
        raise ValueError(
            f"QPCN_BACKEND must be one of cpu|gpu|auto, got {requested!r}")
    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        if not _try_import_gpu():
            raise RuntimeError(
                "QPCN_BACKEND=gpu requested but CuPy/cuTensorNet import "
                f"failed: {_GPU_IMPORT_ERROR!r}")
        return "gpu"
    # auto
    if _try_import_gpu():
        try:
            # Probe: must have a device available.
            _cp.cuda.runtime.getDeviceCount()
            return "gpu"
        except Exception:
            return "cpu"
    return "cpu"


_BACKEND: str = _resolve_backend()

# xp = numpy module on CPU, cupy module on GPU. All allocations done
# via ``xp`` flow through the right backend.
xp: Any = _cp if _BACKEND == "gpu" else np
GPU_ACTIVE: bool = _BACKEND == "gpu"


def backend_name() -> str:
    """Return 'cpu' or 'gpu' as currently active."""
    return _BACKEND


import weakref

# Cache device mirrors of host arrays so a MERA's invariant tensors
# (disentanglers, isometries, top) are uploaded once. Keyed by id(host)
# with weakref-finalizer eviction so we don't pin host memory.
_DEVICE_MIRROR: dict[int, Any] = {}
_DEVICE_MIRROR_FINALIZERS: dict[int, Any] = {}


def _evict_mirror(key: int) -> None:
    _DEVICE_MIRROR.pop(key, None)
    _DEVICE_MIRROR_FINALIZERS.pop(key, None)


def to_device(arr: np.ndarray) -> Any:
    """Move a host ndarray to device. Identity on CPU backend.

    Caches the device mirror so repeated calls with the same host array
    (e.g. invariant MERA tensors during sampling/anneal) reuse the
    existing upload instead of re-allocating per contraction.
    """
    if not GPU_ACTIVE:
        return arr
    if isinstance(arr, _cp.ndarray):  # type: ignore[union-attr]
        return arr
    key = id(arr)
    mirror = _DEVICE_MIRROR.get(key)
    if mirror is not None:
        return mirror
    mirror = _cp.asarray(arr)
    _DEVICE_MIRROR[key] = mirror
    try:
        # Evict the GPU mirror once the host array is garbage collected.
        _DEVICE_MIRROR_FINALIZERS[key] = weakref.finalize(
            arr, _evict_mirror, key)
    except TypeError:
        # Some host objects (very rarely) can't be finalized; skip cache.
        _DEVICE_MIRROR.pop(key, None)
    return mirror


def to_host(arr: Any) -> np.ndarray:
    """Move a device array back to host. Identity on CPU backend."""
    if not GPU_ACTIVE:
        return arr
    if isinstance(arr, np.ndarray):
        return arr
    return _cp.asnumpy(arr)


# Cache for cuTensorNet optimized contraction paths. Keyed by
# (equation, tuple-of-shape-tuples, tuple-of-dtypes); the value is a
# cuquantum.cutensornet.Network ready to be replayed with new operands.
_NETWORK_CACHE: dict[tuple, Any] = {}
_NETWORK_CACHE_MAX = 256


def _network_key(equation: str, operands: tuple) -> tuple:
    return (
        equation,
        tuple(o.shape for o in operands),
        tuple(str(o.dtype) for o in operands),
    )


def _contract_gpu(equation: str, *operands: Any) -> Any:
    """GPU contraction.

    For 2-tensor ops: ``cupy.einsum`` (path-finder overhead would dominate).
    For 3+ tensors: ``cuquantum.cutensornet.Network`` with cached path.
    """
    # Ensure all operands live on device.
    dev_ops = tuple(to_device(o) for o in operands)
    if len(dev_ops) <= 2:
        return _cp.einsum(equation, *dev_ops, optimize="greedy")
    key = _network_key(equation, dev_ops)
    net = _NETWORK_CACHE.get(key)
    if net is None:
        net = _ctn.Network(equation, *dev_ops)
        net.contract_path()
        if len(_NETWORK_CACHE) < _NETWORK_CACHE_MAX:
            _NETWORK_CACHE[key] = net
        else:
            # Don't grow unboundedly; just run this one without caching.
            result = net.contract()
            net.free()
            return result
    else:
        net.reset_operands(*dev_ops)
    return net.contract()


def contract(equation: str, *operands: Any) -> Any:
    """Multi-tensor contraction.

    On CPU: ``numpy.einsum(equation, *operands, optimize='greedy')``.
    On GPU: cupy.einsum for 2-tensor; cutensornet.Network (with cached
    optimized path) for 3+ tensors.

    Always returns an array in the active backend's namespace (numpy on
    CPU, cupy on GPU). Callers that need a host result must explicitly
    call ``to_host``.
    """
    if not GPU_ACTIVE:
        return np.einsum(equation, *operands, optimize="greedy")
    return _contract_gpu(equation, *operands)


__all__ = [
    "backend_name",
    "contract",
    "GPU_ACTIVE",
    "to_device",
    "to_host",
    "xp",
]
