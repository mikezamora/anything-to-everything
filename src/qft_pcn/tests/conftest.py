"""Per-test memory cap (safety rail).

After commits d81f605 (chi_max=32) and e283085 (tobl species, local dim 8192
-> 65536) the MPS-based encoder tests can balloon the resident set into the
tens of GB. This conftest applies an `RLIMIT_AS` cap inside the test process
so any test that exceeds ~6 GiB throws `MemoryError` instead of pushing the
whole system into kernel OOM-killer territory.

Note: RLIMIT_AS is a soft cap on VIRTUAL address space, not RSS. It is set
deliberately generous (8 GiB) to allow legitimate large MPS tensors while
still catching pathological blow-ups well before WSL's 48 GiB system limit.
This is a safety net, not a precise budget.
"""

from __future__ import annotations

import os
import resource

# Default cap: 8 GiB of virtual address space per test process.
# Override via QFT_PCN_TEST_MEM_CAP_GB env var if a specific test needs more.
_DEFAULT_CAP_GB = int(os.environ.get("QFT_PCN_TEST_MEM_CAP_GB", "8"))
_CAP_BYTES = _DEFAULT_CAP_GB * 1024 * 1024 * 1024


def _install_memory_cap() -> None:
    """Apply RLIMIT_AS on import. Raises only if the cap can't be set."""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    except (ValueError, OSError):
        return  # platform doesn't support it
    # If a stricter cap is already in place, leave it alone.
    if soft != resource.RLIM_INFINITY and soft <= _CAP_BYTES:
        return
    new_hard = hard if hard != resource.RLIM_INFINITY else _CAP_BYTES
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_CAP_BYTES, new_hard))
    except (ValueError, OSError):
        # If the kernel rejects (e.g. our cap exceeds hard), fall back to
        # whatever the hard limit is.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (new_hard, new_hard))
        except (ValueError, OSError):
            pass


_install_memory_cap()
