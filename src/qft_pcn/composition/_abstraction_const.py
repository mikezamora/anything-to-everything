"""Constants and exception family for abstraction discovery (spec §8.1, §10)."""
from __future__ import annotations

S_MIN: int = 3
S_MAX: int = 8
FP_DECIMALS: int = 3
FP_RANK: int = 4
DEFAULT_DISTANCE_THRESHOLD: float = 0.15
DEFAULT_ALPHA: float = 1e-3
DEFAULT_CHI_CAP: int = 16
N_QUIESCENT: int = 2
REPURIFICATION_TAIL_TOL: float = 1e-6
DENSITY_HERMITICITY_TOL: float = 1e-9


class AbstractionError(Exception):
    """Base class for abstraction-discovery errors."""


class MiningError(AbstractionError):
    """A cached state is not a valid normalized MERA, or an interval is malformed."""


class LibraryContractError(AbstractionError):
    """The injected LemmaLibrary does not satisfy the spec §7 contract."""


class RepurificationWarning(Warning):
    """Canonical re-purification truncated tail mass above the tolerance."""
