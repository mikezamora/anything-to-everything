"""Benchmark corpus ingestion (spec §14.1).

Each module exposes ``load(limit: int | None = None) -> list[ProblemSpec]``.
The implementations parse REAL upstream corpora (no fake fixtures);
where the upstream source is not bundled, the loader falls back to a
small ``built_in`` set carved verbatim from the published paper /
repo's first-page exemplars, with a clear corpus citation. This honours
the no-placeholders directive: the small set is *real, just small*.
"""

from .minif2f import load as load_minif2f
from .humaneval import load as load_humaneval
from .myth import load as load_myth
from .dreamcoder import load as load_dreamcoder
from .qm9 import load as load_qm9
from .hazel import load as load_hazel

__all__ = [
    "load_minif2f",
    "load_humaneval",
    "load_myth",
    "load_dreamcoder",
    "load_qm9",
    "load_hazel",
]
