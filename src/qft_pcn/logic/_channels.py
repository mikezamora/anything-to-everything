"""Live-binder bookkeeping per bond.

A binder is *live* at bond i (between sites i and i+1) iff:
  - its lam_site <= i
  - it has at least one var use at site >= i+1

The encoder uses this to size the bid-register bond and to assign channels
in declaration order.
"""

from __future__ import annotations

from .encoding import BinderHandle, KIND_LAM, KIND_VAR
from ._serialize import NodeOccupancy


def compute_live_binders(
    sites: list[NodeOccupancy],
) -> list[list[BinderHandle]]:
    """Return, for each of the N-1 bonds, the ordered list of live binders.

    Output[i] = ordered (by declaration / lam_site) list of BinderHandles
    that are live at bond i (between sites i and i+1).
    """
    N = len(sites)
    # First pass: collect Lam metadata and the rightmost var use per binder.
    binders: list[BinderHandle] = []   # in declaration order
    binder_lam_site: dict[int, int] = {}  # lam_site -> position in binders
    last_use_site: dict[int, int] = {}   # lam_site -> last var-use site

    for k, occ in enumerate(sites):
        if occ.kind == KIND_LAM:
            depth = occ.binder_ref.lexical_depth if occ.binder_ref else 0
            handle = BinderHandle(lam_site=k, depth_at_lam=depth)
            binder_lam_site[k] = len(binders)
            binders.append(handle)
            last_use_site[k] = k       # if never used, last_use = lam_site
        elif occ.kind == KIND_VAR and occ.var_ref is not None:
            ls = occ.var_ref.binder_site
            last_use_site[ls] = max(last_use_site.get(ls, ls), k)

    # Second pass: compute liveness per bond.
    # A binder is live across bond i iff lam_site <= i AND last_use_site > i.
    live: list[list[BinderHandle]] = []
    for i in range(N - 1):
        crossing: list[BinderHandle] = []
        for h in binders:
            if h.lam_site <= i and last_use_site[h.lam_site] > i:
                crossing.append(h)
        live.append(crossing)
    return live
