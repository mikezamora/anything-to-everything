from __future__ import annotations

from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic._serialize import serialize_preorder
from src.qft_pcn.logic._channels import compute_live_binders
from src.qft_pcn.logic.encoding import BinderHandle


def test_live_binders_p1():
    """\\x:Int. x — bond 0 (between Lam@0 and Var@1) has Lam_x live."""
    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    live = compute_live_binders(sites)
    # N-1 bonds for length-N MPS.
    assert len(live) == 3
    # Bond 0 (between sites 0 and 1): Lam_x is live.
    assert len(live[0]) == 1
    assert live[0][0].lam_site == 0
    # Bond 1 (between sites 1 and 2): after Var consumed Lam_x, nothing live.
    assert len(live[1]) == 0
    assert len(live[2]) == 0


def test_live_binders_p5():
    """(\\x. (\\y. x + y)(3))(4) — nested binders, compaction after each use."""
    ast = parse(r"(\x:Int. (\y:Int. x + y)(3))(4)")
    sites = serialize_preorder(ast, N=32)
    live = compute_live_binders(sites)
    # Layout: APP@0, LAM_x@1, APP@2, LAM_y@3, BIN+@4, VAR_x@5, VAR_y@6,
    #         INT@7, INT@8, PAD@9..31
    # Bond 0 (between APP@0 and LAM_x@1): no binders live yet.
    assert len(live[0]) == 0
    # Bond 1 (between LAM_x@1 and APP@2): Lam_x live.
    assert len(live[1]) == 1
    assert live[1][0].lam_site == 1
    # Bond 2 (between APP@2 and LAM_y@3): Lam_x still live.
    assert len(live[2]) == 1
    # Bond 3 (between LAM_y@3 and BIN@4): both Lam_x and Lam_y live.
    assert len(live[3]) == 2
    assert {b.lam_site for b in live[3]} == {1, 3}
    # Bond 4 (between BIN@4 and VAR_x@5): both still live (Var_x is on the
    # right side of bond 4, but Var_y is also on the right of bond 4 — both
    # are still needed across this bond).
    assert len(live[4]) == 2
    # Bond 5 (between VAR_x@5 and VAR_y@6): after the use at site 5 consumed
    # the LAST occurrence of Lam_x, x leaves; only Lam_y remains.
    assert len(live[5]) == 1
    assert live[5][0].lam_site == 3
    # Bond 6 (between VAR_y@6 and INT@7): Lam_y also consumed; nothing.
    assert len(live[6]) == 0
    # Tail bonds all empty.
    for i in range(7, 31):
        assert len(live[i]) == 0


def test_live_binders_ordering_is_declaration_order():
    """Channel order follows declaration order."""
    ast = parse(r"\x:Int. \y:Int. \z:Int. x + y + z")
    sites = serialize_preorder(ast, N=12)
    live = compute_live_binders(sites)
    # At the bond between LAM_z and the first BIN site, all three are live.
    # Lam_x@0, Lam_y@1, Lam_z@2.
    # Layout: LAM_x@0, LAM_y@1, LAM_z@2, BIN+@3, BIN+@4, VAR_x@5, VAR_y@6,
    #         VAR_z@7, ...
    # Bond 2 between LAM_z@2 and BIN@3:
    handles = live[2]
    assert len(handles) == 3
    # Ordered by lam_site (declaration order).
    assert [h.lam_site for h in handles] == [0, 1, 2]


def test_compute_channel_param_ty_per_bond_p1():
    """For \\x:Int. x, bond 0 has one channel with param_ty = TYPE_INT."""
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic._channels import (
        compute_live_binders, compute_channel_param_ty_per_bond,
    )
    from src.qft_pcn.logic.encoding import TYPE_INT

    ast = parse(r"\x:Int. x")
    sites = serialize_preorder(ast, N=4)
    live = compute_live_binders(sites)
    pt = compute_channel_param_ty_per_bond(sites, live)
    assert len(pt) == 3   # N-1 bonds
    # Bond 0: one channel (Lam_x), param_ty = Int.
    assert pt[0] == [TYPE_INT]
    # Bonds 1, 2: no channels (after Var consumed x).
    assert pt[1] == []
    assert pt[2] == []


def test_compute_channel_param_ty_per_bond_p3():
    """\\f:Int->Int. \\x:Int. f x — two binders with distinct param_ty."""
    from src.qft_pcn.logic.ast import parse
    from src.qft_pcn.logic._serialize import serialize_preorder
    from src.qft_pcn.logic._channels import (
        compute_live_binders, compute_channel_param_ty_per_bond,
    )
    from src.qft_pcn.logic.encoding import TYPE_INT, TYPE_ARR_II

    ast = parse(r"\f:Int->Int. \x:Int. f x")
    sites = serialize_preorder(ast, N=16)
    live = compute_live_binders(sites)
    pt = compute_channel_param_ty_per_bond(sites, live)
    # Layout: LAM_f@0, LAM_x@1, APP@2, VAR_f@3, VAR_x@4, PAD...
    # Bond 0: between LAM_f@0 and LAM_x@1 — Lam_f live, param_ty = Int->Int.
    assert pt[0] == [TYPE_ARR_II]
    # Bond 1: between LAM_x@1 and APP@2 — both Lam_f and Lam_x live.
    assert pt[1] == [TYPE_ARR_II, TYPE_INT]
    # Bond 2: between APP@2 and VAR_f@3 — both still live.
    assert pt[2] == [TYPE_ARR_II, TYPE_INT]
    # Bond 3: between VAR_f@3 and VAR_x@4 — Lam_f consumed at site 3;
    #         only Lam_x remains.
    assert pt[3] == [TYPE_INT]
    # Bond 4: between VAR_x@4 and PAD@5 — Lam_x consumed; nothing.
    assert pt[4] == []
