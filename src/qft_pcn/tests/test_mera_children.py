"""Tests for MeraEncodingMeta.children_of_node (spec §5.6)."""
from __future__ import annotations
from src.qft_pcn.logic.ast import parse
from src.qft_pcn.logic.mera_encoder import encode_mera


def test_children_of_node_exists():
    _, meta = encode_mera(parse(r"\x:Int. x"))
    assert hasattr(meta, "children_of_node")
    assert isinstance(meta.children_of_node, dict)


def test_lam_has_one_child():
    # \x:Int. x  -> node 0 is Lam, node 1 is Var(x); Var is Lam's child.
    _, meta = encode_mera(parse(r"\x:Int. x"))
    assert meta.children_of_node.get(0) == [1]
    assert meta.children_of_node.get(1, []) == []


def test_app_has_two_children():
    # (\x:Int. x + 1)(2): root App has fn (the Lam) and arg (IntLit 2).
    _, meta = encode_mera(parse(r"(\x:Int. x + 1)(2)"))
    kids = meta.children_of_node.get(0)
    assert kids is not None and len(kids) == 2


def test_bin_has_two_children():
    # node for `x + 1`: lhs Var, rhs IntLit.
    _, meta = encode_mera(parse(r"\x:Int. x + 1"))
    bin_nodes = [n for n, ks in meta.children_of_node.items()
                 if len(ks) == 2]
    assert len(bin_nodes) >= 1


def test_children_indices_are_contiguous_preorder():
    # children always have index > parent (pre-order layout).
    _, meta = encode_mera(parse(r"(\x:Int. (\y:Int. x + y)(3))(4)"))
    for parent, kids in meta.children_of_node.items():
        for k in kids:
            assert k > parent
