"""
Utility functions that are not in networkx.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


def get_edge_tree(T, root):
    """
    Nodes in the edge tree are edges in the original tree.

    The new tree will have a node (None, root) which does not correspond
    to any edge in the original tree.

    """
    dual_root = (None, root)
    T_dual = nx.DiGraph()
    if not T:
        return T_dual, dual_root
    for c in T[root]:
        T_dual.add_edge(dual_root, (root, c))
    for v in T:
        for c in T[v]:
            for g in T[c]:
                T_dual.add_edge((v, c), (c, g))
    return T_dual, dual_root

