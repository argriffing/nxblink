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


def partition_nodes(T, edge_to_blen):
    """
    Partition nodes of a tree.

    The nodes are partitioned into equivalence classes,
    where two nodes are considered equivalent if they are connected
    by a branch of length exactly zero.

    Parameters
    ----------
    T : networkx DiGraph
        The tree.
    edge_to_blen : dict
        Maps directed edges of T to non-negative branch lengths.

    Returns
    -------
    components : list of list of nodes
        Each list contains the nodes of a connected component.

    """
    # Compute an undirected graph representing an equivalence relation.
    G = nx.Graph()
    G.add_nodes_from(T)
    for edge in T.edges():
        if not edge_to_blen[edge]:
            G.add_edge(*edge)

    # Return the connected components of the graph.
    return nx.connected_components(G)

