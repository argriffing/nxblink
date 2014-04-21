"""
Utility functions that are not in networkx.

This module cares about piecewise homogeneity of the process.
In particular, partition_nodes needs to look for edges
with zero rate scaling factor.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


def partition_nodes(T, edge_to_blen, edge_to_rate):
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
    edge_to_rate : dict
        Maps directed edges of T to non-negative rate scaling factors.

    Returns
    -------
    components : list of list of nodes
        Each list contains the nodes of a connected component.

    """
    # Compute an undirected graph representing an equivalence relation.
    G = nx.Graph()
    G.add_nodes_from(T)
    for edge in T.edges():
        blen = edge_to_blen[edge]
        rate = edge_to_rate[edge]
        if (not blen) or (not rate):
            G.add_edge(*edge)

    # Return the connected components of the graph.
    return nx.connected_components(G)

