"""
Helper functions related to transition matrices and uniformization.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


def get_omega(total_rates, uniformization_factor):
    max_rate = max(total_rates.values())
    if not max_rate:
        raise Exception('zero max rate; total_rates : ' + str(total_rates))
    return uniformization_factor * max_rate


def get_poisson_rates(total_rates, omega):
    return dict((s, omega - r) for s, r in total_rates.items())


def get_total_rates(Q_nx):
    """
    
    Parameters
    ----------
    Q_nx : directed networkx graph
        rate matrix

    Returns
    -------
    total_rates : dict
        map from state to total rate away from the state

    """
    total_rates = {}
    for sa in Q_nx:
        total_rate = None
        for sb in Q_nx[sa]:
            rate = Q_nx[sa][sb]['weight']
            if total_rate is None:
                total_rate = 0
            total_rate += rate
        if total_rate is not None:
            total_rates[sa] = total_rate
    return total_rates


def get_uniformized_P_nx(Q_nx, total_rates, omega):
    """

    Parameters
    ----------
    Q_nx : directed networkx graph
        rate matrix
    total_rates : dict
        map from state to sum of Q_nx rates out of that state
    omega : float
        uniformization rate

    Returns
    -------
    P_nx : directed networkx graph
        transition probability matrix

    """
    P_nx = nx.DiGraph()
    for sa in Q_nx:
        total_rate = total_rates.get(sa, 0)

        # define the self-transition probability
        weight = 1.0 - total_rate / omega
        P_nx.add_edge(sa, sa, weight=weight)

        # define probabilities of transitions to other states
        for sb in Q_nx[sa]:
            weight = Q_nx[sa][sb]['weight'] / omega
            P_nx.add_edge(sa, sb, weight=weight)

    return P_nx


def get_identity_P_nx(states):
    """
    Get an identity transition matrix.

    Parameters
    ----------
    states : collection of hashables
        collection of states, not necessarily ordered

    Returns
    -------
    P_nx : networkx DiGraph
        Identity matrix, to be interpreted as a transition matrix.

    """
    P_nx = nx.DiGraph()
    for s in states:
        P_nx.add_edge(s, s, weight=1)
    return P_nx


def get_node_to_tm(T, root, node_to_blen):
    """
    Use branch lengths to compute the distance from each node to the root.

    Parameters
    ----------
    T : networkx DiGraph
        the tree
    root : hashable
        the root of the tree
    node_to_blen : dict
        branch length associated with each directed edge

    Returns
    -------
    node_to_tm : dict
        map from node to distance from the root

    """
    node_to_tm = {root : 0}
    for edge in nx.bfs_edges(T, root):
        va, vb = edge
        node_to_tm[vb] = node_to_tm[va] + node_to_blen[edge]
    return node_to_tm


def hamming_distance(va, vb):
    return sum(1 for a, b in zip(va, vb) if a != b)

