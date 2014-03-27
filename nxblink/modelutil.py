"""
Help build descriptions of blinking process models.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


__all__ = [
        'get_Q_blink', 'get_Q_meta',
        'hamming_distance', 'compound_state_is_ok',
        ]


def get_Q_blink(rate_on=None, rate_off=None):
    Q_blink = nx.DiGraph()
    Q_blink.add_weighted_edges_from((
        (False, True, rate_on),
        (True, False, rate_off),
        ))
    return Q_blink


def get_Q_meta(Q_primary, primary_to_tol):
    """
    Return a DiGraph of rates from primary states into sets of states.

    """
    Q_meta = nx.DiGraph()
    for primary_sa, primary_sb in Q_primary.edges():
        rate = Q_primary[primary_sa][primary_sb]['weight']
        tol_sb = primary_to_tol[primary_sb]
        if not Q_meta.has_edge(primary_sa, tol_sb):
            Q_meta.add_edge(primary_sa, tol_sb, weight=rate)
        else:
            Q_meta[primary_sa][tol_sb]['weight'] += rate
    return Q_meta


def hamming_distance(va, vb):
    return sum(1 for a, b in zip(va, vb) if a != b)


def compound_state_is_ok(primary_to_tol, state):
    primary, tols = state
    tclass = primary_to_tol[primary]
    return True if tols[tclass] else False

