"""
Model specification code.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

__all__ = [
        'get_Q_primary', 'get_Q_blink', 'get_primary_to_tol',
        'get_Q_meta', 'get_T_and_root', 'get_edge_to_blen',
        'hamming_distance', 'compound_state_is_ok',
        ]

def get_Q_primary():
    """
    This is like a symmetric codon rate matrix that is not normalized.

    """
    rate = 1
    Q_primary = nx.DiGraph()
    Q_primary.add_weighted_edges_from((
        (0, 1, rate),
        (0, 2, rate),
        (1, 0, rate),
        (1, 3, rate),
        (2, 0, rate),
        (2, 3, rate),
        (2, 4, rate),
        (3, 1, rate),
        (3, 2, rate),
        (3, 5, rate),
        (4, 2, rate),
        (4, 5, rate),
        (5, 3, rate),
        (5, 4, rate),
        ))
    return Q_primary


def get_Q_blink(rate_on=None, rate_off=None):
    Q_blink = nx.DiGraph()
    Q_blink.add_weighted_edges_from((
        (False, True, rate_on),
        (True, False, rate_off),
        ))
    return Q_blink


def get_primary_to_tol():
    """
    Return a map from primary state to tolerance track name.

    This is like a genetic code mapping codons to amino acids.

    """
    primary_to_tol = {
            0 : 'T0',
            1 : 'T0',
            2 : 'T1',
            3 : 'T1',
            4 : 'T2',
            5 : 'T2',
            }
    return primary_to_tol


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


def get_T_and_root():
    # rooted tree, deliberately without branch lengths
    T = nx.DiGraph()
    T.add_edges_from([
        ('N1', 'N0'),
        ('N1', 'N2'),
        ('N1', 'N5'),
        ('N2', 'N3'),
        ('N2', 'N4'),
        ])
    return T, 'N1'


def get_edge_to_blen():
    edge_to_blen = {
            ('N1', 'N0') : 0.5,
            ('N1', 'N2') : 0.5,
            ('N1', 'N5') : 0.5,
            ('N2', 'N3') : 0.5,
            ('N2', 'N4') : 0.5,
            }
    return edge_to_blen


def hamming_distance(va, vb):
    return sum(1 for a, b in zip(va, vb) if a != b)


def compound_state_is_ok(primary_to_tol, state):
    primary, tols = state
    tclass = primary_to_tol[primary]
    return True if tols[tclass] else False

