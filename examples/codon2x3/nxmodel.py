"""
Model specification code.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

__all__ = [
        'get_Q_primary', 'get_primary_to_tol',
        'get_T_and_root', 'get_edge_to_blen',
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


#TODO this is in nxblink/modelutil
#def get_Q_blink(rate_on=None, rate_off=None):


def get_primary_to_tol():
    """
    Return a map from primary state to tolerance track name.

    This is like a genetic code mapping codons to amino acids.

    """
    primary_to_tol = {
            0 : 0,
            1 : 0,
            2 : 1,
            3 : 1,
            4 : 2,
            5 : 2,
            }
    return primary_to_tol


#TODO this is in nxblink/modelutil
#def get_Q_meta(Q_primary, primary_to_tol):


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


#TODO this is in nxblink/modelutil
#def hamming_distance(va, vb):


#TODO this is in nxblink/modelutil
#def compound_state_is_ok(primary_to_tol, state):

