"""
Functions related to an explicitly compound yet sparse Markov process.

It is designed for testing, so it has a hardcoded state space.

This module does not need to care about piecewise homogeneity of the process.

"""
from __future__ import division, print_function, absolute_import

from collections import namedtuple

import networkx as nx

from .util import hamming_distance

__all__ = ['State', 'compound_state_is_ok', 'get_Q_compound']


State = namedtuple('State', 'P T0 T1 T2')


def compound_state_is_ok(primary_to_tol, state):
    """
    Check whether the primary state is compatible with its tolerance.

    Parameters
    ----------
    primary_to_tol : dict
        Map from primary state to the name of its tolerance class.
    state : State
        The compound state as a named tuple.

    Returns
    -------
    ret : bool
        True if the primary state is compatible with its tolerance state.

    """
    return getattr(state, primary_to_tol[state.P])


def get_Q_compound(
        Q_primary, on_rate, off_rate, primary_to_tol, compound_states):
    """
    Get a compound state rate matrix as a networkx Digraph.

    This is only for testing, because realistically sized processes
    will have combinatorially large compound state spaces.

    """
    Q_compound = nx.DiGraph()
    for i, sa in enumerate(compound_states):

        # skip compound states that have zero probability
        if not compound_state_is_ok(primary_to_tol, sa):
            continue

        for j, sb in enumerate(compound_states):

            # skip compound states that have zero probability
            if not compound_state_is_ok(primary_to_tol, sb):
                continue

            # if hamming distance between compound states is not 1 then skip
            if hamming_distance(sa, sb) != 1:
                continue

            # if a primary transition is not allowed then skip
            if sa.P != sb.P and not Q_primary.has_edge(sa.P, sb.P):
                continue

            # add the primary transition or tolerance transition
            if sa.P != sb.P:
                rate = Q_primary[sa.P][sb.P]['weight']
                if primary_to_tol[sa.P] == primary_to_tol[sb.P]:
                    # synonymous primary process transition
                    Q_compound.add_edge(sa, sb, weight=rate)
                else:
                    # non-synonymous primary process transition
                    Q_compound.add_edge(sa, sb, weight=rate)
            else:
                diff = sum(sb) - sum(sa)
                if diff == 1:
                    Q_compound.add_edge(sa, sb, weight=on_rate)
                elif diff == -1:
                    Q_compound.add_edge(sa, sb, weight=off_rate)
                else:
                    raise Exception
    return Q_compound

