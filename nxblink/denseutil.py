"""
Helper functions for analytically computing expectations.

These helper functions are for the dense compound process,
and this module is not necessary for Rao-Teh sampling
or for the analysis of the track histories sampled using Rao-Teh
stochastic mapping.

This module should possibly be moved elsewhere.
The nxblink package is focused on analysis of sparse matrices
using networkx structures and algorithms, whereas a more appropriate package
for this module would use numpy and scipy structures and algorithms
to analyze dense rate matrices, transition matrices, and distributions.

This module does not need to care about piecewise homogeneity of the process.

"""
from __future__ import division, print_function, absolute_import

from itertools import product
from collections import namedtuple

import networkx as nx
import numpy as np
import scipy.linalg

from .util import hamming_distance
from .compound import State, compound_state_is_ok


__all__ = [
        'get_compound_states',
        'define_compound_process', 'get_expected_rate', 'nx_to_np',
        'nx_to_np_rate_matrix', 'np_to_nx_transition_matrix',
        'compute_edge_expectation', 'compute_dwell_times',
        ]


def get_compound_states():
    """
    Helper function for dense rate matrices.

    Note that some of the compound states in this list may be infeasible.

    """
    nprimary = 6

    # Define the name and state space of each subprocess.
    track_names = State._fields
    track_states = (
            range(nprimary),
            (0, 1),
            (0, 1),
            (0, 1),
            )

    # The compound state space is the cartesian product
    # of subprocess state spaces.
    compound_states = [State(*x) for x in product(*track_states)]

    # Return the ordered compound states.
    return compound_states


def define_compound_process(Q_primary, compound_states, primary_to_tol):
    """
    Compute indicator matrices for the compound process.

    """
    n = len(compound_states)

    # define some dense indicator matrices
    I_syn = np.zeros((n, n), dtype=float)
    I_non = np.zeros((n, n), dtype=float)
    I_on = np.zeros((n, n), dtype=float)
    I_off = np.zeros((n, n), dtype=float)

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

            # set the indicator according to the transition type
            if sa.P != sb.P:
                if primary_to_tol[sa.P] == primary_to_tol[sb.P]:
                    I_syn[i, j] = 1
                else:
                    I_non[i, j] = 1
            else:
                diff = sum(sb) - sum(sa)
                if diff == 1:
                    I_on[i, j] = 1
                elif diff == -1:
                    I_off[i, j] = 1
                else:
                    raise Exception

    return I_syn, I_non, I_on, I_off


def get_expected_rate(Q_dense, dense_distn):
    return -np.dot(np.diag(Q_dense), dense_distn)


def nx_to_np(M_nx, ordered_states):
    state_to_idx = dict((s, i) for i, s in enumerate(ordered_states))
    nstates = len(ordered_states)
    M_np = np.zeros((nstates, nstates))
    for sa, sb in M_nx.edges():
        i = state_to_idx[sa]
        j = state_to_idx[sb]
        M_np[i, j] = M_nx[sa][sb]['weight']
    return M_np


def nx_to_np_rate_matrix(Q_nx, ordered_states):
    Q_np = nx_to_np(Q_nx, ordered_states)
    row_sums = np.sum(Q_np, axis=1)
    Q_np = Q_np - np.diag(row_sums)
    return Q_np


def np_to_nx_transition_matrix(P_np, ordered_states):
    P_nx = nx.DiGraph()
    for i, sa in enumerate(ordered_states):
        for j, sb in enumerate(ordered_states):
            p = P_np[i, j]
            if p:
                P_nx.add_edge(sa, sb, weight=p)
    return P_nx


def compute_edge_expectation(Q, P, J, indicator, t):
    # Q is the rate matrix
    # P is the conditional transition matrix
    # J is the joint distribution matrix
    ncompound = Q.shape[0]
    E = Q * indicator
    interact = scipy.linalg.expm_frechet(Q*t, E*t, compute_expm=False)
    total = 0
    for i in range(ncompound):
        for j in range(ncompound):
            if J[i, j]:
                total += J[i, j] * interact[i, j] / P[i, j]
    return total


def compute_dwell_times(Q, P, J, indicator, t):
    # Q is the rate matrix
    # P is the conditional transition matrix
    # J is the joint distribution matrix
    # the indicator is a dense 1d vector
    ncompound = Q.shape[0]
    E = np.diag(indicator)
    interact = scipy.linalg.expm_frechet(Q*t, E*t, compute_expm=False)
    total = 0
    for i in range(ncompound):
        for j in range(ncompound):
            if J[i, j]:
                total += J[i, j] * interact[i, j] / P[i, j]
    return total


