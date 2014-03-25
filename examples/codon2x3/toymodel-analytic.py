"""
This toy model is described in raoteh/examples/codon2x3.

"""
from __future__ import division, print_function, absolute_import

import itertools

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg

from nxmctree import dynamic_fset_lhood
import util


def get_Q_primary():
    # this is like a symmetric codon rate matrix
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


def get_primary_to_tol():
    # this is like a genetic code mapping codons to amino acids
    primary_to_tol = {
            0 : 0,
            1 : 0,
            2 : 1,
            3 : 1,
            4 : 2,
            5 : 2,
            }
    return primary_to_tol


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


def get_compound_states(primary_to_tol):
    nprimary = len(primary_to_tol)
    all_tols = list(itertools.product((0, 1), repeat=3))
    compound_states = list(itertools.product(range(nprimary), all_tols))
    return compound_states


def compound_state_is_ok(primary_to_tol, state):
    primary, tols = state
    tclass = primary_to_tol[primary]
    return True if tols[tclass] else False


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
        prim_a, tols_a = sa
        tclass_a = primary_to_tol[prim_a]
        if not tols_a[tclass_a]:
            continue

        for j, sb in enumerate(compound_states):

            # skip compound states that have zero probability
            prim_b, tols_b = sb
            tclass_b = primary_to_tol[prim_b]
            if not tols_b[tclass_b]:
                continue

            # if neither or both primary state and tolerance change then skip
            if hamming_distance(sa, sb) != 1:
                continue

            # if the tolerances change at more than one position then skip
            if hamming_distance(tols_a, tols_b) > 1:
                continue

            # if a primary transition is not allowed then skip
            if prim_a != prim_b and not Q_primary.has_edge(prim_a, prim_b):
                continue

            # set the indicator according to the transition type
            if prim_a != prim_b and tclass_a == tclass_b:
                I_syn[i, j] = 1
            elif prim_a != prim_b and tclass_a != tclass_b:
                I_non[i, j] = 1
            elif sum(tols_b) - sum(tols_a) == 1:
                I_on[i, j] = 1
            elif sum(tols_b) - sum(tols_a) == -1:
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


def run(primary_to_tol, compound_states, node_to_data_fset):

    # Get the primary rate matrix and convert it to a dense ndarray.
    nprimary = 6
    Q_primary_nx = get_Q_primary()
    Q_primary_dense = nx_to_np_rate_matrix(Q_primary_nx, range(nprimary))
    primary_distn_dense = np.ones(nprimary, dtype=float) / nprimary

    # The expected rate of the pure primary process
    # will be used for normalization.
    expected_primary_rate = get_expected_rate(
            Q_primary_dense, primary_distn_dense)
    #print('pure primary process expected rate:')
    #print(expected_primary_rate)
    #print

    # Get the rooted directed tree shape.
    T, root = get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = get_edge_to_blen()

    # Define the compound process through some indicators.
    indicators = define_compound_process(
            Q_primary_nx, compound_states, primary_to_tol)
    I_syn, I_non, I_on, I_off = indicators

    # Define the dense compound transition rate matrix through the indicators.
    syn_rate = 1.0
    non_rate = 1.0
    on_rate = 1.0
    off_rate = 1.0
    Q_compound = (
            syn_rate * I_syn / expected_primary_rate +
            non_rate * I_non / expected_primary_rate +
            #syn_rate * I_syn +
            #non_rate * I_non +
            on_rate * I_on +
            off_rate * I_off)
    #Q_compound = Q_compound / expected_primary_rate
    row_sums = np.sum(Q_compound, axis=1)
    Q_compound = Q_compound - np.diag(row_sums)
    
    # Define a sparse stationary distribution over compound states.
    # This should use the rates but for now it will just be
    # uniform over the ok compound states because of symmetry.
    compound_distn = {}
    for state in compound_states:
        if compound_state_is_ok(primary_to_tol, state):
            compound_distn[state] = 1.0
    total = sum(compound_distn.values())
    compound_distn = dict((k, v/total) for k, v in compound_distn.items())
    #print('compound distn:')
    #print(compound_distn)
    #print()

    # Convert the compound state distribution to a dense array.
    # Check that the distribution is at equilibrium.
    compound_distn_np = np.array([
            compound_distn.get(k, 0) for k in compound_states])
    equilibrium_rates = np.dot(compound_distn_np, Q_compound)
    assert_allclose(equilibrium_rates, 0, atol=1e-10)

    # Make the np and nx transition probability matrices.
    # Map each branch to the transition matrix.
    edge_to_P_np = {}
    edge_to_P_nx = {}
    for edge in T.edges():
        t = edge_to_blen[edge]
        P_np = scipy.linalg.expm(Q_compound * t)
        P_nx = np_to_nx_transition_matrix(P_np, compound_states)
        edge_to_P_np[edge] = P_np
        edge_to_P_nx[edge] = P_nx

    # Compute the likelihood
    lhood = dynamic_fset_lhood.get_lhood(
            T, edge_to_P_nx, root, compound_distn, node_to_data_fset)
    print('likelihood:')
    print(lhood)
    print()

    # Compute the map from edge to posterior joint state distribution.
    # Convert the nx transition probability matrices back into dense ndarrays.
    edge_to_nxdistn = dynamic_fset_lhood.get_edge_to_nxdistn(
            T, edge_to_P_nx, root, compound_distn, node_to_data_fset)
    edge_to_J = {}
    for edge, J_nx in edge_to_nxdistn.items():
        J_np = nx_to_np(J_nx, compound_states)
        edge_to_J[edge] = J_np

    # Compute labeled transition count expectations
    # using the rate matrix, the joint posterior state distribution matrices,
    # the indicator matrices, and the conditional transition probability
    # distribution matrix.
    primary_expectation = 0
    blink_expectation = 0
    for edge in T.edges():
        va, vb = edge
        Q = Q_compound
        J = edge_to_J[edge]
        P = edge_to_P_np[edge]
        t = edge_to_blen[edge]

        # primary transition event count expectations
        syn_total = compute_edge_expectation(Q, P, J, I_syn, t)
        non_total = compute_edge_expectation(Q, P, J, I_non, t)
        primary_expectation += syn_total
        primary_expectation += non_total
        print('edge %s -> %s syn expectation %s' % (va, vb, syn_total))
        print('edge %s -> %s non expectation %s' % (va, vb, non_total))

        # blink transition event count expectations
        on_total = compute_edge_expectation(Q, P, J, I_on, t)
        off_total = compute_edge_expectation(Q, P, J, I_off, t)
        blink_expectation += on_total
        blink_expectation += off_total
        print('edge %s -> %s on expectation %s' % (va, vb, on_total))
        print('edge %s -> %s off expectation %s' % (va, vb, off_total))
        
        print()

    print('primary expectation:')
    print(primary_expectation)
    print()

    print('blink expectation:')
    print(blink_expectation)
    print()



def main():

    # Get the analog of the genetic code.
    primary_to_tol = get_primary_to_tol()

    # Define the ordering of the compound states.
    compound_states = get_compound_states(primary_to_tol)

    # No data.
    print ('expectations given no alignment or disease data')
    print()
    node_to_data_fset = {
            'N0' : set(compound_states),
            'N1' : set(compound_states),
            'N2' : set(compound_states),
            'N3' : set(compound_states),
            'N4' : set(compound_states),
            'N5' : set(compound_states),
            }
    run(primary_to_tol, compound_states, node_to_data_fset)
    print()

    # Alignment data only.
    print ('expectations given only alignment data but not disease data')
    print()
    node_to_data_fset = {
            'N0' : {
                (0, (1, 0, 0)),
                (0, (1, 0, 1)),
                (0, (1, 1, 0)),
                (0, (1, 1, 1))},
            'N1' : set(compound_states),
            'N2' : set(compound_states),
            'N3' : {
                (4, (0, 0, 1)),
                (4, (0, 1, 1)),
                (4, (1, 0, 1)),
                (4, (1, 1, 1))},
            'N4' : {
                (5, (0, 0, 1)),
                (5, (0, 1, 1)),
                (5, (1, 0, 1)),
                (5, (1, 1, 1))},
            'N5' : {
                (1, (1, 0, 0)),
                (1, (1, 0, 1)),
                (1, (1, 1, 0)),
                (1, (1, 1, 1))},
            }
    run(primary_to_tol, compound_states, node_to_data_fset)
    print()

    # Alignment and disease data.
    print ('expectations given alignment and disease data')
    print()
    node_to_data_fset = {
            'N0' : {
                (0, (1, 0, 1))},
            'N1' : set(compound_states),
            'N2' : set(compound_states),
            'N3' : {
                (4, (0, 0, 1)),
                (4, (0, 1, 1)),
                (4, (1, 0, 1)),
                (4, (1, 1, 1))},
            'N4' : {
                (5, (0, 0, 1)),
                (5, (0, 1, 1)),
                (5, (1, 0, 1)),
                (5, (1, 1, 1))},
            'N5' : {
                (1, (1, 0, 0)),
                (1, (1, 0, 1)),
                (1, (1, 1, 0)),
                (1, (1, 1, 1))},
            }
    run(primary_to_tol, compound_states, node_to_data_fset)
    print()

    # Alignment and fully observed disease data.
    print ('expectations given alignment and fully observed disease data')
    print ('(all leaf disease states which were previously considered to be')
    print ('unobserved are now considered to be tolerated (blinked on))')
    print()
    node_to_data_fset = {
            'N0' : {
                (0, (1, 0, 1))},
            'N1' : set(compound_states),
            'N2' : set(compound_states),
            'N3' : {
                (4, (1, 1, 1))},
            'N4' : {
                (5, (1, 1, 1))},
            'N5' : {
                (1, (1, 1, 1))},
            }
    run(primary_to_tol, compound_states, node_to_data_fset)
    print()


main()

