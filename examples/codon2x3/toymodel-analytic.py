"""
This toy model is described in raoteh/examples/codon2x3.

"""
from __future__ import division, print_function, absolute_import

import argparse

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose
import scipy.linalg
from scipy.special import xlogy

from nxmctree import dynamic_fset_lhood
from nxblink.compound import State, compound_state_is_ok, get_Q_compound
from nxblink.denseutil import (
        get_compound_states,
        define_compound_process, get_expected_rate, nx_to_np,
        nx_to_np_rate_matrix, np_to_nx_transition_matrix,
        compute_edge_expectation, compute_dwell_times)

import nxmodel
import nxmodelb


def run(model, compound_states, node_to_data_fset):
    """

    Parameters
    ----------
    model : Python module
        a module with hardcoded information about the model

    """
    # Get the analog of the genetic code.
    primary_to_tol = model.get_primary_to_tol()

    # Get the number of compound states.
    ncompound = len(compound_states)

    # Get the prior blink state distribution.
    blink_distn = model.get_blink_distn()

    #TODO check that the primary rate matrix is time-reversible
    # Get the primary rate matrix and convert it to a dense ndarray.
    nprimary = 6
    Q_primary_nx = model.get_Q_primary()
    Q_primary_dense = nx_to_np_rate_matrix(Q_primary_nx, range(nprimary))
    primary_distn = model.get_primary_distn()
    primary_distn_dense = np.array([primary_distn[i] for i in range(nprimary)])

    # Get the expected rate using only the nx rate matrix and the nx distn.
    expected_primary_rate = 0
    for sa, sb in Q_primary_nx.edges():
        p = primary_distn[sa]
        rate = Q_primary_nx[sa][sb]['weight']
        expected_primary_rate += p * rate

    # Normalize the primary rate matrix by dividing all rates
    # by the expected rate.
    for sa, sb in Q_primary_nx.edges():
        Q_primary_nx[sa][sb]['weight'] /= expected_primary_rate

    # Get the rooted directed tree shape.
    T, root = model.get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = model.get_edge_to_blen()

    # Define some indicators for the compound process
    indicators = define_compound_process(
            Q_primary_nx, compound_states, primary_to_tol)
    I_syn, I_non, I_on, I_off = indicators

    # Define the dense compound transition rate matrix
    on_rate = model.get_rate_on()
    off_rate = model.get_rate_off()
    Q_compound_nx = get_Q_compound(
            Q_primary_nx, on_rate, off_rate, primary_to_tol, compound_states)
    Q_compound = nx_to_np(Q_compound_nx, compound_states)
    row_sums = np.sum(Q_compound, axis=1)
    Q_compound = Q_compound - np.diag(row_sums)
    
    # Define a sparse stationary distribution over compound states.
    compound_distn = {}
    for state in compound_states:
        if compound_state_is_ok(primary_to_tol, state):
            p = 1.0
            p *= primary_distn[state.P]
            for tol_name in 'T0', 'T1', 'T2':
                if primary_to_tol[state.P] != tol_name:
                    tol_state = getattr(state, tol_name)
                    p *= blink_distn[tol_state]
            compound_distn[state] = p
    total = sum(compound_distn.values())
    assert_allclose(total, 1)

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

    # Compute the map from node to posterior state distribution.
    # Convert the dict distribution back into a dense distribution.
    # This is used in the calculation of expected log likelihood.
    node_to_distn = dynamic_fset_lhood.get_node_to_distn(
            T, edge_to_P_nx, root, compound_distn, node_to_data_fset)
    root_distn = node_to_distn[root]
    print('prior distribution at the root:')
    for i, p in sorted(compound_distn.items()):
        print(i, p)
    print()
    print('posterior distribution at the root:')
    for i, p in sorted(root_distn.items()):
        print(i, p)
    print()
    root_distn_np = np.zeros(ncompound)
    for i, s in enumerate(compound_states):
        if s in root_distn:
            root_distn_np[i] = root_distn[s]

    # Compute the map from edge to posterior joint state distribution.
    # Convert the nx transition probability matrices back into dense ndarrays.
    edge_to_nxdistn = dynamic_fset_lhood.get_edge_to_nxdistn(
            T, edge_to_P_nx, root, compound_distn, node_to_data_fset)
    edge_to_J = {}
    for edge, J_nx in edge_to_nxdistn.items():
        J_np = nx_to_np(J_nx, compound_states)
        edge_to_J[edge] = J_np


    # Initialize contributions to the expected log likelihood.
    #
    # Compute the contribution of the initial state distribution.
    ell_init = xlogy(root_distn_np, compound_distn_np).sum()
    # Initialize the contribution of the expected transitions.
    I_all = I_on + I_off + I_syn + I_non
    I_log_all = xlogy(I_all, Q_compound)
    ell_trans = 0
    # Initialize the contribution of the dwell times.
    ell_dwell = 0

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

        # Compute expectation of logs of rates of observed transitions.
        # This is part of the expected log likelihood calculation.
        contrib = compute_edge_expectation(Q, P, J, I_log_all, t)
        ell_trans += contrib
        print('edge %s -> %s ell trans contrib %s' % (va, vb, contrib))

        # Compute sum of expectations of dwell times
        contrib = compute_dwell_times(Q, P, J, -row_sums, t)
        ell_dwell += contrib
        print('edge %s -> %s ell dwell contrib %s' % (va, vb, contrib))
        
        print()

    print('expected count of primary process transitions:')
    print(primary_expectation)
    print()

    print('expected count of blink process transitions:')
    print(blink_expectation)
    print()

    print('expected log likelihood:')
    print('contribution of initial state distribution :', ell_init)
    print('contribution of expected transition counts :', ell_trans)
    print('contribution of expected dwell times       :', ell_dwell)
    print('total                                      :', (
        ell_init + ell_trans + ell_dwell))
    print()



def main(args):
    """

    """
    # define the model
    if args.model == 'nxmodel':
        model = nxmodel
    elif args.model == 'nxmodelb':
        model = nxmodelb
    else:
        raise Exception

    # Define the ordering of the compound states.
    compound_states = get_compound_states()

    # Define the data according to the command line selection.
    if args.data == 0:
        # No data.
        node_to_data_fset = {
                'N0' : set(compound_states),
                'N1' : set(compound_states),
                'N2' : set(compound_states),
                'N3' : set(compound_states),
                'N4' : set(compound_states),
                'N5' : set(compound_states),
                }
    elif args.data == 1:
        # Alignment data only.
        node_to_data_fset = {
                'N0' : {
                    State(0, 1, 0, 0),
                    State(0, 1, 0, 1),
                    State(0, 1, 1, 0),
                    State(0, 1, 1, 1)},
                'N1' : set(compound_states),
                'N2' : set(compound_states),
                'N3' : {
                    State(4, 0, 0, 1),
                    State(4, 0, 1, 1),
                    State(4, 1, 0, 1),
                    State(4, 1, 1, 1)},
                'N4' : {
                    State(5, 0, 0, 1),
                    State(5, 0, 1, 1),
                    State(5, 1, 0, 1),
                    State(5, 1, 1, 1)},
                'N5' : {
                    State(1, 1, 0, 0),
                    State(1, 1, 0, 1),
                    State(1, 1, 1, 0),
                    State(1, 1, 1, 1)},
                }
    elif args.data == 2:
        # Alignment and disease data.
        node_to_data_fset = {
                'N0' : {
                    State(0, 1, 0, 1)},
                'N1' : set(compound_states),
                'N2' : set(compound_states),
                'N3' : {
                    State(4, 0, 0, 1),
                    State(4, 0, 1, 1),
                    State(4, 1, 0, 1),
                    State(4, 1, 1, 1)},
                'N4' : {
                    State(5, 0, 0, 1),
                    State(5, 0, 1, 1),
                    State(5, 1, 0, 1),
                    State(5, 1, 1, 1)},
                'N5' : {
                    State(1, 1, 0, 0),
                    State(1, 1, 0, 1),
                    State(1, 1, 1, 0),
                    State(1, 1, 1, 1)},
                }
    elif args.data == 3:
        # Alignment and fully observed disease data.
        node_to_data_fset = {
                'N0' : {
                    State(0, 1, 0, 1)},
                'N1' : set(compound_states),
                'N2' : set(compound_states),
                'N3' : {
                    State(4, 1, 1, 1)},
                'N4' : {
                    State(5, 1, 1, 1)},
                'N5' : {
                    State(1, 1, 1, 1)},
                }
    else:
        raise Exception

    # run the analysis
    run(model, compound_states, node_to_data_fset)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
            choices=('nxmodel', 'nxmodelb'), default='nxmodel',
            help='specify the model complexity')
    parser.add_argument('--data',
            choices=(0, 1, 2, 3), type=int, default=0,
            help=(
                'specify the data level ('
                '0: no data, '
                '1: alignment only, '
                '2: alignment and human disease data, ',
                '3: alignment and human disease data '
                'and assume all others benign)'))
    args = parser.parse_args()
    main(args)

