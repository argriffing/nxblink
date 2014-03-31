"""
Implement the Muse-Gaut 1994 codon rate matrix as an nx.DiGraph.

The Muse-Gaut 1994 (MG94) codon model is a time-reversible
continuous time Markov process that allows a mutational equilibrium,
a mutational transition/transversion rate ratio 'kappa',
and an 'omega' parameter that distinguishes between synonymous
and nonsynonymous codon substitution rates.
This model is nearly the same as the Goldman-Yang 1994 (GY94) codon model.
The MG94 parameterization is slightly better in the sense of having a clearer
mechanistic interpretation, but the MG94 authors were less well known.

TODO:
Add a general purpose python package or module
for analysis of rate matrices of continuous-time finite-state Markov processes.
Because this package would be for prototypes, it could emphasize convenience,
so it could use a networkx DiGraph representation of a rate matrix
of a continuous time Markov process.
This representation will allow the states to be anything hashable,
rather then forcing an artificial ordering to be imposed on categorical states.
Furthermore it will allow the representation of processes that require
huge sparse rate matrices.
This package would avoid doing anything with trees.
At least initially it could also avoid code for sampling
or for matrix exponentiation.
Uniformization could also be out of scope initially;
perhaps later add a separate module or package for
uniformization-based algorithms.
This package should allow some input validation.
For example:
    * check that a finite dict distn is reasonable (non-negative entries
      and entries sum to 1).
    * check that a nx.DiGraph rate matrix has the right form (all edges
      have non-negative weights, and no loops exist, where a loop is an
      edge connecting a state to itself)
    * check that a state distribution is at equilibrium with respect
      to a given rate matrix.
    * check that a state distribution meets the detailed balance equations
      with respect to a given rate matrix.
nxrate


This module has been copied and modified
from cmedb/create-mg94.py -> raoteh/examples/p53/create_mg94 ->
nxblink/examples/p53/create_mg94.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np

import nxmctree
from nxmctree.util import prod, dict_distn

import nxblink
from nxblink.util import hamming_distance


#import cmedbutil

#TODO replace these...
#from raoteh.sampler import _util, _density

#distn = _util.get_normalized_dict_distn(weights)

# check nucleotide distribution
#nt_distn_dense = _density.dict_to_numpy_array(nt_distn)
#cmedbutil.assert_stochastic_vector(nt_distn_dense)

# check time-reversible rate matrix invariants
#Q_dense = _density.rate_matrix_to_numpy_array(Q, nodelist=states)
#distn_dense = _density.dict_to_numpy_array(distn, nodelist=states)
#cmedbutil.assert_stochastic_vector(distn_dense)
#cmedbutil.assert_rate_matrix(Q_dense)
#cmedbutil.assert_equilibrium(Q_dense, distn_dense)
#cmedbutil.assert_detailed_balance(Q_dense, distn_dense)


def create_mg94(
        A, C, G, T,
        kappa, omega, genetic_code,
        target_expected_rate=None,
        target_expected_syn_rate=None,
        ):
    """

    Parameters
    ----------
    A : float
        mutational equilibrium probability of nucleotide A
    C : float
        mutational equilibrium probability of nucleotide C
    G : float
        mutational equilibrium probability of nucleotide G
    T : float
        mutational equilibrium probability of nucleotide T
    kappa : float
        transition/transversion ratio
    omega : float
        nonsynonymous/synonymous ratio
    genetic_code : sequence
        Sequence whose elements are (state, residue, codon) tuples.
    target_expected_rate : float, optional
        rescale to this expected substitution rate
    target_expected_syn_rate : float, optional
        rescale to this expected synonymous substitution rate

    Returns
    -------
    Q : weighted directed networkx graph
        Sparse transition rate matrix.
    distn : dict
        Sparse codon state stationary distribution.
    state_to_residue : sequence
        Sequence of residues indexed by codon state.
    residue_to_part : dict
        Map from residue to tolerance class.

    """
    if (target_expected_rate, target_expected_syn_rate).count(None) > 1:
        raise ValueError('target_expected_rate and target_expected_syn_rate '
                'are mutually exclusive')

    # define state_to_part
    state_to_residue = dict((s, r) for s, r, c in genetic_code)
    alphabetic_residues = sorted(set(r for s, r, c in genetic_code))
    residue_to_part = dict((r, i) for i, r in enumerate(alphabetic_residues))
    state_to_part = dict((s, residue_to_part[r]) for s, r, c in genetic_code)

    nt_distn = {
            'A' : A,
            'C' : C,
            'G' : G,
            'T' : T,
            }

    # construct the mg94 rate matrix
    nstates = len(genetic_code)
    states = range(nstates)
    transitions = ('AG', 'GA', 'CT', 'TC')
    Q = nx.DiGraph()
    for a, (state_a, residue_a, codon_a) in enumerate(genetic_code):
        for b, (state_b, residue_b, codon_b) in enumerate(genetic_code):
            if hamming_distance(codon_a, codon_b) != 1:
                continue
            for nta, ntb in zip(codon_a, codon_b):
                if nta != ntb:
                    rate = nt_distn[ntb]
                    if nta + ntb in transitions:
                        rate *= kappa
            if residue_a != residue_b:
                rate *= omega
            Q.add_edge(a, b, weight=rate)

    # construct the stationary distribution
    weights = {}
    for i, (state, residue, codon) in enumerate(genetic_code):
        weights[i] = prod(nt_distn[nt] for nt in codon)
    distn = dict_distn(weights)

    # compute the expected syn and nonsyn rates for rescaling
    expected_syn_rate = 0.0
    expected_nonsyn_rate = 0.0
    for a, (state_a, residue_a, codon_a) in enumerate(genetic_code):
        for b, (state_b, residue_b, codon_b) in enumerate(genetic_code):
            if hamming_distance(codon_a, codon_b) != 1:
                continue
            if (a in distn) and Q.has_edge(a, b):
                rate = distn[a] * Q[a][b]['weight']
                if residue_a == residue_b:
                    expected_syn_rate += rate
                else:
                    expected_nonsyn_rate += rate

    # rescale the rate matrix to taste
    if target_expected_rate is not None:
        expected_rate = expected_syn_rate + expected_nonsyn_rate
        scale = target_expected_rate / expected_rate
    elif target_expected_syn_rate is not None:
        scale = target_expected_syn_rate / expected_syn_rate
    else:
        raise Exception
    for sa, sb in Q.edges():
        Q[sa][sb]['weight'] *= scale

    # check nucleotide distribution
    nt_distn_dense = _density.dict_to_numpy_array(nt_distn)
    cmedbutil.assert_stochastic_vector(nt_distn_dense)

    # check time-reversible rate matrix invariants
    Q_dense = _density.rate_matrix_to_numpy_array(Q, nodelist=states)
    distn_dense = _density.dict_to_numpy_array(distn, nodelist=states)
    cmedbutil.assert_stochastic_vector(distn_dense)
    cmedbutil.assert_rate_matrix(Q_dense)
    cmedbutil.assert_equilibrium(Q_dense, distn_dense)
    cmedbutil.assert_detailed_balance(Q_dense, distn_dense)

    return Q, distn, state_to_residue, residue_to_part

