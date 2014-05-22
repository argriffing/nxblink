"""
Expectation maximization for the codon blinking model.

The expectation step is mainly accomplished through the
CTBN Rao-Teh sampling of path histories, summarized
through the nxblink.summary.Summary object.
This module provides maximization_step().

The model has the following parameters:
    - kappa
    - omega
    - P(A)
    - P(C)
    - P(G)
    - P(T)
    - blink rate on
    - blink rate off
    - branch-specific rate scaling factor for each branch

The idea is to use log transformed parameter in the search,
with the dense matrix representation of the codon rate matrix
and its dense equilibrium distribution.
The maximization step can be done using the trust-ncg method
of the scipy.optimize minimization with gradient and hessian
provided with automatic differentiation through algopy.

If the sample average approximation of expectation finds an edge
with no sampled event, then set the edge-specific rate to zero
rather than trying to estimate it numerically in the maximization step.

"""
from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np
from scipy.optimize import minimize

import algopy
from algopy import log, exp, square

import nxblink
from nxblink.summary import Summary
from nxblink.em import (
        get_ll_dwell, get_ll_trans, get_ll_root, get_expected_rate)

# algopy boilerplate
def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))


# algopy boilerplate
def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


def hamming_distance(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def objective(summary, edges, genetic_code, log_params):
    """
    Compute the objective function to minimize.

    Computes negative log likelihood of augmented process,
    penalized according to violation of the constraint that
    mutational nucleotide probabilities should sum to 1.
    The log_params input may be any manner of exotic array.

    """
    # compute the number of primary states
    nprimary = len(genetic_code)

    # unpack the parameters
    unpacked, penalty = unpack_params(log_params)
    kappa, omega, A, C, G, T, blink_on, blink_off, edge_rates = unpacked

    # construct the unnormalized pre-rate-matrix from the parameters
    pre_Q = get_pre_Q(genetic_code, kappa, omega, A, C, G, T)
    distn = get_distn(genetic_code, kappa, omega, A, C, G, T)

    # compute expected negative log likelihood
    ll_root = get_ll_root(summary, distn, blink_on, blink_off)
    ll_dwell = get_ll_dwell(summary, pre_Q, distn, blink_on, blink_off,
            edges, edge_rates)
    ll_trans = get_ll_trans(summary, pre_Q, distn, blink_on, blink_off,
            edges, edge_rates)
    neg_ll = -(ll_root + ll_dwell + ll_trans)
    return neg_ll


def maximization_step(summary, genetic_code,
        kappa, omega, A, C, G, T, blink_on, blink_off, edge_to_rate):
    """
    This is the maximization step of EM parameter estimation.

    Parameters
    ----------
    summary : nxblink.summary.Summary object
        Per-edge sample average approximations of expectations,
        across multiple aligned sites and multiple samples per site.
    genetic_code : triples
        Information about the genetic code,
        for the purposes of constructing the rate matrix.
    kappa, omega, A, C, G, T, blink_on, blink_off: floats
        Initial parameter estimates.
    edge_to_rate : dict
        Initial parameter estimates of edge-specific rates.

    """
    # extract the edges and the corresponding edge rates from the dict
    edges, edge_rates = zip(*edge_to_rate.items())

    # pack the initial parameter estimates into a point estimate
    x0 = pack_params(kappa, omega, A, C, G, T, blink_on, blink_off, edge_rates)

    # stash the summary, edges, and genetic code into the rate matrix
    f = partial(objective, summary, edges, genetic_code)
    g = partial(eval_grad, f)
    h = partial(eval_hess, f)

    # maximize the log likelihood
    #result = minimize(f, x0, method='trust-ncg', jac=g, hess=h)
    result = minimize(f, x0, method='L-BFGS-B', jac=g)

    # unpack the parameters
    unpacked, penalty = unpack_params(result.x)
    kappa, omega, A, C, G, T, blink_on, blink_off, edge_rates = unpacked

    # possibly mention the negative log likelihood and the penalty
    print(result)
    print('penalty:', penalty)

    # convert the edge rates back into a dict
    edge_to_rate = dict(zip(edges, edge_rates))

    # return the parameter estimates
    return kappa, omega, A, C, G, T, blink_on, blink_off, edge_to_rate


def pack_params(kappa, omega, A, C, G, T, blink_on, blink_off, edge_rates):
    """
    This function is mainly for constructing initial parameter values.

    Returns log params suitable as an initial vector
    for scipy.optimize.minimize methods.

    """
    global_params = [kappa, omega, A, C, G, T, blink_on, blink_off]
    params = np.array(list(global_params) + list(edge_rates))
    log_params = log(params)
    return log_params


def unpack_params(log_params):
    """
    Unpack the parameters.

    This function also enforces the simplex constraint
    on the mutational nucleotide distribution,
    and it computes a penalty corresponding to violation of this constraint.

    """
    # undo the log transformation of the parameters
    params = exp(log_params)

    # unpack the parameters
    kappa, omega = params[0:2]
    A, C, G, T = params[2:6]
    blink_on, blink_off = params[6:8]
    edge_rates = params[8:]

    # normalize the nucleotide probability distribution and compute a penalty
    nt_prob_sum = A + C + G + T
    A = A / nt_prob_sum
    C = C / nt_prob_sum
    G = G / nt_prob_sum
    T = T / nt_prob_sum
    penalty = square(log(nt_prob_sum))

    # return unpacked parameters and the penalty
    unpacked = (kappa, omega, A, C, G, T, blink_on, blink_off, edge_rates)
    return unpacked, penalty


def get_pre_Q(genetic_code, kappa, omega, A, C, G, T):
    """
    Compute a Muse-Gaut 1994 pre-rate-matrix without any particular scale.

    The dtype of the returned object should be like that of its parameters,
    so that exotic array types like algopy arrays are supported.
    The diagonal should consist of zeros because it is a pre-rate-matrix.
    Subsequent function calls can deal with expected rate.

    """
    # define the states and initialize the pre-rate-matrix
    nstates = len(genetic_code)
    states = range(nstates)
    Q = algopy.zeros((nstates, nstates), dtype=kappa)

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
    transitions = ('AG', 'GA', 'CT', 'TC')
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
            Q[a, b] = rate

    # return the pre-rate-matrix
    return Q


def get_distn(genetic_code, kappa, omega, A, C, G, T):
    """

    """
    # initialize the unweighted distribution
    nstates = len(genetic_code)
    weights = algopy.ones(nstates, dtype=kappa)

    nt_distn = {
            'A' : A,
            'C' : C,
            'G' : G,
            'T' : T,
            }

    # construct the unnormalized distribution
    for i, (state, residue, codon) in enumerate(genetic_code):
        for nt in codon:
            weights[i] = weights[i] * nt_distn[nt]

    # return the normalized distribution
    distn = weights / weights.sum()
    return distn
