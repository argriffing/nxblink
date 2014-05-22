"""
Functions related to expectation maximization.

Most of the expectation step is in the nxblink.summary module.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import algopy
from algopy import exp, log


def get_ll_dwell(summary, pre_Q, distn, blink_on, blink_off):
    """
    Get dwell-related contribution to log likelihood.

    Parameters
    ----------
    summary : Summary object
        Summary of blinking process trajectories.
    pre_Q : dense possibly exotic array
        Unscaled primary process pre-rate-matrix
        whose diagonal entries are zero.
    distn : dense possibly exotic array
        Primary state distribution.
    blink_on : float, or exotic float-like with derivatives information
        blink rate on
    blink_off : float, or exotic float-like with derivatives information
        blink rate off

    Returns
    -------
    ll : float, or exotic float-like with derivatives information
        log likelihood contribution from dwell

    """
    # compute the expected rate of the unnormalized pre-rate-matrix
    expected_rate = get_expected_rate(pre_Q, distn)

    # initialize expected log likelihood using the right data type
    ll = algopy.zeros(1, dtype=distn)[0]

    # contributions per edge
    for edge, edge_rate in zip(edges, edge_rates):

        # extract edge summaries for dwell times
        pri_dwell = summary.edge_to_pri_dwell[edge]
        off_xon_dwell = summary.edge_to_off_xon_dwell[edge]
        xon_off_dwell = summary.edge_to_xon_off_dwell[edge]

        # compute a scaled sum of dwell times
        ll_dwell = algopy.zeros(1, dtype=distn)[0]

        # contribution of dwell times associated with primary transitions
        for sa in pri_dwell:
            for sb in pri_dwell[sa]:
                elapsed = pri_dwell[sa][sb]['weight']
                if elapsed:
                    ll_dwell = ll_dwell + elapsed * pre_Q[sa, sb]

        # contribution of dwell times associated with tolerance transitions
        if off_xon_dwell:
            ll_dwell = ll_dwell + off_xon_dwell * blink_on
        if xon_off_dwell:
            ll_dwell = ll_dwell + xon_off_dwell * blink_off

        # Add the dwell time contribution,
        # adjusted for edge-specific rate and pre-rate-matrix scaling.
        ll = ll - ll_dwell * (edge_rate / expected_rate)

    # return expected log likelihood contribution of dwell
    return ll


def get_ll_trans(summary, pre_Q, distn, blink_on, blink_off):
    """

    Parameters
    ----------
    summary : Summary object
        Summary of blinking process trajectories.
    pre_Q : dense possibly exotic array
        Unscaled primary process pre-rate-matrix
        whose diagonal entries are zero.
    distn : dense possibly exotic array
        Primary state distribution.
    blink_on : float, or exotic float-like with derivatives information
        blink rate on
    blink_off : float, or exotic float-like with derivatives information
        blink rate off

    Returns
    -------
    ll : float, or exotic float-like with derivatives information
        log likelihood contribution from transitions

    """
    # compute the expected rate of the unnormalized pre-rate-matrix
    expected_rate = get_expected_rate(pre_Q, distn)

    # initialize expected log likelihood using the right data type
    ll = algopy.zeros(1, dtype=distn)[0]

    # contributions per edge
    for edge, edge_rate in zip(edges, edge_rates):

        # extract edge summaries for transitions
        pri_trans = summary.edge_to_pri_trans[edge]
        off_xon_trans = summary.edge_to_off_xon_trans[edge]
        xon_off_trans = summary.edge_to_xon_off_trans[edge]

        # Initialize the total number of all types of transitions.
        # This is for making adjustments according to the edge rate
        # and the expected rate of the pre-rate-matrix.
        transition_count_sum = 0

        # contribution of primary transition summary to expected log likelihood
        for sa in pri_trans:
            for sb in pri_trans[sa]:
                count = pri_trans[sa][sb]['weight']
                transition_count_sum += count
                if count:
                    ell = ell + count * log(pre_Q[sa, sb])

        # contribution of blink transition summary to expected log likelihood
        if off_xon_trans:
            transition_count_sum += off_xon_trans
            ll = ll + off_xon_trans * log(blink_on)
        if xon_off_trans:
            transition_count_sum += xon_off_trans
            ll = ll + xon_off_trans * log(blink_off)

        # add the adjustment for edge-specific rate and pre-rate-matrix scaling
        if transition_count_sum:
            ll = ll + transition_count_sum * log(edge_rate / expected_rate)

    # return negative expected log likelihood contribution due to transitions
    return ll


def get_ll_root(summary, pre_Q, distn, blink_on, blink_off):
    """

    Parameters
    ----------
    summary : Summary object
        Summary of blinking process trajectories.
    distn : dense possibly exotic array
        Primary state distribution.
    blink_on : float, or exotic float-like with derivatives information
        blink rate on
    blink_off : float, or exotic float-like with derivatives information
        blink rate off

    Returns
    -------
    ll : float, or exotic float-like with derivatives information
        log likelihood contribution from root state

    """
    # construct the blink distribution with the right data type
    blink_distn = algopy.zeros(2, dtype=distn)
    blink_distn[0] = blink_off / (blink_on + blink_off)
    blink_distn[1] = blink_on / (blink_on + blink_off)

    # initialize expected log likelihood using the right data type
    ll = algopy.zeros(1, dtype=distn)[0]

    # root primary state contribution to expected log likelihood
    obs = algopy.zeros_like(distn)
    for state, count in summary.root_pri_to_count.items():
        if count:
            ll = ll + count * log(distn[state])

    # root blink state contribution to expected log likelihood
    if summary.root_off_count:
        ll = ll + summary.root_off_count * log(blink_distn[0])
    if summary.root_xon_count:
        ll = ll + summary.root_xon_count * log(blink_distn[1])

    # return expected log likelihood contribution of root
    return ll
