"""
Functions specifying the blinking model.

Uniformization, sampling, inference, summary statistics, and likelihood
are out of scope for this module.

This module does not need to care about piecewise homogeneity of the process.

"""
from __future__ import division, print_function, absolute_import

from itertools import product
from collections import defaultdict

import networkx as nx


__all__ = ['get_interaction_map', 'get_Q_blink', 'get_Q_meta']


def _get_primary_interaction_map(primary_to_part):
    """
    Helper function for get_interaction_map.

    For the primary foreground track, for each tolerance background track,
    get the map from each background track state to the set of allowed
    foreground track states.

    """
    # initialize some structures
    all_primary_states = set(primary_to_part)
    part_to_primary_set = defaultdict(set)
    for primary, part in primary_to_part.items():
        part_to_primary_set[part].add(primary)

    # construct the interaction map
    interaction = {}
    interaction['PRIMARY'] = {}
    for part, primary_set in part_to_primary_set.items():
        d = {
                True : all_primary_states,
                False : all_primary_states - primary_set}
        interaction['PRIMARY'][part] = d
    return interaction


def _get_tolerance_interaction_map(primary_to_part):
    """
    Helper function for get_interaction_map.

    For each tolerance foreground track, for the primary background track,
    get the map from each background track state to the set of allowed
    foreground track states.

    """
    # initialize some structures
    all_primary_states = set(primary_to_part)
    part_to_primary_set = defaultdict(set)
    for primary, part in primary_to_part.items():
        part_to_primary_set[part].add(primary)

    # construct the interaction map
    interaction = {}
    for part, primary_set in part_to_primary_set.items():
        interaction[part] = {}
        d = {}
        for primary in all_primary_states:
            if primary in primary_set:
                d[primary] = {True}
            else:
                d[primary] = {False, True}
        interaction[part]['PRIMARY'] = d
    return interaction


def get_interaction_map(primary_to_part):
    interaction = {}
    interaction.update(_get_primary_interaction_map(primary_to_part))
    interaction.update(_get_tolerance_interaction_map(primary_to_part))
    return interaction


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

