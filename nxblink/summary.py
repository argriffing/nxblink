"""
Functions to compute summaries of the tracks of the blinking process.

In particular, compute statistics that additively contribute to
the expected log likelihood.

"""
from __future__ import division, print_function, absolute_import

import math

from .navigation import gen_segments


__all__ = [
        'get_ell_init_contrib',
        'get_ell_dwell_contrib',
        'get_ell_trans_contrib',
        ]


def get_ell_init_contrib(
        root,
        primary_distn, blink_distn,
        primary_track, tolerance_tracks, primary_to_tol):
    """
    """
    ll_init = 0
    primary_state = primary_track.history[root]
    tol_name = primary_to_tol[primary_state]
    ll_init += math.log(primary_distn[primary_state])
    for tol_track in tolerance_tracks:
        if tol_track.name != tol_name:
            tol_state = tol_track.history[root]
            ll_init += math.log(blink_distn[tol_state])
    return ll_init


def get_ell_dwell_contrib(
        T, root, node_to_tm,
        Q_primary, Q_blink, Q_meta,
        primary_track, tolerance_tracks, primary_to_tol):
    """

    Returns
    -------
    edge_to_contrib : dict
        maps directed edges to dwell time contribution to expected log lhood

    """
    tracks = [primary_track] + tolerance_tracks
    edge_to_contrib = {}
    for edge in T.edges():
        va, vb = edge
        contrib = 0
        for tma, tmb, track_to_state in gen_segments(
                edge, node_to_tm, tracks):
            primary_state = track_to_state[primary_track.name]
            primary_tol_name = primary_to_tol[primary_state]

            # Get the context-dependent rate.
            rate = 0
            for tol_track in tolerance_tracks:
                tol_name = tol_track.name
                tol_state = track_to_state[tol_name]

                if primary_tol_name == tol_name:
                    # The tol_state must be True and cannot change to False.
                    # The primary state can change to its synonymous states.
                    if Q_meta.has_edge(primary_state, tol_name):
                        rate += Q_meta[primary_state][tol_name]['weight']
                else:
                    # The tol_state can be True or False
                    # and is free to change from one to the other.
                    # If the tol_state is True then the primary state
                    # can change to any of its neighbors in this tol_name.
                    if tol_state:
                        rate += Q_blink[True][False]['weight']
                        if Q_meta.has_edge(primary_state, tol_name):
                            rate += Q_meta[primary_state][tol_name]['weight']
                    else:
                        rate += Q_blink[False][True]['weight']

            # Accumulate the contribution of the segment.
            contrib += -(rate * (tmb - tma))

        # Store the edge contribution.
        edge_to_contrib[edge] = contrib

    # Return the map of contributions from edges.
    return edge_to_contrib


def get_ell_trans_contrib(
        T, root,
        Q_primary, Q_blink,
        primary_track, tolerance_tracks):
    """
    """
    tracks = [primary_track] + tolerance_tracks
    edge_to_contrib = {}
    for edge in T.edges():
        va, vb = edge
        contrib = 0
        for track in tracks:
            for ev in track.events[edge]:
                if ev.track is primary_track:
                    if Q_primary.has_edge(ev.sa, ev.sb):
                        rate = Q_primary[ev.sa][ev.sb]['weight']
                    else:
                        raise Exception
                else:
                    if ev.sa == False and ev.sb == True:
                        rate = Q_blink[ev.sa][ev.sb]['weight']
                    elif ev.sa == True and ev.sb == False:
                        rate = Q_blink[ev.sa][ev.sb]['weight']
                    else:
                        raise Exception
                contrib += math.log(rate)
        edge_to_contrib[edge] = contrib
    return edge_to_contrib

