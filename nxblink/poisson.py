"""
Functions related to sampling the poisson events.

This module cares about piecewise homogeneity of the process.

"""
from __future__ import division, print_function, absolute_import

import random

import numpy as np
import networkx as nx

from .util import get_total_rates, get_omega, get_uniformized_P_nx
from .navigation import gen_context_segments
from .trajectory import Event


__all__ = ['sample_primary_poisson_events', 'sample_blink_poisson_events']


def _poisson_helper(track, rate, tma, tmb):
    """
    Sample poisson events on a segment.

    Parameters
    ----------
    track : Trajectory
        trajectory object for which the poisson events should be sampled
    rate : float
        poisson rate of events
    tma : float
        initial segment time
    tmb : float
        final segment time

    Returns
    -------
    events : list
        list of event objects

    """
    blen = tmb - tma
    nevents = np.random.poisson(rate * blen)
    times = np.random.uniform(low=tma, high=tmb, size=nevents)
    events = []
    for tm in times:
        ev = Event(track=track, tm=tm)
        events.append(ev)
    return events


def _sample_poisson_events(edge, edge_rate, node_to_tm,
        bg_tracks, fg_track, bg_to_fg_fset, use_local_rates=True):
    """
    A helper function for a resampling step.
    
    This step samples incomplete foreground poisson events,
    and it also uses the uniformization rate to define the uniformized
    transition probability matrices for the new foreground poisson events
    and also for the old foreground transition events.

    """
    ev_to_P_nx = {}
    poisson_events = []
    for info in gen_context_segments(edge, node_to_tm, bg_tracks, fg_track):

        # Unpack the context segment info.
        ctx_tma, ctx_tmb, bg_track_to_state, initial_fg_state, fg_events = info

        # Get the set of foreground states allowed by the background.
        fsets = []
        for bg_track in bg_tracks:
            bg_state = bg_track_to_state[bg_track.name]
            fsets.append(bg_to_fg_fset[bg_track.name][bg_state])
        fg_allowed = set.intersection(*fsets)

        if use_local_rates:

            # Get the local transition rate matrix determined by background.
            # This uses the edge-specific rate scaling factor.
            Q_local = nx.DiGraph()
            for s in fg_track.Q_nx:
                Q_local.add_node(s)
            for sa, sb in fg_track.Q_nx.edges():
                if sb in fg_allowed:
                    rate = edge_rate * fg_track.Q_nx[sa][sb]['weight']
                    Q_local.add_edge(sa, sb, weight=rate)

            # Compute the total local rates.
            local_rates = get_total_rates(Q_local)
            local_omega = get_omega(local_rates, 2)

            # Define a locally uniformized transition probability matrix.
            # This transition matrix will be used for all foreground
            # poisson events and all foreground transition events
            # within the current context.
            P_local = get_uniformized_P_nx(Q_local, local_rates, local_omega)

        else:

            global_rates = fg_track.total_rates
            global_omega = fg_track.omega
            P_local = fg_track.P_nx

        # Iterate over foreground segments within the background segment.
        # Add poisson events.
        fg_state = initial_fg_state
        seq = [(ev.tm, ev) for ev in fg_events]
        seq = [(ctx_tma, None)] + seq + [(ctx_tmb, None)]
        for (tma, eva), (tmb, evb) in zip(seq[:-1], seq[1:]):

            # Update the foreground state
            # and map the foreground event to the local transition matrix.
            if eva is not None:
                fg_state = eva.sb
                ev_to_P_nx[eva] = P_local

            if use_local_rates:

                # Compute the poisson rate for this foreground state segment
                # using the local uniformization rate that depends on the
                # background context.
                poisson_rate = local_omega
                if fg_state in local_rates:
                    poisson_rate -= local_rates[fg_state]

            else:

                poisson_rate = global_omega
                if fg_state in global_rates:
                    poisson_rate -= global_rates[fg_state]

                poisson_rate *= edge_rate


            # Sample some poisson events on the segment.
            # Map the events to the local transition matrix.
            segment_events = _poisson_helper(fg_track, poisson_rate, tma, tmb)
            for ev in segment_events:
                ev_to_P_nx[ev] = P_local
            poisson_events.extend(segment_events)

    # Add the poisson events into the list of foreground
    # track events for this edge.
    fg_track.events[edge].extend(poisson_events)

    # Return the map that associates a transition probability matrix
    # to each foreground event on this edge.
    return ev_to_P_nx


def sample_primary_poisson_events(edge, edge_rate, node_to_tm,
        primary_track, blink_tracks, blink_to_primary_fset):
    """
    Sample poisson events on an edge of the primary codon-like trajectory.

    This function is a specialization of an earlier function
    named sample_poisson_events which had intended to not care about
    primary vs. blinking tracks except through their roles as
    foreground vs. background tracks.
    Some specific details of our model causes this aggregation
    to not work so well in practice, but such a unification will make more
    sense when a fully general CTBN sampler is implemented.

    Parameters
    ----------
    edge : x
        x
    edge_rate : float
        An edge-specific rate scaling factor.
    node_to_tm : x
        x
    primary_track : x
        x
    blink_tracks : x
        x
    blink_to_primary_fset : x
        x

    Returns
    -------
    ev_to_P_nx : dict
        Map from new poisson events to corresponding
        uniformized transition matrices.

    """
    fg_track = primary_track
    bg_tracks = blink_tracks
    bg_to_fg_fset = blink_to_primary_fset
    return _sample_poisson_events(edge, edge_rate, node_to_tm,
            bg_tracks, fg_track, bg_to_fg_fset,
            use_local_rates=True)


def sample_blink_poisson_events(edge, edge_rate, node_to_tm,
        foreground_blink_track, primary_track, bg_to_fg_fset):
    """

    Parameters
    ----------
    edge : x
        x
    edge_rate : float
        An edge-specific rate scaling factor.
    node_to_tm : dict
        x
    foreground_blink_track : x
        x
    primary_track : x
        x
    bg_to_fg_fset : x
        x

    """
    fg_track = foreground_blink_track
    bg_tracks = [primary_track]
    return _sample_poisson_events(edge, edge_rate, node_to_tm,
            bg_tracks, fg_track, bg_to_fg_fset,
            use_local_rates=True)
