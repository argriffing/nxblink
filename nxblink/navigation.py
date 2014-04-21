"""
Functions related to navigation of the tree structure.

This module does not need to care about piecewise homogeneity of the process.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


def partition_nodes(T, edge_to_blen, edge_to_rate):
    """
    Partition nodes of a tree.

    The nodes are partitioned into equivalence classes,
    where two nodes are considered equivalent if they are connected
    by a branch of length exactly zero.

    Parameters
    ----------
    T : networkx DiGraph
        The tree.
    edge_to_blen : dict
        Maps directed edges of T to non-negative branch lengths.
    edge_to_rate : dict
        Maps directed edges of T to non-negative rate scaling factors.

    Returns
    -------
    components : list of list of nodes
        Each list contains the nodes of a connected component.

    """
    # Compute an undirected graph representing an equivalence relation.
    G = nx.Graph()
    G.add_nodes_from(T)
    for edge in T.edges():
        blen = edge_to_blen[edge]
        rate = edge_to_rate[edge]
        if (not blen) or (not rate):
            G.add_edge(*edge)

    # Return the connected components of the graph.
    return nx.connected_components(G)


def gen_segments(edge, node_to_tm, tracks):
    """
    Iterate over segments, tracking all track states.

    This is a helper function for sampling poisson events.
    On each segment, neither the background nor the foreground state changes.

    """
    va, vb = edge
    edge_tma = node_to_tm[va]
    edge_tmb = node_to_tm[vb]

    # Concatenate events from all tracks of interest.
    events = [ev for track in tracks for ev in track.events[edge]]

    # Construct tuples corresponding to sorted events or nodes.
    # No times should coincide.
    seq = [(ev.tm, ev.track, ev.sa, ev.sb) for ev in events]
    info_a = (edge_tma, None, None, None)
    info_b = (edge_tmb, None, None, None)
    seq = [info_a] + sorted(seq) + [info_b]

    # Initialize track states at the beginning of the edge.
    track_to_state = dict((t.name, t.history[va]) for t in tracks)

    # Iterate over segments of the edge.
    for segment in zip(seq[:-1], seq[1:]):
        info_a, info_b = segment
        tma, tracka, saa, sba = info_a
        tmb, trackb, sab, sbb = info_b
        blen = tmb - tma

        # Keep the state of each track up to date.
        if tracka is not None:
            tm, track, sa, sb = info_a
            name = tracka.name
            if track_to_state[name] != sa:
                raise Exception('incompatible transition: '
                        'current state on track %s is %s '
                        'but encountered a transition event from '
                        'state %s to state %s' % (
                            name, track_to_state[name], sa, sb))
            track_to_state[name] = sb

        yield tma, tmb, track_to_state


def gen_context_segments(edge, node_to_tm, bg_tracks, fg_track):
    """
    Iterate over segments, tracking the background state.

    This is a helper function for sampling poisson events
    and for defining the uniformized transition probability matrices
    for foreground events in the forward-filtering-backward-sampling step.
    On each segment the background state is constant.

    Parameters
    ----------
    edge : node pair
        A pair of nodes from the directed rooted networkx tree.
    node_to_tm : dict
        Use this to get the times of the initial and final node of the edge.
    bg_tracks : collection of Trajectory objects
        Collection of relevant background trajectories.
    fg_track : Trajectory object
        Foreground trajectory.

    Returns
    -------
    contexts : generator
        Yields contextual segments over which the background state is constant.

    """
    # Get the initial and final times of the edge.
    va, vb = edge
    edge_tma = node_to_tm[va]
    edge_tmb = node_to_tm[vb]

    # Concatenate all events on this edge.
    tracks = bg_tracks + [fg_track]
    events = [ev for track in tracks for ev in track.events[edge]]

    # Construct tuples corresponding to sorted events or nodes.
    # No times should coincide.
    seq = [(ev.tm, ev) for ev in events]
    seq = [(edge_tma, None)] + sorted(seq) + [(edge_tmb, None)]

    # Initialize track states at the beginning of the edge.
    fg_state = fg_track.history[va]
    bg_track_to_state = dict((t.name, t.history[va]) for t in bg_tracks)

    # Initialize the first context segment info.
    ctx_initial_tm = edge_tma
    ctx_initial_fg_state = fg_state
    ctx_fg_events = []

    # Iterate over segments of the edge.
    for segment in zip(seq[:-1], seq[1:]):

        # Unpack the segment endpoints.
        (tma, eva), (tmb, evb) = segment

        # If the initial endpoint of the segment
        # is a transition event on the background track,
        # then we have completed a context segment.
        if eva is not None and eva.track is not fg_track:

            # Yield the current context segment and initialize the next one.
            yield (
                    ctx_initial_tm,
                    tma,
                    dict(bg_track_to_state), 
                    ctx_initial_fg_state,
                    ctx_fg_events)
            ctx_initial_tm = tma
            ctx_initial_fg_state = fg_state
            ctx_fg_events = []

            # Update the background track states.
            if eva.sa == eva.sb:
                raise Exception('background self transition')
            if eva.sa != bg_track_to_state[eva.track.name]:
                raise Exception('incompatible background transition')
            bg_track_to_state[eva.track.name] = eva.sb

        # If the initial endpoint of the segment
        # is a foreground transition event
        # then update the foreground state
        # and add the foreground event to the list.
        if eva is not None and eva.track is fg_track:
            if eva.sa == eva.sb:
                raise Exception('foreground self transition')
            if eva.sa != fg_state:
                raise Exception('incompatible foreground transition')
            fg_state = eva.sb
            ctx_fg_events.append(eva)

        # If this is the last segment, then yield the current context segment.
        if evb is None:
            if tmb != edge_tmb:
                raise Exception('found an unexpected event time')
            yield (
                    ctx_initial_tm,
                    tmb,
                    dict(bg_track_to_state), 
                    ctx_initial_fg_state,
                    ctx_fg_events)

