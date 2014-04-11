"""
Rao-Teh CTBN sampling, specialized to a class of blinking models.

"""
from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np
import networkx as nx

import nxmctree
from nxmctree.sampling import sample_history

from .util import (
        set_or_confirm_history_state, get_total_rates, get_omega,
        get_uniformized_P_nx)
from .graphutil import get_edge_tree, partition_nodes
from .navigation import MetaNode, gen_meta_segments
from .trajectory import Event
from .poisson import sample_primary_poisson_events, sample_blink_poisson_events


def update_track_data_for_zero_blen(T, edge_to_blen, edge_to_rate, tracks):
    """
    Update track data, accounting for branches with length zero.

    """
    iso_node_lists = list(partition_nodes(T, edge_to_blen, edge_to_rate))
    for track in tracks:
        for pool in iso_node_lists:

            # initialize the fset associated with the pool.
            pool_fset = None
            for v in pool:
                if pool_fset is None:
                    pool_fset = set(track.data[v])
                else:
                    pool_fset &= track.data[v]
            
            # check that the pool fset is not empty
            if not pool_fset:
                raise Exception('no data for the node pool '
                        'consisting of ' + str(pool))

            # set each vertex data fset to the pool data fset.
            for v in pool:
                track.data[v] = set(pool_fset)


def init_blink_history(T, track):
    """
    Initial blink history is True where consistent with the data.

    This defines the blink states at nodes of the tree T.

    """
    # initialize the track history according to the data
    for v in T:
        if True in track.data[v]:
            blink_state = True
        elif False in track.data[v]:
            blink_state = False
        else:
            raise Exception('neither True nor False is feasible')
        track.history[v] = blink_state


def init_complete_blink_events(T, node_to_tm, track):
    """
    Add actual blink transitions near each end of the edge.

    These events are designed so that in the middle 2/3 of each edge
    every primary state is tolerated according to the blinking process.

    """
    for edge in T.edges():
        va, vb = edge
        sa = track.history[va]
        sb = track.history[vb]
        edge_tma = node_to_tm[va]
        edge_tmb = node_to_tm[vb]
        blen = edge_tmb - edge_tma
        events = []
        if blen:
            if not sa:
                tma = edge_tma + blen * np.random.uniform(0, 1/3)
                events.append(Event(track=track, tm=tma, sa=sa, sb=True))
            if not sb:
                tmb = edge_tma + blen * np.random.uniform(2/3, 1)
                events.append(Event(track=track, tm=tmb, sa=True, sb=sb))
        track.events[edge] = events


def init_incomplete_primary_events(T, node_to_tm, edge_to_rate,
        primary_track, diameter):
    """
    This function assigns potential transition times but not the states.

    Parameters
    ----------
    T : nx tree
        tree
    edge_to_rate : dict
        x
    node_to_tm : dict
        maps nodes to times
    primary_track : Trajectory
        current state of the track
    diameter : int
        directed unweighted diameter of the primary transition rate matrix

    Returns
    -------
    ev_to_P_nx : dict
        Map from new poisson events to corresponding
        uniformized transition matrices.

    """
    ev_to_P_nx = {}
    for edge in T.edges():
        
        # Make a plausible transition probability matrix.
        # It does not need to be carefully constructed,
        # because we are using it only to make an initial feasible trajectory.
        Q_local = primary_track.Q_nx.copy()
        for sa, sb in Q_local.edges():
            Q_local[sa][sb]['weight'] *= edge_to_rate[edge]
        local_rates = get_total_rates(Q_local)
        local_omega = get_omega(local_rates, 2)
        P_local = get_uniformized_P_nx(Q_local, local_rates, local_omega)

        # Make the events.
        va, vb = edge
        edge_tma = node_to_tm[va]
        edge_tmb = node_to_tm[vb]
        blen = edge_tmb - edge_tma
        events = []
        if blen:
            times = edge_tma + blen * np.random.uniform(
                    low=1/3, high=2/3, size=diameter)
            events = [Event(track=primary_track, tm=tm) for tm in times]
        primary_track.events[edge] = events

        # Record the transition matrix associated with each event.
        for ev in events:
            ev_to_P_nx[ev] = P_local

    # Return the map that associates a transition probability matrix
    # to each of the events.
    return ev_to_P_nx


def get_node_to_meta(T, root, node_to_tm, fg_track):
    """
    Create meta nodes representing structural nodes in the tree.

    This is a helper function.

    """
    P_nx_identity = fg_track.P_nx_identity
    node_to_meta = {}
    for v in T:
        f = partial(set_or_confirm_history_state, fg_track.history, v)
        fset = fg_track.data[v]
        m = MetaNode(track=None, P_nx=P_nx_identity,
                set_sa=f, set_sb=f, fset=fset,
                tm=node_to_tm[v])
        node_to_meta[v] = m
    return node_to_meta


def resample_using_meta_node_tree(root, meta_node_tree, mroot,
        fg_track, node_to_data_lmap):
    """
    Resample the states of the foreground process using the meta node tree.

    This is a helper function.

    """
    # Build the tree whose vertices are edges of the meta node tree.
    meta_edge_tree, meta_edge_root = get_edge_tree(meta_node_tree, mroot)

    # Create the map from edges of the meta edge tree
    # to primary state transition matrices.
    edge_to_P = {}
    for pair in meta_edge_tree.edges():
        (ma, mb), (mb2, mc) = pair
        if mb != mb2:
            raise Exception('incompatibly constructed meta edge tree')
        edge_to_P[pair] = mb.P_nx

    # Use nxmctree to sample a history on the meta edge tree.
    root_data_fset = fg_track.data[root]
    node_to_data_lmap[meta_edge_root] = dict((s, 1) for s in root_data_fset)
    meta_edge_to_sampled_state = sample_history(
            meta_edge_tree, edge_to_P, meta_edge_root,
            fg_track.prior_root_distn, node_to_data_lmap)

    # Use the sampled history to update the primary history at structural nodes
    # and to update the primary event transitions.
    for meta_edge in meta_edge_tree:
        ma, mb = meta_edge
        state = meta_edge_to_sampled_state[meta_edge]
        if ma is not None:
            ma.set_sb(state)
        if mb is not None:
            mb.set_sa(state)


def sample_blink_transitions(T, root, node_to_tm, edge_to_rate, ev_to_P_nx,
        fg_track, bg_tracks, bg_to_fg_fset, Q_meta):
    """
    Sample the history (nodes to states) and the events (edge to event list).

    This function depends on a foreground track
    and a collection of contextual background tracks.

    """
    P_nx_identity = fg_track.P_nx_identity
    node_to_meta = get_node_to_meta(T, root, node_to_tm, fg_track)
    mroot = node_to_meta[root]

    # Build the tree whose vertices are meta nodes,
    # and map edges of this tree to sets of feasible foreground states,
    # accounting for data at structural nodes and background context
    # along edge segments.
    #
    # Also create the map from edges of this tree
    # to sets of primary states not directly contradicted by data or context.
    #
    meta_node_tree = nx.DiGraph()
    node_to_data_lmap = dict()
    for edge in T.edges():

        for segment, bg_track_to_state, fg_allowed in gen_meta_segments(
                edge, node_to_meta, ev_to_P_nx,
                fg_track, bg_tracks, bg_to_fg_fset):
            ma, mb = segment

            # Get the set of states allowed by data and background interaction.
            fsets = []
            for m in segment:
                if m.fset is not None:
                    fsets.append(m.fset)
            for name, state in bg_track_to_state.items():
                fsets.append(bg_to_fg_fset[name][state])
            fg_allowed = set.intersection(*fsets)

            # For each possible foreground state,
            # use the states of the background tracks and the data
            # to determine foreground feasibility
            # and possibly a multiplicative rate penalty.
            lmap = dict()

            # The lmap has nontrivial penalties
            # depending on both the background (primary) track state
            # and the proposed foreground blink state.
            pri_track = bg_tracks[0]
            pri_state = bg_track_to_state[pri_track.name]
            if False in fg_allowed:
                lmap[False] = 1
            # The blink state choice of True should be penalized
            # according to the sum of rates from the current
            # primary state to primary states controlled by
            # the proposed foreground track.
            if True in fg_allowed:
                if Q_meta.has_edge(pri_state, fg_track.name):
                    rate_sum = Q_meta[pri_state][fg_track.name]['weight']
                    amount = rate_sum * edge_to_rate[edge] * (mb.tm - ma.tm)
                    lmap[True] = np.exp(-amount)
                else:
                    lmap[True] = 1

            # Map the segment to the lmap.
            # Segments will be nodes of the tree whose history will be sampled.
            node_to_data_lmap[segment] = lmap

            # Add the meta node to the meta node tree.
            meta_node_tree.add_edge(ma, mb)

    # Resample the states using the meta tree.
    resample_using_meta_node_tree(root, meta_node_tree, mroot,
            fg_track, node_to_data_lmap)


def sample_primary_transitions(T, root, node_to_tm, ev_to_P_nx,
        fg_track, bg_tracks, bg_to_fg_fset):
    """
    Sample the history (nodes to states) and the events (edge to event list).

    This function depends on a foreground track
    and a collection of contextual background tracks.

    """
    P_nx_identity = fg_track.P_nx_identity
    node_to_meta = get_node_to_meta(T, root, node_to_tm, fg_track)
    mroot = node_to_meta[root]

    # Build the tree whose vertices are meta nodes,
    # and map edges of this tree to sets of feasible foreground states,
    # accounting for data at structural nodes and background context
    # along edge segments.
    #
    # Also create the map from edges of this tree
    # to sets of primary states not directly contradicted by data or context.
    #
    meta_node_tree = nx.DiGraph()
    node_to_data_lmap = dict()
    for edge in T.edges():

        for segment, bg_track_to_state, fg_allowed in gen_meta_segments(
                edge, node_to_meta, ev_to_P_nx,
                fg_track, bg_tracks, bg_to_fg_fset):
            ma, mb = segment

            # Get the set of states allowed by data and background interaction.
            fsets = []
            for m in segment:
                if m.fset is not None:
                    fsets.append(m.fset)
            for name, state in bg_track_to_state.items():
                fsets.append(bg_to_fg_fset[name][state])
            fg_allowed = set.intersection(*fsets)

            # Check feasibility.
            if not fg_allowed:
                raise Exception('no foreground state is allowed')

            # Use the states of the background blinking tracks,
            # together with fsets of the two meta nodes if applicable,
            # to define the set of feasible foreground states
            # at this segment.
            lmap = dict((s, 1) for s in fg_allowed)

            # Map the segment to the lmap.
            # Segments will be nodes of the tree whose history will be sampled.
            node_to_data_lmap[segment] = lmap

            # Add the meta node to the meta node tree.
            #print('adding segment', ma, mb)
            meta_node_tree.add_edge(ma, mb)

    # Resample the states using the meta tree.
    resample_using_meta_node_tree(root, meta_node_tree, mroot,
            fg_track, node_to_data_lmap)


def blinking_model_rao_teh(
        T, root, node_to_tm, edge_to_rate,
        Q_primary, Q_blink, Q_meta,
        primary_track, tolerance_tracks, interaction_map):
    """

    Parameters
    ----------
    T : x
        x
    root : x
        x
    node_to_tm : x
        x
    edge_to_rate : x
        x
    Q_primary : x
        x
    Q_blink : x
        x
    Q_meta : x
        x
    primary_track : hashable
        label of the primary track
    tolerance_tracks : collection of hashables
        labels of tolerance tracks
    interaction_map : dict
        x

    """
    # Initialize blink history and events.
    for track in tolerance_tracks:
        init_blink_history(T, track)
        init_complete_blink_events(T, node_to_tm, track)
        track.remove_self_transitions()

    # Initialize the primary trajectory with many incomplete events.
    diameter = nx.diameter(Q_primary)
    ev_to_P_nx = init_incomplete_primary_events(T, node_to_tm, edge_to_rate,
            primary_track, diameter)

    # print stuff for debugging...
    """
    tracks = [primary_track] + tolerance_tracks
    for track in tracks:
        for edge in T.edges():
            va, vb = edge
            print(track.name, va, vb, track.events[edge])
    """

    #
    # Sample the state of the primary track.
    sample_primary_transitions(T, root, node_to_tm, ev_to_P_nx,
            primary_track, tolerance_tracks, interaction_map['PRIMARY'])
    #
    # Remove self-transition events from the primary track.
    primary_track.remove_self_transitions()

    # Outer loop of the Rao-Teh-Gibbs sampler.
    while True:

        # add poisson events to the primary track
        ev_to_P_nx = {}
        for edge in T.edges():
            edge_rate = edge_to_rate[edge]
            edge_ev_to_P_nx = sample_primary_poisson_events(
                    edge, edge_rate, node_to_tm,
                    primary_track, tolerance_tracks, interaction_map['PRIMARY'])
            ev_to_P_nx.update(edge_ev_to_P_nx)
        # clear state labels for the primary track
        primary_track.clear_state_labels()
        # sample state transitions for the primary track
        sample_primary_transitions(T, root, node_to_tm, ev_to_P_nx,
                primary_track, tolerance_tracks, interaction_map['PRIMARY'])
        # remove self transitions for the primary track
        primary_track.remove_self_transitions()

        # Update each blinking track.
        for track in tolerance_tracks:
            name = track.name
            # add poisson events to this blink track
            ev_to_P_nx = {}
            for edge in T.edges():
                edge_rate = edge_to_rate[edge]
                edge_ev_to_P_nx = sample_blink_poisson_events(
                        edge, edge_rate, node_to_tm,
                        track, primary_track, interaction_map[name])
                ev_to_P_nx.update(edge_ev_to_P_nx)
            # clear state labels for this blink track
            track.clear_state_labels()
            # sample state transitions for this blink track
            sample_blink_transitions(
                    T, root, node_to_tm,
                    edge_to_rate, ev_to_P_nx,
                    track, [primary_track], interaction_map[name], Q_meta)
            # remove self transitions for this blink track
            track.remove_self_transitions()

        """
        # Summarize the sample.
        expected_on = 0
        expected_off = 0
        for track in tolerance_tracks:
            for edge in T.edges():
                for ev in track.events[edge]:
                    transition = (ev.sa, ev.sb)
                    if transition == (False, True):
                        expected_on += 1
                    elif transition == (True, False):
                        expected_off += 1

        yield expected_on, expected_off
        """

        # Yield the track states.
        yield primary_track, tolerance_tracks

