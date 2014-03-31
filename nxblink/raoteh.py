"""
Rao-Teh CTBN sampling, specialized to a class of blinking models.

"""
from __future__ import division, print_function, absolute_import

from functools import partial

import networkx as nx

import nxmctree
from nxmctree.sampling import sample_history

from .graphutil import get_edge_tree
from .util import (
        set_or_confirm_history_state, get_total_rates, get_omega,
        get_uniformized_P_nx)
from .navigation import MetaNode, gen_meta_segments


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


def sample_blink_transitions(T, root, node_to_tm,
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
                edge, node_to_meta, fg_track, bg_tracks, bg_to_fg_fset):
            ma, mb = segment

            # Update the ma transition matrix if it is a foreground event.
            # For the blink tracks use the generic transition matrix.
            if ma.track is fg_track:
                ma.P_nx = fg_track.P_nx

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
                    amount = rate_sum * (mb.tm - ma.tm)
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


def sample_primary_transitions(T, root, node_to_tm,
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
                edge, node_to_meta, fg_track, bg_tracks, bg_to_fg_fset):
            ma, mb = segment

            # Update the ma transition matrix if it is a foreground event.
            if ma.track is fg_track:

                # Uniformize the transition matrix
                # according to the background states.
                Q_local = nx.DiGraph()
                for s in fg_track.Q_nx:
                    Q_local.add_node(s)
                for sa, sb in fg_track.Q_nx.edges():
                    if sb in fg_allowed:
                        rate = fg_track.Q_nx[sa][sb]['weight']
                        Q_local.add_edge(sa, sb, weight=rate)

                # Compute the total local rates.
                local_rates = get_total_rates(Q_local)
                local_omega = get_omega(local_rates, 2)
                P_local = get_uniformized_P_nx(
                        Q_local, local_rates, local_omega)
                ma.P_nx = P_local

            # Get the set of states allowed by data and background interaction.
            fsets = []
            for m in segment:
                if m.fset is not None:
                    fsets.append(m.fset)
            for name, state in bg_track_to_state.items():
                fsets.append(bg_to_fg_fset[name][state])
            fg_allowed = set.intersection(*fsets)

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

