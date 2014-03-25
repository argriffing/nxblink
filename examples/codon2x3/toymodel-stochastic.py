"""
This toy model is described in raoteh/examples/codon2x3.

mini glossary
tol -- tolerance
traj -- trajectory
fg -- foreground track
bg -- background track

The tree is rooted and edges are directed.
For each substate track, each permanent node maps to a list of events.
Each event is a handle mapping to some event info giving the
time of the event along the branch and the nature of the transition,
if any, associated with the event.

We can use the jargon that 'events' are associated with
locations in the tree defined by a directed edge
and a distance along that edge.
Events will usually be associated with a state transition,
but 'incomplete events' will not have such an association.

The process is separated into multiple 'tracks' -- a primary process
track and one track for each of the tolerance processes.
The track trajectories are not independent of each other.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
from functools import partial

import networkx as nx
import numpy as np

import nxmctree
from nxmctree.sampling import sample_history

from nxmodel import (
        get_Q_primary, get_Q_blink, get_primary_to_tol,
        get_Q_meta, get_T_and_root, get_edge_to_blen,
        hamming_distance, compound_state_is_ok)
from poisson import sample_primary_poisson_events, sample_blink_poisson_events
from util import (
        get_node_to_tm, get_total_rates, get_omega, get_uniformized_P_nx,
        do_nothing, set_or_confirm_history_state)
from navigation import MetaNode, gen_meta_segments, gen_segments
from trajectory import Trajectory, Event


RATE_ON = 1.0
RATE_OFF = 1.0

#NTOLS = 3

#alpha = RATE_ON / (RATE_ON + RATE_OFF)
#t = 1 / NTOLS

#P_ON = t * 1 + (1-t) * alpha
#P_OFF = t * 0 + (1-t) * (1-alpha)

P_ON = RATE_ON / (RATE_ON + RATE_OFF)
P_OFF = RATE_OFF / (RATE_ON + RATE_OFF)


###############################################################################
# Primary track and blink track initialization.



def init_blink_history(T, track):
    """
    Initial blink history is True where consistent with the data.

    """
    for v in T:
        track.history[v] = (True in track.data[v])


def init_complete_blink_events(T, node_to_tm, track):
    """
    Init blink track.

    """
    for edge in T.edges():
        va, vb = edge
        sa = track.history[va]
        sb = track.history[vb]
        edge_tma = node_to_tm[va]
        edge_tmb = node_to_tm[vb]
        blen = edge_tmb - edge_tma
        tma = edge_tma + blen * np.random.uniform(0, 1/3)
        tmb = edge_tma + blen * np.random.uniform(2/3, 1)
        eva = Event(track=track, tm=tma, sa=sa, sb=True)
        evb = Event(track=track, tm=tmb, sa=True, sb=sb)
        track.events[edge] = [eva, evb]


def init_incomplete_primary_events(T, node_to_tm, primary_track, diameter):
    """
    Parameters
    ----------
    T : nx tree
        tree
    node_to_tm : dict
        maps nodes to times
    primary_track : Trajectory
        current state of the track
    diameter : int
        directed unweighted diameter of the primary transition rate matrix

    """
    for edge in T.edges():
        va, vb = edge
        edge_tma = node_to_tm[va]
        edge_tmb = node_to_tm[vb]
        blen = edge_tmb - edge_tma
        times = edge_tma + blen * np.random.uniform(
                low=1/3, high=2/3, size=diameter-1)
        events = [Event(track=primary_track, tm=tm) for tm in times]
        primary_track.events[edge] = events


###############################################################################
# Classes and functions for steps of Rao Teh iteration.




def get_edge_tree(T, root):
    """
    Nodes in the edge tree are edges in the original tree.

    The new tree will have a node (None, root) which does not correspond
    to any edge in the original tree.

    """
    dual_root = (None, root)
    T_dual = nx.DiGraph()
    if not T:
        return T_dual, dual_root
    for c in T[root]:
        T_dual.add_edge(dual_root, (root, c))
    for v in T:
        for c in T[v]:
            for g in T[c]:
                T_dual.add_edge((v, c), (c, g))
    return T_dual, dual_root


def get_blink_dwell_times(T, node_to_tm, blink_tracks):
    """
    This function is only for reporting results.

    """
    dwell_off = 0
    dwell_on = 0
    for edge in T.edges():
        va, vb = edge
        for tma, tmb, track_to_state in gen_segments(
                edge, node_to_tm, blink_tracks):
            blen = tmb - tma
            for track in blink_tracks:
                state = track_to_state[track.name]
                if state == False:
                    dwell_off += blen
                elif state == True:
                    dwell_on += blen
                else:
                    raise Exception
    return dwell_off, dwell_on


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


def sample_blink_transitions(T, root, node_to_tm, primary_to_tol,
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


def sample_primary_transitions(T, root, node_to_tm, primary_to_tol,
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


###############################################################################
# Main Rao-Teh-Gibbs sampling function.


def blinking_model_rao_teh(
        T, root, node_to_tm, primary_to_tol,
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
    primary_to_tol : x
        x
    Q_primary : x
        x
    Q_blink : x
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
    diameter = 4
    init_incomplete_primary_events(T, node_to_tm, primary_track, diameter)
    #
    # Sample the state of the primary track.
    sample_primary_transitions(T, root, node_to_tm, primary_to_tol,
            primary_track, tolerance_tracks, interaction_map['P'])
    #
    # Remove self-transition events from the primary track.
    primary_track.remove_self_transitions()

    # Outer loop of the Rao-Teh-Gibbs sampler.
    while True:

        # add poisson events to the primary track
        for edge in T.edges():
            sample_primary_poisson_events(edge, node_to_tm,
                    primary_track, tolerance_tracks, interaction_map['P'])
        # clear state labels for the primary track
        primary_track.clear_state_labels()
        # sample state transitions for the primary track
        sample_primary_transitions(T, root, node_to_tm, primary_to_tol,
                primary_track, tolerance_tracks, interaction_map['P'])
        # remove self transitions for the primary track
        primary_track.remove_self_transitions()

        # Update each blinking track.
        for track in tolerance_tracks:
            name = track.name
            # add poisson events to this blink track
            for edge in T.edges():
                sample_blink_poisson_events(edge, node_to_tm,
                        track, [primary_track], interaction_map[name])
            # clear state labels for this blink track
            track.clear_state_labels()
            # sample state transitions for this blink track
            sample_blink_transitions(T, root, node_to_tm, primary_to_tol,
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


def run(primary_to_tol, interaction_map, track_to_node_to_data_fset):

    # Get the rooted directed tree shape.
    T, root = get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = get_edge_to_blen()
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    # Define the uniformization factor.
    uniformization_factor = 2

    # Define the primary rate matrix.
    Q_primary = get_Q_primary()

    # Define the prior primary state distribution.
    #TODO do not use hardcoded uniform distribution
    nprimary = 6
    primary_distn = dict((s, 1/nprimary) for s in range(nprimary))

    # Normalize the primary rate matrix to have expected rate 1.
    expected_primary_rate = 0
    for sa, sb in Q_primary.edges():
        p = primary_distn[sa]
        rate = Q_primary[sa][sb]['weight']
        expected_primary_rate += p * rate
    #
    #print('pure primary process expected rate:')
    #print(expected_primary_rate)
    #print()
    #
    for sa, sb in Q_primary.edges():
        Q_primary[sa][sb]['weight'] /= expected_primary_rate

    # Define primary trajectory.
    primary_track = Trajectory(
            name='P', data=track_to_node_to_data_fset['P'],
            history=dict(), events=dict(),
            prior_root_distn=primary_distn, Q_nx=Q_primary,
            uniformization_factor=uniformization_factor)

    # Define the rate matrix for a single blinking trajectory.
    Q_blink = get_Q_blink(rate_on=RATE_ON, rate_off=RATE_OFF)

    # Define the prior blink state distribution.
    blink_distn = {False : P_OFF, True : P_ON}

    Q_meta = get_Q_meta(Q_primary, primary_to_tol)

    # Define tolerance process trajectories.
    tolerance_tracks = []
    for name in ('T0', 'T1', 'T2'):
        track = Trajectory(
                name=name, data=track_to_node_to_data_fset[name],
                history=dict(), events=dict(),
                prior_root_distn=blink_distn, Q_nx=Q_blink,
                uniformization_factor=uniformization_factor)
        tolerance_tracks.append(track)

    # sample correlated trajectories using rao teh on the blinking model
    va_vb_type_to_count = defaultdict(int)
    #k = 800
    #k = 400
    k = 200
    #k = 80
    nsamples = k * k
    burnin = nsamples // 10
    ncounted = 0
    total_dwell_off = 0
    total_dwell_on = 0
    for i, (pri_track, tol_tracks) in enumerate(blinking_model_rao_teh(
            T, root, node_to_tm, primary_to_tol,
            Q_primary, Q_blink, Q_meta,
            primary_track, tolerance_tracks, interaction_map)):
        nsampled = i+1
        if nsampled < burnin:
            continue
        # Summarize the trajectories.
        for edge in T.edges():
            va, vb = edge
            for track in tol_tracks:
                for ev in track.events[edge]:
                    transition = (ev.sa, ev.sb)
                    if ev.sa == ev.sb:
                        raise Exception('self-transitions should not remain')
                    if transition == (False, True):
                        va_vb_type_to_count[va, vb, 'on'] += 1
                    elif transition == (True, False):
                        va_vb_type_to_count[va, vb, 'off'] += 1
            for ev in pri_track.events[edge]:
                transition = (ev.sa, ev.sb)
                if ev.sa == ev.sb:
                    raise Exception('self-transitions should not remain')
                if primary_to_tol[ev.sa] == primary_to_tol[ev.sb]:
                    va_vb_type_to_count[va, vb, 'syn'] += 1
                else:
                    va_vb_type_to_count[va, vb, 'non'] += 1
        dwell_off, dwell_on = get_blink_dwell_times(T, node_to_tm, tol_tracks)
        total_dwell_off += dwell_off
        total_dwell_on += dwell_on
        # Loop control.
        ncounted += 1
        if ncounted == nsamples:
            break

    # report infos
    print('burnin:', burnin)
    print('samples after burnin:', nsamples)
    for va_vb_type, count in sorted(va_vb_type_to_count.items()):
        va, vb, s = va_vb_type
        print(va, '->', vb, s, ':', count / nsamples)
    print('dwell off:', total_dwell_off / nsamples)
    print('dwell on :', total_dwell_on / nsamples)


def main():

    # Get the analog of the genetic code.
    primary_to_tol = get_primary_to_tol()

    # Define track interactions.
    # This is analogous to the creation of the compound rate matrices.
    interaction_map = {
            'P' : {
                'T0' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {2, 3, 4, 5},
                    },
                'T1' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 4, 5},
                    },
                'T2' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 2, 3},
                    }
                },
            'T0' : {
                'P' : {
                    0 : {True},
                    1 : {True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                },
            'T1' : {
                'P' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {True},
                    3 : {True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                },
            'T2' : {
                'P' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {True},
                    5 : {True},
                    }
                }
            }


    # No data.
    print ('expectations given no alignment or disease data')
    print()
    data = {
            'P' : {
                'N0' : {0, 1, 2, 3, 4, 5},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {0, 1, 2, 3, 4, 5},
                'N4' : {0, 1, 2, 3, 4, 5},
                'N5' : {0, 1, 2, 3, 4, 5},
                },
            'T0' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T1' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            }
    run(primary_to_tol, interaction_map, data)
    print()


    # Alignment data only.
    print ('expectations given only alignment data but not disease data')
    print()
    data = {
            'P' : {
                'N0' : {0},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {4},
                'N4' : {5},
                'N5' : {1},
                },
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {False, True},
                },
            }
    #run(primary_to_tol, interaction_map, data)
    print()


    # Alignment and disease data.
    print ('expectations given alignment and disease data')
    print()
    data = {
            'P' : {
                'N0' : {0},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {4},
                'N4' : {5},
                'N5' : {1},
                },
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {False, True},
                },
            }
    #run(primary_to_tol, interaction_map, data)
    print()

    # Alignment and fully observed disease data.
    print ('expectations given alignment and fully observed disease data')
    print ('(all leaf disease states which were previously considered to be')
    print ('unobserved are now considered to be tolerated (blinked on))')
    print()
    data = {
            'P' : {
                'N0' : {0},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {4},
                'N4' : {5},
                'N5' : {1},
                },
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                },
            'T2' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                },
            }
    #run(primary_to_tol, interaction_map, data)
    print()


main()

