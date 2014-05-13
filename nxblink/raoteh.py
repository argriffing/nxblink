"""
Rao-Teh CTBN sampling, specialized to a class of blinking models.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx

import nxmctree
from nxmctree.sampling import sample_history

from .util import (get_total_rates, get_omega, get_uniformized_P_nx,
        get_node_to_tm)
from .navigation import partition_nodes
from .trajectory import Trajectory, Event
from .chunking import (get_primary_chunk_tree, get_blinking_chunk_tree,
        resample_using_chunk_tree)
from .poisson import sample_primary_poisson_events, sample_blink_poisson_events
from .model import get_interaction_map, get_Q_blink, get_Q_meta


def gen_samples(model, data, nburnin, nsamples):
    """
    Sample trajectories from the conditional distribution.

    This is a convenience function.

    Parameters
    ----------
    model : object
        A source of information about the model.
        This includes
         * get_blink_distn
         * get_primary_to_tol
         * get_T_and_root
         * get_rate_on
         * get_rate_off
         * get_primary_distn
         * get_Q_primary
         * get_edge_to_blen
    data : object
        A source of information about the data at nodes of the tree graph.
        This includes
         * get_data
         * get_primary_data
         * get_tolerance_data
        The get_data() member function returns a map from a foreground track
        to a map from a background state to a set of allowed foreground states.
    nburnin : integer
        The number of iterations of burn in.
    nsamples: integer
        The number of iterations of yielded sampled trajectories after burn in.

    """
    # Extract information from the model and data.
    primary_to_tol = model.get_primary_to_tol()
    track_to_node_to_data_fset = data.get_data()

    # Pre-compute the interaction map.
    interaction_map = get_interaction_map(primary_to_tol)

    # Get the rooted directed tree shape.
    T, root = model.get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = model.get_edge_to_blen()

    # Initialize the map from edge to rate.
    edge_to_rate = model.get_edge_to_rate()

    # Convert the branch length map to a node time map.
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    # Define the uniformization factor.
    uniformization_factor = 2

    # Define the primary rate matrix.
    Q_primary = model.get_Q_primary()

    # Define the prior primary state distribution.
    primary_distn = model.get_primary_distn()

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
            name='PRIMARY', data=track_to_node_to_data_fset['PRIMARY'],
            history=dict(), events=dict(),
            prior_root_distn=primary_distn, Q_nx=Q_primary,
            uniformization_factor=uniformization_factor)

    # Define the rate matrix for a single blinking trajectory.
    rate_on = model.get_rate_on()
    rate_off = model.get_rate_off()
    Q_blink = get_Q_blink(rate_on=rate_on, rate_off=rate_off)
    blink_distn = model.get_blink_distn()

    # Define rates from a primary state to adjacent primary states
    # controlled by a given tolerance class.
    Q_meta = get_Q_meta(Q_primary, primary_to_tol)

    # Define tolerance process trajectories.
    tolerance_names = set(primary_to_tol.values())
    tolerance_tracks = []
    for name in tolerance_names:
        track = Trajectory(
                name=name, data=track_to_node_to_data_fset[name],
                history=dict(), events=dict(),
                prior_root_distn=blink_distn, Q_nx=Q_blink,
                uniformization_factor=uniformization_factor)
        tolerance_tracks.append(track)

    # Update track data, accounting for branches with length zero.
    tracks = [primary_track] + tolerance_tracks
    update_track_data_for_zero_blen(T, edge_to_blen, edge_to_rate, tracks)

    # Initialize the tracks.
    init_tracks(T, root, node_to_tm, edge_to_rate,
            primary_to_tol, Q_primary,
            primary_track, tolerance_tracks)

    # sample correlated trajectories using rao teh on the blinking model
    ncounted = 0
    for i, (pri_track, tol_tracks) in enumerate(_gen_samples(
            T, root, node_to_tm, edge_to_rate,
            primary_to_tol, Q_meta,
            primary_track, tolerance_tracks, interaction_map)):
        nsampled = i+1
        if nsampled <= nburnin:
            continue

        # Yield some information about the sampled trajectories.
        yield primary_track, tolerance_tracks

        # Loop control.
        ncounted += 1
        if ncounted == nsamples:
            break


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


def init_complete_blink_events(T, node_to_tm, edge_to_rate, track):
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
        edge_length = edge_tmb - edge_tma
        edge_rate = edge_to_rate[edge]
        events = []
        if edge_length and edge_rate:
            if not sa:
                tma = edge_tma + edge_length * np.random.uniform(0, 1/3)
                events.append(Event(track=track, tm=tma, sa=sa, sb=True))
            if not sb:
                tmb = edge_tma + edge_length * np.random.uniform(2/3, 1)
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
    node_to_tm : dict
        maps nodes to times
    edge_to_rate : dict
        x
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
        va, vb = edge
        edge_tma = node_to_tm[va]
        edge_tmb = node_to_tm[vb]
        edge_length = edge_tmb - edge_tma
        edge_rate = edge_to_rate[edge]
        events = []
        if edge_length and edge_rate:

            # Make a plausible transition probability matrix.
            # It does not need to be carefully constructed,
            # because we are using it only to make an initial
            # feasible trajectory.
            Q_local = primary_track.Q_nx.copy()
            for sa, sb in Q_local.edges():
                Q_local[sa][sb]['weight'] *= edge_rate
            local_rates = get_total_rates(Q_local)
            local_omega = get_omega(local_rates, 2)
            P_local = get_uniformized_P_nx(Q_local, local_rates, local_omega)

            # Make the events.
            times = edge_tma + edge_length * np.random.uniform(
                    low=1/3, high=2/3, size=diameter)
            events = [Event(track=primary_track, tm=tm) for tm in times]

            # Record the transition matrix associated with each event.
            for ev in events:
                ev_to_P_nx[ev] = P_local

        # Set the primary track events.
        primary_track.events[edge] = events

    # Return the map that associates a transition probability matrix
    # to each of the events.
    return ev_to_P_nx


def sample_blink_transitions(T, root, node_to_tm, edge_to_rate,
        primary_to_tol, Q_meta, ev_to_P_nx,
        fg_track, primary_track):
    """
    Sample the history (nodes to states) and the events (edge to event list).

    """
    # Get the partition of the tree into chunks.
    info = get_blinking_chunk_tree(T, root, node_to_tm, edge_to_rate,
            primary_to_tol, Q_meta,
            fg_track, primary_track)
    chunk_tree, chunk_root, chunks, chunk_edge_to_event = info

    # Resample the foreground track history
    # and the foreground event transitions using the chunk tree.
    resample_using_chunk_tree(fg_track, ev_to_P_nx,
            chunk_tree, chunk_root, chunks, chunk_edge_to_event)


def sample_primary_transitions(T, root, node_to_tm, edge_to_rate,
        primary_to_tol, ev_to_P_nx,
        fg_track, bg_tracks):
    """
    Sample the history (nodes to states) and the events (edge to event list).

    This function depends on a foreground track
    and a collection of contextual background tracks.

    """
    # Get the partition of the tree into chunks.
    info = get_primary_chunk_tree(T, root, node_to_tm, edge_to_rate,
            primary_to_tol,
            fg_track, bg_tracks)
    chunk_tree, chunk_root, chunks, chunk_edge_to_event = info

    # Resample the foreground track history
    # and the foreground event transitions using the chunk tree.
    resample_using_chunk_tree(fg_track, ev_to_P_nx,
            chunk_tree, chunk_root, chunks, chunk_edge_to_event)


# was part of blinking_model_rao_teh
def init_tracks(T, root, node_to_tm, edge_to_rate,
        primary_to_tol, Q_primary,
        primary_track, tolerance_tracks):
    """
    Initialize trajectories of all tracks.

    """
    # Initialize blink history and events.
    for track in tolerance_tracks:
        init_blink_history(T, track)
        init_complete_blink_events(T, node_to_tm, edge_to_rate, track)
        track.remove_self_transitions()

    # Initialize the primary trajectory with many incomplete events.
    diameter = nx.diameter(Q_primary)
    ev_to_P_nx = init_incomplete_primary_events(T, node_to_tm, edge_to_rate,
            primary_track, diameter)

    # Sample the state of the primary track.
    sample_primary_transitions(T, root, node_to_tm, edge_to_rate,
            primary_to_tol, ev_to_P_nx,
            primary_track, tolerance_tracks)

    # Remove self-transition events from the primary track.
    primary_track.remove_self_transitions()


def _gen_samples(T, root, node_to_tm, edge_to_rate,
        primary_to_tol, Q_meta,
        primary_track, tolerance_tracks, interaction_map):
    """
    Tracks are assumed to have been initialized.

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
    primary_to_tol : x
        x
    Q_meta : x
        x
    primary_track : hashable
        label of the primary track
    tolerance_tracks : collection of hashables
        labels of tolerance tracks
    interaction_map : x
        x

    """
    while True:

        # add poisson events to the primary track
        ev_to_P_nx = {}
        for edge in T.edges():
            edge_rate = edge_to_rate[edge]
            if edge_rate:
                edge_ev_to_P_nx = sample_primary_poisson_events(
                        edge, edge_rate, node_to_tm,
                        primary_track, tolerance_tracks,
                        interaction_map['PRIMARY'])
                ev_to_P_nx.update(edge_ev_to_P_nx)
        # clear state labels for the primary track
        primary_track.clear_state_labels()
        # sample state transitions for the primary track
        sample_primary_transitions(T, root, node_to_tm, edge_to_rate,
                primary_to_tol, ev_to_P_nx,
                primary_track, tolerance_tracks)
        # remove self transitions for the primary track
        primary_track.remove_self_transitions()

        # Update each blinking track.
        for track in tolerance_tracks:
            name = track.name
            # add poisson events to this blink track
            ev_to_P_nx = {}
            for edge in T.edges():
                edge_rate = edge_to_rate[edge]
                if edge_rate:
                    edge_ev_to_P_nx = sample_blink_poisson_events(
                            edge, edge_rate, node_to_tm,
                            track, primary_track, interaction_map[name])
                    ev_to_P_nx.update(edge_ev_to_P_nx)
            # clear state labels for this blink track
            track.clear_state_labels()
            # sample state transitions for this blink track
            sample_blink_transitions(T, root, node_to_tm, edge_to_rate,
                    primary_to_tol, Q_meta, ev_to_P_nx,
                    track, primary_track)
            # remove self transitions for this blink track
            track.remove_self_transitions()

        # Yield the track states.
        yield primary_track, tolerance_tracks

