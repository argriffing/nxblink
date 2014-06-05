"""
Forward sampling of blinking process trajectories.

The samples are conditional on the initial state at the root.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np

from .trajectory import LightTrajectory, Event
from .model import get_Q_blink, get_interaction_map
from .util import get_node_to_tm


def gen_potential_transitions(
        track_to_state, primary_name, primary_to_tol, Q_primary, Q_blink):
    """
    Generate potential transitions given the current state and the process.

    Collect all allowed transitions and their rates.
    The entries in the collection will be structured like
    ((track name, initial state, final state), rate).
    This collection will be used as follows.
    First, it will be used to compute the total rate,
    which in turn will be used to sample a transition time.
    If the transition time does not exceed the tail endpoint
    of the edge, then a transition will be added to the event
    list of the current edge for its corresponding track.
    The transition will be picked according to a weighted
    choice whose weights are proportional to the transition rate.

    """
    # Extract properties of the current state.
    current_pri_state = track_to_state[primary_name]
    current_tol_class = primary_to_tol[current_pri_state]

    # Yield tolerated primary process transitions.
    for sa in Q_primary:
        for sb in Q_primary[sa]:
            sb_tol_class = primary_to_tol[sb]
            if track_to_state[sb_tol_class]:
                rate = Q_primary[sa][sb]['weight']
                trans = (primary_name, sa, sb)
                yield trans, rate

    # Yield tolerance process transitions.
    tol_classes = set(primary_to_tol.values())
    for tol_class in tol_classes:

        # Gain of tolerance is always allowed.
        if not track_to_state[tol_class]:
            trans = (tol_class, 0, 1)
            rate = Q_blink[0][1]['weight']
            yield trans, rate

        # Loss of tolerance is allowed, except for the
        # tolerance class corresponding to the current primary state.
        if track_to_state[tol_class]:
            if track_to_state[current_tol_class]:
                trans = (tol_class, 1, 0)
                rate = Q_blink[1][0]['weight']
                yield trans, rate


def gen_forward_samples(model, seq_of_track_to_root_state):
    """
    Generate forward samples of trajectories from the blinking process.

    Each forward sample consists of a collection of trajectories.
    The number of forward samples corresponds to the length of the
    seq_of_track_to_root_state sequence argument.
    Each entry in that sequence argument is a map from trajectory name
    (e.g. 'PRIMARY', 0, 1, 2, ..., 19) to the state of that trajectory
    at the root.

    The input format of this function is inspired by the more sophisticated
    conditional sampling functions in the nxblink/raoteh.py module.

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
         * get_edge_to_rate
    seq_of_track_to_state : sequence
        For each sample to be generated, the data at the root.

    Returns
    -------
    pri_track : LightTrajectory
        primary process trajectory
    tol_tracks : collection of LightTrajectory objects
        tolerance process trajectories

    """
    # Extract information from the model.
    primary_to_tol = model.get_primary_to_tol()
    tol_names = set(primary_to_tol.values())

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

    # Define the primary rate matrix.
    Q_primary = model.get_Q_primary()
    primary_distn = model.get_primary_distn()

    # Normalize the primary rate matrix to have expected rate 1.
    expected_primary_rate = 0
    for sa, sb in Q_primary.edges():
        p = primary_distn[sa]
        rate = Q_primary[sa][sb]['weight']
        expected_primary_rate += p * rate
    for sa, sb in Q_primary.edges():
        Q_primary[sa][sb]['weight'] /= expected_primary_rate

    # Define the rate matrix for a single blinking trajectory.
    rate_on = model.get_rate_on()
    rate_off = model.get_rate_off()
    Q_blink = get_Q_blink(rate_on=rate_on, rate_off=rate_off)

    # Generate forward samples.
    for track_to_root_state in seq_of_track_to_root_state:

        # Initialize primary trajectory.
        primary_name = 'PRIMARY'
        root_pri_state = track_to_root_state[primary_name]
        pri_track = LightTrajectory(
                name=primary_name,
                history={root : root_pri_state},
                events=dict())

        # Initialize tolerance process trajectories.
        tol_tracks = []
        for name in tol_names:
            track = LightTrajectory(
                    name=name,
                    history={root : track_to_root_state[name]},
                    events=dict())
            tol_tracks.append(track)

        # Aggregate the tracks.
        tracks = [pri_track] + tol_tracks
        name_to_track = dict((t.name, t) for t in tracks)

        # Iterate over edges, from the root towards the leaves.
        for edge in nx.bfs_edges(T, root):

            # For each track, initialize the event list for this edge.
            for track in tracks:
                track.events[edge] = []

            # Unpack the edge.
            na, nb = edge

            # Get the edge rate.
            # The edge length is already accounted for using node times.
            edge_rate = edge_to_rate[edge]

            # Get the time and states at the head of the edge.
            tm = node_to_tm[na]
            track_to_state = dict()
            for track in tracks:
                track_to_state[track.name] = track.history[na]

            # Get the time at the tail of the edge.
            tm_tail = node_to_tm[nb]

            # Add events onto the edge until the tail endpoint is reached.
            while True:

                # Collect all allowed transitions and their rates.
                # The edge rate can be ignored in this step,
                # because the edge rate affects all rates proportionally.
                pot = list(gen_potential_transitions(
                    track_to_state, pri_track.name,
                    primary_to_tol, Q_primary, Q_blink))
                
                # Get the total rate out of the current state.
                # This includes primary and tolerance process rates.
                # Note that the edge rate must be considered in this step,
                # if this rate is to be used directly
                # for sampling the wait time.
                rate = edge_rate * sum(r for t, r in pot)

                # Sample a wait time.
                # The wait time will be small if the rate is large.
                if rate:
                    scale = 1 / rate
                    wait_time = np.random.exponential(scale)
                    tm_ev = tm + wait_time
                else:
                    tm_ev = np.inf

                # If the wait time puts the time of the transition
                # past the tail endpoint of the edge,
                # then we have finished sampling events on this edge.
                if tm_ev > tm_tail:
                    break

                # Sample the transition proportionally to rate.
                n = len(pot)
                ts, rates = zip(*pot)
                weights = np.array(rates)
                distn = weights / weights.sum()
                idx = np.random.choice(range(n), p=distn)
                track_name, sa, sb = ts[idx]
                track = name_to_track[track_name]

                # Create the new event object,
                # add the event to the appropriate track,
                # update the current time,
                # and update the current state.
                ev = Event(track, tm_ev, sa, sb)
                track.events[edge].append(ev)
                tm = tm_ev
                track_to_state[track_name] = sb

            # Update the history at the tail endpoint of the edge.
            for track in tracks:
                track.history[nb] = track_to_state[track.name]

        # Yield the sampled trajectories.
        yield pri_track, tol_tracks

