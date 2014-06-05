"""
Forward sampling of blinking process trajectories.

The samples are conditional on the initial state at the root.

"""
from __future__ import division, print_function, absolute_import

from .trajectory import LightTrajectory


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
    for track_to_root_state in seq_of_track_to_root_states:

        # Initialize primary trajectory.
        pri_track = LightTrajectory(
                name='PRIMARY',
                history={root : root_pri_state},
                events=dict())

        # Initialize tolerance process trajectories.
        tol_tracks = []
        for name in tolerance_names:
            track = Trajectory(
                    name=name,
                    history={root : root_tol_to_state[name]},
                    events=dict())
            tol_tracks.append(track)

        # Aggregate the tracks.
        tracks = [pri_track] + tol_tracks

        # Iterate over edges, from the root towards the leaves.
        for edge in nx.bfs_edges(T, root):

            # Unpack the edge.
            na, nb = edge

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
                # The entries in the collection will be structured like
                # ((track name, initial state, final state), rate).
                # This collection will be used as follows.
                # First, it will be used to compute the total rate,
                # which in turn will be used to sample a transition time.
                # If the transition time does not exceed the tail endpoint
                # of the edge, then a transition will be added to the event
                # list of the current edge for its corresponding track.
                # The transition will be picked according to a weighted
                # choice whose weights are proportional to the transition rate.
                pot_trans = []

                # Add transitions corresponding to 

                # Compute the total rate out of the current state.
                # This includes rates to synonymous codons,
                # rates to tolerated nonsynonymous codons,
                # rates of tolerance loss, and rates of tolerance gain.
                total_primary_rate = 0
                for sa in Q_primary:
                    for sb in Q_primary[sa]:


