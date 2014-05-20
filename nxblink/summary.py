"""
Functions to compute summaries of the tracks of the blinking process.

In particular, compute statistics that additively contribute to
the expected log likelihood.

track:
    overall:
        - sample count
    initial state at root:
        - count of each primary state
        - count of tolerance classes unnecessarily on
        - count of tolerance classes off
    at each edge of the tree:
        transition summary:
            - number of each primary state transition
            - number of on->off transitions
            - number of off->on transitions
        dwell summary:
            - for each potential primary state transition,
              total time spent in a state that allows the transition
            - dwell time weighted by number of unnecessarily on tol classes
            - dwell time weighted by number of off tol classes

"""
from __future__ import division, print_function, absolute_import

import math
from collections import defaultdict

import networkx as nx

from .navigation import gen_segments


__all__ = [
        'get_ell_init_contrib',
        'get_ell_dwell_contrib',
        'get_ell_trans_contrib',
        'BlinkSummary',
        ]


class Summary(object):
    """
    This is a completely generic summary.

    It should certainly capture sufficient statistics for the full trajectory,
    even allowing per-edge rates.
    Note that no rates are required for this summary.
    The tree is required to not change across samples.

    Parameters
    ----------
    T : DiGraph
        tree directed from the root, annotations are ignored
    root : hashable
        root node of the tree
    node_to_tm : dict
        map from node to time elapsed since root
    primary_to_tol : dict
        map from primary state to tolerance class

    """
    def __init__(self, T, root, node_to_tm, primary_to_tol):

        # store some args
        self._T = T
        self._root = root
        self._node_to_tm = node_to_tm
        self._primary_to_tol = primary_to_tol

        # temporary edge list for initializing the summaries
        edges = T.edges()

        # overall summary
        self.nsamples = 0

        # summary of the state at the root
        self.root_pri_to_count = defaultdict(int)
        self.root_xon_count = defaultdict(int)
        self.root_off_count = defaultdict(int)

        # per-edge summary of transitions
        self.edge_to_pri_trans = dict((e, DiGraph()) for e in edges)
        self.edge_to_off_xon_trans = dict((e, defaultdict(int)) for e in edges)
        self.edge_to_xon_off_trans = dict((e, defaultdict(int)) for e in edges)

        # per-edge summary of dwell times
        self.edge_to_pri_dwell = dict((e, DiGraph()) for e in edges)
        self.edge_to_off_xon_dwell = dict(
                (e, defaultdict(float)) for e in edges)
        self.edge_to_xon_off_dwell = dict(
                (e, defaultdict(float)) for e in edges)
    
    def on_sample(self, primary_track, tolerance_tracks):
        """

        """
        pass


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
        T, root, node_to_tm, edge_to_rate,
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
            contrib += -(edge_to_rate[edge] * rate * (tmb - tma))

        # Store the edge contribution.
        edge_to_contrib[edge] = contrib

    # Return the map of contributions from edges.
    return edge_to_contrib


def get_ell_trans_contrib(
        T, root, edge_to_rate,
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
                contrib += math.log(edge_to_rate[edge] * rate)
        edge_to_contrib[edge] = contrib
    return edge_to_contrib


class BlinkSummary(object):
    """
    Summary history relevant for blinking rate estimation.

    This is the sample average expectation step of EM for the estimation of
    blinking rates conditional on the other parameters.
    The term 'xon' will be used for 'nontrivial on state', where 'nontrivial'
    means 'not the blinking track associated with the current primary state'.

    """
    def __init__(self):
        self.xon_root_count = 0
        self.off_root_count = 0
        self.off_xon_count = 0
        self.xon_off_count = 0
        self.off_xon_dwell = 0
        self.xon_off_dwell = 0
        self.nsamples = 0

    def on_sample(self,
            T, root, node_to_tm, edge_to_rate,
            primary_track, tolerance_tracks, primary_to_tol):
        """
        """
        self._on_sample_init(
            root, primary_track, tolerance_tracks, primary_to_tol)
        for edge in T.edges():
            edge_rate = edge_to_rate[edge]
            self._on_sample_trans(edge, tolerance_tracks)
            self._on_sample_dwell(edge, edge_rate, node_to_tm,
                    primary_track, tolerance_tracks, primary_to_tol)
        self.nsamples += 1

    def _on_sample_init(self,
            root, primary_track, tolerance_tracks, primary_to_tol):
        primary_state = primary_track.history[root]
        tol_name = primary_to_tol[primary_state]
        for tol_track in tolerance_tracks:
            if tol_track.name != tol_name:
                tol_state = tol_track.history[root]
                if tol_state:
                    self.xon_root_count += 1
                else:
                    self.off_root_count += 1

    def _on_sample_trans(self, edge, tolerance_tracks):
        for tol_track in tolerance_tracks:
            for ev in tol_track.events[edge]:
                if ev.sa and (not ev.sb):
                    self.xon_off_count += 1
                elif (not ev.sa) and ev.sb:
                    self.off_xon_count += 1
                else:
                    raise Exception

    def _on_sample_dwell(self,
            edge, edge_rate, node_to_tm,
            primary_track, tolerance_tracks, primary_to_tol):
        tracks = [primary_track] + tolerance_tracks
        for tma, tmb, track_to_state in gen_segments(edge, node_to_tm, tracks):
            amount = edge_rate * (tmb - tma)
            primary_state = track_to_state[primary_track.name]
            tol_name = primary_to_tol[primary_state]
            for tol_track in tolerance_tracks:
                if tol_track.name != tol_name:
                    tol_state = track_to_state[tol_track.name]
                    if tol_state:
                        self.xon_off_dwell += amount
                    else:
                        self.off_xon_dwell += amount

