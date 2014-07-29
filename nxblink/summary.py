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
from StringIO import StringIO

import networkx as nx

from .navigation import gen_segments


__all__ = [
        'get_ell_init_contrib',
        'get_ell_dwell_contrib',
        'get_ell_trans_contrib',
        ]


def _gen_segments_for_dwell(edge, node_to_tm, pri_track, tol_tracks):
    tracks = [pri_track] + tol_tracks
    for tma, tmb, track_to_state in gen_segments(edge, node_to_tm, tracks):
        elapsed = tmb - tma
        pri_state = track_to_state[pri_track.name]
        tol_states = dict((t.name, track_to_state[t.name]) for t in tol_tracks)
        yield elapsed, pri_state, tol_states


class BaseSummary(object):
    """
    This is a completely generic summary for blinking model trajectories.

    Derived classes summarize different trajectory formats,
    for example trajectories split among primary and tolerance tracks,
    versus compound state trajectories.

    It should capture sufficient statistics for the full trajectory,
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
    Q_primary : DiGraph
        Primary process rate matrix.
        In this function it is used only for its sparsity pattern.

    """
    def __init__(self, T, root, node_to_tm, primary_to_tol, Q_primary):

        # store some args
        self._T = T
        self._root = root
        self._node_to_tm = node_to_tm
        self._primary_to_tol = primary_to_tol
        self._Q_primary = Q_primary

        # temporary edge list for initializing the summaries
        edges = T.edges()

        # overall summary
        self.nsamples = 0

        # summary of the state at the root
        self.root_pri_to_count = defaultdict(int)
        self.root_xon_count = 0
        self.root_off_count = 0

        # per-edge summary of transitions
        self.edge_to_pri_trans = dict((e, nx.DiGraph()) for e in edges)
        self.edge_to_off_xon_trans = defaultdict(int)
        self.edge_to_xon_off_trans = defaultdict(int)

        # per-edge summary of dwell times
        self.edge_to_pri_dwell = dict((e, nx.DiGraph()) for e in edges)
        self.edge_to_off_xon_dwell = defaultdict(float)
        self.edge_to_xon_off_dwell = defaultdict(float)
    
    def _on_primary_dwell(self, edge, elapsed, pri_state, tol_states):
        # add elapsed time to all available primary state transitions
        G = self.edge_to_pri_dwell[edge]
        for pri_state_b in self._Q_primary[pri_state]:
            tol_class_b = self._primary_to_tol[pri_state_b]
            if tol_states[tol_class_b]:
                trans = (pri_state, pri_state_b)
                if G.has_edge(*trans):
                    G[pri_state][pri_state_b]['weight'] += elapsed
                else:
                    G.add_edge(*trans, weight=elapsed)

    def _on_tolerance_dwell(self, edge, elapsed, pri_state, tol_states):
        # add elapsed time to all available tolerance state transitions
        primary_tol_class = self._primary_to_tol[pri_state]
        for tol_name, tol_state in tol_states.items():
            if tol_name != primary_tol_class:
                if tol_state:
                    self.edge_to_xon_off_dwell[edge] += elapsed
                else:
                    self.edge_to_off_xon_dwell[edge] += elapsed

    def __str__(self):
        s = StringIO()
        print('nsamples:', self.nsamples, file=s)

        # root summary
        print('root summary:', file=s)
        print('  unforced blinked-on count:', self.root_xon_count, file=s)
        print('  blinked-off count:', self.root_off_count, file=s)
        print('  nonzero primary state counts:', file=s)
        for pri_state, count in sorted(self.root_pri_to_count.items()):
            print('    ', pri_state, ':', count, file=s)

        # per-edge summary
        print('edge-specific summaries:', file=s)
        for edge in self._T.edges():

            print('  edge', edge, ':', file=s)

            # extract edge summaries for transitions
            pri_trans = self.edge_to_pri_trans[edge]
            off_xon_trans = self.edge_to_off_xon_trans[edge]
            xon_off_trans = self.edge_to_xon_off_trans[edge]

            # extract edge summaries for dwell times
            pri_dwell = self.edge_to_pri_dwell[edge]
            off_xon_dwell = self.edge_to_off_xon_dwell[edge]
            xon_off_dwell = self.edge_to_xon_off_dwell[edge]

            # transition summary
            print('    trans: blinks:', file=s)
            print('      off -> on :', off_xon_trans, file=s)
            print('      unforced on -> off :', xon_off_trans, file=s)
            print('    trans: primary:', file=s)
            for sa in sorted(pri_trans):
                for sb in sorted(pri_trans[sa]):
                    count = pri_trans[sa][sb]['weight']
                    print('      ', sa, '->', sb, ':', count, file=s)

            # dwell summary
            print('    dwell: blinks:', file=s)
            print('      off -> on :', off_xon_dwell, file=s)
            print('      unforced on -> off :', xon_off_dwell, file=s)
            print('    dwell: primary:', file=s)
            for sa in sorted(pri_dwell):
                for sb in sorted(pri_dwell[sa]):
                    dwell = pri_dwell[sa][sb]['weight']
                    print('      ', sa, '->', sb, ':', dwell, file=s)

        # return the string
        return s.getvalue()


class Summary(BaseSummary):
    """
    A summary for blinking process trajectories split into separate tracks.

    """
    def on_sample(self, primary_track, tolerance_tracks):
        """

        """
        self.nsamples += 1

        # add the root counts
        self._on_sample_root(primary_track, tolerance_tracks)

        # add info per edge
        for edge in self._T.edges():

            # transition summaries
            self._on_primary_trans(edge, primary_track)
            self._on_tolerance_trans(edge, tolerance_tracks)

            # dwell time summaries
            for elapsed, pri_state, tol_states in _gen_segments_for_dwell(
                    edge, self._node_to_tm, primary_track, tolerance_tracks):

                # dwell time contributions per segment
                self._on_primary_dwell(edge, elapsed, pri_state, tol_states)
                self._on_tolerance_dwell(edge, elapsed, pri_state, tol_states)

    def _on_sample_root(self, primary_track, tolerance_tracks):
        pri_state = primary_track.history[self._root]
        self.root_pri_to_count[pri_state] += 1
        for tol_track in tolerance_tracks:
            tol_state = tol_track.history[self._root]
            if not tol_state:
                self.root_off_count += 1
            elif tol_state != self._primary_to_tol[pri_state]:
                self.root_xon_count += 1

    def _on_primary_trans(self, edge, primary_track):
        # the order of the transitions does not matter in this step
        G = self.edge_to_pri_trans[edge]
        for ev in primary_track.events[edge]:
            trans = (ev.sa, ev.sb)
            if G.has_edge(*trans):
                G[ev.sa][ev.sb]['weight'] += 1
            else:
                G.add_edge(*trans, weight=1)

    def _on_tolerance_trans(self, edge, tolerance_tracks):
        # the order of the transitions does not matter in this step
        for track in tolerance_tracks:
            for ev in track.events[edge]:
                if ev.sa and not ev.sb:
                    self.edge_to_xon_off_trans[edge] += 1
                elif not ev.sa and ev.sb:
                    self.edge_to_off_xon_trans[edge] += 1
                else:
                    raise Exception


# TODO a bit obsolete
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


# TODO a bit obsolete
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


# TODO a bit obsolete
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


def get_ll_init(summary, pre_Q, distn, blink_on, blink_off):
    """

    Parameters
    ----------
    summary : Summary object
        Summary of blinking process trajectories.
    pre_Q : dense possibly exotic array
        Unscaled primary process pre-rate-matrix
        whose diagonal entries are zero.
    distn : dense possibly exotic array
        Primary state distribution.
    blink_on : float, or exotic float-like with derivatives information
        blink rate on
    blink_off : float, or exotic float-like with derivatives information
        blink rate off

    """
    # construct the blink distribution with the right data type
    blink_distn = algopy.zeros(2, dtype=distn)
    blink_distn[0] = blink_off / (blink_on + blink_off)
    blink_distn[1] = blink_on / (blink_on + blink_off)

    # compute the expected rate of the unnormalized pre-rate-matrix
    expected_rate = get_expected_rate(pre_Q, distn)

    # initialize expected log likelihood using the right data type
    ell = algopy.zeros(3, dtype=distn)[0]

    # root primary state contribution to expected log likelihood
    obs = algopy.zeros_like(distn)
    for state, count in summary.root_pri_to_count.items():
        if count:
            ell = ell + count * log(distn[state])

    # root blink state contribution to expected log likelihood
    if summary.root_off_count:
        ell = ell + summary.root_off_count * log(blink_distn[0])
    if summary.root_xon_count:
        ell = ell + summary.root_xon_count * log(blink_distn[1])

    # contributions per edge
    for edge, edge_rate in zip(edges, edge_rates):

        # extract edge summaries for transitions
        pri_trans = summary.edge_to_pri_trans[edge]
        off_xon_trans = summary.edge_to_off_xon_trans[edge]
        xon_off_trans = summary.edge_to_xon_off_trans[edge]

        # extract edge summaries for dwell times
        pri_dwell = summary.edge_to_pri_dwell[edge]
        off_xon_dwell = summary.edge_to_off_xon_dwell[edge]
        xon_off_dwell = summary.edge_to_xon_off_dwell[edge]

        # Initialize the total number of all types of transitions.
        # This is for making adjustments according to the edge rate
        # and the expected rate of the pre-rate-matrix.
        transition_count_sum = 0

        # contribution of primary transition summary to expected log likelihood
        for sa in pri_trans:
            for sb in pri_trans[sa]:
                count = pri_trans[sa][sb]['weight']
                transition_count_sum += count
                if count:
                    ell = ell + count * log(pre_Q[sa, sb])

        # contribution of blink transition summary to expected log likelihood
        if off_xon_trans:
            transition_count_sum += off_xon_trans
            ell = ell + off_xon_trans * log(blink_on)
        if xon_off_trans:
            transition_count_sum += xon_off_trans
            ell = ell + xon_off_trans * log(blink_off)

        # add the adjustment for edge-specific rate and pre-rate-matrix scaling
        if transition_count_sum:
            ell = ell + transition_count_sum * log(edge_rate / expected_rate)
        
        # compute a scaled sum of dwell times
        ell_dwell = algopy.zeros(1, dtype=distn)[0]

        # contribution of dwell times associated with primary transitions
        for sa in pri_dwell:
            for sb in pri_dwell[sa]:
                elapsed = pri_dwell[sa][sb]['weight']
                if elapsed:
                    ell_dwell = ell_dwell + elapsed * pre_Q[sa, sb]

        # contribution of dwell times associated with tolerance transitions
        if off_xon_dwell:
            ell_dwell = ell_dwell + off_xon_dwell * blink_on
        if xon_off_dwell:
            ell_dwell = ell_dwell + xon_off_dwell * blink_off

        # Add the dwell time contribution,
        # adjusted for edge-specific rate and pre-rate-matrix scaling.
        ell = ell - ell_dwell * (edge_rate / expected_rate)

    # return the negative expected log likelihood
    return -ell
