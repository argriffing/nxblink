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
        'BlinkSummary',
        ]


def _gen_segments_for_dwell(edge, node_to_tm, pri_track, tol_tracks):
    tracks = [pri_track] + tol_tracks
    for tma, tmb, track_to_state in gen_segments(edge, node_to_tm, tracks):
        elapsed = tmb - tma
        pri_state = track_to_state[pri_track.name]
        tol_states = dict((t.name, track_to_state[t.name]) for t in tol_tracks)
        yield elapsed, pri_state, tol_states


class Summary(object):
    """
    This is a completely generic summary.

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

            print('edge', edge, ':', file=s)

            # extract edge summaries for transitions
            pri_trans = self.edge_to_pri_trans[edge]
            off_xon_trans = self.edge_to_off_xon_trans[edge]
            xon_off_trans = self.edge_to_xon_off_trans[edge]

            # extract edge summaries for dwell times
            pri_dwell = self.edge_to_pri_dwell[edge]
            off_xon_dwell = self.edge_to_off_xon_dwell[edge]
            xon_off_dwell = self.edge_to_xon_off_dwell[edge]

            # transition summary
            print('  blink state transition counts:', file=s)
            print('    off -> on :', off_xon_trans, file=s)
            print('    unforced on -> off :', xon_off_trans, file=s)
            print('  primary state transition counts:', file=s)
            for sa in sorted(pri_trans):
                for sb in sorted(pri_trans[sa]):
                    count = pri_trans[sa][sb]['weight']
                    print('    ', sa, '->', sb, ':', count, file=s)

            # dwell summary
            print('  blink state dwell summary:', file=s)
            print('    off -> on :', off_xon_dwell, file=s)
            print('    unforced on -> off :', xon_off_dwell, file=s)
            print('  primary state dwell summary:', file=s)
            for sa in sorted(pri_dwell):
                for sb in sorted(pri_dwell[sa]):
                    dwell = pri_dwell[sa][sb]['weight']
                    print('    ', sa, '->', sb, ':', dwell, file=s)

        # return the string
        return s.getvalue()


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


# TODO a bit obsolete
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

