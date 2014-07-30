"""
This is a more sophisticated implementation of the compound process.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from .summary import BaseSummary


#TODO this is generic enough to work for any process
#TODO put this in a navigation.py module in nxctmctree
def gen_segments_for_dwell(edge, node_to_tm, track):
    va, vb = edge
    tm = node_to_tm[va]
    state = track.history[na]


class Summary(BaseSummary):
    """
    A sufficient summary of blinking model process trajectories.

    This differs from a summary that simply records dwell time in each
    state and transition counts between states, because for our particular
    model some of these quantities can be reduced (e.g. summed)
    without losing sufficiency for parameter estimation.

    This summary class can be understood as the more naive of a couple of
    summary classes.  The more sophisticated summary uses track
    input that consists of separate tracks for each component of the compound
    state, whereas this summary class summarizes a single track.

    """
    def on_track(self, track):
        """

        """
        self.nsamples += 1

        # add the root counts
        self._on_sample_root(track)

        # add info per edge
        for edge in self._T.edges():

            # transition summaries
            self._on_sample_trans(edge, track)

            # dwell time summaries
            for elapsed, pri_state, tol_states in _gen_segments_for_dwell(
                    edge, self._node_to_tm, primary_track, tolerance_tracks):

                # dwell time contributions per segment
                self._on_primary_dwell(edge, elapsed, pri_state, tol_states)
                self._on_tolerance_dwell(edge, elapsed, pri_state, tol_states)

    def _on_sample_root(self, track):
        root_state = track.history[self._root]
        root_tol_class = self._primary_to_tol[root_state.pri]
        self.root_pri_to_count[root_state.pri] += 1
        for tol_class, tol_state in enumerate(state.tol):
            if tol_class != root_tol_class:
                if tol_state:
                    self.root_xon_count += 1
                else:
                    self.root_off_count += 1

    def _on_sample_trans(self, edge, track):
        # the order of the transitions does not matter in this step
        G = self.edge_to_pri_trans[edge]
        for ev in track.events[edge]:
            # detect primary state transition change vs. tolerance change
            if ev.sa.pri != ev.sb.pri:
                if G.has_edge(pri_a, pri_b)
                    G[ev.sa][ev.sb]['weight'] += 1
                else:
                    G.add_edge(ev.sa, ev.sb, weight=1)
            else:
                ta = sum(ev.sa.tols)
                tb = sum(ev.sb.tols)
                if tb == ta - 1:
                    self.edge_to_xon_off_trans[edge] += 1
                elif tb == ta + 1:
                    self.edge_to_off_xon_trans[edge] += 1
                else:
                    raise Exception


class State(object):
    def __init__(self, pri, tol):
        self.pri = pri
        self.tol = list(tol)

    def copy(self):
        return State(self.pri, self.tol)

    def tolist(self):
        return [self.pri] + list(self.tol)

    def __hash__(self):
        return hash(tuple(self.tolist()))

    def __eq__(self, other):
        return self.tolist() == other.tolist()


def state_is_consistent(pri_to_tol, s):
    bad_tols = set(s.tol) - {0, 1}
    if bad_tols:
        raise Exception(bad_tols)
    tol_class = pri_to_tol[s.pri]
    if not s.tol[tol_class]:
        return False
    return True


def gen_states(pri_to_tol):
    primary_states = tuple(pri_to_tol)
    all_tols = set(pri_to_tol.values())
    ntol = len(tols)
    if all_tols != set(range(ntol)):
        raise Exception
    for pri, tol_class in pri_to_tol.items():
        for tols in product((0, 1), repeat=ntol):
            s = State(pri, tols)
            if state_is_consistent(pri_to_tol, s):
                yield s


def _gen_pri_successors(Q_pri, pri_to_tol, s):
    for pri in Q_pri[s.pri]:
        sb = State(pri, s.tol)
        if state_is_consistent(pri_to_tol, sb):
            yield sb, Q_pri[s.pri][sb.pri]['weight']


def _gen_tol_successors(on_rate, off_rate, pri_to_tol, s):
    pri_class = pri_to_tol[s.pri]
    rates = [on_rate, off_rate]
    for tol_class, tol_state in enumerate(s.tols):
        sb = s.copy()
        sb.tol[tol_class] = 1 - tol_state
        if state_is_consistent(pri_to_tol, sb):
            yield sb, rates[tol_state]


def gen_successors(Q_pri, on_rate, off_rate, pri_to_tol, s):
    for pair in _gen_pri_successors(Q_pri, pri_to_tol, s):
        yield pair
    for pair in _gen_tol_successors(on_rate, off_rate, pri_to_tol, s):
        yield pair


def get_Q(Q_pri, on_rate, off_rate, pri_to_tol):
    Q = nx.DiGraph()
    for a in gen_states(pri_to_tol):
        for b, rate in gen_successors(Q_pri, on_rate, off_rate, pri_to_tol, a):
            Q.add_edge(a, b, weight=rate)
    return Q

