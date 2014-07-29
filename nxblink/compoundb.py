"""
This is a more sophisticated implementation of the compound process.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from .summary import BaseSummary


#TODO
# Instead of summarizing split tracks, summarize a compound state track.
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

