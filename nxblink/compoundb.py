"""
This is a more sophisticated implementation of the compound process.

It is more sophisticated than the named-tuple approach to compound states,
but it is less sophisticated than keeping each component of the combinatorial
state space on a separate track and using CTBN methods.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

import nxctmctree

from .summary import BaseSummary

__all__ = ['State', 'Summary', 'gen_states', 'get_Q']


class State(object):
    def __init__(self, pri, tol):
        self.pri = pri
        self.tol = list(tol)

    def copy(self):
        return State(self.pri, self.tol)

    def tolist(self):
        return [self.pri] + list(self.tol)

    def __lt__(self, other):
        return self.tolist() < other.tolist()

    def __hash__(self):
        return hash(tuple(self.tolist()))

    def __eq__(self, other):
        return self.tolist() == other.tolist()


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
            for tma, tmb, s in gen_segments(edge, self._node_to_tm, track):

                # dwell time contributions per segment
                elapsed = tmb - tma
                self._on_primary_dwell(edge, elapsed, s.pri, s.tol)
                self._on_tolerance_dwell(edge, elapsed, s.pri, s.tol)

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


def get_Q(Q_pri, on_rate, off_rate, pri_to_tol):
    Q = nx.DiGraph()
    for a in gen_states(pri_to_tol):
        for b, rate in gen_successors(Q_pri, on_rate, off_rate, pri_to_tol, a):
            Q.add_edge(a, b, weight=rate)
    return Q


#TODO this is generic enough to work for any process
#TODO put this in a navigation.py module in nxctmctree
def gen_segments(edge, node_to_tm, track):
    """
    Yield (tma, tmb, state) triples along an edge.

    """
    va, vb = edge
    tm, state = node_to_tm[va], track.history[va]
    for tm_next, sb in sorted((ev.tm, ev.sb) for ev in track.events[edge]):
        yield tm, tm_next, state
        tm, state = tm_next, sb
    yield tm, track.history[vb], state


def state_is_consistent(pri_to_tol, s):
    bad_tols = set(s.tol) - {0, 1}
    if bad_tols:
        raise Exception(bad_tols)
    tol_class = pri_to_tol[s.pri]
    if not s.tol[tol_class]:
        return False
    return True


def _gen_pri_successors(Q_pri, pri_to_tol, s):
    """
    Yield (state, rate) pairs.

    """
    for pri in Q_pri[s.pri]:
        sb = State(pri, s.tol)
        if state_is_consistent(pri_to_tol, sb):
            yield sb, Q_pri[s.pri][sb.pri]['weight']


def _gen_tol_successors(on_rate, off_rate, pri_to_tol, s):
    """
    Yield (state, rate) pairs.

    """
    pri_class = pri_to_tol[s.pri]
    rates = [on_rate, off_rate]
    for tol_class, tol_state in enumerate(s.tols):
        sb = s.copy()
        sb.tol[tol_class] = 1 - tol_state
        if state_is_consistent(pri_to_tol, sb):
            yield sb, rates[tol_state]


def gen_successors(Q_pri, on_rate, off_rate, pri_to_tol, s):
    """
    Yield (state, rate) pairs.

    """
    for pair in _gen_pri_successors(Q_pri, pri_to_tol, s):
        yield pair
    for pair in _gen_tol_successors(on_rate, off_rate, pri_to_tol, s):
        yield pair


def gen_raoteh_samples(model, data, nburnin, nsamples):
    """
    Yield sampled tracks.

    The input model and data formats should be compatible
    with that of nxblink.toymodel and nxblink.toydata.

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
    for track in nxctmctree.raoteh.gen_raoteh_trajectories(
            T, edge_to_Q, root, root_prior_distn, node_to_data_fset,
            edge_to_blen, edge_to_rate,
            set_of_all_states, initial_track, ntrajectories)
    yield track

