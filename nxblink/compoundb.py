"""
This is a more sophisticated implementation of the compound process.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx


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

