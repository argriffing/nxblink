"""
"""
from __future__ import division, print_function, absolute_import

import networkx as nx
from numpy.testing import assert_equal

import nxblink
from nxblink.graphutil import partition_nodes


def test_partition():
    T = nx.DiGraph()
    T.add_edge('a', 'b')
    T.add_edge('b', 'c')
    T.add_node('d')
    edge_to_blen = {
            ('a', 'b') : 0,
            ('b', 'c') : 1}
    actual = sorted(sorted(x) for x in partition_nodes(T, edge_to_blen))
    desired = sorted(sorted(x) for x in [['a', 'b'], ['c'], ['d']])
    assert_equal(actual, desired)

