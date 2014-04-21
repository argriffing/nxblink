"""
"""
from __future__ import division, print_function, absolute_import

import networkx as nx
from numpy.testing import assert_equal

import nxblink
from nxblink.navigation import partition_nodes


def test_partition():
    T = nx.DiGraph()
    T.add_edge('a', 'b')
    T.add_edge('b', 'c')
    T.add_node('d')
    d_sparse = {
            ('a', 'b') : 0,
            ('b', 'c') : 1}
    d_dense = {
            ('a', 'b') : 1,
            ('b', 'c') : 1}
    edge_to_blen = d_sparse
    edge_to_rate = d_dense
    node_lists = partition_nodes(T, edge_to_blen, edge_to_rate)
    actual = sorted(sorted(x) for x in node_lists)
    desired = sorted(sorted(x) for x in [['a', 'b'], ['c'], ['d']])
    assert_equal(actual, desired)

