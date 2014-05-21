"""
"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_equal, assert_allclose

import nxblink
from nxblink.navigation import gen_segments
from nxblink.trajectory import LightTrajectory, Event

def test_gen_segments():
    va = 'A'
    vb = 'B'
    edge = (va, vb)
    node_to_tm = {va:4, vb:10}
    track_a = LightTrajectory('ta', {va:1, vb:4})
    track_b = LightTrajectory('tb', {va:'x', vb:'x'})
    track_c = LightTrajectory('tc', {va:'a', vb:'c'})
    track_a.events = {edge : [
        Event(track_a, 7, 3, 2),
        Event(track_a, 5.5, 1, 3),
        Event(track_a, 9, 2, 4)]}
    track_b.events = {edge : []}
    track_c.events = {edge : [
        Event(track_c, 5, 'a', 'b'),
        Event(track_c, 6, 'b', 'c')]}

    # test segmentation for poisson
    tracks = [track_a, track_b, track_c]
    expected_triples = [
            (4.0, 5.0, {'ta':1, 'tb':'x', 'tc':'a'}),
            (5.0, 5.5, {'ta':1, 'tb':'x', 'tc':'b'}),
            (5.5, 6.0, {'ta':3, 'tb':'x', 'tc':'b'}),
            (6.0, 7.0, {'ta':3, 'tb':'x', 'tc':'c'}),
            (7.0, 9.0, {'ta':2, 'tb':'x', 'tc':'c'}),
            (9.0, 10.0, {'ta':4, 'tb':'x', 'tc':'c'}),
            ]
    for i, triple in enumerate(gen_segments(edge, node_to_tm, tracks)):
        assert_equal(triple, expected_triples[i])
