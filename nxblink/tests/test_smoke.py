"""
The tests in this module are smoke tests.

Passing these tests does not mean that the code is correct.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_equal

import nxblink
from nxblink.toymodel import BlinkModelA, BlinkModelB, BlinkModelC
from nxblink.toydata import DataA, DataB, DataC, DataD

from nxblink.compoundb import bar


def test_ctbn_raoteh_sampling():
    k = 4
    nburnin = k
    nsamples = k * k
    for model in BlinkModelA, BlinkModelB, BlinkModelC:
        for data in DataA, DataB, DataC, DataD:
            nsampled = 0
            for sample in nxblink.raoteh.gen_samples(
                    model, data, nburnin, nsamples):
                primary_track, tolerance_tracks = sample
                nsampled += 1
            yield assert_equal, nsampled, nsamples


def test_vanilla_raoteh_sampling_compoundb():
    k = 4
    nburnin = k
    nsamples = k * k
    for model in BlinkModelA, BlinkModelB, BlinkModelC:
        for data in DataA, DataB, DataC, DataD:
            nsampled = 0
            for track in compoundb.gen_raoteh_samples(
                    model, data, nburnin, nsamples):
                nsampled += 1
            yield assert_equal, nsampled, nsamples

