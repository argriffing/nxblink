"""
The tests in this module are smoke tests.

Passing these tests does not mean that the code is correct.

"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_equal

import nxblink
from nxblink.raoteh import gen_samples
from nxblink.toymodel import BlinkModelA, BlinkModelB, BlinkModelC
from nxblink.toydata import DataA, DataB, DataC, DataD

def test_me():
    k = 4
    nburnin = k
    nsamples = k * k
    for model in BlinkModelA, BlinkModelB, BlinkModelC:
        for data in DataA, DataB, DataC, DataD:
            nsampled = 0
            for sample in gen_samples(model, data, nburnin, nsamples):
                primary_track, tolerance_tracks = sample
                nsampled += 1
            yield assert_equal, nsampled, nsamples
