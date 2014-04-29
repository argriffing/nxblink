"""
The tests in this module are smoke tests.

Passing these tests does not mean that the code is correct.

"""
from __future__ import division, print_function, absolute_import

"""
from collections import defaultdict
from functools import partial
import argparse

import networkx as nx
import numpy as np

import nxmctree
from nxmctree.sampling import sample_history

from nxblink.model import get_Q_blink, get_Q_meta
from nxblink.model import get_interaction_map
from nxblink.util import get_node_to_tm
from nxblink.navigation import gen_segments
from nxblink.maxlikelihood import get_blink_rate_mle
from nxblink.trajectory import Trajectory
from nxblink.summary import (BlinkSummary,
        get_ell_init_contrib, get_ell_dwell_contrib, get_ell_trans_contrib)
from nxblink.raoteh import (
        init_tracks, gen_samples,
        update_track_data_for_zero_blen)
"""

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
