"""
"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_allclose

import nxblink
from nxblink.maxlikelihood import (
        get_blink_rate_mle, get_blink_rate_analytical_mle)

def test_blink_rate_mle_a():
    xon_root_count = 6528
    off_root_count = 6272
    off_xon_count = 15776
    xon_off_count = 16139
    off_xon_dwell = 15893.1803168
    xon_off_dwell = 16106.8196832
    nsamples = 6400
    rate_on, rate_off =  get_blink_rate_mle(
            xon_root_count,
            off_root_count,
            off_xon_count,
            xon_off_count,
            off_xon_dwell,
            xon_off_dwell,
            )
    exact_rate_on, exact_rate_off =  get_blink_rate_analytical_mle(
            xon_root_count,
            off_root_count,
            off_xon_count,
            xon_off_count,
            off_xon_dwell,
            xon_off_dwell,
            )
    yield assert_allclose, rate_on, exact_rate_on
    yield assert_allclose, rate_off, exact_rate_off
    yield assert_allclose, rate_on, 1.0, 1e-1
    yield assert_allclose, rate_off, 1.0, 1e-1


def test_blink_rate_mle_b():
    xon_root_count = 8811
    off_root_count = 3989
    off_xon_count = 22025
    xon_off_count = 21989
    off_xon_dwell = 10069.0006186
    xon_off_dwell = 21930.9993814
    nsamples = 6400
    rate_on, rate_off =  get_blink_rate_mle(
            xon_root_count,
            off_root_count,
            off_xon_count,
            xon_off_count,
            off_xon_dwell,
            xon_off_dwell,
            )
    exact_rate_on, exact_rate_off =  get_blink_rate_analytical_mle(
            xon_root_count,
            off_root_count,
            off_xon_count,
            xon_off_count,
            off_xon_dwell,
            xon_off_dwell,
            )
    yield assert_allclose, rate_on, exact_rate_on
    yield assert_allclose, rate_off, exact_rate_off
    yield assert_allclose, rate_on, 2.22, 1e-1
    yield assert_allclose, rate_off, 1.0, 1e-1

