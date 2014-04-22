"""
"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_equal

import nxblink
from nxblink.hello import get_hello


def test_hello():
    assert_equal(get_hello(), 'hello')

