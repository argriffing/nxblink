"""
"""
from __future__ import division, print_function, absolute_import

from numpy.testing import assert_equal

import nxblink
from nxblink.model import get_interaction_map


def test_interaction_map():
    primary_to_tol = {
            0 : 'T0',
            1 : 'T0',
            2 : 'T1',
            3 : 'T1',
            4 : 'T2',
            5 : 'T2',
            }
    actual = get_interaction_map(primary_to_tol)
    desired = {
            'PRIMARY' : {
                'T0' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {2, 3, 4, 5},
                    },
                'T1' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 4, 5},
                    },
                'T2' : {
                    True : {0, 1, 2, 3, 4, 5},
                    False : {0, 1, 2, 3},
                    }
                },
            'T0' : {
                'PRIMARY' : {
                    0 : {True},
                    1 : {True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                },
            'T1' : {
                'PRIMARY' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {True},
                    3 : {True},
                    4 : {False, True},
                    5 : {False, True},
                    }
                },
            'T2' : {
                'PRIMARY' : {
                    0 : {False, True},
                    1 : {False, True},
                    2 : {False, True},
                    3 : {False, True},
                    4 : {True},
                    5 : {True},
                    }
                }
            }
    assert_equal(actual, desired)

