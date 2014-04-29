"""
Four data levels with strictly increasing observed information.

"""
from __future__ import division, print_function, absolute_import

__all__ = ['DataA', 'DataB', 'DataC', 'DataD']


class Data(object):
    """
    Data base class.

    """
    @classmethod
    def get_data(cls):
        data = dict()
        data.update({'PRIMARY' : cls.get_primary_data()})
        data.update(cls.get_tolerance_data())
        return data


class DataA(Data):
    """
    No data.

    """
    @classmethod
    def get_primary_data(cls):
        data = {
                'N0' : {0, 1, 2, 3, 4, 5},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {0, 1, 2, 3, 4, 5},
                'N4' : {0, 1, 2, 3, 4, 5},
                'N5' : {0, 1, 2, 3, 4, 5},
                }
        return data

    @classmethod
    def get_tolerance_data(cls):
        data = {
            'T0' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T1' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            }
        return data


class DataB(Data):
    """
    Alignment data only.

    """
    @classmethod
    def get_primary_data(cls):
        data = {
                'N0' : {0},
                'N1' : {0, 1, 2, 3, 4, 5},
                'N2' : {0, 1, 2, 3, 4, 5},
                'N3' : {4},
                'N4' : {5},
                'N5' : {1},
                }
        return data

    @classmethod
    def get_tolerance_data(cls):
        data = {
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {False, True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {False, True},
                }
            }
        return data


class DataC(DataB):
    """
    Alignment and disease data.

    No additional information about the primary process is observed.

    """
    @classmethod
    def get_tolerance_data(cls):
        data = {
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {False, True},
                'N4' : {False, True},
                'N5' : {False, True},
                },
            'T2' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {False, True},
                }
            }
        return data


class DataD(DataB):
    """
    Alignment and fully observed disease data at leaves.

    No additional information about the primary process is observed.

    """
    @classmethod
    def get_tolerance_data(cls):
        data = {
            'T0' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                },
            'T1' : {
                'N0' : {False},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                },
            'T2' : {
                'N0' : {True},
                'N1' : {False, True},
                'N2' : {False, True},
                'N3' : {True},
                'N4' : {True},
                'N5' : {True},
                }
            }
        return data

