"""
Forward sampling of blinking process trajectories.

The samples are conditional on the initial state at the root.

"""
from __future__ import division, print_function, absolute_import


def forward_sample(model, root_pri_state, root_tol_to_state):
    """

    The input format of this function is inspired by the sampling functions
    in the nxblink/raoteh.py module.

    Parameters
    ----------
    model : object
        A source of information about the model.
        This includes
         * get_blink_distn
         * get_primary_to_tol
         * get_T_and_root
         * get_rate_on
         * get_rate_off
         * get_primary_distn
         * get_Q_primary
         * get_edge_to_blen
         * get_edge_to_rate

    Returns
    -------
    pri_track : Trajectory
        primary process trajectory
    tol_tracks : collection of Trajectory objects
        tolerance process trajectories

    """
    pass

