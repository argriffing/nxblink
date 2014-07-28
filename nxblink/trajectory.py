"""
A trajectory for Rao-Teh sampling on trees, including the blinking model.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from .util import (
        get_total_rates, get_omega, get_uniformized_P_nx, get_identity_P_nx)

import nxctmctree
from nxctmctree.trajectory import Event, LightTrajectory


class Trajectory(LightTrajectory):
    """
    Aggregate data and functions related to a single trajectory.

    """
    def __init__(self, name=None, data=None, history=None, events=None,
            prior_root_distn=None, Q_nx=None, uniformization_factor=None):
        """

        Parameters
        ----------
        name : hashable, optional
            name of the trajectory
        data : dict, optional
            map from permanent node to set of states compatible with data
        history : dict, optional
            Map from permanent node to current state.
            Note that this is not the same as a trajectory.
        events : dict, optional
            map from permanent edge to list of events
        prior_root_distn : dict, optional
            prior state distribution at the root
        Q_nx : networkx DiGraph
            rate matrix as a networkx directed graph
        uniformization_factor : float
            uniformization factor that suggests 2.0

        """
        self.name = name
        self.data = data
        self.history = history
        self.events = events
        self.prior_root_distn = prior_root_distn
        self.Q_nx = Q_nx
        self.uniformization_factor = uniformization_factor

        # Precompute the total rates out of each state.
        self.total_rates = get_total_rates(self.Q_nx)

        # Precompute the uniformization rate.
        self.omega = get_omega(self.total_rates, self.uniformization_factor)

        # Precompute the uniformized transition matrix.
        self.P_nx = get_uniformized_P_nx(
                self.Q_nx, self.total_rates, self.omega)

        # Precompute the identity transition matrix.
        self.P_nx_identity = get_identity_P_nx(set(Q_nx))
