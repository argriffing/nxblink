"""
A trajectory for Rao-Teh sampling on trees, including the blinking model.

Eventually this module should be moved out of the nxmctree package.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from util import get_total_rates, get_omega, get_poisson_rates
from util import get_uniformized_P_nx, get_identity_P_nx


class Trajectory(object):
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
            x
        Q_nx : x
            x
        uniformization_factor : x
            x

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
        self.P_nx_identity = get_identity_P_nx(set(self.P_nx))

        # Precompute poisson rates for Rao-Teh sampling.
        self.poisson_rates = get_poisson_rates(self.total_rates, self.omega)

    def remove_self_transitions(self):
        edges = set(self.events)
        for edge in edges:
            events = self.events[edge]
            self.events[edge] = [ev for ev in events if ev.sb != ev.sa]

    def clear_state_labels(self):
        """
        Clear the state labels but not the event times.

        """
        nodes = set(self.history)
        edges = set(self.events)
        for v in nodes:
            self.history[v] = None
        for edge in edges:
            for ev in self.events[edge]:
                ev.sa = None
                ev.sb = None


class Event(object):
    def __init__(self, track=None, tm=None, sa=None, sb=None):
        """

        Parameters
        ----------
        track : Trajectory object, optional
            the trajectory object on which the event occurs
        tm : float, optional
            time along the edge at which the event occurs
        sa : hashable, optional
            initial state of the transition
        sb : hashable, optional
            final state of the transition

        """
        self.track = track
        self.tm = tm
        self.sa = sa
        self.sb = sb

    def init_sa(self, state):
        if self.sa is not None:
            raise Exception('the initial state is already set')
        self.sa = state

    def init_sb(self, state):
        if self.sb is not None:
            raise Exception('the final state is already set')
        self.sb = state

    def init_or_confirm_sb(self, state):
        if self.sb is None:
            self.sb = state
        if self.sb != state:
            raise Exception('final state incompatibility')

    def __lt__(self, other):
        """
        Give events a partial order.

        """
        if self.tm == other.tm:
            warnings.warn('simultaneous events')
        return self.tm < other.tm


