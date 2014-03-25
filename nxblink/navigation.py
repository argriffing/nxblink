"""
Functions related to navigation of the tree structure.

"""
from __future__ import division, print_function, absolute_import

from util import do_nothing


# OK so this is not really navigation...
class MetaNode(object):
    """
    This is hashable so it can be a node in a networkx graph.

    """
    def __init__(self, track=None, P_nx=None,
            set_sa=None, set_sb=None, fset=None, transition=None, tm=None):
        """

        Parameters
        ----------
        track : Trajectory
            event track if any
        P_nx : nx transition matrix, optional
            the node is associated with this transition matrix
        set_sa : callback, optional
            report the sampled initial state
        set_sb : callback, optional
            report the sampled final state
        fset : set, optional
            Set of foreground state restrictions or None.
            None is interpreted as no restriction rather than
            lack of feasible states.
        transition : triple, optional
            A transition like (trajectory_name, sa, sb) or None.
            None is interpreted as absence of background transition.
        tm : float
            time elapsed since the root

        """
        self.track = track
        self.P_nx = P_nx
        self.set_sa = set_sa
        self.set_sb = set_sb
        self.fset = fset
        self.transition = transition
        self.tm = tm

    def __eq__(self, other):
        return id(self) == id(other)

    def __hash__(self):
        return id(self)


def gen_meta_segments(edge, node_to_meta, fg_track, bg_tracks, bg_to_fg_fset):
    # Sequence meta nodes from three sources:
    # the two structural endpoint nodes,
    # the nodes representing transitions in background tracks,
    # and nodes representing transitions in the foreground track.
    # Note that meta nodes are not meaningfully sortable,
    # but events are sortable.
    va, vb = edge
    tracks = [fg_track] + bg_tracks

    # Concatenate events from all tracks of interest.
    events = [ev for track in tracks for ev in track.events[edge]]

    # Construct the meta nodes corresponding to sorted events.
    seq = []
    for ev in sorted(events):
        if ev.track is fg_track:
            m = MetaNode(track=ev.track, P_nx=None,
                    set_sa=ev.init_sa, set_sb=ev.init_sb,
                    tm=ev.tm)
        else:
            m = MetaNode(track=ev.track, P_nx=fg_track.P_nx_identity,
                    set_sa=do_nothing, set_sb=do_nothing,
                    transition=(ev.track.name, ev.sa, ev.sb),
                    tm=ev.tm)
        seq.append(m)
    ma = node_to_meta[va]
    mb = node_to_meta[vb]
    seq = [ma] + seq + [mb]

    # Initialize background states at the beginning of the edge.
    bg_track_to_state = {}
    for bg_track in bg_tracks:
        bg_track_to_state[bg_track.name] = bg_track.history[va]

    # Add segments of the edge as edges of the meta node tree.
    # Track the state of each background track at each segment.
    for segment in zip(seq[:-1], seq[1:]):
        ma, mb = segment

        # Keep the state of each background track up to date.
        if ma.transition is not None:
            name, sa, sb = ma.transition
            if bg_track_to_state[name] != sa:
                raise Exception('incompatible transition: '
                        'current state on track %s is %s '
                        'but encountered a transition event from '
                        'state %s to state %s' % (
                            name, bg_track_to_state[name], sa, sb))
            bg_track_to_state[name] = sb

        # Get the set of foreground states allowed by the background.
        # Note that this deliberately does not include the data.
        fsets = []
        for name, state in bg_track_to_state.items():
            fsets.append(bg_to_fg_fset[name][state])
        fg_allowed = set.intersection(*fsets)

        yield segment, bg_track_to_state, fg_allowed


def gen_segments(edge, node_to_tm, tracks):
    """
    Iterate over segments, tracking the background state.

    This is a helper function for sampling poisson events.
    On each segment, neither the background nor the foreground state changes.

    """
    va, vb = edge
    edge_tma = node_to_tm[va]
    edge_tmb = node_to_tm[vb]

    # Concatenate events from all tracks of interest.
    events = [ev for track in tracks for ev in track.events[edge]]

    # Construct tuples corresponding to sorted events or nodes.
    # No times should coincide.
    seq = [(ev.tm, ev.track, ev.sa, ev.sb) for ev in sorted(events)]
    info_a = (edge_tma, None, None, None)
    info_b = (edge_tmb, None, None, None)
    seq = [info_a] + seq + [info_b]

    # Initialize track states at the beginning of the edge.
    track_to_state = dict((t.name, t.history[va]) for t in tracks)

    # Iterate over segments of the edge.
    # Within each segment the foreground and background tracks
    # maintain the same state, and therefore the rate
    # of new foreground poisson events is constant within the segment.
    poisson_events = []
    for segment in zip(seq[:-1], seq[1:]):
        info_a, info_b = segment
        tma, tracka, saa, sba = info_a
        tmb, trackb, sab, sbb = info_b
        blen = tmb - tma

        # Keep the state of each track up to date.
        if tracka is not None:
            tm, track, sa, sb = info_a
            name = tracka.name
            if track_to_state[name] != sa:
                raise Exception('incompatible transition: '
                        'current state on track %s is %s '
                        'but encountered a transition event from '
                        'state %s to state %s' % (
                            name, track_to_state[name], sa, sb))
            track_to_state[name] = sb

        yield tma, tmb, track_to_state

