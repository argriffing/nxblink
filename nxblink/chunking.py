"""
This module is related to tree segmentation into iso-foreground state chunks.

Within each chunk of the tree, the foreground state cannot change.
Each edge of the chunk index tree will be mapped to a foreground event.
Each chunk object will have 
 * collection of structural nodes.
 * set of foreground states allowed by data (e.g. alignment data, disease data).
 * set of foreground states allowed by background trajectories.
 * for each foreground state, the sum of background transition expectations,
   where exp(-sum) will be the emission probability for the foreground state.

This module should replace the MetaNode in the navigation.py module
and elsewhere in this nxblink package.

Perhaps some of this module should eventually be rewritten
in a more object oriented style.

"""
from __future__ import division, print_function, absolute_import


class Chunk(object):
    def __init__(self, idx, all_fg_states):
        self.idx = idx
        self.structural_nodes = []
        self.data_allowed_states = set(all_fg_states)
        self.bg_allowed_states = set(all_fg_states)
        self.state_to_bg_penalty = dict((s, 0) for s in all_fg_states)


def _blinking_edge(
        node_to_tm,
        edge, edge_length, edge_rate,
        fg_track, primary_track,
        chunk_tree, chunks, node_to_chunk, chunk_edge_to_event):
    """
    A helper function to build the blinking chunk tree.

    """
    va, vb = edge

    # Initialize the current time, the current chunk,
    # the current foreground blinking state, and the current primary state.
    tm = node_to_tm[va]
    chunk = node_to_chunk[va]
    fg_state = fg_track.history[va]
    primary_state = primary_track.history[va]

    # Iterate over events sorted by time.
    # Include an edge endpoint as a sentinel event.
    # Therefore events are in correspondence to edge segments.
    tracks = (fg_track, primary_track)
    sentinel = (node_to_tm[vb], None)
    seq = sorted((ev.tm, ev) for t in tracks for ev in t.events) + [sentinel]


def get_blinking_chunk_tree(T, root, node_to_tm, edge_to_rate,
        fg_track, primary_track,
        ):
    """
    Get the chunk tree, when the foreground is a blinking process.

    The strategy is to define the structural root node as belonging
    to an initial chunk, and then subsequently edges are visited in preorder.
    Therefore the rootward endpoint of each edge will belong to a known chunk.
    For each foreground transition observed along an edge,
    a new chunk (and chunk edge) is added to the rooted chunk tree.

    """
    # All foreground states.
    all_fg_states = {False, True}

    # Construct the root of the chunk tree, and add it to the list.
    chunks = []
    chunk_root = Chunk(0, all_fg_states)
    chunks.append(chunk_root)

    # Initialize the chunk tree.
    chunk_tree = nx.DiGraph()

    # Initialize the map from structural node to chunk.
    node_to_chunk = {root : chunk_root}
    chunk_edge_to_event = {}

    # Process edges of the original tree one at a time.
    for edge in nx.bfs_edges(T, root):
        _blinking_edge(
                node_to_tm,
                edge, edge_to_rate[edge],
                chunk_tree, chunks, node_to_chunk, chunk_edge_to_event)

    # Return the chunk tree, its root, the list of chunk nodes,
    # and the map from chunk tree edges to foreground events.
    return chunk_tree, chunk_root, chunks, chunk_edge_to_event



def gen_meta_segments(edge, node_to_meta, ev_to_P_nx,
        fg_track, bg_tracks, bg_to_fg_fset):
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
            m = MetaNode(track=ev.track, P_nx=ev_to_P_nx[ev],
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
