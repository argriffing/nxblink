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

"""
from __future__ import division, print_function, absolute_import

import math

import networkx as nx

import nxmctree
from nxmctree.sampling import sample_history


class Chunk(object):
    def __init__(self, idx, all_fg_states):
        self.idx = idx
        self.structural_nodes = set()
        self.data_allowed_states = set(all_fg_states)
        self.bg_allowed_states = set(all_fg_states)
        self.state_to_bg_penalty = dict((s, 0) for s in all_fg_states)

    def get_lmap(self):
        lmap = dict()
        for state in self.data_allowed_states & self.bg_allowed_states:
            lmap[state] = math.exp(-self.state_to_bg_penalty[state])
        return lmap


def _blinking_edge(
        node_to_tm,
        primary_to_tol, Q_meta,
        edge, edge_rate,
        fg_track, primary_track,
        chunk_tree, chunks, node_to_chunk, chunk_edge_to_event,
        use_bg_penalty=True):
    """
    A helper function to build the blinking chunk tree.

    """
    va, vb = edge
    all_states = {False, True}

    # Initialize the current time, the current chunk,
    # and the current primary state.
    tm = node_to_tm[va]
    chunk = node_to_chunk[va]
    primary_state = primary_track.history[va]

    # Iterate over events sorted by time.
    # Include an edge endpoint as a sentinel event,
    # therefore events are in correspondence to edge segments.
    tracks = (fg_track, primary_track)
    sentinel = (node_to_tm[vb], None)
    edge_events = [ev for t in tracks for ev in t.events[edge]]
    seq = sorted((ev.tm, ev) for ev in edge_events) + [sentinel]
    for next_tm, ev in seq:

        # For this segment determine the blink states allowed
        # by the primary process.
        if primary_to_tol[primary_state] == fg_track.name:
            bg_allowed_states = {True}
        else:
            bg_allowed_states = {False, True}
        chunk.bg_allowed_states &= bg_allowed_states

        # Add the codon transition rate penalty for the True blink state.
        if use_bg_penalty:
            if Q_meta.has_edge(primary_state, fg_track.name):
                rate_sum = Q_meta[primary_state][fg_track.name]['weight']
                amount = rate_sum * edge_rate * (next_tm - tm)
                chunk.state_to_bg_penalty[True] += amount

        # If the event is a foreground transition
        # then create a new chunk tree node and add a chunk tree edge.
        # Otherwise if the event is from the primary track,
        # update the primary state.
        if ev is None:
            # this is the sentinel event for the branch endpoint
            pass
        elif ev.track is fg_track:
            next_chunk = Chunk(len(chunks), all_states)
            chunks.append(next_chunk)
            chunk_edge = (chunk.idx, next_chunk.idx)
            chunk_tree.add_edge(*chunk_edge)
            chunk_edge_to_event[chunk_edge] = ev
            chunk = next_chunk
        elif ev.track is primary_track:
            if ev.sa != primary_state:
                raise Exception
            primary_state = ev.sb
        else:
            raise Exception

        # Update the time.
        tm = next_tm

    # Associate the endpoint node with the current chunk.
    chunk.structural_nodes.add(vb)
    node_to_chunk[vb] = chunk


def get_blinking_chunk_tree(T, root, node_to_tm, edge_to_rate,
        primary_to_tol, Q_meta,
        fg_track, primary_track,
        use_bg_penalty=True,
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
    chunk_root = Chunk(len(chunks), all_fg_states)
    chunk_root.structural_nodes.add(root)
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
                primary_to_tol, Q_meta,
                edge, edge_to_rate[edge],
                fg_track, primary_track,
                chunk_tree, chunks, node_to_chunk, chunk_edge_to_event,
                use_bg_penalty=use_bg_penalty)

    # Define the data restriction on the foreground states for each chunk.
    for chunk in chunks:
        for v in chunk.structural_nodes:
            chunk.data_allowed_states &= fg_track.data[v]

    # Return the chunk tree, its root, the list of chunk nodes,
    # and the map from chunk tree edges to foreground events.
    return chunk_tree, chunk_root, chunks, chunk_edge_to_event


def _primary_edge(
        node_to_tm,
        primary_to_tol,
        edge, edge_rate,
        fg_track, bg_tracks,
        chunk_tree, chunks, node_to_chunk, chunk_edge_to_event):
    """
    A helper function to build the codon process chunk tree.

    """
    va, vb = edge
    all_states = set(primary_to_tol)

    # Initialize the current time, the current chunk,
    # and the current background states.
    tm = node_to_tm[va]
    chunk = node_to_chunk[va]
    bg_name_to_state = {}
    for track in bg_tracks:
        bg_name_to_state[track.name] = track.history[va]

    # Iterate over events sorted by time.
    # Include an edge endpoint as a sentinel event,
    # therefore events are in correspondence to edge segments.
    tracks = [fg_track] + bg_tracks
    sentinel = (node_to_tm[vb], None)
    edge_events = [ev for t in tracks for ev in t.events[edge]]
    seq = sorted((ev.tm, ev) for ev in edge_events) + [sentinel]
    for next_tm, ev in seq:

        # For this segment determine the foreground states allowed
        # by the background states.
        bg_allowed_states = set()
        for candidate_state, candidate_tolname in primary_to_tol.items():
            if bg_name_to_state[candidate_tolname]:
                bg_allowed_states.add(candidate_state)
        chunk.bg_allowed_states &= bg_allowed_states

        # If the event is a foreground transition
        # then create a new chunk tree node and add a chunk tree edge.
        # Otherwise if the event is from a background track,
        # update the background state.
        if ev is None:
            # this is the sentinel event for the branch endpoint
            pass
        elif ev.track is fg_track:
            next_chunk = Chunk(len(chunks), all_states)
            chunks.append(next_chunk)
            chunk_edge = (chunk.idx, next_chunk.idx)
            chunk_tree.add_edge(*chunk_edge)
            chunk_edge_to_event[chunk_edge] = ev
            chunk = next_chunk
        else:
            bg_name_to_state[ev.track.name] = ev.sb

        # Update the time.
        tm = next_tm

    # Associate the endpoint node with the current chunk.
    chunk.structural_nodes.add(vb)
    node_to_chunk[vb] = chunk


def get_primary_chunk_tree(T, root, node_to_tm, edge_to_rate,
        primary_to_tol,
        fg_track, tolerance_tracks,
        ):
    """
    Get the chunk tree, when the foreground is the primary process.

    """
    # All foreground states.
    all_fg_states = set(primary_to_tol)

    # Construct the root of the chunk tree, and add it to the list.
    chunks = []
    chunk_root = Chunk(len(chunks), all_fg_states)
    chunk_root.structural_nodes.add(root)
    chunks.append(chunk_root)

    # Initialize the chunk tree.
    chunk_tree = nx.DiGraph()

    # Initialize the map from structural node to chunk.
    node_to_chunk = {root : chunk_root}
    chunk_edge_to_event = {}

    # Process edges of the original tree one at a time.
    for edge in nx.bfs_edges(T, root):
        _primary_edge(
                node_to_tm,
                primary_to_tol,
                edge, edge_to_rate[edge],
                fg_track, tolerance_tracks,
                chunk_tree, chunks, node_to_chunk, chunk_edge_to_event)

    # Define the data restriction on the foreground states for each chunk.
    for chunk in chunks:
        for v in chunk.structural_nodes:
            chunk.data_allowed_states &= fg_track.data[v]

    # Return the chunk tree, its root, the list of chunk nodes,
    # and the map from chunk tree edges to foreground events.
    return chunk_tree, chunk_root, chunks, chunk_edge_to_event


def resample_using_chunk_tree(
        fg_track, ev_to_P_nx,
        chunk_tree, chunk_root, chunks, chunk_edge_to_event,
        ):
    """
    Construct the per-node information, then sample the foreground states,
    then map the foreground states per chunk back onto the foreground states
    at structural nodes and at transition events.

    """
    edge_to_P = dict(
            (edge, ev_to_P_nx[ev]) for edge, ev in chunk_edge_to_event.items())
    node_to_data_lmap = dict()
    for chunk in chunks:
        node_to_data_lmap[chunk.idx] = chunk.get_lmap()
    node_to_state = sample_history(
            chunk_tree, edge_to_P, chunk_root.idx,
            fg_track.prior_root_distn, node_to_data_lmap)
    for chunk_idx, state in node_to_state.items():
        for v in chunks[chunk_idx].structural_nodes:
            fg_track.history[v] = node_to_state[chunk_idx]
    for chunk_edge in chunk_tree.edges():
        idxa, idxb = chunk_edge
        ev = chunk_edge_to_event[chunk_edge]
        ev.sa = node_to_state[idxa]
        ev.sb = node_to_state[idxb]

