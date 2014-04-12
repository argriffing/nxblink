"""
This script computes some Rao-Teh samples.

The trajectories are statistically sampled using the Rao-Teh CTBN
sampling scheme specialized to a blinking process,
and the data (alignment data and disease data) is from a p53 database.

"""
from __future__ import division, print_function, absolute_import

from StringIO import StringIO
from collections import defaultdict
import functools
import argparse
import sys

import networkx as nx
import numpy as np

import create_mg94
import app_helper

import nxblink
from nxblink.model import get_Q_blink, get_Q_meta, get_interaction_map
from nxblink.util import get_node_to_tm
from nxblink.navigation import gen_segments
from nxblink.trajectory import Trajectory
from nxblink.summary import (BlinkSummary,
        get_ell_init_contrib, get_ell_dwell_contrib, get_ell_trans_contrib)
from nxblink.raoteh import (
        blinking_model_rao_teh, update_track_data_for_zero_blen)

from nxmodel import (
        get_Q_primary_and_distn, get_primary_to_tol, get_tree_info)


BENIGN = 'BENIGN'
LETHAL = 'LETHAL'
UNKNOWN = 'UNKNOWN'


def process_alignment_column(
        nsamples_sqrt,
        genetic_code, primary_to_tol,
        T, root, edge_to_blen, edge_to_rate, name_to_leaf, human_leaf,
        Q_primary, Q_blink, Q_meta,
        primary_distn, blink_distn,
        names, codon_column,
        benign_residues, lethal_residues,
        ):
    """

    """
    # Convert a model specification to an interaction map for convenience.
    # This is a reformulation of the interactions between the primary
    # process track and the tolerance process tracks.
    interaction_map = get_interaction_map(primary_to_tol)

    # Further process the genetic code.
    codon_to_state = dict((c, s) for s, r, c in genetic_code)

    # Use the tree information to get a map from node to time from the root.
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    # Define the uniformization factor.
    uniformization_factor = 2

    # Define the column-specific disease states and the benign states
    # for the reference (human) node in the tree.
    benign_states = set()
    lethal_states = set()
    for s, r, c in genetic_code:
        if r in benign_residues:
            benign_states.add(s)
        elif r in lethal_residues:
            lethal_states.add(s)
        else:
            raise Exception(
                    'each amino acid should be considered either '
                    'benign or lethal in this model, '
                    'but residue %s at position %s '
                    'was found to be neither' % (r, pos))

    # add the primary node_to_fset constraints implied by the alignment
    primary_map = {}
    primary_map['PRIMARY'] = {}
    all_primary_states = set(primary_distn)
    for v in T:
        primary_map['PRIMARY'][v] = all_primary_states
    for name, codon in zip(names, codon_column):
        leaf = name_to_leaf[name]
        primary_map['PRIMARY'][leaf] = {codon_to_state[codon]}

    # add the tolerance node_to_fset constraints implied by the alignment
    tolerance_map = {}
    all_parts = set(primary_to_tol.values())
    for part in all_parts:
        tolerance_map[part] = {}
        for v in T:
            tolerance_map[part][v] = {False, True}
        for name, codon in zip(names, codon_column):
            leaf = name_to_leaf[name]
            primary_state = codon_to_state[codon]
            observed_part = primary_to_tol[primary_state]
            if part == observed_part:
                tolerance_map[part][leaf] = {True}
            else:
                tolerance_map[part][leaf] = {False, True}

    # TODO use something like benign_parts instead of benign_states
    # adjust the tolerance constraints using human disease data
    for primary_state in benign_states:
        part = primary_to_tol[primary_state]
        fset = {True} & tolerance_map[part][human_leaf]
        tolerance_map[part][human_leaf] = fset
    for primary_state in lethal_states:
        part = primary_to_tol[primary_state]
        fset = {False} & tolerance_map[part][human_leaf]
        tolerance_map[part][human_leaf] = fset

    # update the data including both the primary and tolerance constraints
    data = {}
    data.update(primary_map)
    data.update(tolerance_map)

    track_to_node_to_data_fset = data

    # Define primary trajectory.
    primary_track = Trajectory(
            name='PRIMARY', data=track_to_node_to_data_fset['PRIMARY'],
            history=dict(), events=dict(),
            prior_root_distn=primary_distn, Q_nx=Q_primary,
            uniformization_factor=uniformization_factor)

    # Define tolerance process trajectories.
    tolerance_tracks = []
    for name in all_parts:
        track = Trajectory(
                name=name, data=track_to_node_to_data_fset[name],
                history=dict(), events=dict(),
                prior_root_distn=blink_distn, Q_nx=Q_blink,
                uniformization_factor=uniformization_factor)
        tolerance_tracks.append(track)

    # Update track data, accounting for branches with length zero.
    tracks = [primary_track] + tolerance_tracks
    update_track_data_for_zero_blen(T, edge_to_blen, edge_to_rate, tracks)

    # Initialize the log likelihood contribution
    # of the initial state at the root.
    ell_init_contrib = 0
    
    # Initialize contributions of the dwell times on each edge
    # to the expected log likelihood.
    edge_to_ell_dwell_contrib = defaultdict(float)

    # Initialize contributions of the transition events on each edge
    # to the expected log likelihood.
    edge_to_ell_trans_contrib = defaultdict(float)

    # sample correlated trajectories using rao teh on the blinking model
    va_vb_type_to_count = defaultdict(int)
    nsamples = nsamples_sqrt * nsamples_sqrt
    burnin = nsamples_sqrt
    ncounted = 0
    #total_dwell_off = 0
    #total_dwell_on = 0
    blink_summary = BlinkSummary()
    for i, (pri_track, tol_tracks) in enumerate(blinking_model_rao_teh(
            T, root, node_to_tm, edge_to_rate,
            Q_primary, Q_blink, Q_meta,
            primary_track, tolerance_tracks, interaction_map)):
        nsampled = i+1
        if nsampled % nsamples_sqrt == 0:
            print('iteration', nsampled)
        if nsampled <= burnin:
            continue

        # Summarize the trajectories with respect to blink parameters.
        blink_summary.on_sample(T, root, node_to_tm, edge_to_rate,
                primary_track, tolerance_tracks, primary_to_tol)

        #TODO dead code commented out; move to nxblink/summary.py or delete?
        """
        # Summarize the trajectories.
        for edge in T.edges():
            va, vb = edge
            for track in tol_tracks:
                for ev in track.events[edge]:
                    transition = (ev.sa, ev.sb)
                    if ev.sa == ev.sb:
                        raise Exception(
                                'self-transitions should not remain')
                    if transition == (False, True):
                        va_vb_type_to_count[va, vb, 'on'] += 1
                    elif transition == (True, False):
                        va_vb_type_to_count[va, vb, 'off'] += 1
            for ev in pri_track.events[edge]:
                transition = (ev.sa, ev.sb)
                if ev.sa == ev.sb:
                    raise Exception(
                            'self-transitions should not remain')
                if primary_to_tol[ev.sa] == primary_to_tol[ev.sb]:
                    va_vb_type_to_count[va, vb, 'syn'] += 1
                else:
                    va_vb_type_to_count[va, vb, 'non'] += 1
        #dwell_off, dwell_on = get_blink_dwell_times(
                #T, node_to_tm, tol_tracks)
        #total_dwell_off += dwell_off
        #total_dwell_on += dwell_on

        # Get the contribution of the prior probabilty of the root state
        # to the expected log likelihood.
        ll_init = get_ell_init_contrib(
                root,
                primary_distn, blink_distn,
                primary_track, tolerance_tracks, primary_to_tol)
        ell_init_contrib += ll_init

        # Get the contributions of the dwell times on each edge
        # to the expected log likelihood.
        d = get_ell_dwell_contrib(
                T, root, node_to_tm,
                Q_primary, Q_blink, Q_meta,
                primary_track, tolerance_tracks, primary_to_tol)
        for k, v in d.items():
            edge_to_ell_dwell_contrib[k] += v

        # Get the contributions of the transition events on each edge
        # to the expected log likelihood.
        d = get_ell_trans_contrib(
                T, root,
                Q_primary, Q_blink,
                primary_track, tolerance_tracks)
        for k, v in d.items():
            edge_to_ell_trans_contrib[k] += v
        """

        # Loop control.
        ncounted += 1
        if ncounted == nsamples:
            break

    # report infos per column
    print(
            blink_summary.xon_root_count,
            blink_summary.off_root_count,
            blink_summary.off_xon_count,
            blink_summary.xon_off_count,
            blink_summary.off_xon_dwell,
            blink_summary.xon_off_dwell,
            blink_summary.nsamples,
            sep='\t')
    print('ml rate on:',
            (blink_summary.xon_root_count + blink_summary.off_xon_count) / (
                blink_summary.off_xon_dwell))
    print('ml rate off:',
            (blink_summary.off_root_count + blink_summary.xon_off_count) / (
                blink_summary.xon_off_dwell))

    """
    # report infos
    #print('burnin:', burnin)
    #print('samples after burnin:', nsamples)
    #for va_vb_type, count in sorted(va_vb_type_to_count.items()):
        #va, vb, s = va_vb_type
        #print(va, '->', vb, s, ':', count / nsamples)
    #print('dwell off:', total_dwell_off / nsamples)
    #print('dwell on :', total_dwell_on / nsamples)
    #print()

    # report transition summaries
    type_to_count = defaultdict(int)
    for (va, vb, t), v in va_vb_type_to_count.items():
        type_to_count[t] += v
    print('expected transitions')
    print(' expected blink transitions')
    print('  expected off -> on        :', type_to_count['on'] / nsamples)
    print('  expected on -> off        :', type_to_count['off'] / nsamples)
    print('  total                     :', (
        type_to_count['on'] + type_to_count['off']) / nsamples)
    print(' expected codon transitions')
    print('  expected synonymous       :', type_to_count['syn'] / nsamples)
    print('  expected non synonymous   :', type_to_count['non'] / nsamples)
    print('  total                     :', (
        type_to_count['syn'] + type_to_count['non']) / nsamples)
    print('total                       :', (
        sum(type_to_count.values()) / nsamples))
    print()

    # edge dwell
    #print('edge dwell time contributions to expected log likelihood:')
    #for edge, contrib in sorted(edge_to_ell_dwell_contrib.items()):
        #va, vb = edge
        #print(va, '->', vb, ':', contrib / nsamples)
    #print()
    total_ell_dwell_contrib = sum(edge_to_ell_dwell_contrib.values())
    #print('total dwell time contribution to expected log likelihood:')
    #print(total_ell_dwell_contrib / nsamples)
    #print()

    # edge transition
    #print('edge transition contributions to expected log likelihood:')
    #for edge, contrib in sorted(edge_to_ell_trans_contrib.items()):
        #va, vb = edge
        #print(va, '->', vb, ':', contrib / nsamples)
    #print()
    total_ell_trans_contrib = sum(edge_to_ell_trans_contrib.values())
    #print('transition event contribution to expected log likelihood:')
    #print(total_ell_trans_contrib / nsamples)
    #print()

    total_ell_init_contrib = ell_init_contrib
    #print('root state contribution to expected log likelihood:')
    #print(total_ell_init_contrib / nsamples)
    #print()

    #print('expected log likelihood for this alignment column:')
    #print(sum((
        #total_ell_dwell_contrib,
        #total_ell_trans_contrib,
        #total_ell_init_contrib)) / nsamples)
    #print()

    ell_init = total_ell_init_contrib / nsamples
    ell_trans = total_ell_trans_contrib / nsamples
    ell_dwell = total_ell_dwell_contrib / nsamples
    print('sample average log likelihood:')
    print('contribution of root state        :', ell_init)
    print('contribution of transition counts :', ell_trans)
    print('contribution of dwell times       :', ell_dwell)
    print('total                             :', (
        ell_init + ell_trans + ell_dwell))
    """


def main(args):
    """

    """
    # Specify the model.
    # Define the rate matrix for a single blinking trajectory,
    # and the prior blink state distribution.

    # Initial blink rate guess.
    #RATE_ON = 1.0
    #RATE_OFF = 1.0

    # For the first column of the p53 alignment,
    # one step of EM for blink rates using sqrt nsamples 8:
    #RATE_ON = 1.315
    #RATE_OFF = 0.993

    # Two iterations of EM.
    #RATE_ON = 1.666
    #RATE_OFF = 0.994

    # Restart using different blinking rates.
    RATE_ON = 0.1
    RATE_OFF = 0.2

    P_ON = RATE_ON / (RATE_ON + RATE_OFF)
    P_OFF = RATE_OFF / (RATE_ON + RATE_OFF)
    primary_to_tol = get_primary_to_tol()
    Q_primary, primary_distn = get_Q_primary_and_distn()
    Q_blink = get_Q_blink(rate_on=RATE_ON, rate_off=RATE_OFF)
    blink_distn = {False : P_OFF, True : P_ON}
    Q_meta = get_Q_meta(Q_primary, primary_to_tol)

    # Specify the tree shape, the root,
    # the branch lengths, and the map from leaf name to leaf node.
    T, root, edge_to_blen, name_to_leaf = get_tree_info()
    human_leaf = name_to_leaf['Has']

    # Flip edge_to_blen and edge_to_rate.
    edge_to_rate = edge_to_blen
    edge_to_blen = dict((edge, 1) for edge in edge_to_rate)

    # Report a tree summary.
    print('sum of branch lengths:', sum(edge_to_blen.values()))
    print()

    # Read the alignment.
    print('reading the alignment...')
    with open('testseq') as fin:
        name_codons_list = list(app_helper.read_phylip(fin))

    # Read the interpreted disease data.
    with open(args.disease) as fin:
        interpreted_disease_data = app_helper.read_interpreted_disease_data(fin)
    pos_to_benign_residues = defaultdict(set)
    pos_to_lethal_residues = defaultdict(set)
    for pos, residue, status in interpreted_disease_data:
        if status == BENIGN:
            pos_to_benign_residues[pos].add(residue)
        elif status == LETHAL:
            pos_to_lethal_residues[pos].add(residue)
        elif status == UNKNOWN:
            raise NotImplementedError(
                    'unknown amino acid status in the reference process '
                    'requires integrating over too many things')
        else:
            raise Exception('invalid disease status: ' + str(status))
    pos_to_benign_residues = dict(pos_to_benign_residues)
    pos_to_lethal_residues = dict(pos_to_lethal_residues)

    # Reformat the alignment data.
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*codon_sequences)

    # Use command line arguments to get the selected codon columns.
    # We do this because we might not want to analyze all codon columns,
    # because analyzing codon columns may be slow.
    if args.ncols is None:
        selected_codon_columns = codon_columns
    else:
        selected_codon_columns = codon_columns[:args.ncols]

    # Re-read the genetic code to pass to the per-column analysis.
    with open('universal.code.txt') as fin:
        genetic_code = app_helper.read_genetic_code(fin)

    # Analyze some codon columns.
    headers = [
            'xon_root_count',
            'off_root_count',
            'off_xon_count',
            'xon_off_count',
            'off_xon_dwell',
            'xon_off_dwell',
            'nsamples',
            ]
    print(*headers, sep='\t')
    for i, codon_column in enumerate(selected_codon_columns):
        pos = i + 1
        benign_residues = pos_to_benign_residues.get(pos, set())
        lethal_residues = pos_to_lethal_residues.get(pos, set())
        #print('codon position', pos)
        process_alignment_column(
                args.k,
                genetic_code, primary_to_tol,
                T, root, edge_to_blen, edge_to_rate, name_to_leaf, human_leaf,
                Q_primary, Q_blink, Q_meta,
                primary_distn, blink_distn,
                names, codon_column,
                benign_residues, lethal_residues,
                )
        #print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ncols', type=int,
            help='limit the number of summarized columns')
    parser.add_argument('--disease', required=True,
            help='csv file with filtered disease data')
    parser.add_argument('--k', type=int, default=10,
            help='square root of number of samples')
    main(parser.parse_args())

