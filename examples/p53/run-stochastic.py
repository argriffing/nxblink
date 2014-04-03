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
from nxblink.raoteh import blinking_model_rao_teh
from nxblink.navigation import gen_segments
from nxblink.trajectory import Trajectory

from nxmodel import (
        get_Q_primary_and_distn, get_primary_to_tol, get_tree_info)


BENIGN = 'BENIGN'
LETHAL = 'LETHAL'
UNKNOWN = 'UNKNOWN'




def old_main(args):

    # Pick some parameters.
    info = get_jeff_params_e()
    kappa, omega, A, C, T, G, rho, tree, root, leaf_name_pairs = info
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)

    # Read the genetic code.
    print('reading the genetic code...')
    with open('universal.code.txt') as fin:
        genetic_code = app_helper.read_genetic_code(fin)
    codon_to_state = dict((c, s) for s, r, c in genetic_code)

    # Check that the states are in the correct order.
    nstates = len(genetic_code)
    if range(nstates) != [s for s, r, c in genetic_code]:
        raise Exception
    states = range(nstates)

    # Define the default process codon rate matrix
    # and distribution and tolerance classes.
    info = create_mg94.create_mg94(
            A, C, G, T,
            kappa, omega, genetic_code,
            target_expected_rate=1.0)
    Q, primary_distn, state_to_residue, residue_to_part = info
    primary_to_part = dict(
            (i, residue_to_part[r]) for i, r in state_to_residue.items())

    # Define the dense default process codon rate matrix and distribution.
    #Q_dense = _density.rate_matrix_to_numpy_array(
            #Q, nodelist=states)
    #primary_distn_dense = _density.dict_to_numpy_array(
            #primary_distn, nodelist=states)

    # Report the genetic code.
    print('genetic code:')
    for triple in genetic_code:
        print(triple)
    print()
    print('codon distn:')
    app_helper.print_codon_distn(codon_to_state, primary_distn)
    print()

    # Define an SD decomposition of the default process rate matrix.
    #D1 = primary_distn_dense
    #S1 = np.dot(Q_dense, np.diag(qtop.pseudo_reciprocal(D1)))

    # Change the root to 'Has' which is typoed from 'H'omo 'sa'piens.
    original_root = root
    root = name_to_leaf['Has']

    # print a summary of the tree
    #print_tree_summary(tree)
    #print()

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

    # compute the log likelihood, column by column
    # using _mjp_dense (the dense Markov jump process module).
    print('preparing to compute log likelihood...')
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*codon_sequences)
    # Get the row index of the homo sapiens name.
    reference_codon_row_index = names.index('Has')
    print('computing log likelihood...')
    total_ll_compound_cont = 0
    total_ll_compound_disc = 0

    # Compute prior switching probabilities per branch.
    # This does not use alignment or disease information.
    """
    if args.prior_switch_tsv_out is not None:

        # Every codon state is benign.
        # No codon state is lethal.
        benign_states = set(states)
        lethal_states = set()

        # For the root node, all reference process states are allowed
        # but no default process state is allowed.
        # For non-root nodes, every state is allowed, including all
        # reference and all default process states.
        reference_states = set(states)
        default_states = set(s+nstates for s in states)
        node_to_allowed_states = dict()
        for node in tree:
            if node == root:
                allowed_states = reference_states
            else:
                allowed_states = reference_states | default_states
            node_to_allowed_states[node] = allowed_states

        # Build the summary using the per-site builder object,
        # even though we are doing something that is not site-specific.
        builder = Builder()
        pos = None
        process_codon_column_helper(
                pos,
                builder,
                Q, primary_distn,
                tree, states, nstates, root,
                rho,
                D1, S1,
                benign_states, lethal_states, node_to_allowed_states,
                )

        # Summarize the prior switch output by writing to a tsv file.
        with open(args.prior_switch_tsv_out, 'w') as fout:
            for row in builder.edge_bucket:
                pos, na, nb, p = row
                print(na, nb, p, sep='\t', file=fout)
    """

    # Compute posterior switching probabilities per site per branch.
    # This uses alignment and disease information.
    """
    if args.posterior_switch_tsv_out is not None:

        # Analyze as many codon alignment columns as requested.
        if args.ncols is None:
            selected_codon_columns = codon_columns
        else:
            selected_codon_columns = codon_columns[:args.ncols]
        builder = Builder()
        for i, codon_column in enumerate(selected_codon_columns):
            pos = i + 1
            process_codon_column(
                    pos,
                    builder,
                    Q, primary_distn,
                    tree, states, nstates, root,
                    rho,
                    D1, S1,
                    names, name_to_leaf, genetic_code, codon_to_state,
                    pos_to_benign_residues, pos_to_lethal_residues,
                    codon_column,
                    )

            # Show some progress.
            print(pos)
            sys.stdout.flush()

        # Summarize the posterior switch output by writing a tsv file.
        with open(args.posterior_switch_tsv_out, 'w') as fout:
            for row in builder.edge_bucket:
                print(*row, sep='\t', file=fout)
    """


    # Optionally report information from the builder.
    #if args.verbose:

        #print('edge summary bucket:')
        #for row in builder.edge_bucket:
            #print(row)
        #print()

        #print('node summary bucket:')
        #for row in builder.node_bucket:
            #print(row)
        #print()

        #print('log likelihood summary bucket:')
        #for row in builder.ll_bucket:
            #print(row)
        #print()

    # Sum of log likelihoods.
    #print('sum of codon position log likelhoods:')
    #print(sum(ll for pos, ll in builder.ll_bucket))
    #print()

    # Write the newick-like string with the branch summary.
    #leaf_to_name = dict(leaf_name_pairs)
    #edge_to_prob_sum = defaultdict(float)
    #for site, na, nb, edge_switch_prob in builder.edge_bucket:
        #edge_to_prob_sum[sorted_pair(na, nb)] += edge_switch_prob
    #s = rsummary(tree, leaf_to_name, edge_to_prob_sum, original_root, None)
    #print(s)

    # Write tab separated node data.
    # TODO the output choices are too hardcoded
    #for row in builder.node_bucket:
        #print(*row, sep='\t')





def toy_run(primary_to_tol, interaction_map, track_to_node_to_data_fset):

    # Get the rooted directed tree shape.
    T, root = get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = get_edge_to_blen()
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    # Define the uniformization factor.
    uniformization_factor = 2

    # Define the primary rate matrix.
    Q_primary = get_Q_primary()

    # Define the prior primary state distribution.
    #TODO do not use hardcoded uniform distribution
    nprimary = 6
    primary_distn = dict((s, 1/nprimary) for s in range(nprimary))

    # Normalize the primary rate matrix to have expected rate 1.
    expected_primary_rate = 0
    for sa, sb in Q_primary.edges():
        p = primary_distn[sa]
        rate = Q_primary[sa][sb]['weight']
        expected_primary_rate += p * rate
    #
    #print('pure primary process expected rate:')
    #print(expected_primary_rate)
    #print()
    #
    for sa, sb in Q_primary.edges():
        Q_primary[sa][sb]['weight'] /= expected_primary_rate

    # Define primary trajectory.
    primary_track = Trajectory(
            name='PRIMARY', data=track_to_node_to_data_fset['PRIMARY'],
            history=dict(), events=dict(),
            prior_root_distn=primary_distn, Q_nx=Q_primary,
            uniformization_factor=uniformization_factor)

    # Define the rate matrix for a single blinking trajectory.
    Q_blink = get_Q_blink(rate_on=RATE_ON, rate_off=RATE_OFF)

    # Define the prior blink state distribution.
    blink_distn = {False : P_OFF, True : P_ON}

    Q_meta = get_Q_meta(Q_primary, primary_to_tol)

    # Define tolerance process trajectories.
    tolerance_tracks = []
    for name in ('T0', 'T1', 'T2'):
        track = Trajectory(
                name=name, data=track_to_node_to_data_fset[name],
                history=dict(), events=dict(),
                prior_root_distn=blink_distn, Q_nx=Q_blink,
                uniformization_factor=uniformization_factor)
        tolerance_tracks.append(track)

    # sample correlated trajectories using rao teh on the blinking model
    va_vb_type_to_count = defaultdict(int)
    #k = 800
    #k = 400
    #k = 200
    #k = 80
    k = 10
    nsamples = k * k
    burnin = nsamples // 10
    ncounted = 0
    total_dwell_off = 0
    total_dwell_on = 0
    for i, (pri_track, tol_tracks) in enumerate(blinking_model_rao_teh(
            T, root, node_to_tm,
            Q_primary, Q_blink, Q_meta,
            primary_track, tolerance_tracks, interaction_map)):
        nsampled = i+1
        if nsampled < burnin:
            continue
        # Summarize the trajectories.
        for edge in T.edges():
            va, vb = edge
            for track in tol_tracks:
                for ev in track.events[edge]:
                    transition = (ev.sa, ev.sb)
                    if ev.sa == ev.sb:
                        raise Exception('self-transitions should not remain')
                    if transition == (False, True):
                        va_vb_type_to_count[va, vb, 'on'] += 1
                    elif transition == (True, False):
                        va_vb_type_to_count[va, vb, 'off'] += 1
            for ev in pri_track.events[edge]:
                transition = (ev.sa, ev.sb)
                if ev.sa == ev.sb:
                    raise Exception('self-transitions should not remain')
                if primary_to_tol[ev.sa] == primary_to_tol[ev.sb]:
                    va_vb_type_to_count[va, vb, 'syn'] += 1
                else:
                    va_vb_type_to_count[va, vb, 'non'] += 1
        #dwell_off, dwell_on = get_blink_dwell_times(T, node_to_tm, tol_tracks)
        #total_dwell_off += dwell_off
        #total_dwell_on += dwell_on
        # Loop control.
        ncounted += 1
        if ncounted == nsamples:
            break

    # report infos
    print('burnin:', burnin)
    print('samples after burnin:', nsamples)
    for va_vb_type, count in sorted(va_vb_type_to_count.items()):
        va, vb, s = va_vb_type
        print(va, '->', vb, s, ':', count / nsamples)
    #print('dwell off:', total_dwell_off / nsamples)
    #print('dwell on :', total_dwell_on / nsamples)


def main(args):
    """

    """
    ###########################################################################
    # Model specification and conversion.

    # Specify the model.
    # Define the rate matrix for a single blinking trajectory,
    # and the prior blink state distribution.
    RATE_ON = 1.0
    RATE_OFF = 1.0
    P_ON = RATE_ON / (RATE_ON + RATE_OFF)
    P_OFF = RATE_OFF / (RATE_ON + RATE_OFF)
    primary_to_tol = get_primary_to_tol()
    Q_primary, primary_distn = get_Q_primary_and_distn()
    Q_blink = get_Q_blink(rate_on=RATE_ON, rate_off=RATE_OFF)
    blink_distn = {False : P_OFF, True : P_ON}
    Q_meta = get_Q_meta(Q_primary, primary_to_tol)

    # Convert a model specification to an interaction map for convenience.
    # This is a reformulation of the interactions between the primary
    # process track and the tolerance process tracks.
    interaction_map = get_interaction_map(primary_to_tol)

    ###########################################################################
    # Tree specification and conversion.

    # Specify the tree shape, the root,
    # the branch lengths, and the map from leaf name to leaf node.
    T, root, edge_to_blen, name_to_leaf = get_tree_info()
    human_leaf = name_to_leaf['Has']

    # Use the tree information to get a map from node to time from the root.
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    ###########################################################################
    # Rao-Teh parameters.

    # Define the uniformization factor.
    uniformization_factor = 2

    ###########################################################################
    # Data specification and conversion.

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

    ###########################################################################
    # Analyze some codon columns.

    with open('universal.code.txt') as fin:
        genetic_code = app_helper.read_genetic_code(fin)
    codon_to_state = dict((c, s) for s, r, c in genetic_code)

    for i, codon_column in enumerate(selected_codon_columns):
        pos = i + 1

        # Define the column-specific disease states and the benign states.
        benign_residues = pos_to_benign_residues.get(pos, set())
        lethal_residues = pos_to_lethal_residues.get(pos, set())
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

        # adjust the tolerance constraints using disease data
        for primary_state in benign_states:
            part = primary_to_tol[primary_state]
            tolerance_map[part][leaf] = {True}
        for primary_state in lethal_states:
            part = primary_to_tol[primary_state]
            tolerance_map[part][leaf] = {False}

        # update the data including both the primary and tolerance constraints
        data = {}
        data.update(primary_map)
        data.update(tolerance_map)

        # run the analysis for the column
        #run(primary_to_tol, interaction_map, data)
        #print()

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

        # sample correlated trajectories using rao teh on the blinking model
        va_vb_type_to_count = defaultdict(int)
        #k = 800
        #k = 400
        #k = 200
        #k = 80
        k = 10
        nsamples = k * k
        burnin = nsamples // 10
        ncounted = 0
        total_dwell_off = 0
        total_dwell_on = 0
        for i, (pri_track, tol_tracks) in enumerate(blinking_model_rao_teh(
                T, root, node_to_tm,
                Q_primary, Q_blink, Q_meta,
                primary_track, tolerance_tracks, interaction_map)):
            nsampled = i+1
            print(nsampled)
            if nsampled < burnin:
                continue
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
            # Loop control.
            ncounted += 1
            if ncounted == nsamples:
                break

        # report infos
        print('burnin:', burnin)
        print('samples after burnin:', nsamples)
        for va_vb_type, count in sorted(va_vb_type_to_count.items()):
            va, vb, s = va_vb_type
            print(va, '->', vb, s, ':', count / nsamples)
        #print('dwell off:', total_dwell_off / nsamples)
        #print('dwell on :', total_dwell_on / nsamples)
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ncols', type=int,
            help='limit the number of summarized columns')
    parser.add_argument('--disease', required=True,
            help='csv file with filtered disease data')
    main(parser.parse_args())

