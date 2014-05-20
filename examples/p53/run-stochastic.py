"""
This script computes some Rao-Teh samples.

The trajectories are statistically sampled using the Rao-Teh CTBN
sampling scheme specialized to a blinking process,
and the data (alignment data and disease data) is from a p53 database.

"""
from __future__ import division, print_function, absolute_import

from StringIO import StringIO
from collections import defaultdict
import itertools
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
from nxblink.maxlikelihood import get_blink_rate_mle
from nxblink.trajectory import Trajectory
from nxblink.summary import (BlinkSummary,
        get_ell_init_contrib, get_ell_dwell_contrib, get_ell_trans_contrib)
from nxblink.raoteh import (
        init_tracks, gen_samples, update_track_data_for_zero_blen)

import p53model
import p53data


BENIGN = 'BENIGN'
LETHAL = 'LETHAL'
UNKNOWN = 'UNKNOWN'


def main(args):
    """

    """
    # Specify the model.
    # Define the rate matrix for a single blinking trajectory,
    # and the prior blink state distribution.

    # Specify the tree shape, the root,
    # the branch lengths, and the map from leaf name to leaf node.
    tree, root, edge_to_blen, name_to_leaf = p53model.get_tree_info()
    all_nodes = set(tree)
    human_leaf = name_to_leaf['Has']
    print('tree type:', type(tree))
    print('tree edges:', tree.edges())
    for edge in tree.edges():
        if edge not in edge_to_blen:
            raise Exception(edge)

    # Flip edge_to_blen and edge_to_rate.
    edge_to_rate = edge_to_blen
    edge_to_blen = dict((edge, 1) for edge in edge_to_rate)

    # get a map from node to time from the root.
    node_to_tm = get_node_to_tm(tree, root, edge_to_blen)

    # Define the number of burn-in iterations
    # and the number of subsequent samples.
    nburnin = args.k
    nsamples = args.k * args.k

    # Report a tree summary.
    print('root:', root)
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
                    'required integrating over too many things in the earlier '
                    'model that Liwen used, so we are avoiding this '
                    'interpretation')
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

    # Further process the genetic code.
    codon_to_state = dict((c, s) for s, r, c in genetic_code)

    # initialize the model
    rate_on = 1.5
    rate_off = 0.5
    param_info = p53model._get_jeff_params_e()
    kappa, omega, A, C, T, G, rho, d_tree, d_root, leaf_name_pairs = param_info
    print('tree edges:', tree.edges())
    model = p53model.Model(
            kappa, omega, A, C, G, T,
            rate_on, rate_off,
            tree, root,
            edge_to_blen, edge_to_rate,
            )

    # extract some summaries from the model
    primary_to_tol = model.get_primary_to_tol()
    Q_primary = model.get_Q_primary()
    primary_distn = model.get_primary_distn()
    Q_meta = get_Q_meta(Q_primary, primary_to_tol)

    # This is a reformulation of the interactions between the primary
    # process track and the tolerance process tracks.
    interaction_map = get_interaction_map(primary_to_tol)

    # Outer EM loop.
    for em_iteration in itertools.count(1):

        # Summaries of blinking states.
        Q_blink = model.get_Q_blink()
        blink_distn = model.get_blink_distn()

        # Summarize some codon column histories.
        blink_summary = BlinkSummary()
        for i, codon_column in enumerate(selected_codon_columns):
            pos = i + 1
            benign_residues = pos_to_benign_residues.get(pos, set())
            lethal_residues = pos_to_lethal_residues.get(pos, set())

            # Define the data object corresponding to this column.
            data = p53data.Data(
                    genetic_code, benign_residues, lethal_residues,
                    all_nodes, names, name_to_leaf, human_leaf,
                    primary_to_tol, codon_to_state,
                    codon_column,
                    )

            print('edge_to_blen:', edge_to_blen)
            for track_info in gen_samples(model, data, nburnin, nsamples):

                # Unpack the tracks.
                primary_track, tolerance_tracks = track_info

                # Summarize the trajectories with respect to blink parameters.
                blink_summary.on_sample(tree, root, node_to_tm, edge_to_rate,
                        primary_track, tolerance_tracks, primary_to_tol)

        # Estimate new blinking rates.
        rate_on, rate_off =  get_blink_rate_mle(
                blink_summary.xon_root_count,
                blink_summary.off_root_count,
                blink_summary.off_xon_count,
                blink_summary.xon_off_count,
                blink_summary.off_xon_dwell,
                blink_summary.xon_off_dwell,
                )
        print('xon_root_count:', blink_summary.xon_root_count)
        print('off_root_count:', blink_summary.off_root_count)
        print('off_xon_count:', blink_summary.off_xon_count)
        print('xon_off_count:', blink_summary.xon_off_count)
        print('off_xon_dwell:', blink_summary.off_xon_dwell)
        print('xon_off_dwell:', blink_summary.xon_off_dwell)
        print('finished EM iteration', em_iteration)
        sys.stdout.flush()


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

