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

import app_helper

BENIGN = 'BENIGN'
LETHAL = 'LETHAL'
UNKNOWN = 'UNKNOWN'

#TODO unfinished


def get_jeff_params_e():
    """
    Added early December 2013 in response to email from Jeff.
    Use these to compute a log likelihood and per-branch
    expected switching counts (equivalently probabilities because
    at most one switch is allowed per branch).
    The log likelihood should be summed over all p53 codon sites
    and the expected switching counts should be averaged
    over all codon sites.
    """

    # FINAL ESTIMATE: rho12 =    0.61610
    # FINAL ESTIMATE: for frequency of purines is    0.50862
    # FINAL ESTIMATE: for freq. of A among purines is    0.49373
    # FINAL ESTIMATE: for freq. of T among pyrimidines is    0.38884
    # FINAL ESTIMATE: kappa =    3.38714
    # FINAL ESTIMATE: omega =    0.37767

    rho = 0.61610
    AG = 0.50862
    CT = 1 - AG
    A = AG * 0.49373
    G = AG - A
    T = CT * 0.38884
    C = CT - T
    kappa = 3.38714
    omega = 0.37767

    tree_string = """((((((Has:  0.0073385245,Ptr:  0.0073385245):  0.0640509640,Ppy:  0.0713894884):  0.0542000118,(((Mmu:  0.0025462071,Mfu:  0.0025462071):  0.0000000000,Mfa:  0.0025462071):  0.0318638454,Cae:  0.0344100525):  0.0911794477):  0.1983006745,(Mim:  0.3238901743,Tgl:  0.3238901743):  0.0000000004):  0.2277808059,((((((Mum:  0.1797319785,Rno:  0.1797319785):  0.1566592047,Mun:  0.3363911832):  0.0192333544,(Cgr:  0.1074213106,Mau:  0.1074213106):  0.2482032271):  0.0447054051,Sju:  0.4003299428):  0.1000000288,(Cpo:  0.4170856630,Mmo:  0.4170856630):  0.0832443086):  0.0250358682,(Ocu:  0.4149196099,Opr:  0.4149196099):  0.1104462299):  0.0263051408):  0.0000000147,(Sar:  0.4524627987,((Fca:  0.2801000848,Cfa:  0.2801000848):  0.1338023902,((Bta:  0.0880000138,Oar:  0.0880000138):  0.1543496707,Dle:  0.2423496845):  0.1715527905):  0.0385603236):  0.0992081966);"""
    fin = StringIO(tree_string)
    tree, root, leaf_name_pairs = app_helper.read_newick(fin)
    return (kappa, omega, A, C, T, G, rho,
            tree, root, leaf_name_pairs)


def main(args):

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
    Q_dense = _density.rate_matrix_to_numpy_array(
            Q, nodelist=states)
    primary_distn_dense = _density.dict_to_numpy_array(
            primary_distn, nodelist=states)

    # Report the genetic code.
    print('genetic code:')
    for triple in genetic_code:
        print(triple)
    print()
    print('codon distn:')
    app_helper.print_codon_distn(codon_to_state, primary_distn)
    print()

    # Define an SD decomposition of the default process rate matrix.
    D1 = primary_distn_dense
    S1 = np.dot(Q_dense, np.diag(qtop.pseudo_reciprocal(D1)))

    # Change the root to 'Has' which is typoed from 'H'omo 'sa'piens.
    original_root = root
    root = name_to_leaf['Has']

    # print a summary of the tree
    print_tree_summary(tree)
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

    # Compute posterior switching probabilities per site per branch.
    # This uses alignment and disease information.
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ncols', type=int,
            help='limit the number of summarized columns')
    parser.add_argument('--disease', required=True,
            help='csv file with filtered disease data')
    parser.add_argument('--dt', type=float,
            help='discretize the tree with this maximum branchlet length')
    parser.add_argument('--prior-switch-tsv-out',
            default='prior.switch.data.tsv',
            help='write prior per-branch switching probabilities '
                'to this file')
    parser.add_argument('--posterior-switch-tsv-out',
            default='posterior.switch.data.tsv',
            help='write posterior per-site per-branch switching probabilities '
                'to this file')
    main(parser.parse_args())

