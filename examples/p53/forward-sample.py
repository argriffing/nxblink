"""
This script computes some Rao-Teh samples.

The trajectories are statistically sampled using the Rao-Teh CTBN
sampling scheme specialized to a blinking process,
and the data (alignment data and disease data) is from a p53 database.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import itertools
from itertools import izip_longest
import argparse

import networkx as nx
import numpy as np

import app_helper

import nxmctree
from nxmctree.sampling import dict_random_choice

import nxblink
from nxblink.util import get_node_to_tm
from nxblink.fwdsample import gen_forward_samples

import p53model


BENIGN = 'BENIGN'
LETHAL = 'LETHAL'
UNKNOWN = 'UNKNOWN'


# official python itertools recipe
def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def prob_arg(s):
    p = float(s)
    if p < 0:
        raise ValueError('expected a non-negative probability')
    if p > 1:
        raise ValueError('expected a probability less than or equal to 1')
    return p


def rate_arg(s):
    r = float(s)
    if r < 0:
        raise ValueError('expected a non-negative rate')
    return r


def rate_ratio_arg(s):
    ratio = float(s)
    if ratio < 0:
        raise ValueError('expected a non-negative rate ratio')
    return ratio


def _tree_helper(tree, root):
    """
    Convert the undirected tree to a DiGraph and get the edge_to_blen map.

    """
    edge_to_blen = {}
    T = nx.DiGraph()
    for va, vb in nx.bfs_edges(tree, root):
        T.add_edge(va, vb)
        edge_to_blen[va, vb] = tree[va][vb]['weight']
    return T, root, edge_to_blen


def get_tree_info(f_newick, root_at_human_leaf=False):
    # f_newick : newick file open for reading
    tree, root, leaf_name_pairs = app_helper.read_newick(f_newick)
    if root_at_human_leaf:
        all_nodes = set(tree)
        name_to_leaf = dict((v, k) for k, v in leaf_name_pairs)
        human_leaf = name_to_leaf['Has']
        root = human_leaf
    tree, root, edge_to_blen = _tree_helper(tree, root)
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)
    return tree, root, edge_to_blen, name_to_leaf


def sample_primary_state(primary_distn, primary_to_tol, track_to_state):
    """
    Sample a primary state conditional on the disease data.

    """
    primary_weights = {}
    for primary_state, tol_class in primary_to_tol.items():
        if track_to_state[tol_class]:
            primary_weights[primary_state] = primary_distn[primary_state]
    return dict_random_choice(primary_weights)


def main(args):
    """

    """
    # Unpack some arguments.
    kappa = args.kappa
    omega = 1.0
    pA = args.pA
    pC = args.pC
    pG = args.pG
    pT = args.pT
    nt_distn = np.array([pA, pC, pG, pT])
    if not np.allclose(nt_distn.sum(), 1):
        raise ValueError('expected nt freqs to sum to 1')
    rate_on = args.rate_on
    rate_off = args.rate_off

    # Specify the tree shape, the root,
    # the branch lengths, and the map from leaf name to leaf node.
    # Root the tree at the human leaf.
    with open(args.newick) as fin:
        tree, root, edge_to_blen, name_to_leaf = get_tree_info(
                fin, root_at_human_leaf=True)

    # Specify the model.
    # Define the rate matrix for a single blinking trajectory,
    # and the prior blink state distribution.

    # Flip edge_to_blen and edge_to_rate.
    edge_to_rate = edge_to_blen
    edge_to_blen = dict((edge, 1) for edge in edge_to_rate)

    # Report some information about the tree.
    edges = list(nx.bfs_edges(tree, root))
    expectation = 0
    for edge in edges:
        expectation += edge_to_blen[edge] * edge_to_rate[edge]
    print('number of branches:', len(edge_to_rate))
    print('unconditional expected number of codon substitutions on the tree:',
            expectation)

    # get a map from node to time from the root.
    node_to_tm = get_node_to_tm(tree, root, edge_to_blen)

    # Read the alignment.
    print('reading the alignment...')
    with open(args.align_in) as fin:
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
    ncols = len(selected_codon_columns)

    # Re-read the genetic code to pass to the per-column analysis.
    with open(args.code) as fin:
        genetic_code = app_helper.read_genetic_code(fin)

    # Further process the genetic code.
    codon_to_state = dict((c, s) for s, r, c in genetic_code)

    # initialize the model
    model = p53model.Model(
            kappa, omega, pA, pC, pG, pT,
            rate_on, rate_off,
            tree, root,
            edge_to_blen, edge_to_rate,
            genetic_code,
            )
    
    # from the model, extract the primary state distribution
    # and the primary to tol map for the purpose of sampling
    # primary states at the root node
    # conditional on the disease states at the root node.
    primary_distn = model.get_primary_distn()
    primary_to_tol = model.get_primary_to_tol()

    # Define the sequence of states at the human node.
    # Each element of the sequence corresponds to a site (column)
    # of a codon alignment.
    # The tolerance states are taken from the human disease data.
    # The codon states are sampled conditional on the tolerance states.
    root_info_seq = []
    for site_index in range(ncols):
        pos = site_index + 1
        benign_residues = pos_to_benign_residues.get(pos, set())
        lethal_residues = pos_to_lethal_residues.get(pos, set())
        track_to_state = dict()
        for residue in benign_residues:
            tol_class = model._residue_to_part[residue]
            track_to_state[tol_class] = 1
        for residue in lethal_residues:
            tol_class = model._residue_to_part[residue]
            track_to_state[tol_class] = 0
        primary_state = sample_primary_state(
                primary_distn, primary_to_tol, track_to_state)
        track_to_state['PRIMARY'] = primary_state
        root_info_seq.append(track_to_state)

    # sample alignment columns
    leaf_names = list(name_to_leaf)
    name_to_codon_state_seq = dict((n, []) for n in leaf_names)
    for i, info in enumerate(gen_forward_samples(model, root_info_seq)):
        print('accumulating samples from site', i+1, '...')
        pri_track, tol_tracks = info
        for leaf_name, node in name_to_leaf.items():
            pri_state = pri_track.history[node]
            name_to_codon_state_seq[leaf_name].append(pri_state)

    # write the alignment in the same format as Jeff's testseq file.
    nleaves = len(leaf_names)
    state_to_codon = dict((v, k) for k, v in codon_to_state.items())
    codons_per_line = 20
    with open(args.align_out, 'w') as fout:
        print(nleaves, ncols, file=fout)
        print(file=fout)
        for name, codon_states in name_to_codon_state_seq.items():
            print(name, file=fout)
            for states in grouper(codon_states, codons_per_line):
                codons = [state_to_codon[s] for s in states if s is not None]
                s = ' '.join(codons)
                print(s, file=fout)
            print(file=fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ncols', type=int,
            help='limit the number of summarized columns')
    parser.add_argument('--disease', required=True,
            help='csv file with filtered disease data')
    parser.add_argument('--kappa', required=True, type=rate_ratio_arg,
            help='mutational nucleotide transition/transversion ratio')
    parser.add_argument('--pA', required=True, type=prob_arg,
            help='mutational nucleotide equilibrium frequency of adenine')
    parser.add_argument('--pC', required=True, type=prob_arg,
            help='mutational nucleotide equilibrium frequency of cytosine')
    parser.add_argument('--pG', required=True, type=prob_arg,
            help='mutational nucleotide equilibrium frequency of guanine')
    parser.add_argument('--pT', required=True, type=prob_arg,
            help='mutational nucleotide equilibrium frequency of thymine')
    parser.add_argument('--rate-on', required=True, type=rate_arg,
            help='rate of amino acid tolerance gain')
    parser.add_argument('--rate-off', required=True, type=rate_arg,
            help='rate of amino acid tolerance loss')
    parser.add_argument('--newick', required=True,
            help='newick tree file')
    parser.add_argument('--align-in', required=True,
            help='alignment file')
    parser.add_argument('--align-out', required=True,
            help='output alignment file')
    parser.add_argument('--code', required=True,
            help='input file defining the genetic code')


    main(parser.parse_args())

