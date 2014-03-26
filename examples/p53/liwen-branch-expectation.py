"""
Compute a posterior summary per branch.

We are given the following ingredients.

* tree shape
* estimated branch lengths
* rate matrix model
* estimated model parameters
* genetic code
* codon alignment among species including humans
* per-site amino acid tolerance benign/lethal data for humans

Using these ingredients we want to compute, for each branch,
some function of the joint state distribution at the branch endpoints.
In particular, we care about the probability that a 'switch'
occurred on the branch.
For each branch, want to sum or average these values
over all sites in the alignment.

A few output formats would be useful for collaboration.
One output format would be a newick-like output that reports the
summed-over-sites switch probability summary for each branch,
instead of reporting the branch length for each branch as in the
traditional newick format.

Another output format would be something useable for D3 javascript
visualization, probably in a json file format.
The idea would be to have an interactive visualization
with a ~20x20 grid of codon sites on the left hand panel of the visualization
(corresponding to the 393 ~= 400 = 20x20 codon sites in p53),
and to have an annotated tree on the right hand panel of the visualization.
The idea is that when the user mouses-over a codon site
in the left hand panel, the tree on the right hand panel is changed
to reflect a site-specific summary.
When no site is moused-over, the tree in the right hand panel
reflects a summary averaged over all sites in the sequence.

In more detail, in the visualization each codon site in the grid
in the left hand pane could display a number
on a correspondingly colored background.
Maybe this number could summarize state diversity/conservation,
for example the minimum number of state changes according to parsimony,
or the posterior expected number of changes at the site,
or just the number of different states at the site.

In the visualization, the tree that summarizes the entire alignment could list
the species names at the leaves and report a switching summary for each branch.
Maybe the posterior probability of belonging to the reference process
could be reported for each node.
The site-specific tree could report the species name and the
site codon at each leaf, the site-specific posterior probability
of belonging to the reference process at each node,
and the site-specific switching probability on each branch.

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
import qtop

from raoteh.sampler import (
        _util,
        _density,
        _mjp_dense,
        _mcy_dense,
        _mc0_dense,
        )

BENIGN = 'BENIGN'
LETHAL = 'LETHAL'
UNKNOWN = 'UNKNOWN'

def print_tree_summary(tree):
    print('number of nodes:', len(tree))
    print('number of leaves:', tree.degree().values().count(1))
    print('number of branches:', tree.size())
    print('total branch length:', tree.size(weight='weight'))

def sorted_pair(a, b):
    """
    Utility function.
    """
    return tuple(sorted((a, b)))

def getp_approx(Q, t):
    """
    Approximate the transition matrix over a small time interval.
    """
    ident = np.eye(*Q.shape)
    return ident + Q*t

def getp_bigt_approx(Q, dt, t):
    """
    Approximate exp(x) = exp(x/n)^n ~= (1 + x/n)^n.
    """
    n = max(1, int(np.ceil(t / dt)))
    psmall = getp_approx(Q, t/n)
    return np.linalg.matrix_power(psmall, n)

def get_expm_augmented_tree(T, root, P_callback=None):
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        edge = T[na][nb]
        weight = edge['weight']
        P = P_callback(weight)
        T_aug.add_edge(na, nb, weight=weight, P=P)
    return T_aug

def get_node_to_distn(tree, node_to_allowed_states, root, nstates,
        root_distn=None, P_callback=None):
    if root not in tree:
        raise ValueError('the specified root is not in the tree')
    T_aug = get_expm_augmented_tree(tree, root, P_callback=P_callback)
    node_to_pmap = _mcy_dense.get_node_to_pmap(T_aug, root, nstates,
            node_to_allowed_states=node_to_allowed_states)
    node_to_distn = _mc0_dense.get_node_to_distn_esd(
            T_aug, root, node_to_pmap, nstates,
            root_distn=root_distn)
    return node_to_distn

def get_likelihood(tree, node_to_allowed_states, root, nstates,
        root_distn=None, P_callback=None):
    if root not in tree:
        raise ValueError('the specified root is not in the tree')
    T_aug = get_expm_augmented_tree(tree, root, P_callback=P_callback)
    return _mcy_dense.get_likelihood(T_aug, root, nstates,
            node_to_allowed_states=node_to_allowed_states,
            root_distn=root_distn, P_default=None)

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


def get_joint_endpoint_distn_tree(
        T, node_to_allowed_states, root, nstates,
        root_distn=None, P_callback=None):
    """

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Edges of this tree are annotated with weights.
    node_to_allowed_states : dict
        Maps each node to a set of allowed states.
    root : integer
        Root node.
    nstates : integer
        Number of states.
    root_distn : 1d ndarray, optional
        Distribution over states at the root.
    P_callback : callback function
        Computes a transition matrix given an amount of elapsed time.

    Returns
    -------
    T_joint : tree as a networkx graph
        Each edge is annotated with joint endpoint
        probability distribution J.

    """
    # Do some input validation for this restricted variant.
    if root not in T:
        raise ValueError('the specified root is not in the tree')

    # Construct the augmented tree by annotating each edge
    # with the appropriate state transition probability matrix.
    T_aug = get_expm_augmented_tree(T, root, P_callback)

    # Construct the node to pmap dict.
    node_to_pmap = _mcy_dense.get_node_to_pmap(T_aug, root, nstates,
            node_to_allowed_states=node_to_allowed_states)

    # Get the marginal state distribution for each node in the tree,
    # conditional on the known states.
    node_to_distn = _mc0_dense.get_node_to_distn(
            T_aug, root, node_to_pmap, nstates,
            root_distn=root_distn)

    # For each edge in the tree, get the joint distribution
    # over the states at the endpoints of the edge.
    T_joint = _mc0_dense.get_joint_endpoint_distn(
            T_aug, root, node_to_pmap, node_to_distn, nstates)

    return T_joint


class Builder(object):
    def __init__(self):
        self.edge_bucket = []
        self.node_bucket = []
        self.ll_bucket = []

    def add_site_log_likelihood(self, site, ll):
        self.ll_bucket.append((site, ll))

    def add_edge_summary(self, site, na, nb, switch_prob):
        """
        Parameters
        ----------
        site : int
            codon site
        na : int
            first endpoint node
        nb : int
            second endpoint node
        switch_prob : float
            Posterior probability that a process switch
            has occurred along the edge.
        """
        self.edge_bucket.append((site, na, nb, switch_prob))

    def add_node_summary(self, site, node, reference_process_prob):
        """
        Parameters
        ----------
        site : int
            codon site
        node : int
            node
        reference_process_prob : float
            Posterior probability that the state of the node
            is in the reference process.

        """
        self.node_bucket.append((site, node, reference_process_prob))


def accumulate_codon_site_summary(
        tree, node_to_allowed_states, root,
        nstates, ncompound,
        compound_root_distn, P_cb_compound,
        site, builder,
        ):
    """
    Compute posterior info for a single codon site.

    """
    # Construct a tree whose branches are annotated
    # with joint endpoint distributions.
    T_joint = get_joint_endpoint_distn_tree(
            tree, node_to_allowed_states, root, ncompound,
            root_distn=compound_root_distn,
            P_callback=P_cb_compound)

    # For each edge,
    # compute the probability that a switch occurred along the edge.
    for na, nb in nx.bfs_edges(T_joint, root):
        J = T_joint[na][nb]['J']
        switch_prob = J[:nstates, nstates:].sum()
        builder.add_edge_summary(site, na, nb, switch_prob)

    # Get the state distribution at each node.
    node_to_distn = get_node_to_distn(
            tree, node_to_allowed_states, root, ncompound,
            root_distn=compound_root_distn,
            P_callback=P_cb_compound)

    # For each node add the summary into the bucket.
    for node, node_compound_distn in node_to_distn.items():
        reference_process_prob = node_compound_distn[:nstates].sum()
        builder.add_node_summary(site, node, reference_process_prob)

    # Compute the log likelihood for the site.
    likelihood = get_likelihood(
            tree, node_to_allowed_states, root, ncompound,
            root_distn=compound_root_distn,
            P_callback=P_cb_compound)
    ll = np.log(likelihood)

    # Add the site log likelihood into the bucket.
    builder.add_site_log_likelihood(site, ll)


def rsummary(tree, leaf_to_name, edge_to_prob_sum, v, parent):
    """
    Recursively build a newick-like summary.

    The number associated with each edge is not the branch length
    but is rather the switching probability.

    Parameters
    ----------
    tree : networkx tree
        original tree
    leaf_to_name : map from int to str
        maps a leaf node to its name
    edge_to_prob_sum : map from sorted node pair to float
        posterior probability sum for each node
    v : int
        root of the subtree under current consideration
    parent : int or None, optional
        parent node of the root of the current subtree

    Returns
    -------
    s : str
        newick-like string

    """
    prob_sum = None
    if parent is not None:
        prob_sum = edge_to_prob_sum[sorted_pair(v, parent)]
    children = [w for w in tree[v] if w != parent]
    if children:
        summaries = [rsummary(
            tree, leaf_to_name, edge_to_prob_sum, w, v) for w in children]
        s = '(' + ', '.join(summaries) + ')'
    else:
        s = leaf_to_name[v]
    if parent is None:
        return s + ';'
    else:
        return s + ':' + str(prob_sum)


def process_codon_column_helper(
        pos,
        builder,
        Q, primary_distn,
        tree, states, nstates, root,
        rho,
        D1, S1,
        benign_states, lethal_states, node_to_allowed_states,
        ):
    """
    Downstream of process_codon_column().

    """
    # Define the reference process.

    # Define the reference process rate matrix.
    Q_reference = nx.DiGraph()
    for sa, sb in Q.edges():
        weight = Q[sa][sb]['weight']
        if sa in benign_states and sb in benign_states:
            Q_reference.add_edge(sa, sb, weight=weight)

    # Define the column-specific initial state distribution.
    reference_weights = {}
    for s in range(nstates):
        if (s in primary_distn) and (s in benign_states):
            reference_weights[s] = primary_distn[s]
    reference_distn = _util.get_normalized_dict_distn(reference_weights)

    # Convert to dense representations of the reference process.
    Q_reference_dense = _density.rate_matrix_to_numpy_array(
            Q_reference, nodelist=states)
    reference_distn_dense = _density.dict_to_numpy_array(
            reference_distn, nodelist=states)

    # Define the diagonal associated with switching processes.
    L = np.array(
            [rho if s in benign_states else 0 for s in range(nstates)],
            dtype=float)

    # Define the compound process.

    # Define the compound process state space.
    ncompound = 2 * nstates
    compound_states = range(ncompound)

    # Initialize the column-specific compound rate matrix.
    Q_compound = nx.DiGraph()
    
    # Add block-diagonal entries of the default process component
    # of the compound process.
    for sa, sb in Q.edges():
        weight = Q[sa][sb]['weight']
        Q_compound.add_edge(nstates + sa, nstates + sb, weight=weight)

    # Add block-diagonal entries of the reference process component
    # of the compound process.
    for sa, sb in Q.edges():
        weight = Q[sa][sb]['weight']
        if sb in benign_states:
            Q_compound.add_edge(sa, sb, weight=weight)

    # Add off-block-diagonal entries directed from the reference
    # to the default process.
    for s in range(nstates):
        Q_compound.add_edge(s, nstates + s, weight=rho)

    # Define the column-specific initial state distribution.
    compound_weights = {}
    for s in range(ncompound):
        if (s in primary_distn) and (s in benign_states):
            compound_weights[s] = primary_distn[s]
    compound_distn = _util.get_normalized_dict_distn(compound_weights)

    # Convert to dense representations.
    Q_compound_dense = _density.rate_matrix_to_numpy_array(
            Q_compound, nodelist=compound_states)
    compound_distn_dense = _density.dict_to_numpy_array(
            compound_distn, nodelist=compound_states)

    # End compound process definition.

    # Define the t -> P callbacks for the true process
    # or for a time-discretized process, depending on cmdline flags.

    # Define an SD decomposition of the reference process.
    D0 = reference_distn_dense
    S0 = qtop.dot_square_diag(Q_reference_dense, qtop.pseudo_reciprocal(D0))

    # Define the decompositions.
    # Define the callbacks that converts branch length to prob matrix.
    sylvester_decomp = qtop.decompose_sylvester_v2(S0, S1, D0, D1, L)
    A0, B0, A1, B1, L, lam0, lam1, XQ = sylvester_decomp
    P_cb_compound_cont = functools.partial(qtop.getp_sylvester_v2,
            D0, A0, B0, A1, B1, L, lam0, lam1, XQ)

    # Accumulate posterior information about the site.
    # NOTE only the continuous non-discretized time model is used.
    P_cb_compound = P_cb_compound_cont
    accumulate_codon_site_summary(
            tree, node_to_allowed_states, root,
            nstates, ncompound,
            compound_distn_dense, P_cb_compound,
            pos, builder)


def process_codon_column(
        pos,
        builder,
        Q, primary_distn,
        tree, states, nstates, root,
        rho,
        D1, S1,
        names, name_to_leaf, genetic_code, codon_to_state,
        pos_to_benign_residues, pos_to_lethal_residues, codon_column,
        ):
    """
    This is called once per codon column.

    """
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

    # Define the compound process state space.
    ncompound = 2 * nstates
    compound_states = range(ncompound)

    # Define the map from node to allowed compound states.
    node_to_allowed_states = dict((n, set(compound_states)) for n in tree)
    for name, codon in zip(names, codon_column):
        leaf = name_to_leaf[name]
        codon = codon.upper()
        codon_state = codon_to_state[codon]
        node_to_allowed_states[leaf] = {codon_state, nstates + codon_state}

    # Continue processing the codon column.
    process_codon_column_helper(
            pos,
            builder,
            Q, primary_distn,
            tree, states, nstates, root,
            rho,
            D1, S1,
            benign_states, lethal_states, node_to_allowed_states,
            )


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

