"""
Model specification code.

This module is defined in analogy to examples/codon2x3/nxmodel.py
It includes details that are specific to a codon rate matrix
and which cannot be generalized across all blinking models.

"""
from __future__ import division, print_function, absolute_import

from StringIO import StringIO

import networkx as nx

from create_mg94 import create_mg94
import app_helper


__all__ = [
        'get_Q_primary', 'get_primary_to_tol',
        'get_tree_info', 'get_edge_to_blen',
        ]


def _get_genetic_code():
    """
    Read the genetic code.

    """
    with open('universal.code.txt') as fin:
        return app_helper.read_genetic_code(fin)


def _get_jeff_params_e():
    """
    Copypasted from raoteh/examples/p53.py
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

    # Use blinking rates instead of omega.
    #omega = 0.37767
    omega = 1.0

    tree_string = """((((((Has:  0.0073385245,Ptr:  0.0073385245):  0.0640509640,Ppy:  0.0713894884):  0.0542000118,(((Mmu:  0.0025462071,Mfu:  0.0025462071):  0.0000000000,Mfa:  0.0025462071):  0.0318638454,Cae:  0.0344100525):  0.0911794477):  0.1983006745,(Mim:  0.3238901743,Tgl:  0.3238901743):  0.0000000004):  0.2277808059,((((((Mum:  0.1797319785,Rno:  0.1797319785):  0.1566592047,Mun:  0.3363911832):  0.0192333544,(Cgr:  0.1074213106,Mau:  0.1074213106):  0.2482032271):  0.0447054051,Sju:  0.4003299428):  0.1000000288,(Cpo:  0.4170856630,Mmo:  0.4170856630):  0.0832443086):  0.0250358682,(Ocu:  0.4149196099,Opr:  0.4149196099):  0.1104462299):  0.0263051408):  0.0000000147,(Sar:  0.4524627987,((Fca:  0.2801000848,Cfa:  0.2801000848):  0.1338023902,((Bta:  0.0880000138,Oar:  0.0880000138):  0.1543496707,Dle:  0.2423496845):  0.1715527905):  0.0385603236):  0.0992081966);"""
    fin = StringIO(tree_string)
    tree, root, leaf_name_pairs = app_helper.read_newick(fin)
    return (kappa, omega, A, C, T, G, rho,
            tree, root, leaf_name_pairs)


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


def get_Q_primary_and_distn():
    """
    """
    # Get the parameters inferred according to the project with Jeff and Liwen.
    genetic_code = _get_genetic_code()
    ret = _get_jeff_params_e()
    kappa, omega, A, C, T, G, rho, tree, root, leaf_name_pairs = ret
    Q_primary, primary_distn, state_to_residue, residue_to_part = create_mg94(
            A, C, G, T,
            kappa, omega, genetic_code,
            target_expected_rate=1.0)
    return Q_primary, primary_distn


def get_primary_to_tol():
    """
    Return a map from primary state to tolerance track name.

    This is like a genetic code mapping codons to amino acids.

    """
    # Define the default process codon rate matrix
    # and distribution and tolerance classes.
    genetic_code = _get_genetic_code()
    ret = _get_jeff_params_e()
    kappa, omega, A, C, T, G, rho, tree, root, leaf_name_pairs = ret
    info = create_mg94(
            A, C, G, T,
            kappa, omega, genetic_code,
            target_expected_rate=1.0)
    Q, primary_distn, state_to_residue, residue_to_part = info
    primary_to_part = dict(
            (i, residue_to_part[r]) for i, r in state_to_residue.items())
    return primary_to_part


def get_tree_info():
    ret = _get_jeff_params_e()
    kappa, omega, A, C, T, G, rho, tree, root, leaf_name_pairs = ret
    tree, root, edge_to_blen = _tree_helper(tree, root)
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)
    return tree, root, edge_to_blen, name_to_leaf


class Model(object):
    """
    This includes everything except the data.

    """
    def __init__(self,
            kappa, omega, A, C, G, T,
            rate_on, rate_off,
            tree, root, edge_to_blen, edge_to_rate,
            ):
        """

        """
        # store input arguments
        self._kappa = kappa
        self._omega = omega
        self._A = A
        self._C = C
        self._G = G
        self._T = T
        self._rate_on = rate_on
        self._rate_off = rate_off
        self._tree = tree
        self._edge_to_blen = edge_to_blen
        self._edge_to_rate = edge_to_rate

        # construct and store more details of the model
        genetic_code = _get_genetic_code()
        info = create_mg94(
                self._A, self._C, self._G, self._T,
                self._kappa, self._omega, genetic_code,
                target_expected_rate=1.0)
        Q, primary_distn, state_to_residue, residue_to_part = info
        primary_to_part = dict(
                (i, residue_to_part[r]) for i, r in state_to_residue.items())
        self._Q = Q
        self._primary_distn = primary_distn
        self._primary_to_part = primary_to_part

    def get_rate_on(self):
        return self._rate_on

    def get_rate_off(self):
        return self._rate_off

    def get_blink_distn(self):
        rate_on = self.get_rate_on()
        rate_off = self.get_rate_off()
        total = rate_on + rate_off
        distn = {
                0 : rate_off / total,
                1 : rate_on / total,
                }
        return distn

    def get_Q_blink(self):
        Q_blink = nx.DiGraph()
        Q_blink.add_weighted_edges_from((
            (0, 1, self.get_rate_on),
            (1, 0, self.get_rate_off),
            ))
        return Q_blink

    def get_primary_to_tol(self):
        return self._primary_to_part

    def get_T_and_root(self):
        """
        Return the rooted tree, deliberately without branch lengths.

        """
        return self._tree, self._root

    def get_edge_to_blen(self):
        return self._edge_to_blen

    def get_edge_to_rate(self):
        return self._edge_to_rate

    def get_primary_distn(self):
        return self._primary_distn

    def get_Q_primary(self):
        return self._Q

