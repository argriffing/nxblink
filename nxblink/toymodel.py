"""
Hardcoded toy models for tests and demos.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

__all__ = ['BlinkModelA', 'BlinkModelB', 'BlinkModelC']


class BlinkModel(object):
    """
    Blinking process toy model base class.

    """
    @classmethod
    def get_blink_distn(self):
        rate_on = self.get_rate_on()
        rate_off = self.get_rate_off()
        total = rate_on + rate_off
        distn = {
                0 : rate_off / total,
                1 : rate_on / total,
                }
        return distn

    @classmethod
    def get_Q_blink(self):
        Q_blink = nx.DiGraph()
        Q_blink.add_weighted_edges_from((
            (0, 1, self.get_rate_on()),
            (1, 0, self.get_rate_off()),
            ))
        return Q_blink

    @classmethod
    def get_primary_to_tol(self):
        """
        Return a map from primary state to tolerance track name.

        This is like a genetic code mapping codons to amino acids.

        """
        primary_to_tol = {
                0 : 'T0',
                1 : 'T0',
                2 : 'T1',
                3 : 'T1',
                4 : 'T2',
                5 : 'T2',
                }
        return primary_to_tol

    @classmethod
    def get_T_and_root(self):
        # rooted tree, deliberately without branch lengths
        T = nx.DiGraph()
        T.add_edges_from([
            ('N1', 'N0'),
            ('N1', 'N2'),
            ('N1', 'N5'),
            ('N2', 'N3'),
            ('N2', 'N4'),
            ])
        return T, 'N1'

    @classmethod
    def get_edge_to_blen(self):
        edge_to_blen = {
                ('N1', 'N0') : 1.0,
                ('N1', 'N2') : 1.0,
                ('N1', 'N5') : 1.0,
                ('N2', 'N3') : 1.0,
                ('N2', 'N4') : 1.0,
                }
        return edge_to_blen


class BlinkModelA(BlinkModel):
    """
    Plain toy blinking model with much symmetry.

    """
    @classmethod
    def get_rate_on(self):
        return 1.0

    @classmethod
    def get_rate_off(self):
        return 1.0

    @classmethod
    def get_primary_distn(self):
        nprimary = 6
        return dict((i, 1/nprimary) for i in range(nprimary))

    @classmethod
    def get_Q_primary(self):
        """
        This is like a symmetric codon rate matrix that is not normalized.

        """
        rate = 1
        Q_primary = nx.DiGraph()
        Q_primary.add_weighted_edges_from((
            (0, 1, rate),
            (0, 2, rate),
            (1, 0, rate),
            (1, 3, rate),
            (2, 0, rate),
            (2, 3, rate),
            (2, 4, rate),
            (3, 1, rate),
            (3, 2, rate),
            (3, 5, rate),
            (4, 2, rate),
            (4, 5, rate),
            (5, 3, rate),
            (5, 4, rate),
            ))
        return Q_primary

    @classmethod
    def get_edge_to_rate(self):
        edge_to_rate = {
                ('N1', 'N0') : 0.5,
                ('N1', 'N2') : 0.5,
                ('N1', 'N5') : 0.5,
                ('N2', 'N3') : 0.5,
                ('N2', 'N4') : 0.5,
                }
        return edge_to_rate


class BlinkModelB(BlinkModel):
    """
    This is a more complicated model than the first code2x3 model.

     * same data (alignment and disease)
     * same tree shape and root
     * let one branch length be twice as long
     * let another branch length be half as long
     * set a couple of branch lengths to zero
     * remove the synonymous transition between states P4 <--> P5
     * force the primary process equilibrium distribution to be non-uniform
       by increasing the equilbrium frequency of state P1 by doubling its
       incoming rates and halving its outgoing rates
     * let the blinking rates be unequal, in particular let the off -> on
       rate be doubled from 1 to 2 and let the on -> off rate be cut in half
       from 1 to 1/2.  This implies a prior blink state distribution
       of 4/5 on, 1/5 off.

    """

    @classmethod
    def get_rate_on(self):
        return 2.0

    @classmethod
    def get_rate_off(self):
        return 0.5

    @classmethod
    def get_primary_distn(self):
        distn = {
                0 : 1 / 9,
                1 : 4 / 9,
                2 : 1 / 9,
                3 : 1 / 9,
                4 : 1 / 9,
                5 : 1 / 9,
                }
        return distn

    @classmethod
    def get_Q_primary(self):
        """
        This is like an unnormalized codon rate matrix.

        This rate matrix has the following differences.
        The primary state 1 gets more probability,
        so it has twice as much rate in, and half as much rate out.
        The synonymous 4 <--> 5 transition edge is removed in this model.

        """
        rate = 1
        Q_primary = nx.DiGraph()
        Q_primary.add_weighted_edges_from((
            (0, 1, 2.0*rate),
            (0, 2, rate),
            (1, 0, 0.5*rate),
            (1, 3, 0.5*rate),
            (2, 0, rate),
            (2, 3, rate),
            (2, 4, rate),
            (3, 1, 2.0*rate),
            (3, 2, rate),
            (3, 5, rate),
            (4, 2, rate),
            (5, 3, rate),
            ))
        return Q_primary

    @classmethod
    def get_edge_to_rate(self):
        """
        Some of these branch lengths are larger and smaller than usual.

        The previous model had all branches with lengths 0.5,
        whereas in this model one arbitrary branch is twice as long,
        and another arbitrary branch is half as long.

        """
        rate = 0.5
        edge_to_rate = {
                ('N1', 'N0') : rate,
                ('N1', 'N2') : rate,
                ('N1', 'N5') : rate,
                ('N2', 'N3') : 2.0*rate,
                ('N2', 'N4') : 0.5*rate,
                }
        return edge_to_rate


class BlinkModelC(BlinkModelB):
    """
    Same as blink model b, except a couple of branch lengths are now zero.

    """
    @classmethod
    def get_edge_to_rate(self):
        rate = 0.5
        edge_to_rate = {
                ('N1', 'N0') : 0,
                ('N1', 'N2') : 0,
                ('N1', 'N5') : rate,
                ('N2', 'N3') : 2.0*rate,
                ('N2', 'N4') : 0.5*rate,
                }
        return edge_to_rate
