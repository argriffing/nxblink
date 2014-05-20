"""
Defines a class to be used with nxblink.raoteh.gen_samples.

"""
from __future__ import division, print_function, absolute_import


class Data(object):

    def __init__(self,
            genetic_code, benign_residues, lethal_residues,
            all_nodes, names, name_to_leaf, human_leaf,
            primary_to_tol, codon_to_state,
            codon_column,
            ):
        """

        """
        # store some args
        self._all_nodes = all_nodes
        self._names = names
        self._name_to_leaf = name_to_leaf
        self._human_leaf = human_leaf
        self._primary_to_tol = primary_to_tol
        self._codon_to_state = codon_to_state
        self._codon_column = codon_column

        # Define and store the column-specific disease states
        # and the benign states for the reference (human) node in the tree.
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
                        'but residue %s '
                        'was found to be neither' % r)
        self._benign_states = benign_states
        self._lethal_states = lethal_states

    def get_data(self):
        data = {}
        data.update({'PRIMARY' : self.get_primary_data()})
        data.update(self.get_tolerance_data())
        return data

    def get_primary_data(self):

        # add the primary node_to_fset constraints implied by the alignment
        primary_map = {}
        all_primary_states = set(self._primary_to_tol)
        for v in self._all_nodes:
            primary_map[v] = all_primary_states
        for name, codon in zip(self._names, self._codon_column):
            leaf = name_to_leaf[name]
            primary_map[leaf] = {self._codon_to_state[codon]}
        return primary_map

    def get_tolerance_data(self):

        # add the tolerance node_to_fset constraints implied by the alignment
        tolerance_map = {}
        all_parts = set(self._primary_to_tol.values())
        for part in all_parts:
            tolerance_map[part] = {}
            for v in self._all_nodes:
                tolerance_map[part][v] = {False, True}
            for name, codon in zip(self._names, self._codon_column):
                leaf = self._name_to_leaf[name]
                primary_state = self._codon_to_state[codon]
                observed_part = self._primary_to_tol[primary_state]
                if part == observed_part:
                    tolerance_map[part][leaf] = {True}
                else:
                    tolerance_map[part][leaf] = {False, True}

        # TODO use something like benign_parts instead of benign_states
        # adjust the tolerance constraints using human disease data
        for primary_state in self._benign_states:
            part = self._primary_to_tol[primary_state]
            fset = {True} & tolerance_map[part][self._human_leaf]
            tolerance_map[part][self._human_leaf] = fset
        for primary_state in self._lethal_states:
            part = self._primary_to_tol[primary_state]
            fset = {False} & tolerance_map[part][self._human_leaf]
            tolerance_map[part][self._human_leaf] = fset

        return tolerance_map

