"""
Interpret molecular cancer disease data in light of the genetic code.

Some possible interpretations of disease data,
as provided by the online p53 mutation databases, are as follows:

1) At each codon position, each amino acid is considered either lethal
or benign.  Every non-reference disease-associated amino acid at the
position is considered lethal and every amino acid that is not
disease-associated is considered benign.

2) As in the previous interpretation, at each codon position each
amino acid is considered either lethal or benign.  Synonymous
mutations and mutations with more than one nucleotide change per codon
are removed from the data set.  From the remaining data, every
disease-associated amino acid at the position is considered lethal and
every amino acid that is not disease-associated is considered benign.
In particular, if for a given codon position the original data
contains evidence of disease association for a given amino acid but
the only such evidence is through >1 nucleotide change from the
reference, then this amino acid will be interpreted as benign at that
position.

3) At each codon position, the disease state of each possible amino
acid is considered either lethal or benign or unknown.  Every
disease-associated amino acid is considered to be lethal.  Among the
amino acids without evidence of disease association, the amino acids
that are reachable by a single nucleotide point mutation from the
reference codon are considered to be benign.  The remaining amino
acids (those without evidence of disease-association and which are not
reachable by a single nucleotide point mutation from the reference
codon) are assumed to have an unknown (missing) disease state.

4) All amino acids except the reference amino acid
are considered to be lethal in the reference process.

"""
from __future__ import division, print_function, absolute_import

import contextlib
import argparse
import csv
import sys
from collections import defaultdict

UNKNOWN = 'UNKNOWN'
BENIGN = 'BENIGN'
LETHAL = 'LETHAL'

def hdist(a, b):
    return sum(1 if x != y else 0 for x, y in zip(a, b))

@contextlib.contextmanager
def open_in(filename):
    if filename == '-':
        yield sys.stdin
    else:
        with open(filename, 'rU') as fin:
            yield fin

@contextlib.contextmanager
def open_out(filename):
    if filename == '-':
        yield sys.stdout
    else:
        with open(filename, 'w') as fout:
            yield fout

def read_genetic_code(fin):
    """

    Parameters
    ----------
    fin : input stream
        Open text stream for reading the genetic code.

    Returns
    -------
    genetic_code : list
        List of (state, residue, codon) triples.

    """
    genetic_code = []
    for line in fin:
        line = line.strip()
        if line:
            state, residue, codon = line.split()
            state = int(state)
            residue = residue.upper()
            codon = codon.upper()
            if residue != 'STOP':
                triple = (state, residue, codon)
                genetic_code.append(triple)
    return genetic_code

def read_raw_disease_data(fin, genetic_code):
    """
    Read disease data that comes directly from the p53 website.

    """
    states, residues, codons = zip(*genetic_code)
    codons = set(codons)
    outrows = []

    codon_offset_index = 3
    wild_codon_index = 4
    mutant_codon_index = 5
    wild_aa_index = 6
    mutant_aa_index = 7

    reader = csv.reader(fin, dialect=csv.excel_tab)
    headers = reader.next()
    for row_index, row in enumerate(reader):

        # filter out mutations that are clearly not amino acid changes
        wild_aa = row[wild_aa_index]
        mutant_aa = row[mutant_aa_index]
        if mutant_aa in ('Stop', 'Fs.'):
            continue
        if wild_aa == mutant_aa:
            continue

        # get the wild codon state
        wild_codon = row[wild_codon_index]
        if wild_codon not in codons:
            continue

        # attempt to add the mutation to the list
        mutant_codon = row[mutant_codon_index]
        if mutant_codon not in codons:
            continue

        codon_offset = int(row[codon_offset_index])

        # put the row into disease data output format
        ntpos = -1
        codonpos = codon_offset
        exon = -1
        wcodon = wild_codon
        mcodon = mutant_codon
        wres = wild_aa.upper()
        mres = mutant_aa.upper()
        outrow = (ntpos, codonpos, exon, wcodon, mcodon, wres, mres)
        outrows.append(outrow)

    return outrows


def read_pre_processed_disease_data(fin):
    """
    Read disease data which has been pre-processed.

    """
    # NOTE nt_pos and codon_pos begin at 1 not 0
    # nt_pos, codon_pos, exon(???), wild_codon, mut_codon, wild_res, mut_res
    # 467 135 5 TGC TAC Cys Tyr
    rows = []
    for line_index, line in enumerate(fin):
        line = line.strip()
        ntpos, codonpos, exon, wcodon, mcodon, wres, mres  = line.split()
        ntpos = int(ntpos)
        codonpos = int(codonpos)
        wres = wres.upper()
        mres = mres.upper()
        wcodon = wcodon.upper()
        mcodon = mcodon.upper()
        if wres == mres:
            # skip synonymous disease
            #raise Exception('synonymous disease: ' + line)
            continue
        if len(mcodon) != 3:
            if not ('INS' in mcodon or 'DEL' in mcodon):
                raise Exception('unrecognized mutant codon: ' + mcodon)
            continue
        row = (ntpos, codonpos, exon, wcodon, mcodon, wres, mres)
        rows.append(row)
    return rows

def write_interpreted_disease_data(fout, interpreted_disease_data):
    header = ('position', 'residue', 'status')
    for row in [header] + interpreted_disease_data:
        print(*row, sep='\t', file=fout)

def interpret_disease_data(
        interpretation, code, disease_data,
        reference_codons, npositions):
    """

    Parameters
    ----------
    interpretation : one of {1, 2, 3, 4}
        Interpretation code; see the script docstring.
    code : sequence of triples
        Genetic code.
    disease_data : sequence of sequences
        Rows of disease data.
    reference_codons: sequence of codons
        Sequence of codons in the reference sequence.
    npositions : positive integer
        Number of codon positions in the sequence alignment.

    Returns
    -------
    out : triples
        (1-based codon position, mutant amino acid, disease state)

    """
    # Get the set of all residues.
    set_of_all_residues = set(r for s, r, c in code)

    # Define genetic translation according to the genetic code.
    codon_to_residue = dict((c, r) for s, r, c in code)

    # Get a sparse map from the codon position to the lethal residue set.
    pos_to_lethal_residues = defaultdict(set)
    for ntpos, codonpos, exon, wcodon, mcodon, wres, mres in disease_data:
        if interpretation in (1, 3):
            # Under these interpretations, each disease-associated
            # amino acid at each position is considered lethal.
            if wres != mres:
                pos_to_lethal_residues[codonpos].add(mres)
        elif interpretation == 2:
            # Under this interpretation, a given amino acid at a given position
            # is considered to be lethal if it is present in the
            # disease-associated data as a nucleotide point mutation
            # from the reference codon.
            if wres != mres:
                if hdist(wcodon, mcodon) == 1:
                    pos_to_lethal_residues[codonpos].add(mres)
        elif interpretation == 4:
            # Under this interpretation, the wild type amino acid
            # at the reference node is considered benign
            # and all other amino acids are considered lethal.
            # In this case, we do not even look at the disease data.
            pass
        else:
            raise ValueError('invalid interpretation code')
    reference_residues = [codon_to_residue[c] for c in reference_codons]
    if interpretation == 4:
        for pos in range(1, npositions+1):
            for r in set_of_all_residues:
                if r != reference_residues[pos-1]:
                    pos_to_lethal_residues[pos].add(r)
    pos_to_lethal_residues = dict(pos_to_lethal_residues)

    # For each codon, get the set of amino acids
    # reachable through zero or one point mutations.
    codon_to_nearby_residues = defaultdict(set)
    for s1, r1, c1 in code:
        for s2, r2, c2 in code:
            if hdist(c1, c2) < 2:
                codon_to_nearby_residues[c1].add(r2)
    codon_to_nearby_residues = dict(codon_to_nearby_residues)

    interpreted_data = []
    for pos in range(1, npositions+1):

        # Partition the residues according to disease state.
        lethal_residues = pos_to_lethal_residues.get(pos, set())
        other_residues = set_of_all_residues - lethal_residues
        if interpretation in (1, 2, 4):
            # Under these interpretations, if an amino acid is not
            # lethal then it is considered to be benign.
            # All residues have known lethal or benign disease state.
            benign_residues = other_residues
            unknown_residues = set()
        elif interpretation == 3:
            # Under this interpretation, an amino acid not known to be lethal
            # is considered benign if it is reachable by a single nucleotide
            # point mutation from the reference codon.
            # Amino acids not known to be lethal and which are not reachable
            # by a single nucleotide point mutation from the reference codon
            # are assumed to have an unknown or missing disease state.
            reference_codon = reference_codons[pos-1]
            nearby_residues = codon_to_nearby_residues[reference_codon]
            benign_residues = other_residues & nearby_residues
            unknown_residues = other_residues - nearby_residues
        else:
            raise ValueError('invalid interpretation code')

        # Report the disease states.
        for r in sorted(set_of_all_residues):
            if r in lethal_residues:
                disease_state = LETHAL
            elif r in benign_residues:
                disease_state = BENIGN
            elif r in unknown_residues:
                disease_state = UNKNOWN
            else:
                raise ValueError('incomplete disease state partition')
            row = (pos, r, disease_state)
            interpreted_data.append(row)
    
    return interpreted_data


def main(args):
    with open(args.code, 'r') as fin:
        code = read_genetic_code(fin)
    with open_in(args.infile) as fin:
        disease_data = read_raw_disease_data(fin, code)
    with open(args.sequence, 'r') as fin:
        reference_nucleotides = ''.join(fin.read().split()).upper()
        n = len(reference_nucleotides)
        ncodons, remainder = divmod(n, 3)
        if remainder:
            raise Exception('unexpected sequence length: ' + str(n))
        reference_codons = [
                reference_nucleotides[3*i : 3*(i+1)] for i in range(ncodons)]
        if ncodons != args.npositions:
            raise Exception(
                    'expected %d codon positions in the reference sequence '
                    'but found %d' % (args.npositions, ncodons))
    interpreted_disease_data = interpret_disease_data(
            args.interpretation, code, disease_data,
            reference_codons, args.npositions)
    with open_out(args.outfile) as fout:
        write_interpreted_disease_data(fout, interpreted_disease_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', '--infile', default='-',
            help='disease data input file')
    parser.add_argument('-o', '--outfile', default='-',
            help='interpreted disease data output file')
    parser.add_argument('--interpretation',
            required=True, type=int, default=1, choices=(1, 2, 3, 4),
            help='interpretation number')
    parser.add_argument('--code', required=True,
            help='genetic code input file')
    parser.add_argument('--sequence', required=True,
            help='reference codon sequence file')
    parser.add_argument('--npositions', default=393,
            help='number of positions in the codon alignment')
    main(parser.parse_args())

