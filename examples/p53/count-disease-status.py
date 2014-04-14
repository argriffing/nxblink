"""

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import argparse
import csv

from numpy.testing import assert_equal


def main(args):
    if args.ncols is None:
        print('analysis of all codon positions')
    else:
        print('analysis of first', args.ncols, 'codon positions')
    print()

    status_dict = defaultdict(int)
    positions = set()
    with open(args.disease) as fin:
        reader = csv.DictReader(fin, delimiter='\t')
        for d in reader:
            pos = int(d['position'])
            if (args.ncols is not None) and pos > args.ncols:
                continue
            positions.add(pos)
            disease_status = d['status']
            status_dict[disease_status] += 1
    npositions = len(positions)
    print('number of positions:', npositions)
    print()

    lethal_count = status_dict['LETHAL']
    free_benign_count = status_dict['BENIGN'] - npositions
    total_count = lethal_count + free_benign_count
    assert_equal(total_count, 19 * npositions)
    print('counts:')
    print('free benign count:', free_benign_count)
    print('lethal count:', lethal_count)
    print('total count:', total_count)
    print()

    free_benign_proportion = free_benign_count / total_count
    lethal_proportion = lethal_count / total_count
    print('proportions:')
    print('free benign proportion:', free_benign_proportion)
    print('lethal proportion:', lethal_proportion)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncols', type=int,
            help='number of codon columns')
    parser.add_argument('--disease', default='int1.out',
            help='disease data file')
    args = parser.parse_args()
    main(args)

