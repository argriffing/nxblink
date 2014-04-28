from __future__ import print_function

#fn = 'em-data.txt'
fn = 'full-sequence-em.out'


def main():

    # read rates on and off from a hardcoded file name
    rates_on = []
    rates_off = []
    with open(fn) as fin:
        for line in fin:
            if line.startswith('rate on'):
                rates_on.append(float(line.split()[-1]))
            if line.startswith('rate off'):
                rates_off.append(float(line.split()[-1]))

    # write a table for R
    print('on', 'off', sep='\t')
    for i, data in enumerate(zip(rates_on, rates_off)):
        pos = i+1
        row = [pos] + list(data)
        print(*row, sep='\t')

main()

