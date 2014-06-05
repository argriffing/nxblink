"""
This toy model is described in raoteh/examples/codon2x3.

mini glossary
tol -- tolerance
traj -- trajectory
fg -- foreground track
bg -- background track
ell -- expected log likelihood

The tree is rooted and edges are directed.
For each substate track, each permanent node maps to a list of events.
Each event is a handle mapping to some event info giving the
time of the event along the branch and the nature of the transition,
if any, associated with the event.

We can use the jargon that 'events' are associated with
locations in the tree defined by a directed edge
and a distance along that edge.
Events will usually be associated with a state transition,
but 'incomplete events' will not have such an association.

The process is separated into multiple 'tracks' -- a primary process
track and one track for each of the tolerance processes.
The track trajectories are not independent of each other.

Begin a work-in-progress to use the nxblink.summary module
instead of the less comprehensive example-specific summary code.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
from functools import partial
import argparse

import networkx as nx
import numpy as np

import nxmctree
from nxmctree.sampling import sample_history

import nxblink
from nxblink.model import get_Q_blink, get_Q_meta
#from nxblink.model import get_interaction_map
from nxblink.util import get_node_to_tm
from nxblink.navigation import gen_segments
from nxblink.maxlikelihood import get_blink_rate_mle
from nxblink.trajectory import Trajectory
from nxblink.summary import (BlinkSummary,
        get_ell_init_contrib, get_ell_dwell_contrib, get_ell_trans_contrib)
from nxblink.raoteh import (
        init_tracks, gen_samples,
        update_track_data_for_zero_blen)
from nxblink.toymodel import BlinkModelA, BlinkModelB, BlinkModelC
from nxblink.toydata import DataA, DataB, DataC, DataD
from nxblink.summary import Summary
from nxblink.em import get_ll_root, get_ll_dwell, get_ll_trans


def get_blink_dwell_times(T, node_to_tm, blink_tracks):
    """
    This function is only for reporting results.

    """
    dwell_off = 0
    dwell_on = 0
    for edge in T.edges():
        va, vb = edge
        for tma, tmb, track_to_state in gen_segments(
                edge, node_to_tm, blink_tracks):
            blen = tmb - tma
            for track in blink_tracks:
                state = track_to_state[track.name]
                if state == False:
                    dwell_off += blen
                elif state == True:
                    dwell_on += blen
                else:
                    raise Exception
    return dwell_off, dwell_on


#TODO update this code to use the more sophisticated raoteh.gen_samples
#TODO in particular avoid boilerplate code related to generating the samples
#TODO and focus on computing and reporting summaries of the sampled histories
def run(model, data, nburnin, nsamples):
    """
    The args are the same as for nxblink.raoteh.gen_samples(...).

    """
    # Extract some information from the model.
    T, root = model.get_T_and_root()
    edge_to_blen = model.get_edge_to_blen()
    edge_to_rate = model.get_edge_to_rate()
    primary_to_tol = model.get_primary_to_tol()
    primary_distn = model.get_primary_distn()
    blink_distn = model.get_blink_distn()
    Q_primary = model.get_Q_primary()
    Q_blink = model.get_Q_blink()
    Q_meta = get_Q_meta(Q_primary, primary_to_tol)
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    # Initialize the log likelihood contribution
    # of the initial state at the root.
    ell_init_contrib = 0
    
    # Initialize contributions of the dwell times on each edge
    # to the expected log likelihood.
    edge_to_ell_dwell_contrib = defaultdict(float)

    # Initialize contributions of the transition events on each edge
    # to the expected log likelihood.
    edge_to_ell_trans_contrib = defaultdict(float)

    # sample correlated trajectories using rao teh on the blinking model
    va_vb_type_to_count = defaultdict(int)
    total_dwell_off = 0
    total_dwell_on = 0
    blink_summary = BlinkSummary()

    # initialize a summary
    summary = Summary(T, root, node_to_tm, primary_to_tol, Q_primary)

    #
    for pri_track, tol_tracks in gen_samples(model, data, nburnin, nsamples):

        # Compute a summary using the more comprehensive code.
        summary.on_sample(pri_track, tol_tracks)

        # Compute a summary using the ad hoc code.
        blink_summary.on_sample(T, root, node_to_tm, edge_to_rate,
                pri_track, tol_tracks, primary_to_tol)

        # Summarize the trajectories.
        for edge in T.edges():
            va, vb = edge
            for track in tol_tracks:
                for ev in track.events[edge]:
                    transition = (ev.sa, ev.sb)
                    if ev.sa == ev.sb:
                        raise Exception('self-transitions should not remain')
                    if transition == (False, True):
                        va_vb_type_to_count[va, vb, 'on'] += 1
                    elif transition == (True, False):
                        va_vb_type_to_count[va, vb, 'off'] += 1
            for ev in pri_track.events[edge]:
                transition = (ev.sa, ev.sb)
                if ev.sa == ev.sb:
                    raise Exception('self-transitions should not remain')
                if primary_to_tol[ev.sa] == primary_to_tol[ev.sb]:
                    va_vb_type_to_count[va, vb, 'syn'] += 1
                else:
                    va_vb_type_to_count[va, vb, 'non'] += 1
        dwell_off, dwell_on = get_blink_dwell_times(T, node_to_tm, tol_tracks)
        total_dwell_off += dwell_off
        total_dwell_on += dwell_on

        # Get the contribution of the prior probabilty of the root state
        # to the expected log likelihood.
        ll_init = get_ell_init_contrib(
                root,
                primary_distn, blink_distn,
                pri_track, tol_tracks, primary_to_tol)
        ell_init_contrib += ll_init

        # Get the contributions of the dwell times on each edge
        # to the expected log likelihood.
        d = get_ell_dwell_contrib(
                T, root, node_to_tm, edge_to_rate,
                Q_primary, Q_blink, Q_meta,
                pri_track, tol_tracks, primary_to_tol)
        for k, v in d.items():
            edge_to_ell_dwell_contrib[k] += v

        # Get the contributions of the transition events on each edge
        # to the expected log likelihood.
        d = get_ell_trans_contrib(
                T, root, edge_to_rate,
                Q_primary, Q_blink,
                pri_track, tol_tracks)
        for k, v in d.items():
            edge_to_ell_trans_contrib[k] += v

    # report infos

    # summary of the run
    print('burnin:', nburnin)
    print('samples after burnin:', nsamples)

    # transition expectations on edges by transition type
    for va_vb_type, count in sorted(va_vb_type_to_count.items()):
        va, vb, s = va_vb_type
        print(va, '->', vb, s, ':', count / nsamples)
    print()

    # initial state contribution
    print('root state contribution to expected log likelihood:')
    print(ell_init_contrib / nsamples)
    print()

    # edge dwell
    print('edge dwell time contributions to expected log likelihood:')
    for edge, contrib in sorted(edge_to_ell_dwell_contrib.items()):
        va, vb = edge
        print(va, '->', vb, ':', contrib / nsamples)
    print()
    print('total dwell time contribution to expected log likelihood:')
    print(sum(edge_to_ell_dwell_contrib.values()) / nsamples)
    print()

    # edge transition
    print('edge transition event contributions to expected log likelihood:')
    for edge, contrib in sorted(edge_to_ell_trans_contrib.items()):
        va, vb = edge
        print(va, '->', vb, ':', contrib / nsamples)
    print()
    print('total transition event contribution to expected log likelihood:')
    print(sum(edge_to_ell_trans_contrib.values()) / nsamples)
    print()

    print('dwell off:', total_dwell_off / nsamples)
    print('dwell on :', total_dwell_on / nsamples)
    print()

    # report infos per column
    print(
            blink_summary.xon_root_count,
            blink_summary.off_root_count,
            blink_summary.off_xon_count,
            blink_summary.xon_off_count,
            blink_summary.off_xon_dwell,
            blink_summary.xon_off_dwell,
            blink_summary.nsamples,
            sep='\t')
    ml_rate_on, ml_rate_off =  get_blink_rate_mle(
            blink_summary.xon_root_count,
            blink_summary.off_root_count,
            blink_summary.off_xon_count,
            blink_summary.xon_off_count,
            blink_summary.off_xon_dwell,
            blink_summary.xon_off_dwell,
            )
    print('ml rate on:', ml_rate_on)
    print('ml rate off:', ml_rate_off)
    print()

    print('comprehensive summary:')
    print(summary)
    print()

    # compute some dense functions related to the rate matrix
    nprimary = len(Q_primary)
    pre_Q_dense = np.zeros((nprimary, nprimary), dtype=float)
    for sa in Q_primary:
        for sb in Q_primary[sa]:
            rate = Q_primary[sa][sb]['weight']
            pre_Q_dense[sa, sb] = rate
    distn_dense = np.zeros(nprimary, dtype=float)
    for state, p in primary_distn.items():
        distn_dense[state] = p
    edges, edge_rates = zip(*edge_to_rate.items())

    # functions of summaries for computing log likelihood
    rate_on = Q_blink[0][1]['weight']
    rate_off = Q_blink[1][0]['weight']
    ll_root = get_ll_root(summary, distn_dense, rate_on, rate_off)
    ll_dwell = get_ll_dwell(summary,
            pre_Q_dense, distn_dense, rate_on, rate_off, edges, edge_rates)
    ll_trans = get_ll_trans(summary,
            pre_Q_dense, distn_dense, rate_on, rate_off, edges, edge_rates)

    print('log likelihood contributions calculated from comprehensive summary:')
    print('root ll contrib:', ll_root)
    print('dwell ll contrib:', ll_dwell)
    print('trans ll contrib:', ll_trans)


def main(args):
    models = {'a' : BlinkModelA, 'b' : BlinkModelB, 'c' : BlinkModelC}
    model = models[args.model]
    data = [DataA, DataB, DataC, DataD][args.data]
    nburnin = args.k
    nsamples = args.k * args.k
    run(model, data, nburnin, nsamples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
            choices=('a', 'b', 'c'), default='a',
            help='specify the model complexity')
    parser.add_argument('--data',
            choices=(0, 1, 2, 3), type=int, default=0,
            help=(
                'specify the data level ('
                '0: no data, '
                '1: alignment only, '
                '2: alignment and human disease data, ',
                '3: alignment and human disease data '
                'and assume all others benign)'))
    parser.add_argument('--k', type=int, default=80,
            help='square root of number of samples')
    parser.add_argument('--out-specific',
            help='model-specific output file')
    parser.add_argument('--out-generic',
            help='generic output file')
    args = parser.parse_args()
    main(args)

