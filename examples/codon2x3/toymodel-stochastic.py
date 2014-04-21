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
from nxblink.model import get_Q_blink, get_Q_meta, get_interaction_map
from nxblink.util import get_node_to_tm
from nxblink.navigation import gen_segments
from nxblink.maxlikelihood import get_blink_rate_mle
from nxblink.trajectory import Trajectory
from nxblink.summary import (BlinkSummary,
        get_ell_init_contrib, get_ell_dwell_contrib, get_ell_trans_contrib)
from nxblink.raoteh import (
        init_tracks, gen_samples,
        update_track_data_for_zero_blen)

import nxmodel
import nxmodelb


###############################################################################
# Classes and functions for steps of Rao Teh iteration.


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


def run(model, primary_to_tol, interaction_map, track_to_node_to_data_fset):

    # Get the rooted directed tree shape.
    T, root = model.get_T_and_root()

    # Get the map from ordered tree edge to branch length.
    # The branch length has complicated units.
    # It is the expected number of primary process transitions
    # along the branch conditional on all tolerance classes being tolerated.
    edge_to_blen = model.get_edge_to_blen()

    # Initialize the map from edge to rate.
    edge_to_rate = dict((k, 1) for k in edge_to_blen)

    #TODO for testing
    edge_to_blen, edge_to_rate = edge_to_rate, edge_to_blen

    # Convert the branch length map to a node time map.
    node_to_tm = get_node_to_tm(T, root, edge_to_blen)

    # Define the uniformization factor.
    uniformization_factor = 2

    # Define the primary rate matrix.
    Q_primary = model.get_Q_primary()

    # Define the prior primary state distribution.
    primary_distn = model.get_primary_distn()
    nprimary = 6

    # Normalize the primary rate matrix to have expected rate 1.
    expected_primary_rate = 0
    for sa, sb in Q_primary.edges():
        p = primary_distn[sa]
        rate = Q_primary[sa][sb]['weight']
        expected_primary_rate += p * rate
    #
    #print('pure primary process expected rate:')
    #print(expected_primary_rate)
    #print()
    #
    for sa, sb in Q_primary.edges():
        Q_primary[sa][sb]['weight'] /= expected_primary_rate

    # Define primary trajectory.
    primary_track = Trajectory(
            name='PRIMARY', data=track_to_node_to_data_fset['PRIMARY'],
            history=dict(), events=dict(),
            prior_root_distn=primary_distn, Q_nx=Q_primary,
            uniformization_factor=uniformization_factor)

    # Define the rate matrix for a single blinking trajectory.
    rate_on = model.get_rate_on()
    rate_off = model.get_rate_off()
    Q_blink = get_Q_blink(rate_on=rate_on, rate_off=rate_off)
    blink_distn = model.get_blink_distn()

    # Define rates from a primary state to adjacent primary states
    # controlled by a given tolerance class.
    Q_meta = get_Q_meta(Q_primary, primary_to_tol)

    # Define tolerance process trajectories.
    tolerance_tracks = []
    for name in ('T0', 'T1', 'T2'):
        track = Trajectory(
                name=name, data=track_to_node_to_data_fset[name],
                history=dict(), events=dict(),
                prior_root_distn=blink_distn, Q_nx=Q_blink,
                uniformization_factor=uniformization_factor)
        tolerance_tracks.append(track)

    # Update track data, accounting for branches with length zero.
    tracks = [primary_track] + tolerance_tracks
    update_track_data_for_zero_blen(T, edge_to_blen, edge_to_rate, tracks)

    # Initialize the tracks.
    init_tracks(T, root, node_to_tm, edge_to_rate,
            primary_to_tol, Q_primary,
            primary_track, tolerance_tracks, interaction_map)

    # Initialize the log likelihood contribution
    # of the initial state at the root.
    ell_init_contrib = 0
    
    # Initialize contributions of the dwell times on each edge
    # to the expected log likelihood.
    edge_to_ell_dwell_contrib = defaultdict(float)
    
    # Initialize contributions of the dwell times on each edge
    # to the expected log likelihood.
    edge_to_ell_dwell_contrib = defaultdict(float)

    # Initialize contributions of the transition events on each edge
    # to the expected log likelihood.
    edge_to_ell_trans_contrib = defaultdict(float)

    # sample correlated trajectories using rao teh on the blinking model
    va_vb_type_to_count = defaultdict(int)
    nsamples = args.k * args.k
    burnin = nsamples // 10
    ncounted = 0
    total_dwell_off = 0
    total_dwell_on = 0
    blink_summary = BlinkSummary()
    for i, (pri_track, tol_tracks) in enumerate(gen_samples(
            T, root, node_to_tm, edge_to_rate,
            primary_to_tol, Q_meta,
            primary_track, tolerance_tracks, interaction_map)):
        nsampled = i+1
        if nsampled <= burnin:
            continue

        # Compute a summary.
        blink_summary.on_sample(T, root, node_to_tm, edge_to_rate,
                primary_track, tolerance_tracks, primary_to_tol)

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
                primary_track, tolerance_tracks, primary_to_tol)
        ell_init_contrib += ll_init

        # Get the contributions of the dwell times on each edge
        # to the expected log likelihood.
        d = get_ell_dwell_contrib(
                T, root, node_to_tm, edge_to_rate,
                Q_primary, Q_blink, Q_meta,
                primary_track, tolerance_tracks, primary_to_tol)
        for k, v in d.items():
            edge_to_ell_dwell_contrib[k] += v

        # Get the contributions of the transition events on each edge
        # to the expected log likelihood.
        d = get_ell_trans_contrib(
                T, root, edge_to_rate,
                Q_primary, Q_blink,
                primary_track, tolerance_tracks)
        for k, v in d.items():
            edge_to_ell_trans_contrib[k] += v

        # Loop control.
        ncounted += 1
        if ncounted == nsamples:
            break

    # report infos

    # summary of the run
    print('burnin:', burnin)
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


def main(args):
    """

    """
    # define the model
    if args.model == 'nxmodel':
        model = nxmodel
    elif args.model == 'nxmodelb':
        model = nxmodelb
    else:
        raise Exception

    # Get the analog of the genetic code.
    # Define track interactions.
    primary_to_tol = model.get_primary_to_tol()
    interaction_map = get_interaction_map(primary_to_tol)

    if args.data == 0:
        # No data.
        data = {
                'PRIMARY' : {
                    'N0' : {0, 1, 2, 3, 4, 5},
                    'N1' : {0, 1, 2, 3, 4, 5},
                    'N2' : {0, 1, 2, 3, 4, 5},
                    'N3' : {0, 1, 2, 3, 4, 5},
                    'N4' : {0, 1, 2, 3, 4, 5},
                    'N5' : {0, 1, 2, 3, 4, 5},
                    },
                'T0' : {
                    'N0' : {False, True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {False, True},
                    'N4' : {False, True},
                    'N5' : {False, True},
                    },
                'T1' : {
                    'N0' : {False, True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {False, True},
                    'N4' : {False, True},
                    'N5' : {False, True},
                    },
                'T2' : {
                    'N0' : {False, True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {False, True},
                    'N4' : {False, True},
                    'N5' : {False, True},
                    },
                }
    elif args.data == 1:
        # Alignment data only.
        data = {
                'PRIMARY' : {
                    'N0' : {0},
                    'N1' : {0, 1, 2, 3, 4, 5},
                    'N2' : {0, 1, 2, 3, 4, 5},
                    'N3' : {4},
                    'N4' : {5},
                    'N5' : {1},
                    },
                'T0' : {
                    'N0' : {True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {False, True},
                    'N4' : {False, True},
                    'N5' : {True},
                    },
                'T1' : {
                    'N0' : {False, True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {False, True},
                    'N4' : {False, True},
                    'N5' : {False, True},
                    },
                'T2' : {
                    'N0' : {False, True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {True},
                    'N4' : {True},
                    'N5' : {False, True},
                    },
                }
    elif args.data == 2:
        # Alignment and disease data.
        data = {
                'PRIMARY' : {
                    'N0' : {0},
                    'N1' : {0, 1, 2, 3, 4, 5},
                    'N2' : {0, 1, 2, 3, 4, 5},
                    'N3' : {4},
                    'N4' : {5},
                    'N5' : {1},
                    },
                'T0' : {
                    'N0' : {True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {False, True},
                    'N4' : {False, True},
                    'N5' : {True},
                    },
                'T1' : {
                    'N0' : {False},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {False, True},
                    'N4' : {False, True},
                    'N5' : {False, True},
                    },
                'T2' : {
                    'N0' : {True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {True},
                    'N4' : {True},
                    'N5' : {False, True},
                    },
                }
    elif args.data == 3:
        # Alignment and fully observed disease data.
        data = {
                'PRIMARY' : {
                    'N0' : {0},
                    'N1' : {0, 1, 2, 3, 4, 5},
                    'N2' : {0, 1, 2, 3, 4, 5},
                    'N3' : {4},
                    'N4' : {5},
                    'N5' : {1},
                    },
                'T0' : {
                    'N0' : {True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {True},
                    'N4' : {True},
                    'N5' : {True},
                    },
                'T1' : {
                    'N0' : {False},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {True},
                    'N4' : {True},
                    'N5' : {True},
                    },
                'T2' : {
                    'N0' : {True},
                    'N1' : {False, True},
                    'N2' : {False, True},
                    'N3' : {True},
                    'N4' : {True},
                    'N5' : {True},
                    },
                }
    else:
        raise Exception

    # Run the stochastic analysis.
    run(model, primary_to_tol, interaction_map, data)
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
            choices=('nxmodel', 'nxmodelb'), default='nxmodel',
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
    args = parser.parse_args()
    main(args)

