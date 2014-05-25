"""
Check expectation-maximization with the toy model.

The blink rates and the edge-specific parameters will be estimated by EM.

"""
from __future__ import division, print_function, absolute_import

import itertools
import argparse

import algopy

import nxblink
from nxblink.toymodel import BlinkModelA, BlinkModelB, BlinkModelC
from nxblink.toydata import DataA, DataB, DataC, DataD
from nxblink.summary import Summary
from nxblink.em import get_ll_root, get_ll_dwell, get_ll_trans

from algopy_boilerplate import eval_grad, eval_hess

#TODO use exact summaries for the EM, using the compound model.
# Three possible levels of estimation:
# 1) max marginal likelihood
# 2) EM using exact expectation and maximization steps
# 3) EM using sample average approximation of expectation


def maximization_step(summary, pre_Q, primary_distn,
        blink_on, blink_off, edge_to_rate):
    """
    This is the maximization step of EM parameter estimation.

    Parameters
    ----------
    summary : nxblink.summary.Summary object
        Per-edge sample average approximations of expectations,
        across multiple aligned sites and multiple samples per site.
    pre_Q : 2d ndarray
        dense primary process pre-rate-matrix
    primary_distn : 1d ndarray
        dense primary state distribution
    blink_on, blink_off: floats
        Initial parameter estimates.
    edge_to_rate : dict
        Initial parameter estimates of edge-specific rates.

    """
    # extract the edges and the corresponding edge rates from the dict
    edges, edge_rates = zip(*edge_to_rate.items())

    # pack the initial parameter estimates into a point estimate
    x0 = pack_params(blink_on, blink_off, edge_rates)

    # stash the summary, edges, and genetic code into the rate matrix
    f = partial(objective, summary, edges, pre_Q, primary_distn)
    g = partial(eval_grad, f)
    h = partial(eval_hess, f)

    # maximize the log likelihood
    #result = minimize(f, x0, method='trust-ncg', jac=g, hess=h)
    result = minimize(f, x0, method='L-BFGS-B', jac=g)

    # unpack the parameters
    unpacked = unpack_params(result.x)
    blink_on, blink_off, edge_rates = unpacked

    # possibly mention the negative log likelihood
    print('minimization results:')
    print(result)

    # convert the edge rates back into a dict
    edge_to_rate = dict(zip(edges, edge_rates))

    # return the parameter estimates
    return blink_on, blink_off, edge_to_rate


def pack_params(blink_on, blink_off, edge_rates):
    """
    This function is mainly for constructing initial parameter values.

    Returns log params suitable as an initial vector
    for scipy.optimize.minimize methods.

    """
    global_params = [blink_on, blink_off]
    params = np.array(list(global_params) + list(edge_rates))
    log_params = log(params)
    return log_params


def unpack_params(log_params):
    """
    Unpack the parameters.

    """
    params = exp(log_params)
    blink_on, blink_off = params[0:2]
    edge_rates = params[2:]
    return blink_on, blink_off, edge_rates


def objective(summary, edges, pre_Q, primary_distn, log_params):
    """
    Compute the objective function to minimize.

    Only the edge-specific scaling parameters and the blink rates
    will be estimated; the objective function is constant with respect to
    the pre_Q and primary_distn arguments.

    """
    # unpack the parameters
    blink_on, blink_off, edge_rates = unpack_params(log_params)

    # compute expected negative log likelihood
    ll_root = get_ll_root(summary, primary_distn,
            blink_on, blink_off)
    ll_dwell = get_ll_dwell(summary, pre_Q, primary_distn,
            blink_on, blink_off, edges, edge_rates)
    ll_trans = get_ll_trans(summary, pre_Q, primary_distn,
            blink_on, blink_off, edges, edge_rates)
    neg_ll = -(ll_root + ll_dwell + ll_trans)
    return neg_ll


def run(model, data, nburnin, nsamples):
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

    # Precompute the dense pre_Q and primary_distn arrays
    # which happen to not depend on the parameters we are estimating.
    nprimary = len(primary_distn)
    pre_Q_dense = np.zeros((nprimary, nprimary), dtype=float)
    for sa in Q_primary:
        for sb in Q_primary[sa]:
            rate = Q_primary[sa][sb]['weight']
            pre_Q_dense[sa, sb] = rate
    primary_distn_dense = np.zeros(nprimary, dtype=float)
    for state, p in primary_distn.items():
        primary_distn_dense[state] = p
    edges, edge_rates = zip(*edge_to_rate.items())

    # Invent some arbitrary intial parameter values that are wrong.
    # The point is to recover the actual values starting with these ones.
    rate_on = 0.4
    rate_off = 1.4
    for edge in edges:
        edge_rates[edge] = 0.1

    # Sample many data points.
    # Each data point consists of an incomplete observation of
    # a state trajectory across the tree.
    # Only some of this trajectory is observed as data:
    # the primary state at the leaves,
    # and the tolerance states at a single distinguished 'reference' leaf.
    # TODO de-hardcode the number of sampled sites
    nsites = 1000
    for site_iteration in range(nsites):
        pass

    # Iterations of expectation maximization.
    for em_iteration in itertools.count(1):

        print('beginning EM iteration', em_iteration)
        print('with parameter values')
        print('blink rate on:', rate_on)

        # Compute statistics of many sampled trajectories
        # without conditioning on any data.
        summary = Summary(T, root, node_to_tm, primary_to_tol, Q_primary)
        for track_info in gen_samples(model, data, nburnin, nsamples):
            pri_track, tol_tracks = track_info
            summary.on_sample(pri_track, tol_tracks)

        # Report the summary for the EM iteration.
        print('comprehensive summary:')
        print(summary)
        print()

        # Use max likelihood to update the parameter values.
        rate_on, rate_off, edge_rates = maximization_step(
                summary, pre_Q, primary_distn,
                blink_on, blink_off, edge_to_rate)


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
    args = parser.parse_args()
    main(args)

