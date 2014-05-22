"""
Check expectation-maximization with the toy model.

The blink rates and the edge-specific parameters will be estimated by EM.

"""
from __future__ import division, print_function, absolute_import

import algopy

import nxblink
from nxblink.toymodel import BlinkModelA, BlinkModelB, BlinkModelC
from nxblink.toydata import DataA, DataB, DataC, DataD
from nxblink.summary import Summary
from nxblink.em import get_ll_root, get_ll_dwell, get_ll_trans


# algopy boilerplate
def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))


# algopy boilerplate
def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))



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


def objective(summary, edges, pre_Q, distn, log_params):
    """
    Compute the objective function to minimize.

    Computes negative log likelihood of augmented process.
    Note that none of the parameters to be estimated


    penalized according to violation of the constraint that
    mutational nucleotide probabilities should sum to 1.
    The log_params input may be any manner of exotic array.

    """
    # unpack the parameters
    blink_on, blink_off, edge_rates = unpack_params(log_params)
    unpacked, penalty = unpack_params(log_params)
    kappa, omega, A, C, G, T, blink_on, blink_off, edge_rates = unpacked

    # construct the unnormalized pre-rate-matrix from the parameters
    pre_Q = get_pre_Q(genetic_code, kappa, omega, A, C, G, T)
    distn = get_distn(genetic_code, kappa, omega, A, C, G, T)

    # compute expected negative log likelihood
    ll_root = get_ll_root(summary, distn, blink_on, blink_off)
    ll_dwell = get_ll_dwell(summary, pre_Q, distn, blink_on, blink_off,
            edges, edge_rates)
    ll_trans = get_ll_trans(summary, pre_Q, distn, blink_on, blink_off,
            edges, edge_rates)
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

    # summarize samples
    summary = Summary(T, root, node_to_tm, primary_to_tol, Q_primary)
    for pri_track, tol_tracks in gen_samples(model, data, nburnin, nsamples):
        summary.on_sample(pri_track, tol_tracks)

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
    args = parser.parse_args()
    main(args)

