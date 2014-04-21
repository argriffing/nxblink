"""
Maximum likelihood estimation for blink rate contribution to log likelihood.

This uses numerical optimization with the exact gradient.
There is a closed form, but it is messy and maybe not much better
than the numerical optimization.
Wolfram-based solvers can find it using something like the following syntax.
solve {-h + u/(x + y) + a/x = 0, -f + u/(x + y) + b/y = 0} for x and y

"""
from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np
import scipy.optimize


def ll_blink_contribution(
        xon_root_count,
        off_root_count,
        off_xon_count,
        xon_off_count,
        off_xon_dwell,
        xon_off_dwell,
        rate_on,
        rate_off,
        ):
    # initial root state contribution
    initial_contrib = (
            xon_root_count * np.log(rate_on) +
            off_root_count * np.log(rate_off) +
            -(xon_root_count + off_root_count) * np.log(rate_on + rate_off))
    # blink transition contribution
    trans_contrib = (
            off_xon_count * np.log(rate_on) +
            xon_off_count * np.log(rate_off))
    # dwell contribution
    dwell_contrib = (
            -off_xon_dwell * rate_on +
            -xon_off_dwell * rate_off)
    return initial_contrib + trans_contrib + dwell_contrib


def ll_blink_contribution_gradient(
        xon_root_count,
        off_root_count,
        off_xon_count,
        xon_off_count,
        off_xon_dwell,
        xon_off_dwell,
        rate_on,
        rate_off,
        ):
    """
    Compute a gradient.

    Notes
    -----
    from sympy import *

    a, b, c, d, e, f = symbols('a b c d e f')
    x, y = symbols('x y')

    init_printing(use_unicode=True)

    expr = (
            (a+c)*log(x) + (b+d)*log(y) +
            -(a+b)*log(x+y) +
            -e*x + -f*y
            )

    print diff(expr, x)
    #-e + (-a - b)/(x + y) + (a + c)/x

    print diff(expr, y)
    #-f + (-a - b)/(x + y) + (b + d)/y
    """
    total_rate = rate_on + rate_off
    d_rate_on = (
            (xon_root_count + off_xon_count) / rate_on +
            -(xon_root_count + off_root_count) / total_rate +
            -off_xon_dwell)
    d_rate_off = (
            (off_root_count + xon_off_count) / rate_off +
            -(off_root_count + xon_root_count) / total_rate +
            -xon_off_dwell)
    return d_rate_on, d_rate_off


def get_blink_rate_analytical_mle(
        xon_root_count,
        off_root_count,
        off_xon_count,
        xon_off_count,
        off_xon_dwell,
        xon_off_dwell,
        ):
    """
    This closed form is not necessarily better than the other approaches.

    """
    # Convert to single letter variables for convenience.
    a = xon_root_count + off_xon_count
    b = off_root_count + xon_off_count
    h = off_xon_dwell
    f = xon_off_dwell
    u = -(xon_root_count + off_root_count)
    #
    z = a*f - 2*a*h - b*h + f*u - h*u
    xopt = (np.sqrt(z*z - 4*a*h*(h - f)*(a + b + u)) + z) / (2*h*(f-h))
    #
    # switch the meanings of variables to make copypasting easier
    a, b, h, f = b, a, f, h
    #
    z = a*f - 2*a*h - b*h + f*u - h*u
    yopt = (np.sqrt(z*z - 4*a*h*(h - f)*(a + b + u)) + z) / (2*h*(f-h))
    #
    return xopt, yopt


def get_blink_rate_mle(
            xon_root_count,
            off_root_count,
            off_xon_count,
            xon_off_count,
            off_xon_dwell,
            xon_off_dwell,
            low_rate=1e-4,
            ):
    def neg_ll(rates):
        rate_on, rate_off = rates
        return -ll_blink_contribution(
                xon_root_count,
                off_root_count,
                off_xon_count,
                xon_off_count,
                off_xon_dwell,
                xon_off_dwell,
                rate_on,
                rate_off,
                )
    def fprime(rates):
        rate_on, rate_off = rates
        d_rate_on, d_rate_off = ll_blink_contribution_gradient(
                xon_root_count,
                off_root_count,
                off_xon_count,
                xon_off_count,
                off_xon_dwell,
                xon_off_dwell,
                rate_on,
                rate_off,
                )
        return -np.array([d_rate_on, d_rate_off])
    initial_rates = np.array([1.0, 1.0])
    bounds = ((low_rate, None), (low_rate, None))
    result = scipy.optimize.fmin_l_bfgs_b(
            neg_ll, initial_rates, fprime, bounds=bounds)
    opt_rates, opt_neg_ll, info = result
    rate_on, rate_off = opt_rates
    return rate_on, rate_off

