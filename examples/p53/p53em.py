"""
Expectation maximization for the codon blinking model.

The model has the following parameters:
    - kappa
    - omega
    - P(A)
    - P(C)
    - P(G)
    - P(T)
    - blink rate on
    - blink rate off
    - branch-specific rate scaling factor for each branch

The idea is to use log transformed parameter in the search,
with the dense matrix representation of the codon rate matrix
and its dense equilibrium distribution.
The maximization step can be done using the trust-ncg method
of the scipy.optimize minimization with gradient and hessian
provided with automatic differentiation through algopy.

"""


