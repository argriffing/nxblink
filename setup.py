#!/usr/bin/env python
"""
Stochastic mapping on a tree with a specific continuous time Bayesian network.

"""

DOCLINES = __doc__.split('\n')

# This setup script is written according to
# http://docs.python.org/2/distutils/setupscript.html
#
# It is meant to be installed through github using pip.

from distutils.core import setup

# This idiom is used by scipy to check if it is running during the setup.
__NXBLINK_SETUP__ = True

setup(
        name='nxblink',
        version='0.1',
        description=DOCLINES[0],
        author='alex',
        url='https://github.com/argriffing/nxblink/',
        download_url='https://github.com/argriffing/nxblink/',
        packages=['nxblink'],
        test_suite='nose.collector',
        package_data={'nxblink' : ['tests/test_*.py']},
        )


