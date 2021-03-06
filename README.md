Stochastic mapping on a tree with a specific continuous time Bayesian network.

UNDER CONSTRUCTION

Required dependencies:
 * [Python 2.7+](http://www.python.org/)
 * [pip](https://pip.readthedocs.org/) (installation)
 * [git](http://git-scm.com/) (installation)
 * [NetworkX](http:/networkx.lanl.gov/) (graph data types and algorithms)
   - `$ pip install --user git+https://github.com/networkx/networkx`
 * [nxmctree](https://github.com/argriffing/nxmctree) (discrete time sampling)
   - `$ pip install --user git+https://github.com/argriffing/nxmctree`
 * [nxrate](https://github.com/argriffing/nxrate) (continuous time rate matrix)
   - `$ pip install --user git+https://github.com/argriffing/nxrate`

Optional dependencies:
 * [nose](https://nose.readthedocs.org/) (testing)
 * [numpy](http://www.numpy.org/) (more testing infrastructure and assertions)
 * [coverage](http://nedbatchelder.com/code/coverage/) (test coverage)
   - `$ apt-get install python-coverage`


User
----

Install:

    $ pip install --user git+https://github.com/argriffing/nxblink

Test:

    $ python -c "import nxblink; nxblink.test()"

Uninstall:

    $ pip uninstall nxblink


Developer
---------

Install:

    $ git clone git@github.com:argriffing/nxblink.git

Test:

    $ python runtests.py

Coverage:

    $ python-coverage run --branch runtests.py
    $ python-coverage html
    $ chromium-browser htmlcov/index.html

Profiling:

    $ python -m cProfile -o profile_data.pyprof runtests.py
    $ pyprof2calltree -i profile_data.pyprof -k

Build docs locally (NOT IMPLEMENTED):

    $ sh make-docs.sh
    $ chromium-browser /tmp/nxdocs/index.html

Subsequently update online docs (NOT IMPLEMENTED):

    $ git checkout gh-pages
    $ cp /tmp/nxdocs/. ./ -R
    $ git add .
    $ git commit -am "update gh-pages"
    $ git push

