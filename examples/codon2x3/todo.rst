
* Redo the Rao-Teh CTBN blinking model sampling code
  to use edge-specific rates rather than branch lengths.
  The point would be to facilitate edge-specific rate parameters
  using sample average statistics of path histories,
  and that these rate parameters could be re-interpreted as branch lengths.

 * It would be possible to have not only edge-specific rates but also
   to have edge-specific processes.  This degree of generality is not
   currently needed.

 * When events are poisson-sampled using the previous foreground state
   and the background states and the edge specific rate,
   use the same poisson rate to build the uniformized
   transition probability matrices.

