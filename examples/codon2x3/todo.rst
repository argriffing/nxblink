Add a new example:
 * code2x3b
 * same data (alignment and disease) as code2x3
 * same tree shape and root
 * let one branch length be twice as long
 * let another branch length be half as long
 * remove the synonymous transition between states P4 <-> P5
 * force the primary process equilibrium distribution to be non-uniform
   by increasing the equilbrium frequency of state P1 by doubling its
   incoming rates and halving its outgoing rates
 * let the blinking rates be unequal, in particular let the off -> on
   rate be doubled from 1 to 2 and let the on -> off rate be cut in half
   from 1 to 1/2.  This implies a prior blink state distribution
   of 4/5 on, 1/5 off.

