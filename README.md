# exo-nailer

The EXOplanet traNsit rAdIal veLocity fittER (EXO-NAILER), is an easy-to-use code 
that allows you to efficiently fit exoplanet transit lightcurves, radial velocities 
or both. It makes use of two important pieces of code:

    + The Bad-Ass Transit Model cAlculatioN (batman) for transit modelling.
      -> http://astro.uchicago.edu/~kreidberg/batman/
    + emcee for MCMC sampling.
      -> http://dan.iel.fm/emcee/current/

DEPENDENCIES
------------

This code makes use of three important libraries:

        + Numpy.
        + Scipy.

All of them are open source and can be easily installed in any machine.

WHISH-LIST
----------
    + Allow the code to handle priors from the outside (i.e., easier 
      than going inside the utilities code).

    + Add example datasets.

    + Create a tutorial.
