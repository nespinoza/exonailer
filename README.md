# exo-nailer

The **EXO**planet tra**N**sits and r**A**d**I**al ve**L**ocity fitt**ER** (**EXO**-**NAILER**), is 
an easy-to-use code that allows you to efficiently fit exoplanet transit lightcurves, radial velocities 
or both. 

Author: Néstor Espinoza (nespino@astro.puc.cl)

DEPENDENCIES
------------

This code makes use of four important libraries:

    + Numpy.
    + Scipy.
    + The Bad-Ass Transit Model cAlculatioN (batman) for transit modelling.
      -> http://astro.uchicago.edu/~kreidberg/batman/
    + emcee for MCMC sampling.
      -> http://dan.iel.fm/emcee/current/

All of them are open source and can be easily installed in any machine.

WHISH-LIST
----------
    + Allow the code to handle priors from the outside (i.e., easier 
      than going inside the utilities code).

    + Add example datasets.

    + Create a tutorial.
