# exo-nailer

The _EXO_planet tra_N_sits and r_A_d_I_al ve_L_ocity fitt_ER_ (_EXO_-_NAILER_), is an easy-to-use code 
that allows you to efficiently fit exoplanet transit lightcurves, radial velocities 
or both. It makes use of two important pieces of code:

    + The _B_ad-_A_ss _T_ransit _M_odel c_A_lculatio_N_ (batman) for transit modelling.
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
