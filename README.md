# exofit

EXOFIT is the Exoplanet Fitter, which makes use of two important pieces of code:

    + The Bad-Ass Transit Model cAlculatioN (batman) for transit modelling.
      -> http://astro.uchicago.edu/~kreidberg/batman/
    + emcee for MCMC sampling.
      -> http://dan.iel.fm/emcee/current/

The idea is for this code to be able to:
    - Fit transit lightcurves (Done!).
    - Fit RV data (Under development).
    - Both simultaneously (Under development).

DEPENDENCIES
------------

This code makes use of three important libraries:

        + Numpy.
        + Scipy.

All of them are open source and can be easily installed in any machine.
