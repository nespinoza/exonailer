# exonailer

The **EXO**planet tra**N**sits and r**A**d**I**al ve**L**ocity fitt**ER** (**EXO**-**NAILER**), is 
an easy-to-use code that allows you to efficiently fit exoplanet transit lightcurves, radial velocities 
or both. 

Author: Néstor Espinoza (nespino@astro.puc.cl)

DEPENDENCIES
------------

This code makes use of six important libraries:

- **Numpy**.
- **Scipy**.
- **The Bad-Ass Transit Model cAlculatioN (batman) for transit modelling** (http://astro.uchicago.edu/~kreidberg/batman/).
- **emcee for MCMC sampling** (http://dan.iel.fm/emcee/current/)
- **Astropy for time conversions** (http://www.astropy.org)

All of them are open source and can be easily installed in any machine. Be 
sure to install them before running the installer (see below), otherwise, it 
will complain. This code also makes use of the `ajplanet` module for 
radial-velocity modelling (https://github.com/andres-jordan/ajplanet) and the 
`flicker-noise` module (https://github.com/nespinoza/flicker-noise), for modelling 
1/f noise. Copies of the source codes of those modules are included in this repository 
and will be installed automatically anyways.

INSTALL
-------
To install the code, simply run the `install.py` code by doing:

    python install.py

After this is done, the code will be ready to use!

USAGE
-----

To use the code is very simple. Suppose we have a target that we named 
TARGET-001:

    1. Put the photometry under 'transit_data/TARGET-001_lc.dat'. Similarly, 
       put the RVs (if you have any) under 'rv_data/TARGET-001_rvs.dat'. These 
       are expected to have three columns: times, data and error; however, if you 
       put only two is ok: the code will recognize that you don't have errors on 
       your variables. The flux is expected to be normalized to 1. The RVs are expected 
       to be in km/s.

    2. Create a prior file under 'priors_data/TARGET-001_priors.dat'. The code 
       expects this file to have three columns: the parameter name, the prior 
       type and the hyperparameters of the prior separated by commas (see below). 
       If you want a parameter to be fixed, put 'FIXED' on the Prior Type column 
       and define the value you want to keep it fixed in the hyperparameters column.

Next, you can modify the options in the exonailer.py code. The options are:

    target:             The name of your target (in this case, 'TARGET-001').

    phot_noise_model:   This parameter defines the noise model used for the photometry. If set 
                        to 'white', it assumes the underlying noise is white-noise. If set to 
                        '1/f', it assumes it is a white + 1/f.

    phot_detrend:       This performs a small detrend on the photometry. If set to 'mfilter' 
                        it will median filter and then smooth this filter with a gaussian filter. 
                        It works pretty well for Kepler data. If you don't want to do any kind 
                        of detrending, set this to None.

    window:             This defines the window of the 'mfilter'. Usually way longer than your 
                        transit event.

    phot_get_outliers:  This automatically sigma-clips any outliers in your data. It relies on 
                        having decent priors on the ephemeris (t0 and P).

    n_omit:             Is an array that lets you ommit transit in the fitting procedure (e.g., 
                        transits with spots). Just put the number of the transits (counted from 
                        the first event in time, with this event counted as 0) that you want 
                        to ommit in the list and the code will do the rest.

    ld_law:             Limb-darkening law to use. For all the laws but the logarithmic the 
                        sampling is done using the transformations defined in Kipping (2013). 
                        The logarithmic law is sampled according to Espinoza & Jordán (2015b).

    mode:               Can be set to three different modes. 'full' performs a full transit + rv 
                        fit, 'transit' performs only a transit fit to the photometry, while 'rvs' 
                        performs a fit to the RVs only.

    rv_jitter:          If set to True, an extra 'jitter' term is added to the RVs error to account 
                        for stellar jitter.

    transit_time_def:   Defines the input and output time scales (the times are assumed to be in the 
                        JD format, i.e., JD, BJD, MDJ, etc.) of the transit times. If input transit times 
                        are, for example, in utc and you want results in tdb, this has to be 'utc->tdb'.

    rv_time_def:        Same as for transit times but for the times in the RVs.

GENERATING THE PRIOR FILE
-------------------------

The priors currently supported by the code are:

    Normal:         Expects that the third column in the prior file has the form 

                                       mu,sigma 

                    where mu is the mean value and sigma the standard-deviation.

    Uniform:        Expects that the third column in the prior file has the form

                                         a,b, 

                    where a is the minimum value and b is the maximum value.

    Jeffreys:       Expects that the third column in the prior file has the form

                                        low,up

                    where low is the lower limit of the variable and up is the upper 
                    limit of the variable.

    FIXED:          This assumes you are giving the fixed value of the variable in 
                    the third column.

The mandatory variables that must have some of the above defined priors are:

    Period:         The period of the orbit of the exoplanet. Same units as the time.
    
    t0:             The time of transit center. Same units as the time.

    a:              Semi-major axis in stellar units.

    p:              Planet-to-star radius ratio.

    inc:            Inclination of the orbit in degrees.

    sigma_w:        Standard-deviation of the underlying white noise process giving rise to 
                    the observed noise (in ppm).

    mu:             Systematic radial velocity.

    K:              Radial-velocity semi-amplitude.

If in the options of the exonailer.py code you set phot_noise_model to '1/f', then you 
must also define a sigma_r parameter (see Carter & Winn, 2009). If you set rv_jitter to 
True, you must also set a sigma_w_rv parameter for the jitter term.

WHISH-LIST
----------

    + Add example datasets.

    + Create a tutorial.


TODO
----

    + Add option to define differnt mu's to RVs taken from different instruments.

    + GPs for detrending and for noise models.

    + Automated TTV analysis.

    + Transit and RVs for multi-planet systems.

    + Noise models for RVs.
