# exonailer

The **EXO**planet tra**N**sits and r**A**d**I**al ve**L**ocity fitt**ER** (**EXO**-**NAILER**), is 
an easy-to-use code that allows you to efficiently fit exoplanet transit lightcurves, radial velocities (RVs) 
or both.

![Exonailer fit to data](exonailer.png?raw=true "Example of exonailer fit to data") 

Author: Néstor Espinoza (espinoza@mpia.de)

*If you make use of this code, please cite Espinoza et al., 2016, ApJ, 830, 43 (http://arxiv.org/abs/1601.07608)*

DEPENDENCIES
------------

This code makes use of seven important libraries:

- **Numpy**.
- **Scipy**.
- **The Bad-Ass Transit Model cAlculatioN (batman) for transit modelling** (http://astro.uchicago.edu/~kreidberg/batman/).
- **The RadVel package** (http://radvel.readthedocs.io/en/latest/).
- **emcee for MCMC sampling** (http://dan.iel.fm/emcee/current/).
- **Astropy for time conversions** (http://www.astropy.org).
- **The GNU Scientific Library** (https://www.gnu.org/software/gsl/)

All of them are open source and can be easily installed in any machine. Be 
sure to install them before running the installer (see below), otherwise, it 
will complain. This code also makes use of the `flicker-noise` module 
(https://github.com/nespinoza/flicker-noise), for modelling 1/f noise. A copy of the 
source code of this module is included in this repository 
and will be installed automatically.

INSTALLATION
------------
To install the code, simply run the `install.py` code by doing:

    python install.py

After this is done, the code will be ready to use!

USAGE
-----

To use the code is very simple, and to help you understand how to use it, we have 
added a synthetic dataset for the target `my_target`, along with this package. `my_target`, 
of course, is a generic target name. You can put any target name you want as long as it 
includes no spaces. With this decided:

    1. Put your photometry under the `transit_data` folder. The file containing the photometry 
       has to be named `target_lc.dat`, where `target` is the name of your target. In our case, 
       `my_target`. Similarly, put the RVs (if you have any) under the `rv_data` folder. The file 
       containing the RVs has to be named `target_rvs.dat`, where `target`, again, is the name of 
       your target. In our case, `my_target`. These are expected to have four columns: times, data, 
       error and name of the instrument (which is a string); however, only the two first are mandatory: 
       the code will recognize that you don't have errors on your variables and if no 
       instrument names are given, it will assume all come from the same instrument. 
       The flux is expected to be normalized to 1. The RVs are expected to be in km/s.

    2. Create a prior file under the `priors_data` folder. The file containing the photometry 
       has to be named `target_priors.dat`, where `target` is (you guessed!) the name of your target. 
       The code expects this file to have three columns: the parameter name, the prior 
       type and the hyperparameters of the prior separated by commas (see below). 
       If you want a parameter to be fixed, put `FIXED` on the Prior Type column 
       and define the value you want to keep it fixed in the hyperparameters column. 
       Also, if a prior other than `FIXED` is defined, a fourth column can be entered for 
       each parameter where you can specify the starting point of the parameter (say, one 
       obtained by a previous least-square fit, or a value you know to be close to the 
       true parameter, etc.).

As can be seen from the above, the code can handle data taken with different instruments. 
For RVs, this means that a different center-of-mass velocity can be fitted for each instrument 
in order to account for offsets between them, and if jitter is included, a different jitter term 
can also fitted for each instrument. For transits, this means a different photometric jitter can be 
fitted to each instrument, as well as different limb-darkening coefficients and different transit depths. 

As previously stated, there is a synthetic dataset along with this code which is useful to understand 
how to get your fit started. The lightcurves for this dataset are under the `transit_data` folder and is 
labeled `my_target_lc.dat`, while the RVs are under the `rv_data` folder and is named `my_target_rvs.dat`.


Next, you can modify the options of your fit in the `options_file.dat` file.

The **GENERAL OPTIONS** are:

    TARGET:             The name of your target.

    MODE:               This defines which kind of fit you want to perform. `full` means a full 
                        transit and radial-velocity fit. `transit` means you only will fit the 
                        transit lightcurves and `rvs` means you will only fit the radial-velocities.

    NWALKERS:           This is the number of walkers on the MCMC runs (for more information on this 
                        parameter, check out the `emcee` documentation).

    NJUMPS:             This is the number of jumps on the MCMC (for more information on this 
                        parameter, check out the `emcee` documentation).

    NBURNIN:            This is the number of burn-in runs of the MCMC (for more information on this
                        parameter, check out the `emcee` documentation). 

    PLOT:               If set to `NO`, no plots will me shown at the end. If set to `YES`, a plot at the 
                        end of the `exonailer` run will be shown similar to the one shown above.

The **PHOTOMETRY OPTIONS** have to be defined for each instrument. For each one, you must define:

    INSTRUMENT:           The name of the instrument. These have to match the instruments in the transit 
                          lightcurves.

    PHOT_NOISE_MODEL:     This parameter defines the noise model used for the photometry. If set 
                          to 'white', it assumes the underlying noise is white-noise. If set to 
                          'flicker', it assumes it is a white + 1/f.

    PHOT_DETREND:         This performs a small detrend on the photometry. If set to 'mfilter' 
                          it will median filter and then smooth this filter with a gaussian filter. 
                          It works pretty well for Kepler data. If you don't want to do any kind 
                          of detrending, set this to `NO`.

    WINDOW:               This defines the window of the 'mfilter'. Usually way longer than your 
                          transit event, and is defined in number of datapoints.

    PHOT_GET_OUTLIERS:    This automatically sigma-clips any outliers in your data if set to `YES`. 
                          It relies on having decent priors on the ephemeris (t0 and P). If you don't want 
                          to remove them, set this to `NO`.

    NOMIT:                It is a sequence of numbers, separated by commas, that lets you ommit transit in 
                          the fitting procedure (e.g., transits with spots). Just put the number of the transits 
                          (counted from the first event in time, with this event counted as 0) that you want 
                          to ommit. If you don't want to ommit any transit, don't put this option.

    RESAMPLING:           Set this to `YES` if you want to use the selective resampling scheme of Kipping (2010, MNRAS, 
                          408, 1758), applied to 30-minute cadence Kepler lightcurves. 

    PHASE_MAX_RESAMPLING: This define the maximum phase at which the data will be resampled if `RESAMPLING` is set to 
                          `YES`.

    NRESAMPLING:          This defines the number of instantaneous lightcurve points used to resample the lightcurve 
                          if `RESAMPLING` is set to `YES`.

    LD_LAW:               Limb-darkening law to use. For all the laws but the logarithmic the 
                          sampling is done using the transformations defined in Kipping (2013). 
                          The logarithmic law is sampled according to Espinoza & Jordán (2015b).

    TRANSIT_TIME_DEF:     Defines the input and output time scales (the times are assumed to be in the 
                          JD format, i.e., JD, BJD, MDJ, etc.) of the transit times. If input transit times 
                          are, for example, in utc and you want results in tdb, this has to be 'utc->tdb'.

The **RADIAL-VELOCITY OPTIONS** have to be defined for each instrument as well. For each one, you must define:

    INSTRUMENT:           The name of the instrument. These have to match the instruments in the radial-velocity 
                          data.

    RV_TIME_DEF:          Defines the input and output time scales (the times are assumed to be in the
                          JD format, i.e., JD, BJD, MDJ, etc.) of the radial-velocity times. If input RV times
                          are, for example, in utc and you want results in tdb, this has to be 'utc->tdb'.

Once you are done with this, just run the code by doing:

    python exonailer.py

GENERATING THE PRIOR FILE
-------------------------

The priors currently supported by the code are:

    Normal:         Expects that the third column in the prior file has the form 

                                       mu,sigma 

                    where mu is the mean value and sigma the standard-deviation.

    Uniform:        Expects that the third column in the prior file has the form

                                         a,b 

                    where a is the minimum value and b is the maximum value.

    Jeffreys:       Expects that the third column in the prior file has the form

                                        low,up

                    where low is the lower limit of the variable and up is the upper 
                    limit of the variable.

    Beta:           Expects that the third column in the prior file has the form

                                        alpha,beta

                    where alpha and beta are the parameters that define the beta distribution. 

    FIXED:          This assumes you are giving the fixed value of the variable in 
                    the third column.

The mandatory variables that must have some of the above defined priors in the case of a `transit` fit are::

    P:              The period of the orbit of the exoplanet. Same units as the time.
    
    t0:             The time of transit center. Same units as the time.

    a:              Semi-major axis in stellar units.

    p:              Planet-to-star radius ratio. If you want to define a different one for each 
                    instrument, add a lower-dash and put the name of the instrument (e.g., `p_Telescope`).

    q1:             Limb-darkening parameter. This corresponds to the transformation of the uninformative 
                    limb-darkening threatment scheme defined in Kipping et al. (2013, MNRAS, 435, 2152). 
                    The exact transformation is defined by the limb-darkening law used. If you want to define 
                    a different one for each instrument, add a lower-dash and put the name of the instrument 
                    (e.g., `q1_Telescope`).

    q2:             Second limb-darkening parameter. Same as for `q1`. 

    inc:            Inclination of the orbit in degrees.

    sigma_w:        Standard-deviation of the underlying white noise process giving rise to 
                    the observed noise (in ppm). If you want to define a different one for each 
                    instrument, add a lower-dash and put the name of the instrument (e.g., `sigma_w_Telescope`).

    ecc:            Eccentricity of the orbit.

    omega:          Argument of periapsis (in degrees)

Of course, e.g., for a circular fit, you might want to fix `ecc` (to 0) and `omega` (e.g., to 90). If you 
define the `PHOT_NOISE_MODEL` as `flicker`, you must add an extra parameter, `sigma_r` (see Carter & Winn, 2009).
The variables which have to be defined in case of a `rvs` fit, in addition to the eccentricity, period, 
time of transit-center and omega, are:

    mu:             Center-of-mass velocity of the RVs. If you want to define a different one for each 
                    instrument, add a lower-dash and put the name of the instrument (e.g., `mu_Spectrograph`).

    K:              Radial-velocity semi-amplitude.
    
    sigma_w_rv:     Jitter term for radial-velocities (see below). If you want to define a different one for each
                    instrument, add a lower-dash and put the name of the instrument (e.g., `sigma_w_rv_Spectrograph`). 
                    If you do not want to include jitter, set this to `FIXED` in the prior file, and set it to zero.

In the case of `full` exonailer fits, all of these parameters have to be defined.

OUTPUTS
-------

The outputs of exonailer will be under the `results` folder. In this folder, you will find a folder for 
each of your fits and, inside, three files:

    posterior_parameters.dat:             This file saves the posterior parameters for each variable in 
                                          the fit. The first column lists the variable name, the second 
                                          the median of the posterior of that parameter (50th percentile), 
                                          the third the 84th percentile of the posterior ("upper 1-sigma 
                                          error") and the fourth the 16th percentile of the posterior 
                                          ("lower 1-sigma error").

    posteriors.pkl:                       This file has the actual posterior distributions for each parameter.

    priors.dat:                           This file saves which prior you used for the given dataset (useful 
                                          in case you are trying different priors to see how your results 
                                          change).

In addition, the data, model and residuals of the transit, radial-velocities or both will be exported as .dat files 
to this folder, so you can easily plot them yourself.

WHISH-LIST
----------

    + Create a tutorial explaining usage of GPs for transits (implemented, but not yet documented)
 
    + Add MULTI-NEST support.


TODO
----

    + Transit and RVs for multi-planet systems.

    + Noise models (e.g. GPs) for RVs.
