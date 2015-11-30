# -*- coding: utf-8 -*-
from math import floor,ceil
import matplotlib.pyplot as plt
import numpy as np
import batman

def get_sigma(x,median):
    """
    This function returns the MAD-based standard-deviation.
    """
    mad = np.median(np.abs(x-median))
    return 1.4826*mad

def get_phases(t,P,t0):
    phase = ((t - np.median(t0))/np.median(P)) % 1
    ii = np.where(phase>=0.5)[0]
    phase[ii] = phase[ii]-1.0
    return phase

def read_transit_params(prior_dict):
    names = ['P','inc','a','p','t0','q1','q2']
    vals = len(names)*[[]]
    for i in range(len(names)):
        param = prior_dict[names[i]]
        vals[i] = param['object'].value
    return vals

def pre_process(t,f,f_err,detrend,get_outliers,n_ommit,window,parameters,ld_law):
    # Now, the first phase in transit fitting is to 'detrend' the 
    # data. This is done with the 'detrend' flag. If 
    # the data is already detrended, set the flag to None:
    if detrend is not None:
        if detrend == 'mfilter':
            # Get median filter, and smooth it with a gaussian filter:
            from scipy.signal import medfilt
            from scipy.ndimage.filters import gaussian_filter
            filt = gaussian_filter(medfilt(f,window),5)
            f = f/filt
            if f_err is not None:
                f_err = f_err/filt

    # Extract transit parameters from prior dictionary:
    P,inc,a,p,t0,q1,q2 = read_transit_params(parameters)

    # If the user wants to ommit transit events:
    if len(n_ommit)>0:
        # Get the phases:
        phases = (t-t0)/P

        # Get the transit events in phase space:
        transit_events = np.arange(ceil(np.min(phases)),floor(np.max(phases))+1)

        # Convert to zeros fluxes at the events you want to eliminate:
        for n in n_ommit:
            idx = np.where((phases>n-0.5)&(phases<n+0.5))[0]
            f[idx] = np.zeros(len(idx))

        # Eliminate them from the t,f and phases array:
        idx = np.where(f!=0.0)[0]
        t = t[idx]
        f = f[idx]
        phases = phases[idx]
        if f_err is not None:
            f_err = f_err[idx]

    # Generate the phases:
    phases = get_phases(t,P,t0)
    # If outlier removal is on, remove them:
    if get_outliers:
        model = get_transit_model(t,t0,P,p,a,inc,q1,q2,ld_law)
        # Get approximate transit duration in phase space:
        idx = np.where(model == 1.0)[0]
        phase_dur = np.abs(phases[idx][np.where(np.abs(phases[idx]) == \
                           np.min(np.abs(phases[idx])))])[0] + 0.01
        # Get precision:
        median_flux = np.median(f)
        sigma = get_sigma(f,median_flux)
        # Perform sigma-clipping for out-of-transit data using phased data:
        good_times = np.array([])
        good_fluxes = np.array([])
        good_phases = np.array([])
        if f_err is not None:
            good_errors = np.array([])

        # Iterate through the dataset:
        for i in range(len(t)):
                if np.abs(phases[i])<phase_dur:
                        good_times = np.append(good_times,t[i])
                        good_fluxes = np.append(good_fluxes,f[i])
                        good_phases = np.append(good_phases,phases[i])
                        if f_err is not None:
                           good_errors = np.append(good_errors,f_err[i])
                else:
                        if (f[i]<median_flux + 3*sigma) and (f[i]>median_flux - 3*sigma):
                                good_times = np.append(good_times,t[i])
                                good_fluxes = np.append(good_fluxes,f[i])
                                good_phases = np.append(good_phases,phases[i])
                                if f_err is not None:
                                    good_errors = np.append(good_errors,f_err[i])
        t = good_times
        f = good_fluxes
        phases = good_phases
        if f_err is not None:
            f_err = good_errors

    return t, phases, f, f_err

def init_batman(t,law):
    """
    This function initializes the batman code.
    """
    params = batman.TransitParams()
    params.t0 = 0.
    params.per = 1.
    params.rp = 0.1
    params.a = 15.
    params.inc = 87.
    params.ecc = 0.
    params.w = 90.
    params.u = [0.1,0.3]
    params.limb_dark = law
    m = batman.TransitModel(params,t)
    return params,m

def get_transit_model(t,t0,P,p,a,inc,q1,q2,ld_law):
    params,m = init_batman(t,law=ld_law)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    params.u = [coeff1,coeff2]
    return m.light_curve(params)

def convert_ld_coeffs(ld_law, coeff1, coeff2):
    if ld_law == 'quadratic':
        q1 = (coeff1 + coeff2)**2
        q2 = coeff1/(2.*(coeff1+coeff2))
    elif ld_law=='square-root':
        q1 = (coeff1 + coeff2)**2
        q2 = coeff2/(2.*(coeff1+coeff2))
    elif ld_law=='logarithmic':
        q1 = (1-coeff2)**2
        q2 = (1.-coeff1)/(1.-coeff2)
    return q1,q2

def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == 'quadratic':
        coeff1 = 2.*np.sqrt(q1)*q2
        coeff2 = np.sqrt(q1)*(1.-2.*q2)
    elif ld_law=='square-root':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    return coeff1,coeff2

import emcee
import Wavelets
import scipy.optimize as op
def exonailer_mcmc_fit(times, relative_flux, error, times_rv, rv, rv_err, \
                       parameters, ld_law, mode, rv_jitter = False, \
                       njumps = 500, nburnin = 500, nwalkers = 100, noise_model = 'white'):
    """
    This function performs an MCMC fitting procedure using a transit model 
    fitted to input data using the batman package (Kreidberg, 2015) assuming 
    the underlying noise process is either 'white' or '1/f-like' (see Carter & 
    Winn, 2010). It makes use of the emcee package (Foreman-Mackey et al., 2014) 
    to perform the MCMC, and the sampling scheme explained in Kipping (2013) to 
    sample coefficients from two-parameter limb-darkening laws; the logarithmic 
    law is sampled according to Espinoza & Jordán (2015b). 

    This transit models assumes 0 eccentricity.

    The inputs are:

      times:            Times (in same units as the period and time of transit center).

      relative_flux:    Relative flux; it is assumed out-of-transit flux is 1.

      error:            If you have errors on the fluxes, put them here. Otherwise, set 
                        this to None.

      times_rv:         Times (in same units as the period and time of transit center) 
                        of RV data.

      rv:               Radial velocity measurements.

      rv_err:           If you have errors on the RVs, put them here. Otherwise, set 
                        this to None.

      parameters:       Dictionary containing the information regarding the parameters (including priors).

      ld_law:           Two-coefficient limb-darkening law to be used. Can be 'quadratic',
                        'logarithmic', 'square-root' or 'exponential'. The last one is not 
                        recommended because it is non-physical (see Espinoza & Jordán, 2015b).


      mode:             'full' = transit + rv fit, 'transit' = only transit fit, 'rv' = only RV fit.

      n_jumps:          Number of jumps to be done by each of the walkers in the MCMC. 

      n_burnin:         Number of burnins to use for the MCMC.

      n_walkers:        Number of walkers to be used to run the MCMC.

      noise_model:      Currently supports two types: 
 
                          'white':   It assumes the underlying noise model is white, gaussian
                                     noise.

                          '1/f'  :   It assumes the underlying noise model is a sum of a 
                                     white noise process plus a 1/f noise model.

    The outputs are the chains of each of the parameters in the theta_0 array in the same 
    order as they were inputted. This includes the sampled parameters from all the walkers.
    """

    if mode != 'rv':
        # Initialize the parameters of the transit model:
        params,m = init_batman(times,law=ld_law)

    # Initialize the variable names:
    transit_params = ['P','t0','a','p','inc','sigma_w','sigma_r','q1','q2']
    rv_params = ['mu','K','sigma_w_rv']

    # Create lists that will save parameters to check the limits on and:
    parameters_to_check = []

    # Eliminate from the parameter list parameters that are being fixed:
    if parameters['P']['type'] == 'FIXED':
        transit_params.pop(transit_params.index('P'))
    elif parameters['P']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('P')
    if parameters['t0']['type'] == 'FIXED':
        transit_params.pop(transit_params.index('t0'))
    elif parameters['t0']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('t0')
    if parameters['a']['type'] == 'FIXED':
        transit_params.pop(transit_params.index('a'))
    elif parameters['a']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('a')
    if parameters['p']['type'] == 'FIXED':
        transit_params.pop(transit_params.index('p'))
    elif parameters['p']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('p')
    if parameters['inc']['type'] == 'FIXED':
        transit_params.pop(transit_params.index('inc'))
    elif parameters['inc']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('inc')
    if parameters['sigma_w']['type'] == 'FIXED':
        transit_params.pop(transit_params.index('sigma_w'))
    elif parameters['sigma_w']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('sigma_w')
    if parameters['q1']['type'] == 'FIXED':
        transit_params.pop(transit_params.index('q1'))
    elif parameters['q1']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('q1')
    if parameters['q2']['type'] == 'FIXED':
        transit_params.pop(transit_params.index('q2'))
    elif parameters['q2']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('q2')
    if parameters['mu']['type'] == 'FIXED':
        rv_params.pop(rv_params.index('mu'))
    elif parameters['mu']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('mu')
    if parameters['K']['type'] == 'FIXED':
        rv_params.pop(rv_params.index('K'))
    elif parameters['K']['type'] in ['Uniform','Jeffreys']:
        parameters_to_check.append('K')
    if noise_model == '1/f':
        if parameters['sigma_r']['type'] == 'FIXED':
            transit_params.pop(transit_params.index('sigma_r'))
        elif parameters['sigma_r']['type'] in ['Uniform','Jeffreys']:
            parameters_to_check.append('sigma_r')
    else:
        transit_params.pop(transit_params.index('sigma_r'))
    if mode != 'transit' and rv_jitter:
        if parameters['sigma_w_rv']['type'] == 'FIXED':
            rv_params.pop(rv_params.index('sigma_w_rv'))       
        elif parameters['sigma_w_rv']['type'] in ['Uniform','Jeffreys']:
            parameters_to_check.append('sigma_w_rv')
    else: 
        sigma_w_rv = 0.0 
        rv_params.pop(rv_params.index('sigma_w_rv'))

    all_mcmc_params = transit_params + rv_params
    n_params = len(all_mcmc_params)

    def get_fn_likelihood(residuals, sigma_w, sigma_r, gamma=1.0):
        like=0.0
        # Arrays of zeros to be passed to the likelihood function
        aa,bb,M = Wavelets.getDWT(residuals)
        # Calculate the g(gamma) factor used in Carter & Winn...
        if(gamma==1.0):
           g_gamma=1.0/(2.0*np.log(2.0))  # (value assuming gamma=1)
        else:
           g_gamma=(2.0)-(2.0)**gamma
        # log-Likelihood of the aproximation coefficients
        sigmasq_S=(sigma_r**2)*g_gamma+(sigma_w)**2
        tau_a =  1.0/sigmasq_S
        like += normal_like( bb[0], 0.0 , tau_a )
        k=long(0)
        SS=range(M)
        for ii in SS:
                # log-Likelihood of the detail coefficients with m=i...
                if(ii==0):
                  sigmasq_W=(sigma_r**2)*(2.0**(-gamma*np.double(1.0)))+(sigma_w)**2
                  tau=1.0/sigmasq_W
                  like += normal_like( bb[1], 0.0, tau )
                else:
                  sigmasq_W=(sigma_r**2)*(2.0**(-gamma*np.double(ii+1)))+(sigma_w)**2
                  tau=1.0/sigmasq_W
                  for j in range(2**ii):
                      like += normal_like( aa[k], 0.0 , tau )
                      k=k+1
        return like

    def lnlike_transit(t, y, yerr, gamma=1.0):

        coeff1,coeff2 = reverse_ld_coeffs(ld_law, \
                        parameters['q1']['object'].value,parameters['q2']['object'].value)
        params.t0 = parameters['t0']['object'].value
        params.per = parameters['P']['object'].value
        params.rp = parameters['p']['object'].value
        params.a = parameters['a']['object'].value
        params.inc = parameters['inc']['object'].value
        params.ecc = 0.0
        params.w = 90.0
        params.u = [coeff1,coeff2]
        model = m.light_curve(params)
        if noise_model == '1/f':
           residuals = (y-model)*1e6
           log_like = get_fn_likelihood(residuals,parameters['sigma_w']['object'].value,\
                           parameters['sigma_r']['object'].value)
        else:
           inv_sigma2 = 1.0/((yerr)**2 + (parameters['sigma_w']['object'].value)**2)
           log_like = -0.5*(np.sum(((y-model)*1e6)**2*inv_sigma2 - np.log(inv_sigma2)))
        return log_like

    def lnlike_rv(t, y, yerr):
        model = parameters['mu']['object'].value - \
                        parameters['K']['object'].value*\
                        np.sin(2.*np.pi*(t-parameters['t0']['object'].value)/parameters['P']['object'].value)
        residuals = (y-model)
        inv_sigma2 = 1.0/((yerr)**2 + (parameters['sigma_w_rv']['object'].value)**2)
        log_like = -0.5*(np.sum((residuals)**2*inv_sigma2 - np.log(inv_sigma2)))
        return log_like

    def lnlike_full(tt, yt, yerrt, trv, yrv, yerrrv):
        return lnlike_rv(trv, yrv, yerrrv) + \
               lnlike_transit(tt, yt, yerrt)

    def lnprior_full(theta):
        # Read in the values of the parameter vector and update values of the objects.
        # For each one, if everything is ok, get the total prior, which is the sum 
        # of the independant priors for each parameter:
        total_prior = 0.0
        for i in range(n_params):
            c_param = all_mcmc_params[i]
            parameters[c_param]['object'].set_value(theta[i])
            if c_param in parameters_to_check:
                if not parameters[c_param]['object'].check_value(theta[i]):
                    return -np.inf
            total_prior += parameters[c_param]['object'].get_ln_prior()

        return total_prior

    def lnprob_full(theta, xt, yt, yerrt, xrv, yrv, yerrrv):
        lp = lnprior_full(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_full(xt, yt, yerrt, xrv, yrv, yerrrv)

    if mode == 'full':
        # Prepare the data:
        xt = times.astype('float64')
        yt = relative_flux.astype('float64')
        if error is None:
            yerrt = 0.0
        else:
            yerrt = error.astype('float64')

        xrv = times_rv.astype('float64')
        yrv = rv.astype('float64')
        if rv_err is None:
            yerrrv = 0.0
        else:
            yerrrv = rv_err.astype('float64')

        # Define the posterior to use:
        lnprob = lnprob_full 

        # Start at the maximum likelihood value:
        nll = lambda *args: -lnprob_full(*args)

        # Extract initial input values of the parameters to be fitted:
        theta_0 = []
        for i in range(n_params):
            print all_mcmc_params[i],parameters[all_mcmc_params[i]]['object'].value
            theta_0.append(parameters[all_mcmc_params[i]]['object'].value)

        # Get ML estimate:
        result = op.minimize(nll, theta_0, args=(xt, yt, yerrt, xrv, yrv, yerrrv))
        theta_ml = result["x"]

        # Now define parameters for emcee:
        ndim = len(theta_ml)
        pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

        # Run the MCMC:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(xt, yt, yerrt, xrv, yrv, yerrrv))

        sampler.run_mcmc(pos, njumps+nburnin)

        # Save the parameter chains for the parameters that were actually varied:
        for i in range(n_params):
            c_param = all_mcmc_params[i]
            c_p_chain = np.array([])
            for walker in range(nwalkers):
                c_p_chain = np.append(c_p_chain,sampler.chain[walker,nburnin:,i])
            parameters[c_param]['object'].set_posterior(np.copy(c_p_chain))
    else:
        print 'Mode not supported. Doing nothing.'

import matplotlib.pyplot as plt
def plot_transit(t,f,parameters,ld_law):
        
    # Extract transit parameters:
    P = parameters['P']['object'].value
    inc = parameters['inc']['object'].value
    a = parameters['a']['object'].value
    p = parameters['p']['object'].value
    t0 = parameters['t0']['object'].value
    q1 = parameters['q1']['object'].value
    q2 = parameters['q2']['object'].value

    # Get data phases:
    phases = get_phases(t,P,t0)

    # Generate model times by super-sampling the times:
    model_t = np.linspace(np.min(t),np.max(t),len(t)*100)
    model_phase = get_phases(model_t,P,t0)

    # Generate model lightcurve and predicted one:
    params,m = init_batman(model_t,law=ld_law)
    params2,m2 = init_batman(t,law=ld_law)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a 
    params.inc = inc
    params.u = [coeff1,coeff2]
    model_lc = m.light_curve(params)
    model_pred = m2.light_curve(params)

    # Now plot:
    plt.style.use('ggplot')
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    idx = np.argsort(model_phase)
    plt.plot(phases,f,'.',color='black',alpha=0.4)
    plt.plot(model_phase[idx],model_lc[idx])
    plt.plot(phases,f-model_pred+(1-1.4*(p**2)),'.',color='black',alpha=0.4)
    plt.show()

def plot_transit_and_rv(t,f,trv,rv,rv_err,parameters,ld_law,rv_jitter):
    # Extract parameters:
    P = parameters['P']['object'].value
    inc = parameters['inc']['object'].value
    a = parameters['a']['object'].value
    p = parameters['p']['object'].value
    t0 = parameters['t0']['object'].value
    q1 = parameters['q1']['object'].value
    q2 = parameters['q2']['object'].value
    mu = parameters['mu']['object'].value
    K = parameters['K']['object'].value

    # Get data phases:
    phases = get_phases(t,P,t0)

    # Generate model times by super-sampling the times:
    model_t = np.linspace(np.min(t),np.max(t),len(t)*100)
    model_phase = get_phases(model_t,P,t0)

    # Generate model lightcurve and predicted one:
    params,m = init_batman(model_t,law=ld_law)
    params2,m2 = init_batman(t,law=ld_law)
    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    params.u = [coeff1,coeff2]
    model_lc = m.light_curve(params)
    model_pred = m2.light_curve(params)

    # Now plot:
    plt.style.use('ggplot')
    plt.subplot(211)
    #plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    idx = np.argsort(model_phase)
    plt.plot(phases,f,'.',color='black',alpha=0.4)
    plt.plot(model_phase[idx],model_lc[idx])
    plt.plot(phases,f-model_pred+(1-1.4*(p**2)),'.',color='black',alpha=0.4)

    plt.subplot(212)
    plt.ylabel('Radial velocity (m/s)')
    plt.xlabel('Phase')
    model_rv = mu - K*np.sin(2.*np.pi*model_phase)
    rv_phases = get_phases(trv,P,t0)
    plt.errorbar(rv_phases,(rv-mu)*1e3,yerr=rv_err*1e3,fmt='o')
    plt.plot(model_phase[idx],(model_rv[idx]-mu)*1e3)
    plt.show()
