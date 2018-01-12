# -*- coding: utf-8 -*-
from math import floor,ceil
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.gridspec as gridspec
from george import kernels
import george
import celerite
from celerite import terms
import numpy as np
import batman
import radvel

# This defines prior distributions that need samples to be
# controlled so they don't get out of their support:
prior_distributions = ['Uniform','Jeffreys','Beta']

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

def read_transit_params(prior_dict,instrument):
    names = ['P','inc','a','p','t0','q1','q2']
    vals = len(names)*[[]]
    for i in range(len(names)):
        try:
            param = prior_dict[names[i]]
        except:
            param = prior_dict[names[i]+'_'+instrument]
        vals[i] = param['object'].value
    return vals

def pre_process(all_t,all_f,all_f_err,options,transit_instruments,parameters):
    out_t = np.array([])
    out_f = np.array([])
    out_phases = np.array([])
    out_f_err = np.array([])
    out_transit_instruments = np.array([])
    all_phases = np.zeros(len(all_t))
    for instrument in options['photometry'].keys():
        all_idx = np.where(transit_instruments==instrument)[0]
        t = all_t[all_idx]
        f = all_f[all_idx]
        if all_f_err is not None:
            f_err = all_f_err[all_idx]
        
        # Now, the first phase in transit fitting is to 'detrend' the 
        # data. This is done with the 'detrend' flag. If 
        # the data is already detrended, set the flag to None:
        if options['photometry'][instrument]['PHOT_DETREND'] is not None:
            if options['photometry'][instrument]['PHOT_DETREND'] == 'mfilter':
                # Get median filter, and smooth it with a gaussian filter:
                from scipy.signal import medfilt
                from scipy.ndimage.filters import gaussian_filter
                filt = gaussian_filter(medfilt(f,options['photometry'][instrument]['WINDOW']),5)
                f = f/filt
                if f_err is not None:
                    f_err = f_err/filt

        # Extract transit parameters from prior dictionary:
        if options['MODE'] != 'transit_noise':
            P,inc,a,p,t0,q1,q2 = read_transit_params(parameters,instrument)

        # If the user wants to ommit transit events:
        if len(options['photometry'][instrument]['NOMIT'])>0:
            # Get the phases:
            phases = (t-t0)/P

            # Get the transit events in phase space:
            transit_events = np.arange(ceil(np.min(phases)),floor(np.max(phases))+1)

            # Convert to zeros fluxes at the events you want to eliminate:
            for n in options['photometry'][instrument]['NOMIT']:
                idx = np.where((phases>n-0.5)&(phases<n+0.5))[0]
                f[idx] = np.zeros(len(idx))

            # Eliminate them from the t,f and phases array:
            idx = np.where(f!=0.0)[0]
            t = t[idx]
            f = f[idx]
            phases = phases[idx]
            if f_err is not None:
                f_err = f_err[idx]

        if options['MODE'] != 'transit_noise':
            # Generate the phases:
            phases = get_phases(t,P,t0)
        # If outlier removal is on, remove them:
        if options['photometry'][instrument]['PHOT_GET_OUTLIERS'] and options['MODE'] != 'transit_noise':
            model = get_transit_model(t.astype('float64'),t0,P,p,a,inc,q1,q2,options['photometry'][instrument]['LD_LAW'])
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
        out_t = np.append(out_t,t)
        out_f = np.append(out_f,f)
        out_transit_instruments = np.append(out_transit_instruments,np.array(len(t)*[instrument]))
        out_f_err = np.append(out_f_err,f_err)
        #all_t[all_idx] = t
        #all_f[all_idx] = f 
        #all_f_err[all_idx] = f_err
        if options['MODE'] != 'transit_noise':
            out_phases = np.append(out_phases,phases)
            #all_phases[all_idx] = phases
        else:
            out_phases = np.append(out_phases,np.zeros(len(t)))
            #all_phases = np.zeros(len(t))

    if f_err is not None:
       return out_t.astype('float64'), out_phases.astype('float64'), out_f.astype('float64'), out_f_err.astype('float64'),out_transit_instruments
       #return all_t.astype('float64'), all_phases.astype('float64'), all_f.astype('float64'), all_f_err.astype('float64')
    else:
       return out_t.astype('float64'), out_phases.astype('float64'), out_f.astype('float64'), f_err,out_transit_instruments
       #return all_t.astype('float64'), all_phases.astype('float64'), all_f.astype('float64'), f_err

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

def init_radvel(nplanets=1):
    return radvel.model.Parameters(nplanets,basis='per tc e w k')

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
    elif ld_law=='squareroot':
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
    elif ld_law=='squareroot':
        coeff1 = np.sqrt(q1)*(1.-2.*q2)
        coeff2 = 2.*np.sqrt(q1)*q2
    elif ld_law=='logarithmic':
        coeff1 = 1.-np.sqrt(q1)*q2
        coeff2 = 1.-np.sqrt(q1)
    return coeff1,coeff2

def count_instruments(instrument_list):
    all_instruments = []
    for instrument in instrument_list:
        if instrument not in all_instruments:
            all_instruments.append(instrument)
    all_idxs = len(all_instruments)*[[]]
    all_ndata = len(all_instruments)*[[]]
    for i in range(len(all_instruments)):
        all_idxs[i] = np.where(all_instruments[i] == instrument_list)[0]
        all_ndata[i] = len(all_idxs[i])
    return all_instruments,all_idxs,np.array(all_ndata)

import emcee
import Wavelets
import scipy.optimize as op
def exonailer_mcmc_fit(times, relative_flux, error, tr_instruments, times_rv, rv, rv_err, rv_instruments,\
                       parameters, idx_resampling, options,texp = 0.01881944):
                       #ld_law, mode, rv_jitter = False, \
                       #njumps = 500, nburnin = 500, nwalkers = 100, noise_model = 'white',\
                       #resampling = False, idx_resampling = [], texp = 0.01881944, N_resampling = 5):
    """
    This function performs an MCMC fitting procedure using a transit model 
    fitted to input data using the batman package (Kreidberg, 2015) assuming 
    the underlying noise process is either 'white' or '1/f-like' (see Carter & 
    Winn, 2010). It makes use of the emcee package (Foreman-Mackey et al., 2014) 
    to perform the MCMC, and the sampling scheme explained in Kipping (2013) to 
    sample coefficients from two-parameter limb-darkening laws; the logarithmic 
    law is sampled according to Espinoza & Jordán (2016). 

    The inputs are:

      times:            Times (in same units as the period and time of transit center).

      relative_flux:    Relative flux; it is assumed out-of-transit flux is 1.

      error:            If you have errors on the fluxes, put them here. Otherwise, set 
                        this to None.

      tr_instruments:   Instruments of each time/flux pair.

      times_rv:         Times (in same units as the period and time of transit center) 
                        of RV data.

      rv:               Radial velocity measurements.

      rv_err:           If you have errors on the RVs, put them here. Otherwise, set 
                        this to None.

      rv_instruments:   Instruments of each time/RV pair.

      parameters:       Dictionary containing the information regarding the parameters (including priors).

      idx_resampling:   This defines the indexes over which you want to perform such resampling 
                        (selective resampling). It is a dictionary over the instruments; idx_resampling[instrument] 
                        has the indexes for the given instrument.

      options:          Dictionary containing the information inputted by the user.

      texp          :   Exposure time in days of each datapoint (default is Kepler long-cadence, 
                        taken from here: http://archive.stsci.edu/mast_faq.php?mission=KEPLER)

    The outputs are the chains of each of the parameters in the theta_0 array in the same 
    order as they were inputted. This includes the sampled parameters from all the walkers.

    """

    # If mode is not RV:
    if options['MODE'] != 'rv':
        params = {}
        m = {}
        t_resampling = {}
        transit_flat = {}
        # Count instruments:
        all_tr_instruments,all_tr_instruments_idxs,n_data_trs = count_instruments(tr_instruments)
        # Prepare data for batman:
        xt = times.astype('float64')
        yt = relative_flux.astype('float64')
        yerrt = error.astype('float64')
        if options['MODE'] != 'transit_noise':
          for k in range(len(all_tr_instruments)):
            instrument = all_tr_instruments[k]
            params[instrument],m[instrument] = init_batman(xt[all_tr_instruments_idxs[k]],\
                                               law=options['photometry'][instrument]['LD_LAW'])
            # Initialize the parameters of the transit model, 
            # and prepare resampling data if resampling is True:
            if options['photometry'][instrument]['RESAMPLING']:
               t_resampling[instrument] = np.array([])
               for i in range(len(idx_resampling[instrument])):
                   tij = np.zeros(options['photometry'][instrument]['NRESAMPLING'])
                   for j in range(1,options['photometry'][instrument]['NRESAMPLING']+1):
                       # Eq (35) in Kipping (2010)    
                       tij[j-1] = xt[all_tr_instruments_idxs[k]][idx_resampling[instrument][i]] + ((j - \
                                  ((options['photometry'][instrument]['NRESAMPLING']+1)/2.))*(texp/np.double(\
                                  options['photometry'][instrument]['NRESAMPLING'])))
                   t_resampling[instrument] = np.append(t_resampling[instrument], np.copy(tij))

               params[instrument],m[instrument] = init_batman(t_resampling[instrument],\
                                                  law=options['photometry'][instrument]['LD_LAW'])
               transit_flat[instrument] = np.ones(len(xt[all_tr_instruments_idxs[k]]))
               transit_flat[instrument][idx_resampling[instrument]] = np.zeros(len(idx_resampling[instrument]))

    # Initialize the variable names:
    if len(all_tr_instruments)>1:
        transit_params = ['P','inc']
    else:
        the_instrument = options['photometry'].keys()[0]
        transit_params = ['P','inc','t0','a','p','sigma_w','sigma_r','q1','q2']
    common_params = ['ecc','omega']

    # If mode is not transit, prepare the RV data too:
    if 'transit' not in options['MODE']:
       xrv = times_rv.astype('float64')
       yrv = rv.astype('float64')
       if rv_err is None:
           yerrrv = 0.0
       else:
           yerrrv = rv_err.astype('float64')
       all_rv_instruments,all_rv_instruments_idxs,n_data_rvs = count_instruments(rv_instruments)

       rv_params = ['K']
       #if len(all_rv_instruments)>1:
       #   for instrument in all_rv_instruments:
       #       rv_params.append('mu_'+instrument)
       #       rv_params.append('sigma_w_rv_'+instrument)
       #else:
       #   rv_params.append('mu')
       #   rv_params.append('sigma_w_rv')
       radvel_params = init_radvel()
    # Create lists that will save parameters to check the limits on:
    parameters_to_check = []

    # Check common parameters:
    if options['MODE'] != 'transit_noise':
        if parameters['ecc']['type'] == 'FIXED':
           common_params.pop(common_params.index('ecc'))
        elif parameters['ecc']['type'] in prior_distributions:
           parameters_to_check.append('ecc')

        if parameters['omega']['type'] == 'FIXED':
           common_params.pop(common_params.index('omega'))
        elif parameters['omega']['type'] in prior_distributions:
           parameters_to_check.append('omega')


    # Eliminate from the parameter list parameters that are being fixed:
    # First, generate a sufix dictionary, which will add the sufix _instrument to
    # each instrument in the MCMC, in order to keep track of the parameters that
    # are being held constant between instruments and those that vary with instrument:
    sufix = {}
    if options['MODE'] != 'rv' and options['MODE'] != 'transit_noise':
        if len(all_tr_instruments)>1:
            # Check parameters that always will be constant amongst transits:
            for par in ['P','inc']:
                if parameters[par]['type'] == 'FIXED':
                    transit_params.pop(transit_params.index(par))
                elif parameters[par]['type'] in prior_distributions:
                    parameters_to_check.append(par)

            # Now check parameters that might change between instruments:
            for i in range(len(all_tr_instruments)):
                instrument = all_tr_instruments[i]
                sufix[instrument] = {}
                for par in ['t0','a','p','sigma_w','q1','q2']:
                    orig_par = par
                    sufix[instrument][orig_par] = ''
                    if par not in parameters.keys():
                        par = par+'_'+instrument
                        sufix[instrument][orig_par] = '_'+instrument
                        if par not in parameters.keys():
                            print 'Error: parameter '+orig_par+' not defined. Exiting...'
                            sys.exit()
                    if par not in transit_params:
                        transit_params.append(par)    
                    if parameters[par]['type'] == 'FIXED':
                        transit_params.pop(transit_params.index(par))
                    elif parameters[par]['type'] in prior_distributions:
                        parameters_to_check.append(par)
                if options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'flicker':
                    for noise_param in ['sigma_r']:
                        transit_params.append(noise_param+'_'+instrument)
                        if parameters[noise_param+'_'+instrument]['type'] == 'FIXED':
                            transit_params.pop(transit_params.index(noise_param+'_'+instrument))
                        elif parameters[noise_param+'_'+instrument]['type'] in prior_distributions:
                            parameters_to_check.append(noise_param+'_'+instrument)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPExpSquaredKernel':
                    for noise_param in ['lnh','lnlambda']:
                        transit_params.append(noise_param+'_'+instrument)
                        if parameters[noise_param+'_'+instrument]['type'] == 'FIXED':
                            transit_params.pop(transit_params.index(noise_param+'_'+instrument))
                        elif parameters[noise_param+'_'+instrument]['type'] in prior_distributions:
                            parameters_to_check.append(noise_param+'_'+instrument)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPGranulation':
                    for noise_param in ['lnomega','lnS']:
                        transit_params.append(noise_param+'_'+instrument)
                        if parameters[noise_param+'_'+instrument]['type'] == 'FIXED':
                            transit_params.pop(transit_params.index(noise_param+'_'+instrument))
                        elif parameters[noise_param+'_'+instrument]['type'] in prior_distributions:
                            parameters_to_check.append(noise_param+'_'+instrument)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPAsteroseismology':
                    for noise_param in ['lnomega','lnS','lnQ','lnA','epsilon','lnW','lnnu','lnDeltanu']:
                        transit_params.append(noise_param+'_'+instrument)
                        if parameters[noise_param+'_'+instrument]['type'] == 'FIXED':
                            transit_params.pop(transit_params.index(noise_param+'_'+instrument))
                        elif parameters[noise_param+'_'+instrument]['type'] in prior_distributions:
                            parameters_to_check.append(noise_param+'_'+instrument)
                            
        else:
            for par in ['P','t0','a','p','inc','sigma_w','q1','q2']:
                 if parameters[par]['type'] == 'FIXED':
                     transit_params.pop(transit_params.index(par))
                 elif parameters[par]['type'] in prior_distributions:
                    parameters_to_check.append(par)
            if options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'flicker':
                if parameters['sigma_r']['type'] == 'FIXED':
                    transit_params.pop(transit_params.index('sigma_r'))
                elif parameters['sigma_r']['type'] in prior_distributions:
                    parameters_to_check.append('sigma_r')
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPExpSquaredKernel':
                transit_params.pop(transit_params.index('sigma_r'))
                for noise_param in ['lnh','lnlambda']:
                    transit_params.append(noise_param)
                    if parameters[noise_param]['type'] == 'FIXED':
                        transit_params.pop(transit_params.index(noise_param))
                    elif parameters[noise_param]['type'] in prior_distributions:
                        parameters_to_check.append(noise_param)
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPGranulation':
                transit_params.pop(transit_params.index('sigma_r'))
                for noise_param in ['lnomega','lnS']:
                    transit_params.append(noise_param)
                    if parameters[noise_param]['type'] == 'FIXED':
                        transit_params.pop(transit_params.index(noise_param))
                    elif parameters[noise_param]['type'] in prior_distributions:
                        parameters_to_check.append(noise_param)
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPAsteroseismology':
                transit_params.pop(transit_params.index('sigma_r'))
                for noise_param in ['lnomega','lnS','lnQ','lnA','epsilon','lnW','lnnu','lnDeltanu']:
                    transit_params.append(noise_param)
                    if parameters[noise_param]['type'] == 'FIXED':
                        transit_params.pop(transit_params.index(noise_param))
                    elif parameters[noise_param]['type'] in prior_distributions:
                        parameters_to_check.append(noise_param)
            else:
                transit_params.pop(transit_params.index('sigma_r'))

    if options['MODE'] != 'transit' and options['MODE'] != 'transit_noise':
        if parameters['K']['type'] == 'FIXED':
            rv_params.pop(rv_params.index('K'))
        elif parameters['K']['type'] in prior_distributions:
            parameters_to_check.append('K')
        if len(all_rv_instruments)>1:
            sigma_w_rv = {}
            for instrument in all_rv_instruments:
                sufix[instrument] = {}
                for par in ['mu','sigma_w_rv']:
                    orig_par = par
                    sufix[instrument][orig_par] = ''
                    if par not in parameters.keys():
                        par = par+'_'+instrument
                        sufix[instrument][orig_par] = '_'+instrument
                        if par not in parameters.keys():
                            print 'Error: parameter '+orig_par+' not defined. Exiting...'
                            sys.exit()
                    if par not in rv_params:
                        rv_params.append(par)
                    if parameters[par]['type'] == 'FIXED':
                        rv_params.pop(rv_params.index(par))
                    elif parameters[par]['type'] in prior_distributions:
                        if par not in parameters_to_check:
                            parameters_to_check.append(par)
        else:
            if parameters['K']['type'] == 'FIXED':
                rv_params.pop(rv_params.index('K'))
            elif parameters['K']['type'] in prior_distributions:
                parameters_to_check.append('K')
            for rvpar in ['sigma_w_rv','mu']:
                if parameters[rvpar]['type'] in prior_distributions:
                    parameters_to_check.append(rvpar)
                    rv_params.append(rvpar)
                elif parameters[rvpar]['type'] != 'FIXED':
                    rv_params.append(rvpar)
    if options['MODE'] == 'transit':
            all_mcmc_params = transit_params + common_params
    elif options['MODE'] == 'rv':
            all_mcmc_params = rv_params + common_params
    elif options['MODE'] == 'transit_noise':
            all_mcmc_params = []
            parameters_to_check = []
            if options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'white':
                noise_parameters = ['sigma_w']
            if options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'flicker':
                noise_parameters = ['sigma_w','sigma_r']
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPExpSquaredKernel':
                noise_parameters in ['lnh','lnlambda','sigma_w']
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPGranulation':
                noise_parameters = ['lnomega','lnS','sigma_w']
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPAsteroseismology':
                noise_parameters = ['lnomega','lnS','lnQ','lnA','epsilon','lnW','lnnu','lnDeltanu','sigma_w']
            for noise_param in noise_parameters:
                    all_mcmc_params.append(noise_param)
                    if parameters[noise_param]['type'] == 'FIXED':
                        all_mcmc_params.pop(all_mcmc_params.index(noise_param))
                    elif parameters[noise_param]['type'] in prior_distributions:
                        parameters_to_check.append(noise_param) 
    else:
            all_mcmc_params = transit_params + rv_params + common_params

    n_params = len(all_mcmc_params)
    log2pi = np.log(2.*np.pi)
    def normal_like(x,mu,tau):
        return 0.5*(np.log(tau) - log2pi - tau*( (x-mu)**2))

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

    def get_sq_exp_likelihood(t,residuals,errors,sigma_w,lnh,lnlambda):
        kernel = (np.exp(lnh)**2)*george.kernels.ExpSquaredKernel(np.exp(lnlambda)**2)
        gp = george.GP(kernel,solver=george.HODLRSolver)
        try:
            gp.compute(t,np.sqrt(errors**2 + sigma_w**2))
        except:
            return -np.inf
        return gp.lnlikelihood(residuals)

    def get_granulation_likelihood(t,residuals,errors,sigma_w,lnomega,lnS):
        bounds = dict(log_S0=(-1e15, 1e15), log_Q=(-1e15, 1e15), log_omega0=(-1e15, 1e15),log_sigma=(-1e15,1e15))
        kernel = terms.SHOTerm(log_S0=lnS, log_Q=np.log(1./np.sqrt(2.)), log_omega0=lnomega,\
                                   bounds=bounds)
        kernel.freeze_parameter("log_Q")
        kernel += terms.JitterTerm(log_sigma=np.log(sigma_w),\
                  bounds=bounds)
        gp = celerite.GP(kernel, mean=np.mean(residuals))
        try:
            gp.compute(t,errors)
        except:
            return -np.inf
        return gp.log_likelihood(residuals)

    def get_asteroseismology_likelihood(t,residuals,errors,sigma_w,lnomega,lnS,lnQ,lnA,epsilon,\
                                        lnW,lnnu,lnDeltanu,instrument):
        bounds = dict(log_S0=(-1e15, 1e15), log_Q=(-1e15, 1e15), log_omega0=(-1e15, 1e15),log_sigma=(-1e15,1e15))
        # First, the granulation noise component:
        kernel = terms.SHOTerm(log_S0=lnS, log_Q=np.log(1./np.sqrt(2.)), log_omega0=lnomega,\
                                   bounds=bounds)
        kernel.freeze_parameter("log_Q")

        # Next, the frequency kernels:
        nu = np.exp(lnnu)
        Deltanu = np.exp(lnDeltanu)
        W = np.exp(lnW)
        n = options['photometry'][instrument]["NASTEROSEISMOLOGY"]
        for j in range(-(n-1)/2,(n-1)/2+1):
            lnSj = lnA - 2.*lnQ - (j*Deltanu+epsilon)**2/(2.*(W**2))
            wj = 2.*np.pi*(nu+j*Deltanu+epsilon)*0.0864 # Last factor converts from muHz to 1/day (assuming t is in days)
            if wj>0.:
                kernel += terms.SHOTerm(log_S0=lnSj, log_Q=lnQ, log_omega0=np.log(wj),
                            bounds=bounds)
            else:
                return -np.inf

        # Finally, a "jitter" term component for the photometric noise:
        kernel += terms.JitterTerm(log_sigma=np.log(sigma_w),\
                  bounds=bounds)

        # Set the GP:
        gp = celerite.GP(kernel, mean=np.mean(residuals))
        try:
            gp.compute(t,errors)
            lnlike = gp.log_likelihood(residuals)
        except:
            return -np.inf

        # Return the likelihood:
        if not np.isnan(lnlike):
            return lnlike
        else:
            return -np.inf

    def lnlike_transit_noise(gamma=1.0):
            residuals = (yt-1.0)*1e6
            if options['photometry'][the_instrument]['PHOT_NOISE_MODEL'] == 'flicker':
               log_like = get_fn_likelihood(residuals,parameters['sigma_w']['object'].value,\
                               parameters['sigma_r']['object'].value)
            elif options['photometry'][the_instrument]['PHOT_NOISE_MODEL'] == 'GPExpSquaredKernel':
               log_like = get_sq_exp_likelihood(xt,residuals,yerrt*1e6,\
                              parameters['sigma_w']['object'].value,\
                              parameters['lnh']['object'].value,\
                              parameters['lnlambda']['object'].value)
            elif options['photometry'][the_instrument]['PHOT_NOISE_MODEL'] == 'GPGranulation':
               log_like = get_granulation_likelihood(xt,residuals,yerrt*1e6,\
                              parameters['sigma_w']['object'].value,\
                              parameters['lnomega']['object'].value,\
                              parameters['lnS']['object'].value)
            elif options['photometry'][the_instrument]['PHOT_NOISE_MODEL'] == 'GPAsteroseismology':
               log_like = get_asteroseismology_likelihood(xt,residuals,yerrt*1e6,\
                              parameters['sigma_w']['object'].value,\
                              parameters['lnomega']['object'].value,\
                              parameters['lnS']['object'].value,\
                              parameters['lnQ']['object'].value,\
                              parameters['lnA']['object'].value,\
                              parameters['epsilon']['object'].value,\
                              parameters['lnW']['object'].value,\
                              parameters['lnnu']['object'].value,\
                              parameters['lnDeltanu']['object'].value,\
                              the_instrument)
            else:
               taus = 1.0/((yerrt*1e6)**2 + (parameters['sigma_w']['object'].value)**2)
               log_like = -0.5*(n_data_trs[0]*log2pi+np.sum(np.log(1./taus)+taus*(residuals**2)))
            return log_like

    def lnlike_transit(gamma=1.0):
        if len(all_tr_instruments) == 1:
            coeff1,coeff2 = reverse_ld_coeffs(options['photometry'][the_instrument]['LD_LAW'], \
                            parameters['q1']['object'].value,parameters['q2']['object'].value)
            params[the_instrument].t0 = parameters['t0']['object'].value
            params[the_instrument].per = parameters['P']['object'].value
            params[the_instrument].rp = parameters['p']['object'].value
            params[the_instrument].a = parameters['a']['object'].value
            params[the_instrument].inc = parameters['inc']['object'].value
            params[the_instrument].ecc = parameters['ecc']['object'].value
            params[the_instrument].w = parameters['omega']['object'].value
            params[the_instrument].u = [coeff1,coeff2]
            model = m[the_instrument].light_curve(params[the_instrument])
            if options['photometry'][the_instrument]['RESAMPLING']:
               for i in range(len(idx_resampling[the_instrument])):
                   transit_flat[the_instrument][idx_resampling[the_instrument][i]] = \
                   np.mean(model[i*options['photometry'][the_instrument]['NRESAMPLING']:options['photometry'][the_instrument]['NRESAMPLING']*(i+1)])
               residuals = (yt-transit_flat[the_instrument])*1e6
            else:
               residuals = (yt-model)*1e6
            if options['photometry'][the_instrument]['PHOT_NOISE_MODEL'] == 'flicker':
               log_like = get_fn_likelihood(residuals,parameters['sigma_w']['object'].value,\
                               parameters['sigma_r']['object'].value)
            elif options['photometry'][the_instrument]['PHOT_NOISE_MODEL'] == 'GPExpSquaredKernel':
               log_like = get_sq_exp_likelihood(xt,residuals,yerrt*1e6,\
                              parameters['sigma_w']['object'].value,\
                              parameters['lnh']['object'].value,\
                              parameters['lnlambda']['object'].value)
            elif options['photometry'][the_instrument]['PHOT_NOISE_MODEL'] == 'GPGranulation':
               log_like = get_granulation_likelihood(xt,residuals,yerrt*1e6,\
                              parameters['sigma_w']['object'].value,\
                              parameters['lnomega']['object'].value,\
                              parameters['lnS']['object'].value)
            elif options['photometry'][the_instrument]['PHOT_NOISE_MODEL'] == 'GPAsteroseismology':
               log_like = get_asteroseismology_likelihood(xt,residuals,yerrt*1e6,\
                              parameters['sigma_w']['object'].value,\
                              parameters['lnomega']['object'].value,\
                              parameters['lnS']['object'].value,\
                              parameters['lnQ']['object'].value,\
                              parameters['lnA']['object'].value,\
                              parameters['epsilon']['object'].value,\
                              parameters['lnW']['object'].value,\
                              parameters['lnnu']['object'].value,\
                              parameters['lnDeltanu']['object'].value,\
                              the_instrument)
            else:
               taus = 1.0/((yerrt*1e6)**2 + (parameters['sigma_w']['object'].value)**2)
               log_like = -0.5*(n_data_trs[0]*log2pi+np.sum(np.log(1./taus)+taus*(residuals**2)))
            #print 'Median residuals:',np.median(residuals)
            #print 'Transit log-like:',log_like
            return log_like
        else:
            log_like = 0.0
            for k in range(len(all_tr_instruments)):
                instrument = all_tr_instruments[k]
                coeff1,coeff2 = reverse_ld_coeffs(options['photometry'][instrument]['LD_LAW'], \
                                parameters['q1'+sufix[instrument]['q1']]['object'].value,\
                                parameters['q2'+sufix[instrument]['q2']]['object'].value)
                params[instrument].t0 = parameters['t0'+sufix[instrument]['t0']]['object'].value
                params[instrument].per = parameters['P']['object'].value
                params[instrument].rp = parameters['p'+sufix[instrument]['p']]['object'].value
                params[instrument].a = parameters['a'+sufix[instrument]['a']]['object'].value
                params[instrument].inc = parameters['inc']['object'].value
                params[instrument].ecc = parameters['ecc']['object'].value
                params[instrument].w = parameters['omega']['object'].value
                params[instrument].u = [coeff1,coeff2]
                model = m[instrument].light_curve(params[instrument])
                if options['photometry'][instrument]['RESAMPLING']:
                   for i in range(len(idx_resampling[instrument])):
                       transit_flat[instrument][idx_resampling[instrument][i]] = \
                       np.mean(model[i*options['photometry'][instrument]['NRESAMPLING']:options['photometry'][instrument]['NRESAMPLING']*(i+1)])
                   residuals = (yt[all_tr_instruments_idxs[k]]-transit_flat[instrument])*1e6
                else:
                   residuals = (yt[all_tr_instruments_idxs[k]]-model)*1e6
                if options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'flicker':
                   log_like = log_like + get_fn_likelihood(residuals,parameters['sigma_w'+sufix[instrument]['sigma_w']]['object'].value,\
                                   parameters['sigma_r'+sufix[instrument]['sigma_r']]['object'].value)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPExpSquaredKernel':
                   log_like = log_like + get_sq_exp_likelihood(xt[all_tr_instruments_idxs[k]],residuals,yerrt[all_tr_instruments_idxs[k]]*1e6,\
                              parameters['sigma_w'+sufix[instrument]['sigma_w']]['object'].value,\
                              parameters['lnh'+sufix[instrument]['lnh']]['object'].value,\
                              parameters['lnlambda'+sufix[instrument]['lnlambda']]['object'].value)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPGranulation':
                   log_like = log_like + get_granulation_likelihood(xt[all_tr_instruments_idxs[k]],residuals,yerrt[all_tr_instruments_idxs[k]]*1e6,\
                              parameters['sigma_w'+sufix[instrument]['sigma_w']]['object'].value,\
                              parameters['lnomega'+sufix[instrument]['lnomega']]['object'].value,\
                              parameters['lnS'+sufix[instrument]['lnS']]['object'].value)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPAsteroseismology':
                   log_like = log_like + get_asteroseismology_likelihood(xt[all_tr_instruments_idxs[k]],residuals,yerrt[all_tr_instruments_idxs[k]]*1e6,\
                              parameters['sigma_w'+sufix[instrument]['sigma_w']]['object'].value,\
                              parameters['lnomega'+sufix[instrument]['lnomega']]['object'].value,\
                              parameters['lnS'+sufix[instrument]['lnS']]['object'].value,\
                              parameters['lnQ'+sufix[instrument]['lnQ']]['object'].value,\
                              parameters['lnA'+sufix[instrument]['lnA']]['object'].value,\
                              parameters['epsilon'+sufix[instrument]['epsilon']]['object'].value,\
                              parameters['lnW'+sufix[instrument]['lnW']]['object'].value,\
                              parameters['lnnu'+sufix[instrument]['lnnu']]['object'].value,\
                              parameters['lnDeltanu'+sufix[instrument]['lnDeltanu']]['object'].value,\
                              instrument)
                else:
                   taus = 1.0/((yerrt[all_tr_instruments_idxs[k]]*1e6)**2 + (parameters['sigma_w'+sufix[instrument]['sigma_w']]['object'].value)**2)
                   log_like = log_like - 0.5*(n_data_trs[k]*log2pi+np.sum(np.log(1./taus)+taus*(residuals**2)))
            return log_like
            

    def lnlike_rv():
        #print 'RVs:'
        #print 'mu',parameters['mu']['object'].value
        #print 'K',parameters['K']['object'].value
        #print 'ecc',parameters['ecc']['object'].value
        if len(all_rv_instruments) == 1:
            radvel_params['per1'] = parameters['P']['object'].value
            radvel_params['tc1'] = parameters['t0']['object'].value
            radvel_params['w1'] = parameters['omega']['object'].value*np.pi/180.
            radvel_params['e1'] = parameters['ecc']['object'].value
            radvel_params['k1'] = parameters['K']['object'].value
            model = parameters['mu']['object'].value + radvel.model.RVModel(radvel_params).__call__(xrv)

            residuals = (yrv-model)
            #print 'Median residuals:',np.median(residuals)
            taus = 1.0/((yerrrv)**2 + (parameters['sigma_w_rv']['object'].value)**2)
            log_like = -0.5*(n_data_rvs[0]*log2pi+np.sum(np.log(1./taus)+taus*(residuals**2)))
            #print 'RV log-like:',log_like
            return log_like
        else:
            log_like = 0.0
            for i in range(len(all_rv_instruments)):
                radvel_params['per1'] = parameters['P']['object'].value
                radvel_params['tc1'] = parameters['t0']['object'].value
                radvel_params['w1'] = parameters['omega']['object'].value*np.pi/180.
                radvel_params['e1'] = parameters['ecc']['object'].value
                radvel_params['k1'] = parameters['K']['object'].value
                model = parameters['mu'+sufix[all_rv_instruments[i]]['mu']]['object'].value + \
                        radvel.model.RVModel(radvel_params).__call__(xrv[all_rv_instruments_idxs[i]])
                residuals = (yrv[all_rv_instruments_idxs[i]]-model)
                taus = 1.0/((yerrrv[all_rv_instruments_idxs[i]])**2 + (parameters['sigma_w_rv'+sufix[all_rv_instruments[i]]['sigma_w_rv']]['object'].value)**2)
                log_like = log_like -0.5*(n_data_rvs[i]*log2pi+np.sum(np.log(1./taus)+taus*(residuals**2)))
            return log_like

    def lnprior(theta):
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

    def lnprob_full(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        lnrv = lnlike_rv()
        return lp + lnrv + lnlike_transit()

    def lnprob_transit(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_transit()

    def lnprob_transit_noise(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_transit_noise()

    def lnprob_rv(theta):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike_rv()

    # Define the posterior to use:
    if options['MODE'] == 'full':
        lnprob = lnprob_full 
    elif options['MODE'] == 'transit':
        lnprob = lnprob_transit
    elif options['MODE'] == 'transit_noise':
        lnprob = lnprob_transit_noise
    elif options['MODE'] == 'rv':
        lnprob = lnprob_rv
    else:
        print 'Mode not supported. Doing nothing.'

    # If already not done, get posterior samples:
    if len(parameters[all_mcmc_params[0]]['object'].posterior) == 0:
        # Make a first MCMC run to search for optimal parameter values 
        # in (almost) all the parameter space defined by the priors if 
        # no initial guess is given:
        ndim = n_params
        pos = []
        for j in range(200):
            while True:
                theta_vector = np.array([])
                for i in range(n_params):
                    current_parameter = all_mcmc_params[i]
                    # If parameter has a guess, sample a value from prior distribution, multiply it by 1e-3 and 
                    # add it to the real value (this is just to have the walkers move around a sphere around the 
                    # guess with orders of magnitude defined by the prior). If no initial guess, sample from the 
                    # prior:
                    if parameters[current_parameter]['object'].has_guess:
                        theta_vector = np.append(theta_vector,parameters[current_parameter]['object'].init_value + \
                                                 (parameters[current_parameter]['object'].init_value-\
                                                  parameters[current_parameter]['object'].sample())*1e-3)
                    else:
                        theta_vector = np.append(theta_vector,parameters[current_parameter]['object'].sample())
                lnprob(theta_vector)
                val = lnprob(theta_vector)
                try:
                    val = lnprob(theta_vector)
                except:
                    val = np.inf
                if np.isfinite(val):
                    break
            pos.append(theta_vector)

        # Run the sampler for a bit (300 walkers, 300 jumps, 300 burnin):
        print '\t Starting first iteration run...'
        sampler = emcee.EnsembleSampler(200, ndim, lnprob)
        sampler.run_mcmc(pos, 200)

        # Now sample the walkers around the values found in previous iteration:
        pos = []
        first_time = True
        init_vals = np.zeros(n_params)
        init_vals_sigma = np.zeros(n_params)
        for j in range(options['NWALKERS']):
            while True:
                theta_vector = np.array([])
                for i in range(n_params):
                    if first_time:
                        c_p_chain = np.array([])
                        for walker in range(200):
                            c_p_chain = np.append(c_p_chain,sampler.chain[walker,100:,i])
                        init_vals[i] = np.median(c_p_chain)
                        init_vals_sigma[i] = get_sigma(c_p_chain,np.median(c_p_chain))
                    current_parameter = all_mcmc_params[i]
                    # Put the walkers around a small gaussian sphere centered on the best value 
                    # found in previous iteration. Walkers will run away from sphere eventually:
                    theta_vector = np.append(theta_vector,np.random.normal(init_vals[i],\
                                                                           init_vals_sigma[i]*1e-3))
                if first_time:
                    first_time = False
                try:
                    val = lnprob(theta_vector)
                except:
                    val = np.inf
                if np.isfinite(val):
                    break
            pos.append(theta_vector)

        # Run the (final) MCMC:
        print '\t Done! Starting MCMC...'
        sampler = emcee.EnsembleSampler(options['NWALKERS'], ndim, lnprob)

        sampler.run_mcmc(pos, options['NJUMPS']+options['NBURNIN'])

        print '\t Done! Saving...'
        # Save the parameter chains for the parameters that were actually varied:
        for i in range(n_params):
            c_param = all_mcmc_params[i]
            c_p_chain = np.array([])
            for walker in range(options['NWALKERS']):
                c_p_chain = np.append(c_p_chain,sampler.chain[walker,options['NBURNIN']:,i])
            parameters[c_param]['object'].set_posterior(np.copy(c_p_chain))

    # When done or if MCMC already performed, save results:
    initial_values = {}
    for i in range(len(all_mcmc_params)):
        initial_values[all_mcmc_params[i]] = parameters[all_mcmc_params[i]]['object'].value

import matplotlib.pyplot as plt
def plot_transit(t,f,parameters,ld_law,transit_instruments,\
                 resampling = False, phase_max = 0.025, \
                 idx_resampling_pred = [], texp = 0.01881944, N_resampling = 5):
        
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
    model_t = np.linspace(np.min(t),np.max(t),len(t)*N_resampling)
    model_phase = get_phases(model_t,P,t0)

    # Initialize the parameters of the transit model, 
    # and prepare resampling data if resampling is True:
    if resampling:
        idx_resampling = np.where((model_phase>-phase_max)&(model_phase<phase_max))[0]
        t_resampling = np.array([])
        for i in range(len(idx_resampling)):
            tij = np.zeros(N_resampling)
            for j in range(1,N_resampling+1):
                # Eq (35) in Kipping (2010)    
                tij[j-1] = model_t[idx_resampling[i]] + ((j - ((N_resampling+1)/2.))*(texp/np.double(N_resampling)))
            t_resampling = np.append(t_resampling, np.copy(tij))

        idx_resampling_pred = np.where((phases>-phase_max)&(phases<phase_max))[0]
        t_resampling_pred = np.array([])
        for i in range(len(idx_resampling_pred)):
            tij = np.zeros(N_resampling)
            for j in range(1,N_resampling+1):
                tij[j-1] = t[idx_resampling_pred[i]] + ((j - ((N_resampling+1)/2.))*(texp/np.double(N_resampling)))
            t_resampling_pred = np.append(t_resampling_pred, np.copy(tij))
        params,m = init_batman(t_resampling, law=ld_law)
        params2,m2 = init_batman(t_resampling_pred, law=ld_law)
        transit_flat = np.ones(len(model_t))
        transit_flat[idx_resampling] = np.zeros(len(idx_resampling))
        transit_flat_pred = np.ones(len(t))
        transit_flat_pred[idx_resampling_pred] = np.zeros(len(idx_resampling_pred))

    else:
        params,m = init_batman(model_t,law=ld_law)
        params2,m2 = init_batman(t,law=ld_law)
    #####################################################################

    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    params.u = [coeff1,coeff2]

    # Generate model and predicted lightcurves:
    if resampling:
        model = m.light_curve(params)
        for i in range(len(idx_resampling)):
            transit_flat[idx_resampling[i]] = np.mean(model[i*N_resampling:N_resampling*(i+1)])
        model_lc = transit_flat

        model = m2.light_curve(params)
        for i in range(len(idx_resampling_pred)):
            transit_flat_pred[idx_resampling_pred[i]] = np.mean(model[i*N_resampling:N_resampling*(i+1)])
        model_pred = transit_flat_pred
    else:
        model_lc = m.light_curve(params)
        model_pred = m2.light_curve(params)

    # Now plot:
    plt.style.use('ggplot')
    plt.title('exonailer final fit + phased data')
    plt.xlabel('Phase')
    plt.ylabel('Relative flux')
    idx = np.argsort(model_phase)
    plt.plot(phases,f,'.',color='black',alpha=0.4)
    plt.plot(model_phase[idx],model_lc[idx])
    idx_ph = np.argsort(phases)
    plt.plot(phases[idx_ph],np.ones(len(phases))*(1-2.5*p**2),'--',color='r')
    plt.plot(phases,(f-model_pred) + (1-2.5*p**2),'.',color='black',alpha=0.4)
    plt.show()

def plot_transit_and_rv(times, relative_flux, error, tr_instruments, times_rv, rv, rv_err, rv_instruments,\
                       parameters, idx_resampling, options, texp = 0.01881944):
    # Generate out_dir folder name (for saving residuals, models, etc.):
    mode = options['MODE']
    target = options['TARGET']
    fname = target+'_'+mode+'_'
    for instrument in options['photometry'].keys():
        fname = fname + instrument +'_'+options['photometry'][instrument]['PHOT_NOISE_MODEL']+\
                      '_'+options['photometry'][instrument]['LD_LAW']+'_'
    out_dir = 'results/'+fname[:-1]+'/'

    plt.title('exonailer final fit + data')
    # If mode is not RV:
    if options['MODE'] != 'rv':
        params = {}
        m = {}
        t_resampling = {}
        transit_flat = {}
        # Count instruments:
        all_tr_instruments,all_tr_instruments_idxs,n_data_trs = count_instruments(tr_instruments)
        # Prepare data for batman:
        xt = times.astype('float64')
        yt = relative_flux.astype('float64')
        yerrt = error.astype('float64')
        for k in range(len(all_tr_instruments)):
            instrument = all_tr_instruments[k]
            params[instrument],m[instrument] = init_batman(xt[all_tr_instruments_idxs[k]],\
                                               law=options['photometry'][instrument]['LD_LAW'])
            # Initialize the parameters of the transit model, 
            # and prepare resampling data if resampling is True:
            if options['photometry'][instrument]['RESAMPLING']:
               t_resampling[instrument] = np.array([])
               for i in range(len(idx_resampling[instrument])):
                   tij = np.zeros(options['photometry'][instrument]['NRESAMPLING'])
                   for j in range(1,options['photometry'][instrument]['NRESAMPLING']+1):
                       # Eq (35) in Kipping (2010)    
                       tij[j-1] = xt[all_tr_instruments_idxs[k]][idx_resampling[instrument][i]] + ((j - \
                                  ((options['photometry'][instrument]['NRESAMPLING']+1)/2.))*(texp/np.double(\
                                  options['photometry'][instrument]['NRESAMPLING'])))
                   t_resampling[instrument] = np.append(t_resampling[instrument], np.copy(tij))

               params[instrument],m[instrument] = init_batman(t_resampling[instrument],\
                                                  law=options['photometry'][instrument]['LD_LAW'])
               transit_flat[instrument] = np.ones(len(xt[all_tr_instruments_idxs[k]]))
               transit_flat[instrument][idx_resampling[instrument]] = np.zeros(len(idx_resampling[instrument]))

    # Initialize the variable names:
    if len(all_tr_instruments)>1:
        transit_params = ['P','inc']
    else:
        the_instrument = options['photometry'].keys()[0]
        transit_params = ['P','inc','t0','a','p','inc','sigma_w','sigma_r','q1','q2']
    common_params = ['ecc','omega']

    # If mode is not transit, prepare the data too:
    if 'transit' not in options['MODE']:
       xrv = times_rv.astype('float64')
       yrv = rv.astype('float64')
       if rv_err is None:
           yerrrv = 0.0
       else:
           yerrrv = rv_err.astype('float64')
       all_rv_instruments,all_rv_instruments_idxs,n_data_rvs = count_instruments(rv_instruments)
       rv_params = ['K']

       #if len(all_rv_instruments)>1:
       #   for instrument in all_rv_instruments:
       #       rv_params.append('mu_'+instrument)
       #       rv_params.append('sigma_w_rv_'+instrument)
       #else:
       #   rv_params.append('mu')
       #   rv_params.append('sigma_w_rv')

    # Create lists that will save parameters to check the limits on:
    parameters_to_check = []

    # Check common parameters:
    if parameters['ecc']['type'] == 'FIXED':
       common_params.pop(common_params.index('ecc'))
    elif parameters['ecc']['type'] in prior_distributions:
       parameters_to_check.append('ecc')

    if parameters['omega']['type'] == 'FIXED':
       common_params.pop(common_params.index('omega'))
    elif parameters['omega']['type'] in prior_distributions:
       parameters_to_check.append('omega')

    # Eliminate from the parameter list parameters that are being fixed:
    # First, generate a sufix dictionary, which will add the sufix _instrument to 
    # each instrument in the MCMC, in order to keep track of the parameters that 
    # are being held constant between instruments and those that vary with instrument:
    sufix = {}
    if options['MODE'] != 'rv':
        if len(all_tr_instruments)>1:
            # First, generate a sufix dictionary, which will add the sufix _instrument to 
            # each instrument in the MCMC, in order to keep track of the parameters that 
            # are being held constant between instruments and those that vary with instrument:
            sufix = {}
            # Check parameters that always will be constant amongst transits:
            for par in ['P','inc']:
                if parameters[par]['type'] == 'FIXED':
                    transit_params.pop(transit_params.index(par))
                elif parameters[par]['type'] in prior_distributions:
                    parameters_to_check.append(par)

            # Now check parameters that might change between instruments:
            for i in range(len(all_tr_instruments)):
                instrument = all_tr_instruments[i]
                sufix[instrument] = {}
                for par in ['t0','a','p','sigma_w','q1','q2']:
                    orig_par = par
                    sufix[instrument][orig_par] = ''
                    if par not in parameters.keys():
                        par = par+'_'+instrument
                        sufix[instrument][orig_par] = '_'+instrument
                        if par not in parameters.keys():
                            print 'Error: parameter '+orig_par+' not defined. Exiting...'
                            sys.exit()
                    if par not in transit_params:
                        transit_params.append(par)        
                    if parameters[par]['type'] == 'FIXED':
                        transit_params.pop(transit_params.index(par))
                    elif parameters[par]['type'] in prior_distributions:
                        parameters_to_check.append(par)
                if options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'flicker':
                    for noise_param in ['sigma_r']:
                        transit_params.append(noise_param+'_'+instrument)
                        if parameters[noise_param+'_'+instrument]['type'] == 'FIXED':
                            transit_params.pop(transit_params.index(noise_param+'_'+instrument))
                        elif parameters[noise_param+'_'+instrument]['type'] in prior_distributions:
                            parameters_to_check.append(noise_param+'_'+instrument)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPExpSquaredKernel':
                    for noise_param in ['lnh','lnlambda']:
                        transit_params.append(noise_param+'_'+instrument)
                        if parameters[noise_param+'_'+instrument]['type'] == 'FIXED':
                            transit_params.pop(transit_params.index(noise_param+'_'+instrument))
                        elif parameters[noise_param+'_'+instrument]['type'] in prior_distributions:
                            parameters_to_check.append(noise_param+'_'+instrument)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPGranulation':
                    for noise_param in ['lnomega','lnS']:
                        transit_params.append(noise_param+'_'+instrument)
                        if parameters[noise_param+'_'+instrument]['type'] == 'FIXED':
                            transit_params.pop(transit_params.index(noise_param+'_'+instrument))
                        elif parameters[noise_param+'_'+instrument]['type'] in prior_distributions:
                            parameters_to_check.append(noise_param+'_'+instrument)
                elif options['photometry'][instrument]['PHOT_NOISE_MODEL'] == 'GPAsteroseismology':
                    for noise_param in ['lnomega','lnS','lnQ','lnA','epsilon','lnW','lnnu','lnDeltanu']:
                        transit_params.append(noise_param+'_'+instrument)
                        if parameters[noise_param+'_'+instrument]['type'] == 'FIXED':
                            transit_params.pop(transit_params.index(noise_param+'_'+instrument))
                        elif parameters[noise_param+'_'+instrument]['type'] in prior_distributions:
                            parameters_to_check.append(noise_param+'_'+instrument)
        else:
            for par in ['P','t0','a','p','inc','sigma_w','q1','q2']:
                 if parameters[par]['type'] == 'FIXED':
                     transit_params.pop(transit_params.index(par))
                 elif parameters[par]['type'] in prior_distributions:
                    parameters_to_check.append(par)
            if options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'flicker':
                if parameters['sigma_r']['type'] == 'FIXED':
                    transit_params.pop(transit_params.index('sigma_r'))
                elif parameters['sigma_r']['type'] in prior_distributions:
                    parameters_to_check.append('sigma_r')
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPExpSquaredKernel':
                transit_params.pop(transit_params.index('sigma_r'))
                for noise_param in ['lnh','lnlambda']:
                    transit_params.append(noise_param)
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPGranulation':
                transit_params.pop(transit_params.index('sigma_r'))
                for noise_param in ['lnomega','lnS']:
                    transit_params.append(noise_param)
            elif options['photometry'][options['photometry'].keys()[0]]['PHOT_NOISE_MODEL'] == 'GPAsteroseismology':
                transit_params.pop(transit_params.index('sigma_r'))
                for noise_param in ['lnomega','lnS','lnQ','lnA','epsilon','lnW','lnnu','lnDeltanu']:
                    transit_params.append(noise_param)
            else:
                transit_params.pop(transit_params.index('sigma_r'))
    if options['MODE'] != 'transit':
        if parameters['K']['type'] == 'FIXED':
            rv_params.pop(rv_params.index('K'))
        elif parameters['K']['type'] in prior_distributions:
            parameters_to_check.append('K')
        if len(all_rv_instruments)>1:
            sigma_w_rv = {}
            for instrument in all_rv_instruments:
                sufix[instrument] = {}
                for par in ['mu','sigma_w_rv']:
                    orig_par = par
                    sufix[instrument][orig_par] = ''
                    if par not in parameters.keys():
                        par = par+'_'+instrument
                        sufix[instrument][orig_par] = '_'+instrument
                        if par not in parameters.keys():
                            print 'Error: parameter '+orig_par+' not defined. Exiting...'
                            sys.exit()
                    if par not in rv_params:
                        rv_params.append(par)
                    if parameters[par]['type'] == 'FIXED':
                        rv_params.pop(rv_params.index(par))
                    elif parameters[par]['type'] in prior_distributions:
                        parameters_to_check.append(par)

            #sigma_w_rv = {}
            #for instrument in all_rv_instruments:
            #    if parameters['mu_'+instrument]['type'] == 'FIXED':
            #        rv_params.pop(rv_params.index('mu_'+instrument))
            #    elif parameters['mu_'+instrument]['type'] in prior_distributions:
            #        parameters_to_check.append('mu_'+instrument)
            #    if parameters['sigma_w_rv_'+instrument]['type'] == 'FIXED':
            #        rv_params.pop(rv_params.index('sigma_w_rv_'+instrument))
            #    elif parameters['sigma_w_rv_'+instrument]['type'] in prior_distributions:
            #        parameters_to_check.append('sigma_w_rv_'+instrument)
            #    else:
            #        sigma_w_rv[instrument] = 0.0
            #        rv_params.pop(rv_params.index('sigma_w_rv_'+instrument))
        else:
            if parameters['K']['type'] == 'FIXED':
                rv_params.pop(rv_params.index('K'))
            elif parameters['K']['type'] in prior_distributions:
                parameters_to_check.append('K')
            for rvpar in ['sigma_w_rv','mu']:
                if parameters[rvpar]['type'] in prior_distributions:
                    parameters_to_check.append(rvpar)
                    rv_params.append(rvpar)
                elif parameters[rvpar]['type'] != 'FIXED':
                    rv_params.append(rvpar)

    if options['MODE'] == 'transit':
       all_mcmc_params = transit_params + common_params
    elif options['MODE'] == 'rv':
       all_mcmc_params = rv_params + common_params
    elif options['MODE'] == 'transit_noise':
       all_mcmc_params = ['sigma_w','sigma_r']
    else:
       all_mcmc_params = transit_params + rv_params + common_params

    # First, generate plot with gridspec according to the number of 
    # instruments used for transits:
    if options['MODE'] == 'full':
        nrows = 3
        ncols = len(all_tr_instruments)
        gridspec.GridSpec(3,len(all_tr_instruments))
    elif options['MODE'] == 'transit':
        nrows = 1
        ncols = len(all_tr_instruments)

    gridspec.GridSpec(nrows,len(all_tr_instruments))
    # Plot transits:
    if len(all_tr_instruments) == 1:
            plt.subplot2grid((nrows,ncols),(0,0),colspan=2)
            coeff1,coeff2 = reverse_ld_coeffs(options['photometry'][the_instrument]['LD_LAW'], \
                            parameters['q1']['object'].value,parameters['q2']['object'].value)
            params[the_instrument].t0 = parameters['t0']['object'].value
            params[the_instrument].per = parameters['P']['object'].value
            params[the_instrument].rp = parameters['p']['object'].value
            params[the_instrument].a = parameters['a']['object'].value
            params[the_instrument].inc = parameters['inc']['object'].value
            params[the_instrument].ecc = parameters['ecc']['object'].value
            params[the_instrument].w = parameters['omega']['object'].value
            params[the_instrument].u = [coeff1,coeff2]
            model = m[the_instrument].light_curve(params[the_instrument])
            model_t = np.linspace(np.min(xt),np.max(xt),len(xt)*4)
            model_phase = get_phases(model_t,params[the_instrument].per,params[the_instrument].t0)
            phase = get_phases(xt,params[the_instrument].per,params[the_instrument].t0)
            if options['photometry'][the_instrument]['RESAMPLING']:
               # Generate residuals for plot:
               for i in range(len(idx_resampling[the_instrument])):
                   transit_flat[the_instrument][idx_resampling[the_instrument][i]] = \
                   np.mean(model[i*options['photometry'][the_instrument]['NRESAMPLING']:options['photometry'][the_instrument]['NRESAMPLING']*(i+1)])
               residuals = (yt-transit_flat[the_instrument])
               # Now model (resampled) transit:
               idx_resampling_pred = np.where((model_phase>-options['photometry'][the_instrument]['PHASE_MAX_RESAMPLING'])&\
                                              (model_phase<options['photometry'][the_instrument]['PHASE_MAX_RESAMPLING']))[0]
               t_resampling_pred = np.array([])
               for i in range(len(idx_resampling_pred)):
                   tij = np.zeros(options['photometry'][the_instrument]['NRESAMPLING'])
                   for j in range(1,options['photometry'][the_instrument]['NRESAMPLING']+1):
                       tij[j-1] = model_t[idx_resampling_pred[i]] + ((j - ((options['photometry'][the_instrument]['NRESAMPLING']+1)/2.))*(texp/\
                                  np.double(options['photometry'][the_instrument]['NRESAMPLING'])))
                   t_resampling_pred = np.append(t_resampling_pred, np.copy(tij))
               params2,m2 = init_batman(t_resampling_pred, law=options['photometry'][the_instrument]['LD_LAW'])
               transit_flat_pred = np.ones(len(model_t))
               transit_flat_pred[idx_resampling_pred] = np.zeros(len(idx_resampling_pred))
               model = m2.light_curve(params[the_instrument])
               for i in range(len(idx_resampling_pred)):
                   transit_flat_pred[idx_resampling_pred[i]] = \
                        np.mean(model[i*options['photometry'][the_instrument]['NRESAMPLING']:options['photometry'][the_instrument]['NRESAMPLING']*(i+1)])
               model = transit_flat_pred
            else:
               residuals = (yt-model)
               params2,m2 = init_batman(model_t, law=options['photometry'][the_instrument]['LD_LAW'])
               model = m2.light_curve(params[the_instrument])
            idx_phase = np.argsort(phase)
            idx_model_phase = np.argsort(model_phase)
            plt.plot(phase[idx_phase],yt[idx_phase],'.',color='black',alpha=0.4)
            plt.plot(model_phase[idx_model_phase],model[idx_model_phase],'r-')
            sigma = get_sigma(residuals[idx_phase],0.0)
            plt.plot(phase[idx_phase],residuals[idx_phase]+(1-1.8*(parameters['p']['object'].value**2))-10*sigma,'.',color='black',alpha=0.4)
            plt.title(the_instrument)
            plt.ylabel('Relative flux')
            plt.xlabel('Phase')
            # Save phased model, data and residuals for the transit:
            fout_model = open(out_dir+'tr_model.dat','w')
            for i in idx_model_phase:
                fout_model.write('{0:.10f} {1:.10f}\n'.format(model_phase[i],model[i]))
            fout_model.close()
            fout_data = open(out_dir+'tr_data.dat','w')
            for i in range(len(idx_phase)):
                fout_data.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(xt[i],phase[i],yt[i]))
            fout_data.close()
            fout_res = open(out_dir+'tr_residuals.dat','w')
            for i in range(len(idx_phase)):
                fout_res.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(xt[i],phase[i],residuals[i]))
            fout_res.close()
    else:
            #sufix[instrument][orig_par]
            for k in range(len(all_tr_instruments)):
                plt.subplot2grid((nrows,ncols),(0,k))
                if k == 0:
                    plt.ylabel('Relative flux')
                #plt.xlabel('Phase')
                instrument = all_tr_instruments[k]
                coeff1,coeff2 = reverse_ld_coeffs(options['photometry'][instrument]['LD_LAW'], \
                                parameters['q1'+sufix[instrument]['q1']]['object'].value,\
                                parameters['q2'+sufix[instrument]['q2']]['object'].value)
                params[instrument].t0 = parameters['t0'+sufix[instrument]['t0']]['object'].value
                params[instrument].per = parameters['P']['object'].value
                params[instrument].rp = parameters['p'+sufix[instrument]['p']]['object'].value
                params[instrument].a = parameters['a'+sufix[instrument]['a']]['object'].value
                params[instrument].inc = parameters['inc']['object'].value
                params[instrument].ecc = parameters['ecc']['object'].value
                params[instrument].w = parameters['omega']['object'].value
                params[instrument].u = [coeff1,coeff2]
                model = m[instrument].light_curve(params[instrument])
                model_t = np.linspace(np.min(xt[all_tr_instruments_idxs[k]]),np.max(xt[all_tr_instruments_idxs[k]]),len(all_tr_instruments_idxs[k])*4)
                model_phase = get_phases(model_t,params[instrument].per,params[instrument].t0)
                phase = get_phases(xt[all_tr_instruments_idxs[k]],params[instrument].per,params[instrument].t0)
                if options['photometry'][instrument]['RESAMPLING']:
                   for i in range(len(idx_resampling[instrument])):
                       transit_flat[instrument][idx_resampling[instrument][i]] = \
                       np.mean(model[i*options['photometry'][instrument]['NRESAMPLING']:options['photometry'][instrument]['NRESAMPLING']*(i+1)])
                   residuals = (yt[all_tr_instruments_idxs[k]]-transit_flat[instrument])*1e6
                   idx_resampling_pred = np.where((model_phase>-options['photometry'][instrument]['PHASE_MAX_RESAMPLING'])&\
                                              (model_phase<options['photometry'][instrument]['PHASE_MAX_RESAMPLING']))[0]
                   t_resampling_pred = np.array([])
                   for i in range(len(idx_resampling_pred)):
                       tij = np.zeros(options['photometry'][instrument]['NRESAMPLING'])
                       for j in range(1,options['photometry'][instrument]['NRESAMPLING']+1):
                           tij[j-1] = model_t[idx_resampling_pred[i]] + ((j - ((options['photometry'][instrument]['NRESAMPLING']+1)/2.))*(texp/\
                                  np.double(options['photometry'][instrument]['NRESAMPLING'])))
                       t_resampling_pred = np.append(t_resampling_pred, np.copy(tij))
                   params2,m2 = init_batman(t_resampling_pred, law=options['photometry'][instrument]['LD_LAW'])
                   transit_flat_pred = np.ones(len(model_t))
                   transit_flat_pred[idx_resampling_pred] = np.zeros(len(idx_resampling_pred))
                   model = m2.light_curve(params[instrument])
                   for i in range(len(idx_resampling_pred)):
                       transit_flat_pred[idx_resampling_pred[i]] = np.mean(model[i*options['photometry'][instrument]['NRESAMPLING']:\
                                                                           options['photometry'][instrument]['NRESAMPLING']*(i+1)])
                   model = transit_flat_pred

                else:
                   residuals = (yt[all_tr_instruments_idxs[k]]-model)*1e6    
                   params2,m2 = init_batman(model_t, law=options['photometry'][instrument]['LD_LAW'])
                   model = m2.light_curve(params[instrument])
                idx_phase = np.argsort(phase)
                idx_model_phase = np.argsort(model_phase)
                plt.plot(phase[idx_phase],yt[all_tr_instruments_idxs[k]][idx_phase],'.',color='black',alpha=0.4)
                plt.plot(model_phase[idx_model_phase],model[idx_model_phase],'r-')
                sigma = get_sigma(residuals[idx_phase]*1e-6,0.0)
                plt.plot(phase[idx_phase],residuals[idx_phase]*1e-6+(1-1.8*(parameters['p'+sufix[instrument]['p']]['object'].value**2))-3*sigma,'.',color='black',alpha=0.4)
                plt.title(instrument)
                # Save phased model, data and residuals for the transit:
                fout_model = open(out_dir+'tr_model_'+instrument+'.dat','w')
                for i in idx_model_phase:
                    fout_model.write('{0:.10f} {1:.10f}\n'.format(model_phase[i],model[i]))
                fout_model.close()
                fout_data = open(out_dir+'tr_data_'+instrument+'.dat','w')
                for i in range(len(idx_phase)):
                    fout_data.write('{0:.10f} {1:.10f}\n'.format(phase[i],yt[i]))
                fout_data.close()
                fout_res = open(out_dir+'tr_residuals_'+instrument+'.dat','w')
                for i in range(len(idx_phase)):
                    fout_res.write('{0:.10f} {1:.10f}\n'.format(phase[i],residuals[i]))
                fout_res.close()

    # Plot RVs:
    if options['MODE'] != 'transit':
        radvel_params = init_radvel()
        if options['MODE'] == 'full':
            plt.subplot2grid((nrows,ncols),(1,0),colspan=ncols)
        if len(all_rv_instruments) == 1:
            radvel_params['per1'] = parameters['P']['object'].value
            radvel_params['tc1'] = parameters['t0']['object'].value
            radvel_params['w1'] = parameters['omega']['object'].value*np.pi/180.
            radvel_params['e1'] = parameters['ecc']['object'].value
            radvel_params['k1'] = parameters['K']['object'].value
            model = parameters['mu']['object'].value + radvel.model.RVModel(radvel_params).__call__(xrv)

            residuals = (yrv-model)
            model_t = parameters['t0']['object'].value + np.linspace(-0.5,0.5,500)*parameters['P']['object'].value
            model_pred = parameters['mu']['object'].value + radvel.model.RVModel(radvel_params).__call__(model_t)

            phase = get_phases(xrv,parameters['P']['object'].value,parameters['t0']['object'].value)
            plt.errorbar(phase,(yrv-parameters['mu']['object'].value),yerr=rv_err,fmt='o',label=all_rv_instruments[0])
            model_phase = get_phases(model_t,parameters['P']['object'].value,parameters['t0']['object'].value)
            idx_rv_model = np.argsort(model_phase)
            plt.plot(model_phase[idx_rv_model],model_pred[idx_rv_model]-parameters['mu']['object'].value)
            plt.ylabel('Radial velocity')
            plt.subplot2grid((3,len(all_tr_instruments)),(2,0),colspan=len(all_tr_instruments))
            plt.errorbar(phase,residuals,rv_err,fmt='o')
            plt.ylabel('RV Residuals')
            plt.xlabel('Phase')
            # Save phased model, data and residuals for the RVs:
            fout_model = open(out_dir+'rv_model.dat','w')
            for i in range(len(model_t)):
                fout_model.write('{0:.10f} {1:.10f}\n'.format(model_phase[i],(model_pred-parameters['mu']['object'].value)[i]))
            fout_model.close()
            fout_data = open(out_dir+'rv_data.dat','w')
            for i in range(len(phase)):
                fout_data.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(phase[i],((yrv-parameters['mu']['object'].value))[i],rv_err[i]))
            fout_data.close()
            fout_res = open(out_dir+'rv_residuals.dat','w')
            for i in range(len(phase)):
                fout_res.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(phase[i],residuals[i],rv_err[i]))
            fout_res.close()
        else:
            log_like = 0.0
            all_residuals = []
            all_phases = []
            radvel_params['per1'] = parameters['P']['object'].value
            radvel_params['tc1'] = parameters['t0']['object'].value
            radvel_params['w1'] = parameters['omega']['object'].value*np.pi/180.
            radvel_params['e1'] = parameters['ecc']['object'].value
            radvel_params['k1'] = parameters['K']['object'].value
            model_t = parameters['t0']['object'].value + np.linspace(-0.5,0.5,500)*parameters['P']['object'].value

            model_pred = radvel.model.RVModel(radvel_params).__call__(model_t)
            for i in range(len(all_rv_instruments)):
                model = parameters['mu_'+all_rv_instruments[i]]['object'].value + \
                        radvel.model.RVModel(radvel_params).__call__(xrv[all_rv_instruments_idxs[i]])
                residuals = (yrv[all_rv_instruments_idxs[i]]-model)
                all_residuals.append(residuals)
                phase = get_phases(xrv[all_rv_instruments_idxs[i]],parameters['P']['object'].value,parameters['t0']['object'].value)
                all_phases.append(phase)
                plt.errorbar(phase,(yrv[all_rv_instruments_idxs[i]]-parameters['mu_'+all_rv_instruments[i]]['object'].value),\
                             yerr=rv_err[all_rv_instruments_idxs[i]],label=all_rv_instruments[i],fmt='o')
                # Save data and residuals:
                fout_data = open(out_dir+'rv_data_'+all_rv_instruments[i]+'.dat','w')
                for ii in range(len(phase)):
                    fout_data.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(phase[ii],(yrv[all_rv_instruments_idxs[i]]-\
                                    parameters['mu_'+all_rv_instruments[i]]['object'].value)[ii],rv_err[all_rv_instruments_idxs[i]][ii]))
                fout_data.close()
                fout_res = open(out_dir+'rv_residuals_'+all_rv_instruments[i]+'.dat','w')
                for ii in range(len(phase)):
                    fout_res.write('{0:.10f} {1:.10f} {2:.10f}\n'.format(phase[ii],residuals[ii],rv_err[all_rv_instruments_idxs[i]][ii]))
            fout_res.close()
            model_phase = get_phases(model_t,parameters['P']['object'].value,parameters['t0']['object'].value)
            idx_rvs_sorted = np.argsort(model_phase)
            plt.plot(model_phase[idx_rvs_sorted],model_pred[idx_rvs_sorted],'-',color='red')
            # Save model:
            fout_model = open(out_dir+'rv_model.dat','w')
            for i in idx_rvs_sorted:
                fout_model.write('{0:.10f} {1:.10f}\n'.format(model_phase[i],model_pred[i]))
            fout_model.close()
            plt.legend()
            plt.ylabel('Radial velocity')
            plt.subplot2grid((3,len(all_tr_instruments)),(2,0),colspan=len(all_tr_instruments))
            for i in range(len(all_rv_instruments)):
                plt.errorbar(all_phases[i],all_residuals[i],yerr=rv_err[all_rv_instruments_idxs[i]],fmt='o')
            plt.ylabel('RV Residuals')
            plt.xlabel('Phase')
    if options['PLOT'].lower() != 'no' and options['PLOT'].lower() != 'false' and options['PLOT'].lower() != 'save':
        plt.show()
    elif options['PLOT'].lower() == 'save':
        plt.savefig(out_dir+'fig.png',dpi=300)
    else:
        plt.clf()

    """
    # Extract parameters:
    P = parameters['P']['object'].value
    inc = parameters['inc']['object'].value
    a = parameters['a']['object'].value
    p = parameters['p']['object'].value
    t0 = parameters['t0']['object'].value
    q1 = parameters['q1']['object'].value
    q2 = parameters['q2']['object'].value
    K = parameters['K']['object'].value
    ecc = parameters['ecc']['object'].value
    omega = parameters['omega']['object'].value
    all_rv_instruments,all_rv_instruments_idxs,n_data_rvs = count_instruments(rv_instruments)
    if len(all_rv_instruments)>1:
        mu = {}
        for instrument in all_rv_instruments:
            mu[instrument] = parameters['mu_'+instrument]['object'].value
    else:
        mu = parameters['mu']['object'].value
        print mu

    # Get data phases:
    phases = get_phases(t,P,t0)

    # Generate model times by super-sampling the times:
    model_t = np.linspace(np.min(t),np.max(t),len(t)*4)
    model_phase = get_phases(model_t,P,t0)

    # Initialize the parameters of the transit model, 
    # and prepare resampling data if resampling is True:
    if resampling:
        idx_resampling = np.where((model_phase>-phase_max)&(model_phase<phase_max))[0]
        t_resampling = np.array([])
        for i in range(len(idx_resampling)):
            tij = np.zeros(N_resampling)
            for j in range(1,N_resampling+1):
                # Eq (35) in Kipping (2010)    
                tij[j-1] = model_t[idx_resampling[i]] + ((j - ((N_resampling+1)/2.))*(texp/np.double(N_resampling)))
            t_resampling = np.append(t_resampling, np.copy(tij))    

        idx_resampling_pred = np.where((phases>-phase_max)&(phases<phase_max))[0]
        t_resampling_pred = np.array([])
        for i in range(len(idx_resampling_pred)):
            tij = np.zeros(N_resampling)
            for j in range(1,N_resampling+1):
                tij[j-1] = t[idx_resampling_pred[i]] + ((j - ((N_resampling+1)/2.))*(texp/np.double(N_resampling)))
            t_resampling_pred = np.append(t_resampling_pred, np.copy(tij))
        params,m = init_batman(t_resampling, law=ld_law)
        params2,m2 = init_batman(t_resampling_pred, law=ld_law)
        transit_flat = np.ones(len(model_t))
        transit_flat[idx_resampling] = np.zeros(len(idx_resampling))
        transit_flat_pred = np.ones(len(t))
        transit_flat_pred[idx_resampling_pred] = np.zeros(len(idx_resampling_pred))

    else:
        params,m = init_batman(model_t,law=ld_law)
        params2,m2 = init_batman(t,law=ld_law)

    coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
    params.t0 = t0
    params.per = P
    params.rp = p
    params.a = a
    params.inc = inc
    params.ecc = ecc
    params.omega = omega
    params.u = [coeff1,coeff2]

    # Generate model and predicted lightcurves:
    if resampling:
        model = m.light_curve(params)
        for i in range(len(idx_resampling)):
            transit_flat[idx_resampling[i]] = np.mean(model[i*N_resampling:N_resampling*(i+1)])
        model_lc = transit_flat

        model = m2.light_curve(params)
        for i in range(len(idx_resampling_pred)):
            transit_flat_pred[idx_resampling_pred[i]] = np.mean(model[i*N_resampling:N_resampling*(i+1)])
        model_pred = transit_flat_pred
    else:
        model_lc = m.light_curve(params)
        model_pred = m2.light_curve(params)

    # Now plot:
    plt.style.use('ggplot')
    plt.subplot(211)
    #plt.xlabel('Phase')
    plt.title('exonailer final fit + data')
    plt.ylabel('Relative flux')
    idx = np.argsort(model_phase)
    plt.plot(phases,f,'.',color='black',alpha=0.4)
    plt.plot(model_phase[idx],model_lc[idx])
    plt.plot(phases,f-model_pred+(1-1.8*(p**2)),'.',color='black',alpha=0.4)

    plt.subplot(212)
    plt.ylabel('Radial velocity (m/s)')
    plt.xlabel('Phase')
    model_rv = rv_model.pl_rv_array(model_t,0.0,K,omega*np.pi/180.,ecc,t0,P)
    predicted_rv = rv_model.pl_rv_array(trv,0.0,K,omega*np.pi/180.,ecc,t0,P)
    if len(all_rv_instruments)>1:
        for i in range(len(all_rv_instruments)):
            rv_phases = get_phases(trv[all_rv_instruments_idxs[i]],P,t0)
            plt.errorbar(rv_phases,(rv[all_rv_instruments_idxs[i]]-mu[all_rv_instruments[i]])*1e3,yerr=rv_err[all_rv_instruments_idxs[i]]*1e3,fmt='o',label='RVs from '+all_rv_instruments[i])
        plt.legend()
    else:
        rv_phases = get_phases(trv,P,t0)
        plt.errorbar(rv_phases,(rv-mu)*1e3,yerr=rv_err*1e3,fmt='o')
    plt.plot(model_phase[idx],(model_rv[idx])*1e3)
    plt.show()
    opt = raw_input('\t Save lightcurve and RV data and models? (y/n)')
    if opt == 'y':
        fname = raw_input('\t Output filename (without extension): ')   
        fout = open('results/'+fname+'_lc_data.dat','w')
        fout.write('# Time    Phase     Normalized flux \n')
        for i in range(len(phases)):
            fout.write(str(t[i])+' '+str(phases[i])+' '+str(f[i])+'\n')
        fout.close()
        fout = open('results/'+fname+'_lc_model.dat','w')
        fout.write('# Phase     Normalized flux \n')
        for i in range(len(model_phase[idx])):
            fout.write(str(model_phase[idx][i])+' '+str(model_lc[idx][i])+'\n')
        fout.close()
        fout = open('results/'+fname+'_o-c_lc.dat','w')
        for i in range(len(phases)):
            fout.write(str(t[i])+' '+str(phases[i])+' '+str(f[i]-model_pred[i])+'\n')
        fout.close()
        fout = open('results/'+fname+'_rvs_data.dat','w')
        fout2 = open('results/'+fname+'_o-c_rvs.dat','w')
        fout.write('# Phase     RV (m/s)  Error (m/s)  Instrument\n')
        if len(all_rv_instruments)>1:
            for i in range(len(all_rv_instruments)):
                rv_phases = get_phases(trv[all_rv_instruments_idxs[i]],P,t0)
                for j in range(len(rv_phases)):
                    fout.write(str(rv_phases[j])+' '+str(((rv[all_rv_instruments_idxs[i]]-mu[all_rv_instruments[i]])*1e3)[j])+\
                                    ' '+str(((rv_err[all_rv_instruments_idxs[i]])*1e3)[j])+\
                                    ' '+all_rv_instruments[i]+'\n')
        else:
            for i in range(len(rv_phases)):
                fout.write(str(rv_phases[i])+' '+str((rv[i]-mu)*1e3)+' '+str(rv_err[i]*1e3)+' \n')
                fout2.write(str(rv_phases[i])+' '+str((rv[i]-mu-predicted_rv[i])*1e3)+' '+str(rv_err[i]*1e3)+' \n')
        fout.close()
        fout2.close()
        fout = open('results/'+fname+'_rvs_model.dat','w')
        fout.write('# Phase     RV (m/s) \n')
        for i in range(len(model_phase[idx])):
            fout.write(str(model_phase[idx][i])+' '+str(((model_rv[idx])*1e3)[i])+'\n')
        fout.close()
    """
