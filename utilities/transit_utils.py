# -*- coding: utf-8 -*-
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
        coeff2 = 1.-np.sqrt(q1)#(1.-coeff1)/(1.-coeff2.)
    return coeff1,coeff2

import Wavelets
import scipy.optimize as op
def exonailer_mcmc_fit(times, relative_flux, error, times_rv, rv, rv_err, \
                       theta_0, sigma_theta_0, ld_law, mode, \
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

      theta_input_0:    Array with the priors on the parameters to be fitted. They are 
                        assumed to be in the following order in 'full' mode :
 
                      theta_input_0 = [P,inc,a,p,t0,q1,q2,sigma_w,sigma_r,mu,K,sigma_w_rv]

                        The parameter sigma_r only has to be included if noise_model = '1/f' 
                        (see below). Here:

                          P:            Period (in the units of the times).

                          inc:          Inclination of the orbit (in degrees).

                          a:            Semi-major axis in stellar units.

                          p:            Planet-to-star radius ratio.

                          t0:           Time of transit center (same units as the times)

                          q1:           Prior on the first converted coefficient of whatever law you 
                                        are going to use for the limb-darkening (see Kipping 2013).

                          q2:           Same thing for the second coefficient.

                          sigma_w:      Standard deviation of the underlying white-noise process.
 
                          sigma_r:      Parameter of the 1/f noise process (used only if noise_model
                                        set to 1/f; see below).

                          mu:           Mean RVs.

                          K:            RV semi-amplitude.

                          sigma_w_Rv:   RV jitter 

      sigma_theta_0:    Array with the standard-deviations of the priors stated above,
                        in the same order as the parameters.

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
    # Initialize the parameters:
    params,m = init_batman(times,law=ld_law)

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

    def lnlike(theta, t, y, yerr, gamma=1.0):
        if noise_model == '1/f':
            P,inc,a,p,t0,q1,q2,sigma_w,sigma_r = theta
        else:
            P,inc,a,p,t0,q1,q2,sigma_w = theta

        coeff1,coeff2 = reverse_ld_coeffs(ld_law, q1, q2)
        params.t0 = t0
        params.per = P
        params.rp = p
        params.a = a
        params.inc = inc
        params.ecc = 0.0
        params.w = 90.0
        params.u = [coeff1,coeff2]
        model = m.light_curve(params)
        if noise_model == '1/f':
           residuals = (y-model)*1e6
           log_like = get_fn_likelihood(residuals,sigma_w,sigma_r)
        else:
           inv_sigma2 = 1.0/((yerr)**2 + (sigma_w)**2)
           log_like = -0.5*(np.sum(((y-model)*1e6)**2*inv_sigma2 - np.log(inv_sigma2)))
        return log_like

    def lnprior(theta):
        if noise_model == '1/f':
            P,inc,a,p,t0,q1,q2,sigma_w,sigma_r = theta
            if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1 or sigma_w < 1.0 or sigma_r < 1.0 \
               or p < 0 or p>1 or P < 0 or inc<0 or inc>90.0\
               or a<1:
               return -np.inf

            P_0,inc_0,a_0,p_0,t0_0,q1_0,q2_0,sigma_w_0,sigma_r_0 = theta_0
            sigma_P, sigma_inc, sigma_a, sigma_p, sigma_t0, up_lim_sigma_w, up_lim_sigma_r = sigma_theta_0

            if sigma_w > up_lim_sigma_w or sigma_r > up_lim_sigma_r:
               return -np.inf

            lnp_sigma_r = -np.log((sigma_r)*np.log((up_lim_sigma_r)))
        else:
            P,inc,a,p,t0,q1,q2,sigma_w  = theta
            if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1 or sigma_w < 1.0 \
               or p < 0 or p>1 or P < 0 or inc<0 or inc>90.0\
               or a<1:
               return -np.inf

            P_0,inc_0,a_0,p_0,t0_0,q1_0,q2_0,sigma_w_0 = theta_0
            sigma_P, sigma_inc, sigma_a, sigma_p, sigma_t0, up_lim_sigma_w = sigma_theta_0

            if sigma_w > up_lim_sigma_w:
               return -np.inf
            lnp_sigma_r = 0.0

        # Jeffrey's prior on sigma_w and sigma_r; uniforms on q1,q2:
        lnp_sigma = -np.log((sigma_w)*np.log(up_lim_sigma_w))
        lnp_inc = -0.5*(((inc-inc_0)**2/(sigma_inc**2)))
        lnp_a = -0.5*(((a-a_0)**2/(sigma_a**2)))
        lnp_p = -0.5*(((p-p_0)**2/(sigma_p**2)))
        lnp_t0 = -0.5*(((t0-t0_0)**2/(sigma_t0**2)))
        lnp_P = -0.5*(((P-P_0)**2/(sigma_P**2)))

        return lnp_sigma + lnp_sigma_r + lnp_inc + lnp_a + lnp_p + lnp_t0 + lnp_P

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr)

    # Define the variables for the MCMC:
    x = times.astype('float64')
    y = relative_flux.astype('float64')
    if error is None:
       yerr = 0.0
    else:
       yerr = error.astype('float64')

    # Start at the maximum likelihood value:
    nll = lambda *args: -lnprob(*args)
    result = op.minimize(nll, theta_0, args=(x, y, yerr))
    theta_ml = result["x"]

    # Now define parameters for emcee:
    ndim = len(theta_ml)
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

    # Run the MCMC:
    import emcee
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))

    sampler.run_mcmc(pos, njumps+nburnin)

    # Save the parameter chains:
    P = np.array([])
    inc = np.array([])
    a = np.array([])
    p = np.array([])
    t0 = np.array([])
    q1 = np.array([])
    q2 = np.array([])
    sigma_w = np.array([])
    if noise_model == '1/f':
       sigma_r = np.array([])
    for walker in range(nwalkers):
        P = np.append(P,sampler.chain[walker,nburnin:,0])
        inc = np.append(inc,sampler.chain[walker,nburnin:,1])
        a = np.append(a,sampler.chain[walker,nburnin:,2])
        p = np.append(p,sampler.chain[walker,nburnin:,3])
        t0 = np.append(t0,sampler.chain[walker,nburnin:,4])
        q1 = np.append(q1,sampler.chain[walker,nburnin:,5])
        q2 = np.append(q2,sampler.chain[walker,nburnin:,6])
        sigma_w = np.append(sigma_w,sampler.chain[walker,nburnin:,7])
        if noise_model == '1/f':
            sigma_r = np.append(sigma_r,sampler.chain[walker,nburnin:,8])

    # Return the chains:
    if noise_model == '1/f':
        return P,inc,a,p,t0,q1,q2,sigma_w,sigma_r
    else:
        return P,inc,a,p,t0,q1,q2,sigma_w

import matplotlib.pyplot as plt
def plot_transit(t,f,theta,ld_law):
        
    # Extract transit parameters:
    P_c,inc_c,a_c,p_c,t0_c,q1_c,q2_c,sigma_w_c = theta
    P = np.median(P_c)
    inc = np.median(inc_c)
    a = np.median(a_c)
    p = np.median(p_c)
    t0 = np.median(t0_c)
    q1 = np.median(q1_c)
    q2 = np.median(q2_c)

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
    plt.plot(phases,f-model_pred+(1-1.2*(p**2)),'.',color='black',alpha=0.4)
    plt.show()
