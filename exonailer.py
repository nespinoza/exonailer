# -*- coding: utf-8 -*-
import sys
sys.path.append('utilities')
import transit_utils
import numpy as np

from math import floor,ceil
################# OPTIONS ######################

# Define the target name, detrending method and parameters of this
# method:
target = 'CL001-04'
detrend = 'mfilter'
window = 41
get_outliers = True

# If you want any transits to be ommited in the fit (e.g., spots),
# put the number of the transit (counted from first transit) in 
# this list:
n_ommit = []#[3,9]

# Initial transit parameters for the MCMC:
P = 41.6927973464#4.09844735818
t0 = 2457151.91171#2457067.90563
a = 31.42363601700875#9.867860165894255
p = 0.026944416288#0.129865221295
inc = 88.4325339589#85.3172129026
sigma_w = 500#500. # ppm

# Initial ld law and (converted) coefficients:
ld_law = 'quadratic'
q1 = 0.659286671241
q2 = 0.532312795382

# Define the mode:
#    'full'     :   Full transit + rvs fit.
#    'transit'  :   Only fit transit lightcurve.
#    'rvs'      :   Only fit RVs.
mode = 'full' 

# Define noise properties:
rv_jitter = False

################################################
############ DATA PRE-PROCESSING ###############
# First, get the transit data:
if mode != 'rvs':
    try:
        t,f,f_err = np.loadtxt('transit_data/'+target+'_lc.dat',unpack=True,usecols=(0,1,2))
    except:
        t,f = np.loadtxt('transit_data/'+target+'_lc.dat',unpack=True,usecols=(0,1))
        f_err = None

if mode != 'transit':
# Now the RV data:
    try:
        t_rv,rv,rv_err = np.loadtxt('rv_data/'+target+'_rvs.dat',unpack=True,usecols=(0,1,2))
    except:
        t_rv,rv = np.loadtxt('rv_data/'+target+'_rvs.dat',unpack=True,usecols=(0,1)) 
        rv_err = None

if mode != 'rvs':
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

    # If outlier removal is on, remove them:
    if get_outliers:
        params,m = transit_utils.init_batman(t,law=ld_law)
        coeff1,coeff2 = transit_utils.reverse_ld_coeffs(ld_law, q1, q2)
        params.t0 = t0
        params.per = P
        params.rp = p
        params.a = a
        params.inc = inc
        params.u = [coeff1,coeff2]
        model = m.light_curve(params)
        # Get approximate transit duration in phase space:
        phases = transit_utils.get_phases(t,P,t0)
        idx = np.where(model == 1.0)[0]
        phase_dur = np.abs(phases[idx][np.where(np.abs(phases[idx]) == np.min(np.abs(phases[idx])))])[0] + 0.01
        # Get precision:
        median_flux = np.median(f)
        sigma = transit_utils.get_sigma(f,median_flux)
        # Perform sigma-clipping for out-of-transit data using phased data:
        good_times = np.array([])
        good_fluxes = np.array([])
        good_phases = np.array([])
        for i in range(len(t)):
                if np.abs(phases[i])<phase_dur:
                        good_times = np.append(good_times,t[i])
                        good_fluxes = np.append(good_fluxes,f[i])
                        good_phases = np.append(good_phases,phases[i])
                else:
                        if (f[i]<median_flux + 3*sigma) and (f[i]>median_flux - 3*sigma):
                                good_times = np.append(good_times,t[i])
                                good_fluxes = np.append(good_fluxes,f[i])
                                good_phases = np.append(good_phases,phases[i])
        t = good_times
        f = good_fluxes
        phases = good_phases
        import matplotlib.pyplot as plt
        plt.plot(phases,f,'.')
        plt.show()

if mode == 'full':
    # Generate the priors on the mean RV and RV semi-amplitude:    
    mu = np.median(rv)
    K = np.sqrt(np.var(rv))
    sigma_w_rv = np.sqrt(np.var(rv))

    # Define the priors on the transit + rvs:
    if rv_jitter:
        theta_0 = P,inc,a,p,t0,q1,q2,sigma_w,mu,K,sigma_w_rv
    else:
        theta_0 = P,inc,a,p,t0,q1,q2,sigma_w,mu,K

    # And the std-devs of the transit parameters + noise params
    # (note no std-devs on ld coeffs) + same for rvs. Also, sigma 
    # on noise params is the upper limit on the noise, which has a Jeffrey's prior:
    if rv_jitter:
        sigma_theta_0 = 0.1,5.,10.,0.1,0.1,2000.0,1.0,1.0,10.0
    else:
        sigma_theta_0 = 0.1,5.,10.,0.1,0.1,2000.0,1.0,1.0

elif mode == 'transit':
    # Fit white-noise model. For this, fill the priors:
    theta_0 = P,inc,a,p,t0,q1,q2,sigma_w

    # And the std-devs of the transit parameters + noise params
    # (note no std-devs on ld coeffs). Also, sigma on noise params
    # is the upper limit on the noise, which has a Jeffrey's prior:
    sigma_theta_0 = 0.1,5.,10.,0.1,0.1,2000.0

    t_rv = None
    rv = None
    rv_err = None
# Run the MCMC:
theta_out = transit_utils.exonailer_mcmc_fit(t, f, f_err, t_rv, rv, rv_err, \
                                             theta_0, sigma_theta_0, \
                                             ld_law, mode, rv_jitter = rv_jitter, \
                                             njumps=100000, nburnin = 100000, \
                                             nwalkers = 100,  noise_model = 'white')
for chain_var in theta_out:
    print np.median(chain_var),'+-',np.sqrt(np.var(chain_var))

# Get plot of the transit-fit:
if mode == 'transit':
    transit_utils.plot_transit(t,f,theta_out,ld_law)
elif mode == 'full':
    transit_utils.plot_transit_and_rv(t,f,t_rv,rv,rv_err,theta_out,ld_law,rv_jitter)
