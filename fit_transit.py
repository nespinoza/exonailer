# -*- coding: utf-8 -*-
import sys
sys.path.append('utilities')
import transit_utils
import numpy as np

from math import floor,ceil
################# OPTIONS ######################

# Define the target name, detrending method and parameters of this
# method:
target = 'CL005-04'
detrend = 'mfilter'
window = 21

# If you want any transits to be ommited in the fit (e.g., spots),
# put the number of the transit (counted from first transit) in 
# this list:
n_ommit = [3,9]

# Initial transit parameters for the MCMC:
P = 4.09844735818
t0 = 2457067.90563
a = 9.867860165894255
p = 0.129865221295
inc = 85.3172129026
sigma_w = 500. # ppm

# Initial ld law and (converted) coefficients:
ld_law = 'quadratic'
q1 = 0.5
q2 = 0.5

################################################
############ DATA PRE-PROCESSING ###############
# First, get the data:
t,f = np.loadtxt('transit_data/'+target+'_lc.dat',unpack=True)

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

######## FIT INITIAL WHITE NOISE MODEL #########
# Now fit initial white-noise model. For this, fill the priors:
theta_0 = P,inc,a,p,t0,q1,q2,sigma_w

# And the std-devs of the transit parameters + noise params
# (note no std-devs on ld coeffs). Also, sigma on noise params
# is the upper limit on the noise, which has a Jeffrey's prior:
sigma_theta_0 = 0.1,5.,10.,0.1,0.1,2000.0

# Run the MCMC:
theta_out_wn = transit_utils.transit_mcmc_fit(t.astype('float64'), \
                                             f.astype('float64'), None, theta_0, sigma_theta_0, \
                                             ld_law,njumps=100, nburnin = 100, \
                                             nwalkers = 100,  noise_model = 'white')

P_c,inc_c,a_c,p_c,t0_c,q1_c,q2_c,sigma_w_c = transit_utils.transit_mcmc_fit(t.astype('float64'), \
                                             f.astype('float64'), None, theta_0, sigma_theta_0, \
                                             ld_law,njumps=100, nburnin = 100, \
                                             nwalkers = 100,  noise_model = 'white')

# Get plot of the white-noise fit:
transit_utils.plot_transit(t,f,theta_out_wn,ld_law)
