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
# First, get the transit data:
try:
    t,f,f_err = np.loadtxt('transit_data/'+target+'_lc.dat',unpack=True,usecols=(0,1,2))
except:
    t,f = np.loadtxt('transit_data/'+target+'_lc.dat',unpack=True,usecols=(0,1))
    f_err = None
# Now the RV data:
try:
    t_rv,rv,rv_err = np.loadtxt('rv_data/'+target+'_rvs.dat',unpack=True,usecols=(0,1,2))
except:
    t_rv,rv = np.loadtxt('rv_data/'+target+'_rvs.dat',unpack=True,usecols=(0,1)) 
    rv_err = None

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

######## FIT INITIAL WHITE NOISE MODEL #########
# Now fit initial white-noise model. For this, fill the priors:
theta_0 = P,inc,a,p,t0,q1,q2,sigma_w

# And the std-devs of the transit parameters + noise params
# (note no std-devs on ld coeffs). Also, sigma on noise params
# is the upper limit on the noise, which has a Jeffrey's prior:
sigma_theta_0 = 0.1,5.,10.,0.1,0.1,2000.0

# Run the MCMC:
theta_out_wn = transit_utils.transit_mcmc_fit(t, f, f_err, theta_0, sigma_theta_0, \
                                             ld_law, njumps=100, nburnin = 100, \
                                             nwalkers = 100,  noise_model = 'white')

# Get plot of the white-noise fit:
transit_utils.plot_transit(t,f,theta_out_wn,ld_law)
