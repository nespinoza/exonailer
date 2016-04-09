# -*- coding: utf-8 -*-
import sys
sys.path.append('utilities')
import os
import data_utils
import general_utils
import numpy as np

################# OPTIONS ######################

# Define the target name, detrending method and parameters of it:
target = 'my_lightcurve'
phot_noise_model = 'white'
phot_detrend = None
window = 41

# Define if you want to perform automatic outlier removal (sigma-clipping):
phot_get_outliers = None

# Define which transits you want to ommit (counted from first transit):
n_ommit = []

# Define if you want to perform the resampling technique and in 
# which phase range you want to perform such resampling. Additionally, 
# define how many samples you want to resample:
resampling = False
phase_max = 0.01
N_resampling = 10

# Limb-darkening law to be used:
ld_law = 'quadratic'

# Define the mode to be used:
mode = 'transit' 

# Define noise properties:
rv_jitter = False

# Define emcee parameters:
nwalkers = 500
njumps = 500
nburnin = 500

# Define time conversions:
transit_time_def = 'utc->utc'
rv_time_def = 'utc->utc'

################################################

# ---------- DATA PRE-PROCESSING ------------- #

# First, get the transit and RV data:
t_tr,f,f_err,transit_instruments,t_rv,rv,rv_err,rv_instruments = general_utils.read_data(target,mode,transit_time_def,rv_time_def)

# Initialize the parameters:
parameters = general_utils.read_priors(target,transit_instruments,rv_instruments,mode)

# Pre-process the transit data if available:
if mode != 'rvs':
    t_tr,phases,f, f_err = data_utils.pre_process(t_tr,f,f_err,phot_detrend,\
                                                  phot_get_outliers,n_ommit,\
                                                  window,parameters,ld_law, mode)
    if resampling:
        # Define indexes between which data will be resampled:
        idx_resampling = np.where((phases>-phase_max)&(phases<phase_max))[0]
    else:
        idx_resampling = []

# Create results folder if not already created:
if not os.path.exists('results'):
    os.mkdir('results')

# If chains not ran, run the MCMC and save results:
if not os.path.exists('results/'+target+'_'+mode+'_'+phot_noise_model+'_'+ld_law):
    data_utils.exonailer_mcmc_fit(t_tr, f, f_err, transit_instruments, t_rv, rv, rv_err, rv_instruments,\
                                     parameters, ld_law, mode, rv_jitter = rv_jitter, \
                                     njumps = njumps, nburnin = nburnin, \
                                     nwalkers = nwalkers,  noise_model = phot_noise_model,\
                                     resampling = resampling, idx_resampling = idx_resampling,\
                                     N_resampling = N_resampling)

    general_utils.save_results(target,mode,phot_noise_model,ld_law,parameters)

else:
    parameters = general_utils.read_results(target,mode,phot_noise_model,ld_law,transit_instruments, rv_instruments)

# Get plot of the transit-fit:
if mode == 'transit':
    data_utils.plot_transit(t_tr,f,parameters,ld_law,transit_instruments)
elif mode == 'full':
    data_utils.plot_transit_and_rv(t_tr,f,t_rv,rv,rv_err,parameters,ld_law,rv_jitter, \
                                      transit_instruments, rv_instruments,\
                                      resampling = resampling, phase_max = phase_max, \
                                      N_resampling=N_resampling)
