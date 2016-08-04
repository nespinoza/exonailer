# -*- coding: utf-8 -*-
import sys
sys.path.append('utilities')
import os
import argparse
import data_utils
import general_utils
import numpy as np

################# OPTIONS ######################
# Get user input:
parser = argparse.ArgumentParser()
parser.add_argument('-target',default=None)
parser.add_argument('-phot_noise_model',default=None)
parser.add_argument('--phot_detrend', dest='phot_detrend', action='store_true')
parser.set_defaults(phot_detrend=False)
parser.add_argument('-window',default = 41)
parser.add_argument('--phot_get_outliers', dest='phot_get_outliers', action='store_true')
parser.set_defaults(phot_get_outliers=False)
# Define if you want to perform the resampling technique and in 
# which phase range you want to perform such resampling. Additionally, 
# define how many samples you want to resample:
parser.add_argument('--resampling', dest='resampling', action='store_true')
parser.set_defaults(resampling=False)
parser.add_argument('-phase_max',default = 0.1)
parser.add_argument('-N_resampling',default = 10)
# Limb-darkening law to be used:
parser.add_argument('-ld_law',default=None)
# Define the mode to be used:
parser.add_argument('-mode',default=None)
# Define noise properties:
parser.add_argument('--rv_jitter', dest='rv_jitter', action='store_true')
parser.set_defaults(rv_jitter=False)
# Define emcee parameters:
parser.add_argument('-nwalkers',default = 500)
parser.add_argument('-njumps',default = 500)
parser.add_argument('-nburnin',default = 500)

# Define time conversions:
parser.add_argument('-transit_time_def',default='tdb->utc')
parser.add_argument('-rv_time_def',default='utc->utc')

args = parser.parse_args()
for argument in ['target','phot_noise_model','phot_detrend','window','phot_get_outliers','resampling',\
                 'phase_max','N_resampling','ld_law','mode','rv_jitter','nwalkers','njumps','nburnin',\
                 'transit_time_def','rv_time_def']:
    if argument in ['window','N_resampling','nwalkers','njumps','nburnin']:
        exec argument+' = np.int(args.'+argument+')'
    elif argument in ['phase_max']:
        exec argument+' = np.double(args.'+argument+')'
    else:
        exec argument+' = args.'+argument

################################################
n_ommit = []
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
    data_utils.plot_transit(t_tr,f,parameters,ld_law,transit_instruments, resampling = resampling, \
                                      phase_max = phase_max, N_resampling=N_resampling)
elif mode == 'full':
    data_utils.plot_transit_and_rv(t_tr,f,t_rv,rv,rv_err,parameters,ld_law,rv_jitter, \
                                      transit_instruments, rv_instruments,\
                                      resampling = resampling, phase_max = phase_max, \
                                      N_resampling=N_resampling)
