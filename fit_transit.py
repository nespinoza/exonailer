# -*- coding: utf-8 -*-
import transit_utils
import numpy as np

################# OPTIONS ######################

target = 'CL005-004'
detrend = 'mfilter'
window = 21

P = 4.09844735818
t0 = 2457067.90563
a = 1./(0.101339093095)
p = 0.129865221295
################################################
# First, get the data:
t,f = np.loadtxt('transit_data/'+target+'_lc.dat',unpack=True)

# First phase in transit fitting is to 'detrend' the 
# data. This is done with the 'detrend' flag. If 
# the data is already detrended, set the flag to None:
if detrend is not None:
    if detrend == 'mfilter':
        # Get median filter, and smooth it with a gaussian filter:
        from scipy.signal import medfilt
        from scipy.ndimage.filters import gaussian_filter
        filt = gaussian_filter(medfilt(f,window),5)
        f = f/filt

# Phase the data:
phase = ((t - t0)/P) % 1
ii = np.where(phase>=0.5)[0]
phase[ii] = phase[ii]-1.0
# Once data is normalized,
import matplotlib.pyplot as plt
plt.plot(phase,f,'.')
plt.show()
