import numpy as np

def read_priors(target):
    # Open the file containing the priors:
    f = open('priors_data/'+target+'_priors.dat','r')
    # Generate dictionary that will save the data on the priors:
    priors = {}
    while True:
        line = f.readline()
        if line == '':
            break
        elif line[0] != '#':
            param_name,prior_type,hyper_params = line.split()
            priors[param_name] = {}
            priors[param_name]['type'] = prior_type
            priors[param_name]['hyperparams'] = \
                            np.array(hyper_params.split(',')).astype('float64')
    f.close()
    return priors

def read_data(target,mode):
    t,f,f_err = None,None,None
    t_rv,rv,rv_err = None,None,None
    if mode != 'rvs':
        try:
            t,f,f_err = np.loadtxt('transit_data/'+target+'_lc.dat',unpack=True,usecols=(0,1,2))
        except:
            t,f = np.loadtxt('transit_data/'+target+'_lc.dat',unpack=True,usecols=(0,1))
    if mode != 'transit':
        try:
            t_rv,rv,rv_err = np.loadtxt('rv_data/'+target+'_rvs.dat',unpack=True,usecols=(0,1,2))
        except:
            t_rv,rv = np.loadtxt('rv_data/'+target+'_rvs.dat',unpack=True,usecols=(0,1))
        
    return t,f,f_err,t_rv,rv,rv_err

def get_quantiles(dist,alpha = 0.68, method = 'median'):
    """
    get_quantiles function

    DESCRIPTION

        This function returns, in the default case, the parameter median and the error% 
        credibility around it. This assumes you give a non-ordered 
        distribution of parameters.

    OUTPUTS

        Median of the parameter,upper credibility bound, lower credibility bound

    """
    ordered_dist = dist[np.argsort(dist)]
    param = 0.0
    # Define the number of samples from posterior
    nsamples = len(dist)
    nsamples_at_each_side = int(nsamples*(alpha/2.)+1)
    if(method == 'median'):
       med_idx = 0
       if(nsamples%2 == 0.0): # Number of points is even
          med_idx_up = int(nsamples/2.)
          med_idx_down = med_idx_up-1
          param = (ordered_dist[med_idx_up]+ordered_dist[med_idx_down])/2.
          return param,ordered_dist[med_idx_up+nsamples_at_each_side],\
                 ordered_dist[med_idx_down-nsamples_at_each_side]
       else:
          med_idx = int(nsamples/2.)
          param = ordered_dist[med_idx]
          return param,ordered_dist[med_idx+nsamples_at_each_side],\
                 ordered_dist[med_idx-nsamples_at_each_side]
