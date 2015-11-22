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
