import numpy as np

def read_priors(target,filename = None):
    # Open the file containing the priors:
    if filename is None:
        f = open('priors_data/'+target+'_priors.dat','r')
    else:
        f = open(filename)
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
            if prior_type == 'Normal':
               priors[param_name]['object'] = normal_parameter(np.array(hyper_params.split(',')).astype('float64'))
            elif prior_type == 'Uniform':
               priors[param_name]['object'] = uniform_parameter(np.array(hyper_params.split(',')).astype('float64'))
            elif prior_type == 'Jeffreys':
               priors[param_name]['object'] = jeffreys_parameter(np.array(hyper_params.split(',')).astype('float64'))
            elif prior_type == 'FIXED':
               priors[param_name]['object'] = constant_parameter(np.array(hyper_params.split(',')).astype('float64')[0])
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

import pickle,os
def save_results(target,mode,phot_noise_model,ld_law,parameters):
    out_dir = 'results/'+target+'_'+mode+'_'+phot_noise_model+'_'+ld_law+'/'
    os.mkdir(out_dir)
    # Copy used prior file to the results folder:
    os.system('cp priors_data/'+target+'_priors.dat '+out_dir+'priors.dat')
    out_posterior_file = open(out_dir+'posterior_parameters.dat','w')
    out_posterior_file.write('# This file has the final parameters obtained from the MCMC chains.\n')

    # Generate an output dictionary with the posteriors:
    out_dict = {}
    for parameter in parameters.keys():
        if parameters[parameter]['type'] != 'FIXED':
            # Save parameter values in posterior file:
            param = parameters[parameter]['object'].value
            up_error = parameters[parameter]['object'].value_u-param
            low_error = param-parameters[parameter]['object'].value_l
            out_dict[parameter] = parameters[parameter]['object'].posterior
        else:
            param = parameters[parameter]['object'].value
            up_error = 0
            low_error = 0

        out_posterior_file.write('{0:10}  {1:10.10f}  {2:10.10f}  {3:10.10f}\n'.format(\
                                   parameter, param, up_error, low_error))
    # Save posterior dict:
    f = open(out_dir+'posteriors.pkl','w')
    pickle.dump(out_dict,f)
    f.close()

def read_results(target,mode,phot_noise_model,ld_law):
    out_dir = 'results/'+target+'_'+mode+'_'+phot_noise_model+'_'+ld_law+'/'
    parameters = read_priors(target,filename = out_dir+'priors.dat')
    thefile = open(out_dir+'posteriors.pkl','r')
    posteriors = pickle.load(thefile)
    for parameter in parameters.keys():
        if parameters[parameter]['type'] != 'FIXED':
            parameters[parameter]['object'].set_posterior(posteriors[parameter])  
    thefile.close()
    return parameters

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

class normal_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a normal prior. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """   
      def __init__(self,prior_hypp):
          self.value = prior_hypp[0]
          self_value_u = 0.0
          self_value_l = 0.0
          self.prior_hypp = prior_hypp
          self.posterior = []

      def get_ln_prior(self):
          return np.log(1./np.sqrt(2.*np.pi*(self.prior_hypp[1]**2)))-\
                 0.5*(((self.prior_hypp[0]-self.value)**2/(self.prior_hypp[1]**2)))

      def set_value(self,new_val):
          self.value = new_val

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l
class uniform_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a uniform prior. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """
      def __init__(self,prior_hypp):
          self.value = (prior_hypp[0]+prior_hypp[1])/2.
          self_value_u = 0.0
          self_value_l = 0.0
          self.prior_hypp = prior_hypp
          self.posterior = []

      def get_ln_prior(self):
          return np.log(1./(self.prior_hypp[1]-self.prior_hypp[0]))

      def check_value(self,x):
          if x > self.prior_hypp[0] and  x < self.prior_hypp[1]:
              return True
          else:
              return False  
 
      def set_value(self,new_val):
          self.value = new_val

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l

log1 = np.log(1)
class jeffreys_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a Jeffreys prior. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """
      def __init__(self,prior_hypp):
          self.value = np.sqrt(prior_hypp[0]*prior_hypp[1])
          self_value_u = 0.0
          self_value_l = 0.0
          self.prior_hypp = prior_hypp
          self.posterior = []

      def get_ln_prior(self):
          return log1 - np.log(self.value*np.log(self.prior_hypp[1]/self.prior_hypp[0]))

      def check_value(self,x):
          if x > self.prior_hypp[0] and  x < self.prior_hypp[1]:
              return True
          else:
              return False

      def set_value(self,new_val):
          self.value = new_val

      def set_posterior(self,posterior_chain):
          self.posterior = posterior_chain
          param, param_u, param_l = get_quantiles(posterior_chain)
          self.value = param
          self.value_u = param_u
          self.value_l = param_l

class constant_parameter:
      """
      Description
      -----------

      This class defines a parameter object which has a constant value. It serves 
      to save both the prior and the posterior chains for an easier check of the parameter.

      """
      def __init__(self,val):
          self.value = val

