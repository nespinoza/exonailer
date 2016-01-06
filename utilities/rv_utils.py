import ajplanet as rv_model
import numpy as np

def get_model_rv(t,K,mu,omega,ecc,t0,P):
    rv_model.pl_rv_array(t,mu,K,omega,ecc,t0,P)
