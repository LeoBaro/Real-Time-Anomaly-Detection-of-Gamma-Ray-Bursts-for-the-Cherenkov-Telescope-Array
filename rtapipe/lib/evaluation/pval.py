import numpy as np
from scipy.stats import norm


def get_sigma_from_pvalue(pval, decimals=3):
    return np.abs(np.round(norm.ppf(pval), decimals))

def get_prob_from_sigma(sigma, decimals=10):
    return np.round(1-(norm.sf(sigma)*2), decimals)

def get_prob_from_pvalue(pval, decimals=10):
    return np.round(1-pval*2, decimals)

def get_pval_from_prob(p, decimals=8):
    return np.round((1-p)/2, decimals)
    
def get_pvalue_from_sigma(sigma, decimals=10):
    p = get_prob_from_sigma(sigma, decimals=decimals)
    return np.round((1-p)/2, decimals)