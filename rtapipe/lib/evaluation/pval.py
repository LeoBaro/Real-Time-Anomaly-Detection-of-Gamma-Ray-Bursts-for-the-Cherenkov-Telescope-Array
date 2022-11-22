import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import interpolate


def get_sigma_from_pvalue(pval, decimals=3):
    sigma = np.abs(np.round(norm.ppf(pval), decimals))
    return sigma  

def get_prob_from_sigma(sigma, decimals=10):
    return np.round(1-(norm.sf(sigma)*2), decimals)

def get_prob_from_pvalue(pval, decimals=10):
    return np.round(1-pval*2, decimals)

def get_pval_from_prob(p, decimals=8):
    return np.round((1-p)/2, decimals)
    
def get_pvalue_from_sigma(sigma, decimals=10):
    p = get_prob_from_sigma(sigma, decimals=decimals)
    return np.round((1-p)/2, decimals)

def get_pval_table(pvalues_path, gt_sigma=None, show=True):

    pvalues_table = pd.read_csv(pvalues_path, sep=" ", header=0)
    pvalues_table = pvalues_table.rename(columns={'x': 'threshold', 'xerr': 'threshold_err', 'y': 'pvalue', 'yerr':'pvalue_err'})
    
    # TEMPORARY FIX
    pvalues_table = pvalues_table[ pvalues_table["pvalue"] < 0.15 ] 
    
    # Compute sigma column
    pvalues_table["sigma"] = pvalues_table["pvalue"].apply(get_sigma_from_pvalue)
    
    # Delete +- inf values
    pvalues_table.replace([np.inf, -np.inf], np.nan, inplace=True)    
    pvalues_table.dropna(subset=["sigma"], how="all", inplace=True)
    
    # Filter on sigma values
    if gt_sigma is not None:
        pvalues_table = pvalues_table[ pvalues_table["sigma"] > gt_sigma ]    
    
    pvalues_table.reset_index(drop=True, inplace=True)

    return pvalues_table

def get_threshold_for_sigma(pval_table, min_sigma):
    return pval_table[pval_table["sigma"]>min_sigma].iloc[0,:]["threshold"]
    
def get_sigma_from_ts(pvalues_table, ts, verbose=False):
    # If there is no suitable index, return either 0 or N (where N is the length of a).
    ii_left = np.searchsorted(pvalues_table["threshold"], ts)

    if ii_left == 0 and ts < pvalues_table["threshold"].iloc[ii_left]:
        if verbose:
            print(f"[Warning] The ts value {ts} is below the ts range.")
        return 0

    if ii_left == pvalues_table.shape[0]:
        if verbose:
            print(f"[Warning] The ts value {ts} exceeds the ts range.")
        return pvalues_table["sigma"].iloc[ii_left-1]
    
    f = interpolate.interp1d(pvalues_table["threshold"], pvalues_table["pvalue"])

    return get_sigma_from_pvalue(f(ts), decimals=10)
