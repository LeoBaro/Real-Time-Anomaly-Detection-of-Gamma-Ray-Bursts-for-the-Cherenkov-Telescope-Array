import numpy as np
import pandas as pd
from scipy.stats import norm


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

def get_pval_table(pvalues_path, gt_sigma, show=True):
    pvalues_table = pd.read_csv(pvalues_path, sep=" ")
    pvalues_table = pvalues_table.rename(columns={'x': 'threshold', 'xerr': 'threshold_err', 'y': 'pvalue', 'yerr':'pvalue_err'})
    pvalues_table["sigma"] = pvalues_table["pvalue"].apply(get_sigma_from_pvalue)
    pvalues_table = pvalues_table.drop_duplicates(subset='sigma', keep="first")
    pvalues_table = pvalues_table[ pvalues_table["sigma"] > gt_sigma ]
    pvalues_table.replace([np.inf, -np.inf], np.nan, inplace=True)    
    pvalues_table.dropna(subset=["sigma"], how="all", inplace=True)
    if show:
        print(pvalues_table)
    return pvalues_table

def get_threshold_for_sigma(pval_table, min_sigma):
    return pval_table[pval_table["sigma"]>min_sigma].iloc[0,:]["threshold"]