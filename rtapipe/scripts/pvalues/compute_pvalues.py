import os
import argparse
import numpy as np
import pandas as pd
from time import time, strftime
from pathlib import Path

from RTAscience.lib.RTAStats import *

# python compute_pvalues.py -p /data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs/datasetid_601-modelname_m4-trainingtype_heavy-timestamp_20220109-161654/epochs/epoch_10/pvalues/merged_ts_for_pvalues.pickle.npy

def compute_pval(input_path):

    data = np.load(input_path)

    output_path = Path(input_path).parent.joinpath(f'pval_{strftime("%Y%m%d-%H%M%S")}')
    output_path.mkdir(parents=True, exist_ok=True)
    for nbin in [100, 1000, 10000]:

        output_wilks_png = output_path.joinpath(f'ts_distribution_bins_{nbin}.png')
        output_pvalue_png = output_path.joinpath(f'pvalue_bins_{nbin}.png')

        print(f"Generating p-value plot: {output_pvalue_png}")

        fig, ax = ts_wilks([data], df=1, nbin=nbin, figsize=(7, 8), xrange=(0,0.3), title='Reconstruction errors distribution', overlay=False, filename=output_wilks_png)
        fig, ax = p_values([data], df=1, nbin=nbin, figsize=(7, 8), title='Reconstruction errors p-values', filename=output_pvalue_png, sigma5=True, write_data=True,  overlay=False, dpi=400, fmt='+', ecolor='red', markersize=0.5, elinewidth=0.5, alpha=0.8)

    #fig, ax = ts_wilks([data], df=1, nbin=nbin, figsize=(7, 8), xrange=(0,0.5), title='Reconstruction errors distribution', usetex=False, filename=output_wilks_svg)
    #fig, ax = p_values([data], df=1, nbin=nbin, figsize=(7, 8), xrange=(0,0.5), title='Reconstruction errors p-values', usetex=False, filename=output_pvalue_svg, overlay=None, sigma5=True, write_data=True)

def compare_pval(input_paths, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    for nbin in [100, 1000, 10000]:

        output_wilks_png = output_dir.joinpath(f'ts_distribution_bins_{nbin}.png')
        output_pvalue_png = output_dir.joinpath(f'pvalue_bins_{nbin}.png')
        
        print(f"Generating p-value plot: {output_pvalue_png}")

        slabel = [Path(input_path).parent.parent.parent.parent.name[:-26] for input_path in input_paths]
        print(f"slabel: {slabel}")
        data = np.stack([np.load(input_path) for input_path in input_paths])

        print(len(slabel), data.shape)
        #fig, ax = ts_wilks([data], slabel=[], df=1, nbin=nbin, figsize=(7, 8), xrange=(0,0.3), title='Reconstruction errors distribution', overlay=False, filename=output_wilks_png)
        fig, ax = p_values(data, slabel=slabel, df=1, nbin=nbin, figsize=(7, 8), xrange=(0,0.3), title='Reconstruction errors p-values', filename=output_pvalue_png, sigma5=True, write_data=False,  overlay=False, dpi=400, fmt='+', ecolor='red', markersize=0.5, elinewidth=0.5, alpha=0.8, legendprop={"size":8}, legendloc=1)



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_path", type=str, required=True, help="")
    args = parser.parse_args()
    compute_pval(args.input_path)

