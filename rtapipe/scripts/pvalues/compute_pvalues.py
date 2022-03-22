import os
import argparse
import numpy as np
import pandas as pd
from time import time, strftime
from pathlib import Path

from RTAscience.lib.RTAStats import *

# python compute_pvalues.py -p /data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs/datasetid_601-modelname_m4-trainingtype_heavy-timestamp_20220109-161654/epochs/epoch_10/pvalues/merged_ts_for_pvalues.pickle.npy

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_path", type=str, required=True, help="")
    args = parser.parse_args()

    output_wilks_png = Path(args.input_path).parent.joinpath(f'ts_distribution_{strftime("%Y%m%d-%H%M%S")}.png')
    output_pvalue_png = Path(args.input_path).parent.joinpath(f'pvalue_{strftime("%Y%m%d-%H%M%S")}.png')
    output_wilks_svg = Path(args.input_path).parent.joinpath(f'ts_distribution_{strftime("%Y%m%d-%H%M%S")}.svg')
    output_pvalue_svg = Path(args.input_path).parent.joinpath(f'pvalue_{strftime("%Y%m%d-%H%M%S")}.svg')


    nbin = 100
    
    data = np.load(args.input_path)
    
    fig, ax = ts_wilks([data], df=1, nbin=nbin, figsize=(7, 8), xrange=(0,1), title='Reconstruction errors distribution', show=False, usetex=False, filename=output_wilks_png)
    fig, ax = p_values([data], df=1, nbin=nbin, figsize=(7, 8), xrange=(0,1), title='Reconstruction errors p-values', show=False, usetex=False, filename=output_pvalue_png, overlay=None, sigma5=False, write_data=True)


    fig, ax = ts_wilks([data], df=1, nbin=nbin, figsize=(7, 8), xrange=(0,1), title='Reconstruction errors distribution', show=False, usetex=False, filename=output_wilks_svg)
    fig, ax = p_values([data], df=1, nbin=nbin, figsize=(7, 8), xrange=(0,1), title='Reconstruction errors p-values', show=False, usetex=False, filename=output_pvalue_svg, overlay=None, sigma5=False, write_data=True)
