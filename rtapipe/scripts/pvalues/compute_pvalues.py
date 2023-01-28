import os
import argparse
import numpy as np
import pandas as pd
from time import time, strftime
from pathlib import Path

from RTAscience.lib.RTAStats import ts_wilks, p_values

"""
compute_pvalues -p /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_cnn_l2_u32_dataset_1201_tsl_5/epochs/epoch_117/pvalues/merged_ts_for_pvalues.pickle.npy
compute_pvalues -p /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_rnn_l2_u32_dataset_1201_tsl_5/epochs/epoch_114/pvalues/merged_ts_for_pvalues.pickle.npy
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_path", type=str, required=True, help="")
    parser.add_argument("-xm", "--x_max", type=float, required=False, default=0.15, help="")
    args = parser.parse_args()
    compute_pval(args.input_path, args.x_max)

def compute_pval(input_path, xmax=0.15):

    data = np.load(input_path)

    output_path = Path(input_path).parent.joinpath(f'pval_{strftime("%Y%m%d-%H%M%S")}')
    output_path.mkdir(parents=True, exist_ok=True)
    for nbin in [100]:

        for extension in [".png", ".svg"]:
            output_wilks  = output_path.joinpath(f'ts_distribution_bins_{nbin}{extension}')
            output_pvalue = output_path.joinpath(f'pvalue_bins_{nbin}{extension}')

            print(f"Generating p-value plot: {output_pvalue}")

            _, _ = ts_wilks([data], df=1, nbin=nbin, figsize=(7, 8), xrange=(0, xmax), title='TS distribution', xlabel="TS (reconstruction errors)", ylabel='Normalised counts', overlay=False, filename=output_wilks)
            _, _ = p_values([data], df=1, nbin=nbin, figsize=(7, 8), title='p-values', filename=output_pvalue, sigma5=True, write_data=True,  overlay=False, dpi=400, fmt='+', ecolor='red', markersize=0.5, elinewidth=0.5, alpha=0.8)


def compare_pval(input_paths, output_dir):

    output_dir.mkdir(parents=True, exist_ok=True)

    for nbin in [100]:

        output_wilks_png = output_dir.joinpath(f'ts_distribution_bins_{nbin}.png')
        output_pvalue_png = output_dir.joinpath(f'pvalue_bins_{nbin}.png')
        
        print(f"Generating p-value plot: {output_pvalue_png}")

        slabel = [Path(input_path).parent.parent.parent.parent.name[:-26] for input_path in input_paths]
        print(f"slabel: {slabel}")
        data = np.stack([np.load(input_path) for input_path in input_paths])

        _, _ = p_values(data, slabel=slabel, df=1, nbin=nbin, figsize=(7, 8), xrange=(0,0.15), title='p-values', filename=output_pvalue_png, sigma5=True, write_data=False,  overlay=False, dpi=400, fmt='+', ecolor='red', markersize=0.5, elinewidth=0.5, alpha=0.8, legendprop={"size":8}, legendloc=1)



if __name__ == '__main__':
    main()

