import os
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from time import time, strftime
from pathlib import Path

from RTAscience.lib.RTAStats import *
from compute_pvalues import compute_pval

# python compute_pvalues_multiple_folders.py -p /data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs

def compute_pval_multi(args):

    tasks = []

    for root, dirs, files in os.walk(args.input_path, topdown=False):
    
        for name in files:
    
            if name == "merged_ts_for_pvalues.pickle.npy":

                print(f"Computing p-val for {root}")
                dataF = Path(root).joinpath(name)

                tasks.append(dataF)

    print(f"{len(tasks)} jobs..")

    with multiprocessing.Pool() as pool:
        pool.map(compute_pval, tasks)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_path", type=str, required=True, help="")
    args = parser.parse_args()
    compute_pval_multi(args)

