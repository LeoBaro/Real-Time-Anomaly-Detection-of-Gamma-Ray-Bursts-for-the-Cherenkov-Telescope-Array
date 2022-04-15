import os
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from time import time, strftime
from pathlib import Path
from enum import Enum
from RTAscience.lib.RTAStats import *
from compute_pvalues import compare_pval

#python compute_pvalues_comparison.py -p /data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs/ --add-ds 400 --add-model m1 --add-training l m h

def parse_folder_name(folder_name):
    
    dataset_id      = folder_name.split("datasetid_")[1][:3]
    model_id        = folder_name.split("modelname_")[1][:2]
    training_type   = folder_name.split("trainingtype_")[1][:1]

    return (dataset_id, model_id, training_type)

def compute_pval_comp(args):

    tasks = []

    for root, dirs, files in os.walk(args.input_path, topdown=False):
          
        for name in files:
    
            if name == "merged_ts_for_pvalues.pickle.npy":

                dataset_id, model_id, training_type = parse_folder_name(str(Path(root).parent.parent))
                #print(dataset_id, model_id, training_type)
                dataF = Path(root).joinpath(name)

                if dataset_id in args.add_ds and model_id in args.add_model and training_type in args.add_training:
                    tasks.append(dataF)     

    print(f"{len(tasks)} jobs..")
    
    if len(tasks) > 0:
        Path(args.out).mkdir(parents=True, exist_ok=True)

    compare_pval(tasks, args.out)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_path", type=str, required=True, help="")
    parser.add_argument('--add-ds', nargs='+', help='400, 401, 500, 501, ..', required=False, default=[])
    parser.add_argument('--add-model', nargs='+', help='m1, m2, m3, m4', required=False, default=[])
    parser.add_argument('--add-training', nargs='+', help='l,m,h', required=False, default=[])
    parser.add_argument('--out', type=str, required=False, default="./")
    # TODO parser.add_argument('--add-epoch', nargs='+', help='10, 100, 200', required=True)
    args = parser.parse_args()
    compute_pval_comp(args)

