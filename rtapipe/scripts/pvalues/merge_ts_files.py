import os
import argparse
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

from RTAscience.lib.RTAStats import *

# python merge_ts_files.py -p /data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs/datasetid_601-modelname_m4-trainingtype_heavy-timestamp_20220109-161654/epochs/epoch_10/pvalues/jobs 

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_path", type=str, required=True, help="")

    args = parser.parse_args()

    c = 0
    data = []
    s = time()
    p = Path(args.input_path)
    for root, _, files in os.walk(p):
        for f in files:
            pfile = p.joinpath(root,f)
            if f == "ts_for_pvalues.txt":
                continue
            if str(pfile).endswith(".txt"):
                print(pfile)
                data.append(
                    pd.read_csv(pfile, header=None)
                )
                c += 1

    merge = pd.concat(data, axis=0, ignore_index=True)
    merge.to_csv(p.parent.joinpath("merged_ts_for_pvalues.csv"), index=False, header=False)
    np.save(p.parent.joinpath("merged_ts_for_pvalues.pickle"), merge.values[:,0])
    print(f"Merged {c} files for a total of {len(merge)} values. Took {time()-s} seconds.")