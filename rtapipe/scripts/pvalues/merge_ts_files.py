import os
import tqdm
import argparse
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

# python merge_ts_files.py -p /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_cnn_l2_u32_dataset_1201_tsl_5/epochs/epoch_117/pvalues/ts_values

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--input_path", type=str, required=True, help="")

    args = parser.parse_args()

    c = 0
    data = []
    s = time()
    p = Path(args.input_path)
    print("Loading files..")
    for root, _, files in os.walk(p):
        for f in tqdm.tqdm(files):
            pfile = p.joinpath(root,f)
            if str(pfile).endswith(".txt"):
                data.append(
                    pd.read_csv(pfile, header=None)
                )
                c += 1
    print("Merging..")
    merge = pd.concat(data, axis=0, ignore_index=True)
    #merge.to_csv(p.parent.joinpath("merged_ts_for_pvalues.csv"), index=False, header=False)
    output_file = p.parent.joinpath("merged_ts_for_pvalues.pickle")
    np.save(output_file, merge.values[:,0])
    print(f"Merged {c} files for a total of {len(merge)} values. Took {time()-s} seconds. File saved at {output_file}")

if __name__ == '__main__':
    main()