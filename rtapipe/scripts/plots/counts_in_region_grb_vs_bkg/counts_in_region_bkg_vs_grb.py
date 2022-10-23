import argparse
from os import listdir
from pathlib import Path
from os.path import isfile, join

import numpy as np
from astropy.io import fits

def count_photons(ph_list_file):
    with fits.open(ph_list_file) as hdul:
        #header = hdul[1].header
        #print(header)
        #data = hdul[1].data
        #columns = hdul[1].columns
        #print(columns)
        #print(data)
        return len(hdul[1].data)

def get_average_counts(ph_lists_path):
    counts = np.array([count_photons(ph_list_path) for ph_list_path in ph_lists_path])
    return counts.mean(), counts.std() 

def get_files(dir_path):
    return [join(dir_path, f) for f in listdir(dir_path) if isfile(join(dir_path, f)) and ".fits" in f and not ".skymap" in f]

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-od',  '--output_dir', type=str, default="./tmp-output", help='Output dir')
    args = parser.parse_args()

    data_dirs=["./data/backgrounds","./data/run0406_ID000126_onset_100","./data/run0406_ID000126_onset_0"]

    for dd in data_dirs:
        if Path(dd).is_dir():
            files = get_files(dd)
            avg,std = get_average_counts(files)
            print(f"Dir: {dd} \n\tavg: {avg} std: {std}")
