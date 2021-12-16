import argparse
import numpy as np
from pathlib import Path
from RTAscience.lib.RTAStats import ts_wilks, p_values, ts_wilks_cumulative

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-o", "--outname", type=str, required=True)
    args = parser.parse_args()

    outDir=Path("./tmp")
    data = np.genfromtxt(args.file, usecols=(0), skip_header=0, dtype=float)

    print("len(data)=",len(data))
    print(data[0:10])
    #ts_wilks(data, trials=data.shape[0], nbin=100, width=None, filename = outDir.joinpath(f"ts_wilks"))
    #ts_wilks_cumulative(data, trials=data.shape[0], nbin=100, width=None, filename = outDir.joinpath(f"ts_wilks_cumulative"))
    p_values(data, trials=data.shape[0], nbin=100, width=None, xlim=[0, 0.3], filename = outDir.joinpath(f"{args.outname}"))
