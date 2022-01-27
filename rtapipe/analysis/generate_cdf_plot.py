import argparse
import numpy as np
from pathlib import Path
from RTAscience.lib.RTAStats import ts_wilks, p_values, ts_wilks_cumulative

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs='+', type=str, required=True)
    parser.add_argument("-o", "--outname", type=str, required=True)
    args = parser.parse_args()

    outDir=Path("./tmp")
    data = [np.genfromtxt(file, usecols=(0), skip_header=0, dtype=float) for file in args.files]

    dataShape = data[0].shape
    for i, d in enumerate(data):
        assert d.shape == dataShape
    print("max:", np.array(data).max())
    print("sorted:", sorted(data[0], reverse=True))

    ts_wilks(data[0], trials=len(data[0]), nbin=100, width=None, df=1, xlim=[0, 0.3], filename=f"{args.outname}_tswilks")
    p_values(data, trials=len(data[0]), nbin=100, width=None, xlim=[0, 0.3], filename = outDir.joinpath(f"{args.outname}_pvalues"))
