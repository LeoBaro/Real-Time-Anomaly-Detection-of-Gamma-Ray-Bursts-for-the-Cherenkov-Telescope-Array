import argparse 
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fitter import Fitter


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.file, header=None, squeeze=True)
    df = df[:50]
    trials = len(df)
    print("trials",trials)

    maxVal = max(df)
    print("maxVal:",maxVal)

    input("Enter any key to start the fitting..")
    f = Fitter(df, xmin=0, xmax=maxVal, timeout=240)
    f.fit(progress=False, n_jobs=-1)
    # may take some time since by default, all distributions are tried
    # but you call manually provide a smaller set of distributions
    fitDF = f.summary()
    print(fitDF)


