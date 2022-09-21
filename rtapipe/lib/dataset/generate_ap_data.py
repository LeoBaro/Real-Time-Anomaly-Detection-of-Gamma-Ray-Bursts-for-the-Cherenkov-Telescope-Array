import yaml
import argparse
from time import time
from pathlib import Path

from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot
from rtapipe.lib.datasource.integrationstrat.IntegrationStrategies import IntegrationType, TimeIntegration, EnergyIntegration, TimeEnergyIntegration, FullIntegration
from rtapipe.lib.rtapipeutils.PhotometryUtils import PhotometryUtils

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(args):

    with open(Path(args.dataDir).joinpath("config.yaml"), "r") as stream:
        phListConfig = yaml.safe_load(stream)
        wmin = 0
        tobs = phListConfig["simulation"]["tobs"]
        emin = phListConfig["simulation"]["emin"]
        emax = phListConfig["simulation"]["emax"]

    Path(args.outputdir).mkdir(parents=True, exist_ok=True)

    configFile = Path(args.outputdir).joinpath(f"config_{args.integrationtype}.txt")
    with open(configFile, "w") as cfg:
        cfg.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    it = args.integrationtime

    # 100 sec pht list with 5 sec of integration time -> 100/5 = 20 points time series (maximum lenght)
    tsl = int(tobs / it)

    print(f"\nAP data generation --> it: {it} rr: {args.regionradius} tsl: {tsl} ({args.integrationtype} integration) Normalization: {args.normalize}")

    # Windows
    tWindows = PhotometryUtils.getLinearWindows(wmin, tobs, int(it), int(it))

    ph = Photometry2(args.dataDir, args.outputdir)

    start = time()

    ## TIME INTEGRATION
    if args.integrationtype == "t":
        eWindows = None
        integrationTypeEnum = IntegrationType.TIME
    
    ## TIME-ENERGY INTEGRATION
    elif args.integrationtype == "te":
        eWindows = PhotometryUtils.getLogWindows(emin, emax, args.energybins)
        integrationTypeEnum = IntegrationType.TIME_ENERGY

    else:
        raise ValueError(f"Integration type {args.integrationtype} not supported")

    outputFilesCounts, totalCounts = ph.integrateAll(integrationTypeEnum, args.regionradius, tWindows=tWindows, eWindows=eWindows, limit=args.limit, parallel=True, procNumber=args.procnumber, normalize=args.normalize)


    elapsed = round(time()-start, 2)
    print(f"Took: {elapsed} sec. Total counts: {totalCounts}")

    with open(configFile, "a") as cfg:
        cfg.write(f"\nTook={elapsed}")


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generating aperture photometry data')
    parser.add_argument("-dd",    "--dataDir",          type=str,      required=True,  help="The path to the folder containing the input files")
    parser.add_argument("-itype", "--integrationtype",  type=str,      required=True,  choices=["t", "te"], help="")
    parser.add_argument("-itime", "--integrationtime",  type=int,      required=True,  help="")
    parser.add_argument("-rr",    "--regionradius",     type=float,    required=True,  help="A list of region radius values: a different output file will be created for each of those values")
    parser.add_argument("-out",   "--outputdir",        type=str,      required=True,  help="The path to the output directory.")
    parser.add_argument("-norm",  "--normalize",        type=str2bool, required=True,  help="If 'yes' the counts will be normalized")
    parser.add_argument("-proc",  "--procnumber",       type=int,      required=True,  help="The number of processes to use for parallelization")
    parser.add_argument("-lim",   "--limit",            type=int,      required=False, default=None, help="The number of input files to use")
    parser.add_argument("-eb",   "--energybins",        type=int,      required=False, default=4, help="The number of the energy bins")

    args = parser.parse_args()

    main(args)