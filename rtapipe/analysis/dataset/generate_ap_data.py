import argparse
from os import environ
from pathlib import Path
from time import time
from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot
import yaml
# background ROI=2.5
# python generate_ap_data.py -dd $DATA/obs/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5 -t bkg

# grb with onset ROI=2.5
# python generate_ap_data.py -dd $DATA/obs/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5 -t grb

def makePlots(args, integrationType, outputFiles, onset):

    for idx, outputFile in enumerate(outputFiles):

        plot = PhotometrySinglePlot(title=paramsString)

        if args.type == "bkg":
            _ = plot.addData(outputFile, labelPrefix=f"{args.type}-{idx}", marker="x", color="blue")
        else:
            _ = plot.addData(outputFile, labelPrefix=f"{args.type}-{idx}", marker="D", color="red")

        verticalLine = False
        if onset > 0:
            verticalLine = True

        _ = plot.plotScatter(0, integrationType, plotError=False, verticalLine=verticalLine, verticalLineX=onset)

        plot.save(outputDir, f"{idx}_{integrationType}_{paramsString}")

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generating aperture photometry data')
    parser.add_argument("-dd", "--dataDir", type=str, required=True, help="The path to the folder containing the input files")
    parser.add_argument("-t", "--type", type=str, choices=["bkg", "grb"], required=True, help="The type of the input files")
    parser.add_argument("-mp", "--makeplots", type=str2bool, required=True, help="If 'yes' plots will be produced")
    parser.add_argument("-lim", "--limit", type=int, required=None, default=None, help="The number of input files to use")
    parser.add_argument("-itype", "--integrationtype", type=str, required=True, choices=["t", "te"], help="")
    parser.add_argument("-itime", "--integrationtime", type=int, required=True, help="")

    # TODO test multiple integration region radius
    parser.add_argument("-rr", "--regionradius", type=float, required=True, help="A list of region radius values: a different output file will be created for each of those values")
    parser.add_argument("-norm", "--normalize", type=str2bool, required=True, help="If 'yes' the counts will be normalized")
    parser.add_argument("-tsl", "--timeserieslenght", type=int, required=True, default=None, help="The number of the csv files rows (time series lenght)")
    parser.add_argument("-out", "--outputdir", type=str, required=True, help="The path to the output directory. If not provided the script's directory will be used")
    # TODO verify ebins argument
    parser.add_argument("-ebins", "--energybins", type=int, required=False, default=4, help="")
    parser.add_argument("-proc", "--procnumber", type=int, required=False, default=10, help="")
    args = parser.parse_args()

    with open(Path(args.dataDir).joinpath("config.yaml"), "r") as stream:
        phListConfig = yaml.safe_load(stream)
        wmin = 0
        wmax = phListConfig["simulation"]["tobs"]
        emin = phListConfig["simulation"]["emin"]
        emax = phListConfig["simulation"]["emax"]
        onset = phListConfig["simulation"]["onset"]

    if args.outputdir is not None:
        outputRootDir = Path(__file__).parent.joinpath(args.outputdir, Path(args.dataDir).name)
    else:
        outputRootDir = Path(args.outputdir).parent.joinpath(Path(args.dataDir).name)

    Path(args.outputdir).mkdir(parents=True, exist_ok=True)

    configFile = Path(args.outputdir).joinpath(f"config_{args.integrationtype}.txt")
    with open(configFile, "w") as cfg:
        cfg.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))


    ## TIME INTEGRATION
    if args.integrationtype == "t":

        it = args.integrationtime
        rr = args.regionradius

        if args.timeserieslenght is not None:

            wmax = int(it) * args.timeserieslenght

        else:

            args.timeserieslenght = int(wmax / int(it))

        print(f"AP data generation --> integration time: {it} region radius: {rr} (T integration) Normalization: {args.normalize}")

        paramsString = f"integration_t_integration_time_{it}_region_radius_{rr}_timeseries_lenght_{args.timeserieslenght}"

        outputDir = outputRootDir.joinpath(paramsString)

        tWindows = Photometry2.getLinearWindows(wmin, wmax, int(it), int(it))
        print(tWindows)
        ph = Photometry2(args.dataDir, outputDir)

        start = time()

        outputFiles, counts = ph.integrateAll("T", None, rr, tWindows, None, limit=args.limit, parallel=True, procNumber=args.procnumber, normalize=args.normalize)

        elapsed = round(time()-start, 2)
        print(f"Took: {elapsed} sec. Produced: {len(outputFiles)} files.")

        with open(configFile, "a") as cfg:
            cfg.write(f"\nTook={elapsed}")

        if args.makeplots:

            makePlots(args, "T", outputFiles, onset)


    ## TIME + ENERGY INTEGRATION
    else:

        it = args.integrationtime
        rr = args.regionradius

        if args.timeserieslenght is not None:

            wmax = int(it) * args.timeserieslenght

        else:

            args.timeserieslenght = int(wmax / int(it))

        print(f"AP data generation --> integration time: {it} region radius: {rr} (TE integration) Normalization: {args.normalize}")

        paramsString = f"integration_te_integration_time_{it}_region_radius_{rr}_timeseries_lenght_{args.timeserieslenght}"

        outputDir = outputRootDir.joinpath(paramsString)

        tWindows = Photometry2.getLinearWindows(wmin, wmax, int(it), int(it))

        eWindows = Photometry2.getLogWindows(emin, emax, args.energybins)

        ph = Photometry2(args.dataDir, outputDir)

        start = time()

        outputFiles, counts = ph.integrateAll("TE", None, rr, tWindows, eWindows, limit=args.limit, parallel=True, procNumber=args.procnumber, normalize=args.normalize)

        elapsed = round(time()-start, 2)
        print(f"Took: {elapsed} sec. Produced: {len(outputFiles)} files.")

        with open(configFile, "a") as cfg:
            cfg.write(f"\nTook={elapsed}")

        if args.makeplots:

            makePlots(args, "TE", outputFiles, onset)
