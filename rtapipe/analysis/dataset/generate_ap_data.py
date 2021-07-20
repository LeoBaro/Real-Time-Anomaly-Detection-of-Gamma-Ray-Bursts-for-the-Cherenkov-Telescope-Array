import argparse
from os import environ
from pathlib import Path
from time import time
from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot

# background ROI=2.5
# python generate_ap_data.py -dd $DATA/obs/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5 -t bkg

# grb with onset ROI=2.5
# python generate_ap_data.py -dd $DATA/obs/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5 -t grb

def makePlots(args, integrationType, outputFiles):

    for idx, outputFile in enumerate(outputFiles):   

        plot = PhotometrySinglePlot(title=paramsString)

        if args.type == "bkg":
            _ = plot.addData(outputFile, labelPrefix=f"{args.type}-{idx}", marker="x", color="blue")
        else:
            _ = plot.addData(outputFile, labelPrefix=f"{args.type}-{idx}", marker="D", color="red")

        _ = plot.plotScatter(0, integrationType, plotError=False, verticalLine=True, verticalLineX=900)

        plot.save(outputDir, f"{integrationType}_{paramsString}")

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
    parser.add_argument("-lim", "--limit", type=int, required=True, help="The number of input files to use")
    parser.add_argument("-ws", "--windowsize", nargs="+", default=[5], required=False, help="A list of window size values: a different output file will be created for each of those values")
    parser.add_argument("-rr", "--regionradius", nargs="+", default=[0.5], required=False, help="A list of region radius values: a different output file will be created for each of those values")
    parser.add_argument("-norm", "--normalize", type=str2bool, required=False, help="If 'yes' the counts will be normalized")
    parser.add_argument("-out", "--outputdir", type=str, required=False, help="The path to the output directory. If not provided the script's directory will be used")

    parser.add_argument("-wmin", "--windowmin", type=int, required=False, default=0, help="")
    parser.add_argument("-wmax", "--windowmax", type=int, required=False, default=1800, help="")
    parser.add_argument("-emin", "--energymin", type=float, required=False, default=0.03, help="")
    parser.add_argument("-emax", "--energymax", type=float, required=False, default=0.15, help="")
    parser.add_argument("-ebins", "--energybins", type=int, required=False, default=4, help="")

    args = parser.parse_args()


    if args.outputdir is not None:
        outputRootDir = Path(__file__).parent.joinpath(args.outputdir, Path(args.dataDir).name)
    else:
        outputRootDir = Path(args.outputdir).parent.joinpath(Path(args.dataDir).name)
    
    Path(args.outputdir).mkdir(parents=True, exist_ok=True)

    configFile = Path(args.outputdir).joinpath(f"config_{args.type}.txt")
    with open(configFile, "w") as cfg:
        cfg.write('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    
    ## TIME INTEGRATION

    for ws in args.windowsize:

        for rr in args.regionradius:

            print(f"AP data generation --> window size: {ws} region radius: {rr} (T integration)")

            paramsString = f"integration_t_type_{args.type}_window_size_{ws}_region_radius_{rr}"

            outputDir = outputRootDir.joinpath(paramsString)

            tWindows = Photometry2.getLinearWindows(args.windowmin, args.windowmax, int(ws), int(ws))

            ph = Photometry2(args.dataDir, outputDir)

            start = time() 

            outputFiles, counts = ph.integrateAll("T", None, rr, tWindows, None, limit=args.limit, parallel=True, normalize=args.normalize)

            elapsed = round(time()-start, 2)
            print(f"Took: {elapsed} sec. Produced: {len(outputFiles)} files.")

            with open(configFile, "a") as cfg:
                cfg.write(f"\nTook={elapsed}")

            if args.makeplots:

                makePlots(args, "T", outputFiles)




    ## TIME + ENERGY INTEGRATION

    for ws in args.windowsize:

        for rr in args.regionradius:
            
            print(f"AP data generation --> window size: {ws} region radius: {rr} (TE integration)")

            paramsString = f"integration_te_type_{args.type}_window_size_{ws}_region_radius_{rr}"

            outputDir = outputRootDir.joinpath(paramsString)

            tWindows = Photometry2.getLinearWindows(args.windowmin, args.windowmax, int(ws), int(ws))

            eWindows = Photometry2.getLogWindows(args.energymin, args.energymax, args.energybins)
            
            ph = Photometry2(args.dataDir, outputDir)

            outputFiles, counts = ph.integrateAll("TE", None, rr, tWindows, eWindows, limit=args.limit, parallel=True, normalize=args.normalize)

            elapsed = round(time()-start, 2)
            print(f"Took: {elapsed} sec. Produced: {len(outputFiles)} files.")

            with open(configFile, "a") as cfg:
                cfg.write(f"\nTook={elapsed}")

            if args.makeplots:

                makePlots(args, "TE", outputFiles)


