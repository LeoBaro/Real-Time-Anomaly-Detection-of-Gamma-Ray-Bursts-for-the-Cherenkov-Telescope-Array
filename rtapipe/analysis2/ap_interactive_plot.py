import argparse
from os import environ
from pathlib import Path
from time import strftime
from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot

# python ap_interactive_plot.py -dd $DATA/obs/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_0.5 -f $DATA/obs/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_0.5/bkg000002.fits -ws 1 -rr 1

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generating aperture photometry data')
    parser.add_argument("-dd", "--dataDir", required=True)
    parser.add_argument("-f", "--file", type=str, required=True)
    parser.add_argument("-ws", "--windowSize", type=int, required=True)
    parser.add_argument("-rr", "--regionRadius", type=float, required=True)
    args = parser.parse_args()

    paramsString = f"window_size_{args.windowSize}_region_radius_{args.regionRadius}"
    outputDir = Path(__file__).parent.joinpath("ap_interactive_plot_output", paramsString)
    tWindows = Photometry2.getLinearWindows(0, 1800, args.windowSize, args.windowSize)
    eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)
    ph = Photometry2(args.dataDir, outputDir)
    outputFile, counts = ph.integrate(args.file, args.regionRadius, tWindows=tWindows, eWindows=eWindows)
    plot = PhotometrySinglePlot(title=paramsString)
    _ = plot.addData(outputFile, "T_E", labelPrefix=f"Background")
    plot.show()  


