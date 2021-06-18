import argparse
from os import environ
from pathlib import Path
from time import strftime
from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot

# python generate_ap_data.py -dd $DATA/obs/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_0.5

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Generating aperture photometry data')
    parser.add_argument("-dd", "--dataDir", required=True)
    args = parser.parse_args()

    windowSize = [1, 5, 10, 25, 50, 100]
    regionRadius = [1]#, 0.5, 1]

    for ws in windowSize:
        for rr in regionRadius:
            paramsString = f"window_size_{ws}_region_radius_{rr}"
            outputDir = Path(__file__).parent.joinpath("output", paramsString)
            tWindows = Photometry2.getLinearWindows(0, 1800, ws, ws)
            eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)
            ph = Photometry2(args.dataDir, outputDir)
            outputFiles, counts = ph.integrateAll(rr, limit=1, tWindows=tWindows, eWindows=eWindows)
            plot = PhotometrySinglePlot(title=paramsString)
            for idx, outputFile in enumerate(outputFiles):
                _ = plot.addData(outputFile, "T_E", labelPrefix=f"Background-{idx}")
            plot.save(outputDir, paramsString)  


