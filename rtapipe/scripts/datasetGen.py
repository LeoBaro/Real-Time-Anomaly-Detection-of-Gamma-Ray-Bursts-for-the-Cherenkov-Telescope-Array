from pathlib import Path
from rtapipe.lib.datasource.Photometry2 import Photometry2

if __name__=='__main__':

    DATA_IN="/data01/home/baroncelli/phd/DATA/obs/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5"
    dataDir = Path(DATA_IN)
    DATA_OUT = Path(__file__).parent.joinpath("datasetGenOutput")
    outputDir = Path(DATA_OUT)

    tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
    eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

    ph = Photometry2(dataDir, outputDir)
    regionRadius = 1
    outputFiles, counts = ph.integrateAll(regionRadius, tWindows=tWindows, eWindows=eWindows)

