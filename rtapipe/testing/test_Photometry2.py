import pandas as pd
from pathlib import Path

from rtapipe.lib.datasource.Photometry2 import Photometry2

class TestPhotometry2:

    def test_getLinearWindows(self):
                
        w = Photometry2.getLinearWindows(0, 1800, 25, 25)

        assert len(w) == 72
        assert w[0] == (0, 25)
        assert w[-1] == (1775, 1800)

        w = Photometry2.getLinearWindows(0, 1000, 100, 10)

        assert len(w) == 91
        assert w[0] == (0, 100)
        assert w[-1] == (900, 1000)


    def test_getLogWindows(self):

        w = Photometry2.getLogWindows(0.03, 0.15, 4)

        assert len(w) == 4
        assert w[0] == (0.03, 0.0449)       
        assert w[-1] == (0.1003, 0.15)



    def test_init(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output")

        ph = Photometry2(dataDir, outputDir)

        assert len(ph.dataFiles) == 2
        assert ph.runId == "run0406_ID000126"

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output")

        ph = Photometry2(dataDir, outputDir)

        assert len(ph.dataFiles) == 1
        assert ph.runId == "run0406_ID000126"

    def test_getOutputFilePath(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output")

        ph = Photometry2(dataDir, outputDir)

        inputFilename = dataDir.joinpath("bkg000002.fits")


        integrationType = ph.setIntegrationType(None, None)

        outputFilePath = ph.getOutputFilePath(inputFilename, integrationType)

        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_full_simtype_bkg_onset_0.csv"


        integrationType = ph.setIntegrationType(ph.getLinearWindows(0,10,2,1), None)

        outputFilePath = ph.getOutputFilePath(inputFilename, integrationType)

        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_t_simtype_bkg_onset_0.csv"


        integrationType = ph.setIntegrationType(None, Photometry2.getLogWindows(0.03, 0.15, 4))

        outputFilePath = ph.getOutputFilePath(inputFilename, integrationType)

        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_e_simtype_bkg_onset_0.csv"


        integrationType = ph.setIntegrationType(ph.getLinearWindows(0,10,2,1), Photometry2.getLogWindows(0.03, 0.15, 4))

        outputFilePath = ph.getOutputFilePath(inputFilename, integrationType)

        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_te_simtype_bkg_onset_0.csv"



    def test_integrate_full(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_full")
        inputFilename = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius)

        assert Path(outputFilePath).is_file() == True
        assert counts == 46706

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilename = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius)
        
        assert Path(outputFilePath).is_file() == True
        assert counts == 48803
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

    def test_integrate_t_e(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_t_e")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        inputFilename = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius, tWindows=tWindows, eWindows=eWindows)
        assert Path(outputFilePath).is_file() == True
        assert counts == 46706

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilename = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius, tWindows=tWindows, eWindows=eWindows)
        assert Path(outputFilePath).is_file() == True
        assert counts == 48803
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

        
    def test_integrate_t(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_t")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)

        inputFilename = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius, tWindows=tWindows)
        assert Path(outputFilePath).is_file() == True
        assert counts == 46706
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilename = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius, tWindows=tWindows)
        assert Path(outputFilePath).is_file() == True
        assert counts == 48803
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False


    def test_integrate_e(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_e")
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        inputFilename = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius, eWindows=eWindows)
        assert Path(outputFilePath).is_file() == True
        assert counts == 46706
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilename = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius, eWindows=eWindows)
        assert Path(outputFilePath).is_file() == True
        assert counts == 48803
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

    def test_integrateAll(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_all")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFiles, counts = ph.integrateAll(regionRadius, tWindows=tWindows, eWindows=eWindows)

        for outputFilePath in outputFiles:
            assert Path(outputFilePath).is_file() == True
            assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False
        assert counts == 93662


        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilename = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 1
        outputFilePath, counts = ph.integrate(inputFilename, regionRadius, tWindows=tWindows, eWindows=eWindows)
        
        for outputFilePath in outputFiles:
            assert Path(outputFilePath).is_file() == True
            assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False
        assert counts == 48803

