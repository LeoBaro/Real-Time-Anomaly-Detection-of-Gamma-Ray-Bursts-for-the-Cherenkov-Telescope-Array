import pandas as pd
from pathlib import Path

from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.datasource.integrationstrat.IntegrationStrategies import IntegrationType

tolerance = 1e-14

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

        # assert len(ph.dataFiles) == 2


        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output")

        ph = Photometry2(dataDir, outputDir)

        # assert len(ph.dataFiles) == 1


    def test_getOutputFilePath(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output")
        inputFilename = dataDir.joinpath("bkg000002.fits")

        ph = Photometry2(dataDir, outputDir)

        outputFilePath = ph.getOutputFilePath(inputFilename, IntegrationType.TIME, True)
        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_t_simtype_bkg_onset_0_normalized_True.csv"

        outputFilePath = ph.getOutputFilePath(inputFilename, IntegrationType.ENERGY, True)
        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_e_simtype_bkg_onset_0_normalized_True.csv"

        outputFilePath = ph.getOutputFilePath(inputFilename, IntegrationType.TIME_ENERGY, True)
        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_te_simtype_bkg_onset_0_normalized_True.csv"

        outputFilePath = ph.getOutputFilePath(inputFilename, IntegrationType.FULL, True)
        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_full_simtype_bkg_onset_0_normalized_True.csv"



    def test_integrate_full_not_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_full")
        inputFilePath = dataDir.joinpath("bkg000002.fits")

        ph = Photometry2(dataDir, outputDir)
        customRegion = None
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateF(inputFilePath, customRegion, regionRadius, parallel=False, normalize=False)

        assert Path(outputFilePath).is_file() == True
        assert totalCounts == 1945

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilePath = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        customRegion = None
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateF(inputFilePath, customRegion, regionRadius, parallel=False, normalize=False)

        assert Path(outputFilePath).is_file() == True
        assert totalCounts == 3316
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False


    def test_integrate_full_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_full")
        inputFilePath = dataDir.joinpath("bkg000002.fits")

        ph = Photometry2(dataDir, outputDir)
        customRegion = None
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateF(inputFilePath, customRegion, regionRadius, parallel=False, normalize=True)

        assert Path(outputFilePath).is_file() == True
        assert (totalCounts - 1.8267166610338249e-09) < tolerance

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilePath = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        customRegion = None
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateF(inputFilePath, customRegion, regionRadius, parallel=False, normalize=True)

        assert Path(outputFilePath).is_file() == True
        assert (totalCounts - 3.114340590225277e-09) < tolerance
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False


    def test_integrate_t_not_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_t")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)

        inputFilePath = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 0.2
        customRegion = None
        outputFilePath, totalCounts = ph.integrateT(inputFilePath, customRegion, regionRadius, tWindows, parallel=False, normalize=False)
        assert Path(outputFilePath).is_file() == True
        assert totalCounts == 1945
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilePath = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 0.2
        customRegion = None
        outputFilePath, totalCounts = ph.integrateT(inputFilePath, customRegion, regionRadius, tWindows, parallel=False, normalize=False)
        assert Path(outputFilePath).is_file() == True
        assert totalCounts == 3316
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False



    def test_integrate_t_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_t")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)

        inputFilePath = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 0.2
        customRegion = None
        outputFilePath, totalCounts = ph.integrateT(inputFilePath, customRegion, regionRadius, tWindows, parallel=False, normalize=True)
        assert Path(outputFilePath).is_file() == True
        assert (totalCounts - 3.288089989860884e-08) < tolerance
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilePath = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 0.2
        customRegion = None
        outputFilePath, totalCounts = ph.integrateT(inputFilePath, customRegion, regionRadius, tWindows, parallel=False, normalize=True)
        assert Path(outputFilePath).is_file() == True
        assert (totalCounts - 5.6058130624054985e-08) < tolerance
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False


    def test_integrate_e_not_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_e")
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        inputFilePath = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 0.2
        customRegion = None
        outputFilePath, totalCounts = ph.integrateE(inputFilePath, customRegion, regionRadius, eWindows, parallel=False, normalize=False)
        assert Path(outputFilePath).is_file() == True
        assert totalCounts == 1945
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilePath = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateE(inputFilePath, customRegion, regionRadius, eWindows, parallel=False, normalize=False)
        assert Path(outputFilePath).is_file() == True
        assert totalCounts == 3316
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False



    def test_integrate_e_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_e")
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        inputFilePath = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 0.2
        customRegion = None
        outputFilePath, totalCounts = ph.integrateE(inputFilePath, customRegion, regionRadius, eWindows, parallel=False, normalize=True)
        assert Path(outputFilePath).is_file() == True
        assert (totalCounts - 2.4770990166353848e-09) < tolerance
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilePath = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateE(inputFilePath, customRegion, regionRadius, eWindows, parallel=False, normalize=True)
        assert Path(outputFilePath).is_file() == True
        assert (totalCounts - 3.766885883843366e-09) < tolerance
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False



    def test_integrate_t_e_not_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_t_e")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        inputFilePath = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        customRegion = None
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateTE(inputFilePath, customRegion, regionRadius, tWindows, eWindows, parallel=False, normalize=False)
        assert Path(outputFilePath).is_file() == True
        assert totalCounts == 1945

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilePath = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        customRegion = None
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateTE(inputFilePath, customRegion, regionRadius, tWindows, eWindows, parallel=False, normalize=False)
        assert Path(outputFilePath).is_file() == True
        assert totalCounts == 3316
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False




    def test_integrate_t_e_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_t_e")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        inputFilePath = dataDir.joinpath("bkg000002.fits")
        ph = Photometry2(dataDir, outputDir)
        customRegion = None
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateTE(inputFilePath, customRegion, regionRadius, tWindows, eWindows, parallel=False, normalize=True)
        assert Path(outputFilePath).is_file() == True
        assert (totalCounts - 4.458778229943692e-08) < tolerance

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")
        inputFilePath = dataDir.joinpath("grb000002.fits")
        ph = Photometry2(dataDir, outputDir)
        customRegion = None
        regionRadius = 0.2
        outputFilePath, totalCounts = ph.integrateTE(inputFilePath, customRegion, regionRadius, tWindows, eWindows, parallel=False, normalize=True)
        assert Path(outputFilePath).is_file() == True
        assert (totalCounts - 6.780394590918058e-08) < tolerance
        assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False


    def checkOutput(self, outputFiles, expected_files, counts_expected, counts_found):
        assert len(outputFiles) == expected_files
        for outputFilePath in outputFiles:
            assert Path(outputFilePath).is_file() == True
            assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False
        assert (counts_expected - counts_found) <= tolerance


    def test_integrate_all_not_normalized(self):

        dataDirBkg = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        dataDirGrb = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_all")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        customRegion = None
        regionRadius = 0.2


        ph = Photometry2(dataDirBkg, outputDir)

        outputFiles, totalCounts = ph.integrateAll("T", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 2, 3989, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("E", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 2, 3989, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("TE", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 2, 3989, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("F", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 2, 3989, totalCounts)


        ph = Photometry2(dataDirGrb, outputDir)

        outputFiles, totalCounts = ph.integrateAll("T", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 1, 3316, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("E", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 1, 3316, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("TE", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 1, 3316, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("F", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 1, 3316, totalCounts)



    def test_integrate_all_normalized(self):

        dataDirBkg = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        dataDirGrb = Path(__file__).parent.joinpath("test_data", "fits", "grb_onset")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_all")
        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        customRegion = None
        regionRadius = 0.2


        ph = Photometry2(dataDirBkg, outputDir)


        outputFiles, totalCounts = ph.integrateAll("T", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 2, 6.743542914938338e-08, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("E", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 2, 5.117216049208847e-09, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("TE", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 2, 9.210988888575927e-08, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("F", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 2, 3.746412730521299e-09, totalCounts)


        ph = Photometry2(dataDirGrb, outputDir)

        outputFiles, totalCounts = ph.integrateAll("T", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 1, 5.6058130624054985e-08, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("E", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 1, 3.766885883843366e-09, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("TE", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 1, 6.780394590918058e-08, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("F", customRegion, regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 1, 3.114340590225277e-09, totalCounts)
