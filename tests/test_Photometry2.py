import pandas as pd
from time import time
from os import listdir
from pathlib import Path
from shutil import rmtree
from astropy.io import fits
from datetime import datetime
from rtapipe.lib.datasource.Photometry2 import Photometry2
from rtapipe.lib.datasource.integrationstrat.IntegrationStrategies import IntegrationType

tolerance = 1e-14

class TestPhotometry2:



    def read_photons_list_rows(self, fitsFile):
        with fits.open(fitsFile) as hdul:
            return len(hdul[1].data)

    def count_photons(self, fitsFile):
        with fits.open(fitsFile) as hdul:
            return len(hdul[1].data["TIME"])

    def count_photons_in_csv(self, csv_file):
        df_counts = 0
        df = pd.read_csv(csv_file, sep=",")
        for col in df.columns:
            if "COUNTS" in col:
                df_counts += sum(df[col])
        return df_counts
        
    def destroy_output(self, outputDir):
        if outputDir.exists():
            rmtree(outputDir)
        outputDir.mkdir(parents=True)        

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


        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_only_single_file")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output")

        ph = Photometry2(dataDir, outputDir)

        # assert len(ph.dataFiles) == 1


    def test_getOutputFilePath(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output")
        self.destroy_output(outputDir)
        inputFilename = dataDir.joinpath("bkg000001.fits")

        ph = Photometry2(dataDir, outputDir)

        outputFilePath = ph.getOutputFilePath(inputFilename, IntegrationType.TIME, True)
        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_t_simtype_bkg_onset_0_normalized_True.csv"

        outputFilePath = ph.getOutputFilePath(inputFilename, IntegrationType.ENERGY, True)
        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_e_simtype_bkg_onset_0_normalized_True.csv"

        outputFilePath = ph.getOutputFilePath(inputFilename, IntegrationType.TIME_ENERGY, True)
        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_te_simtype_bkg_onset_0_normalized_True.csv"

        outputFilePath = ph.getOutputFilePath(inputFilename, IntegrationType.FULL, True)
        assert f"{outputFilePath}" == f"{outputDir}/bkg000002_full_simtype_bkg_onset_0_normalized_True.csv"

    ################################################################################
    # NOT NORMALIZED 
    ################################################################################

    def test_integrate_full_not_normalized(self):
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_full_not_normalized")
        self.destroy_output(outputDir)

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only_single_file")
        dataFile = dataDir.joinpath("bkg000001.fits")

        ph = Photometry2(dataDir, outputDir)
        computedOutputFilesNumber, computedTotalCounts = ph.integrateAll("F", 10, tWindows=None, eWindows=None, normalize=False, limit=None, parallel=False)

        # Check output files
        realOutputFiles = listdir(outputDir)
        assert computedOutputFilesNumber == len(realOutputFiles)
        outputFile = realOutputFiles[0]
        assert ".csv" in outputFile

        # Check counts
        realTotalCounts = self.count_photons(dataFile)
        csvTotalCounts = self.count_photons_in_csv(outputDir.joinpath(outputFile))
        assert computedTotalCounts == realTotalCounts
        assert csvTotalCounts == realTotalCounts
        print(f"[BKG] Real total counts: {realTotalCounts}, Csv Total Counts: {csvTotalCounts}, Computed Total Counts: {computedTotalCounts}")

        self.destroy_output(outputDir)
        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_only_single_file")
        dataFile = dataDir.joinpath("grb000001.fits")

        ph = Photometry2(dataDir, outputDir)
        computedOutputFilesNumber, computedTotalCounts = ph.integrateAll("F", 10, tWindows=None, eWindows=None, normalize=False, limit=None, parallel=False)

        # Check output files
        realOutputFiles = listdir(outputDir)
        assert computedOutputFilesNumber == len(realOutputFiles)
        outputFile = realOutputFiles[0]
        assert ".csv" in outputFile

        # Check counts
        realTotalCounts = self.count_photons(dataFile)
        csvTotalCounts = self.count_photons_in_csv(outputDir.joinpath(outputFile))
        assert computedTotalCounts == realTotalCounts
        assert csvTotalCounts == realTotalCounts
        print(f"[GRB] Real total counts: {realTotalCounts}, Csv Total Counts: {csvTotalCounts}, Computed Total Counts: {computedTotalCounts}")


    def test_integrate_t_not_normalized_bkg(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only_single_file")
        dataFile = dataDir.joinpath("bkg000001.fits")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_t_not_normalized_bkg")
        self.destroy_output(outputDir)

        ph = Photometry2(dataDir, outputDir)

        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)

        computedOutputFilesNumber, computedTotalCounts = ph.integrateAll("T", 10, tWindows=tWindows, eWindows=None, limit=None, parallel=False, normalize=False)

        # Check output files
        realOutputFiles = listdir(outputDir)
        assert computedOutputFilesNumber == len(realOutputFiles)
        outputFile = realOutputFiles[0]
        assert ".csv" in outputFile

        # Check counts
        realTotalCounts = self.count_photons(dataFile)
        csvTotalCounts = self.count_photons_in_csv(outputDir.joinpath(outputFile))
        assert computedTotalCounts == realTotalCounts
        assert csvTotalCounts == realTotalCounts

        

    def test_integrate_te_not_normalized_bkg(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only_single_file")
        dataFile = dataDir.joinpath("bkg000001.fits")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_te_not_normalized_bkg")
        self.destroy_output(outputDir)

        ph = Photometry2(dataDir, outputDir)

        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        computedOutputFilesNumber, computedTotalCounts = ph.integrateAll("TE", 10, tWindows=tWindows, eWindows=eWindows, limit=None, parallel=False, normalize=False)

        # Check output files
        realOutputFiles = listdir(outputDir)
        assert computedOutputFilesNumber == len(realOutputFiles)
        outputFile = realOutputFiles[0]
        assert ".csv" in outputFile

        # Check counts
        realTotalCounts = self.count_photons(dataFile)
        csvTotalCounts = self.count_photons_in_csv(outputDir.joinpath(outputFile))
        assert computedTotalCounts == realTotalCounts
        assert csvTotalCounts == realTotalCounts



    def test_integrate_te_not_normalized_grb(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_only_single_file")
        dataFile = dataDir.joinpath("grb000001.fits")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_te_not_normalized_grb")
        self.destroy_output(outputDir)


        ph = Photometry2(dataDir, outputDir)

        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)

        computedOutputFilesNumber, computedTotalCounts = ph.integrateAll("TE", 10, tWindows=tWindows, eWindows=eWindows, limit=None, parallel=False, normalize=False)

        # Check output files
        realOutputFiles = listdir(outputDir)
        assert computedOutputFilesNumber == len(realOutputFiles)
        outputFile = realOutputFiles[0]
        assert ".csv" in outputFile

        # Check counts
        realTotalCounts = self.count_photons(dataFile)
        csvTotalCounts = self.count_photons_in_csv(outputDir.joinpath(outputFile))
        assert computedTotalCounts == realTotalCounts
        assert csvTotalCounts == realTotalCounts




    ################################################################################
    # NORMALIZED 
    ################################################################################

    def test_integrate_full_normalized(self):
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_full_normalized")
        self.destroy_output(outputDir)


        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only_single_file")
        ph = Photometry2(dataDir, outputDir)
        outputFiles, totalCounts = ph.integrateAll("F", 0.2, tWindows=None, eWindows=None, limit=None, parallel=False, normalize=True)
        assert (totalCounts - 1.831337521220006e-09) < tolerance
        assert outputFiles == 1


        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_only_single_file")
        self.destroy_output(outputDir)
        ph = Photometry2(dataDir, outputDir)
        outputFiles, totalCounts = ph.integrateAll("F", 0.2, tWindows=None, eWindows=None, limit=None, parallel=False, normalize=True)
        assert (totalCounts - 3.114340590225277e-09) < tolerance





    def test_integrate_t_normalized_bkg(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only_single_file")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_t_normalized_bkg")
        self.destroy_output(outputDir)


        ph = Photometry2(dataDir, outputDir)

        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)

        outputFiles, totalCounts = ph.integrateAll("T", 0.2, tWindows=tWindows, eWindows=None, normalize=True, limit=None, parallel=False)

        assert (totalCounts - 3.296407538196011e-08) < tolerance
        assert outputFiles == 1

        for fileName in listdir(outputDir):
            assert ".csv" in fileName
            assert pd.read_csv(outputDir.joinpath(fileName), sep=",").isnull().values.any() == False






    """
    def test_integrate_te_normalized_grb(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "grb_only_single_file")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_te_normalized_grb")
        try: rmtree(outputDir)
        except: pass
        outputDir.mkdir(parents=True)

        ph = Photometry2(dataDir, outputDir)

        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)
        print("eWindows",eWindows)
        outputFiles, totalCounts = ph.integrateAll("TE", 0.2, tWindows=tWindows, eWindows=eWindows, normalize=True, limit=None, parallel=False)

        assert (totalCounts - 4.881458728926591e-08) < tolerance
        assert outputFiles == 1

        for fileName in listdir(outputDir):
            assert ".csv" in fileName
            assert pd.read_csv(outputDir.joinpath(fileName), sep=",").isnull().values.any() == False




    

    def checkOutput(self, outputFiles, expected_files, counts_expected, counts_found):
        assert len(outputFiles) == expected_files
        for outputFilePath in outputFiles:
            assert Path(outputFilePath).is_file() == True
            assert pd.read_csv(outputFilePath, sep=",").isnull().values.any() == False
        assert (counts_expected - counts_found) <= tolerance


    def test_integrate_all_not_normalized(self):

        dataDirBkg = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        dataDirGrb = Path(__file__).parent.joinpath("test_data", "fits", "grb_only_single_file")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_all")
        rmtree(outputDir)
        outputDir.mkdir(parents=True)

        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)


        regionRadius = 0.2


        ph = Photometry2(dataDirBkg, outputDir)

        outputFiles, totalCounts = ph.integrateAll("T", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 2, 3989, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("E", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 2, 3989, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("TE", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 2, 3989, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("F", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 2, 3989, totalCounts)


        ph = Photometry2(dataDirGrb, outputDir)

        outputFiles, totalCounts = ph.integrateAll("T", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 1, 3316, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("E", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 1, 3316, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("TE", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 1, 3316, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("F", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=False)
        self.checkOutput(outputFiles, 1, 3316, totalCounts)



    def test_integrate_all_normalized(self):

        dataDirBkg = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only")
        dataDirGrb = Path(__file__).parent.joinpath("test_data", "fits", "grb_only_single_file")

        outputDir = Path(__file__).parent.joinpath("photometry_test_output", "test_integrate_all")
        rmtree(outputDir)
        outputDir.mkdir(parents=True)

        tWindows = Photometry2.getLinearWindows(0, 1800, 100, 100)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)


        regionRadius = 0.2


        ph = Photometry2(dataDirBkg, outputDir)


        outputFiles, totalCounts = ph.integrateAll("T", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 2, 6.743542914938338e-08, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("E", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 2, 5.117216049208847e-09, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("TE", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 2, 9.210988888575927e-08, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("F", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 2, 3.746412730521299e-09, totalCounts)


        ph = Photometry2(dataDirGrb, outputDir)

        outputFiles, totalCounts = ph.integrateAll("T", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 1, 5.6058130624054985e-08, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("E", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 1, 3.766885883843366e-09, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("TE", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 1, 6.780394590918058e-08, totalCounts)

        outputFiles, totalCounts = ph.integrateAll("F", regionRadius, tWindows, eWindows, limit=None, parallel=False, normalize=True)
        self.checkOutput(outputFiles, 1, 3.114340590225277e-09, totalCounts)


    def test_benchmark_t_e_integration_without_normalization_with_offset(self):
        folderName = "test_benchmark_t_e_integration_without_normalization_with_offset_ver6"
        dataDir = Path(__file__).parent.joinpath("test_data", "fits", "bkg_only_benchmark_2")
        outputDir = Path(__file__).parent.joinpath("photometry_test_output", folderName)
        try: rmtree(outputDir)
        except: pass
        outputDir.mkdir(parents=True)

        tWindows = Photometry2.getLinearWindows(0, 100, 10, 10)
        eWindows = Photometry2.getLogWindows(0.03, 0.15, 4)
        print("eWindows",eWindows)
        customRegion = "pippoplutopaperino"
        rr = 0.2
        procnumber = 40
        normalize = False
        batchSize = 40

        ph = Photometry2(dataDir, outputDir)

        start = time()
        outputFiles, counts = ph.integrateAll("TE", rr, tWindows, eWindows, parallel=False, procNumber=procnumber, normalize=normalize, batchSize=batchSize)
        end = time() - start
        timeStr = f"[{datetime.now()}] Total time: {round(end,2)} seconds\n"
        with open(outputDir.parent.joinpath(f"{folderName}.txt"), "a") as bf:
            bf.write(timeStr)
        print(timeStr)
        assert len(listdir(outputDir)) == len(listdir(dataDir)) - 1

        """
