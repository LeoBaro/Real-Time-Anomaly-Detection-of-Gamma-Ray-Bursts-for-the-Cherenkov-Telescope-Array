import pytest
from pathlib import Path

from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot

class TestPhotometryPlot:

    def test_getData(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv")
        singleplot = PhotometrySinglePlot()

        inputFile = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_t.csv")

        dataframe = singleplot.getData(inputFile)

        assert dataframe["TCENTER"].isnull().values.any() == False

        assert dataframe["ECENTER"].isnull().values.all() == True


        inputFile = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_e.csv")

        dataframe = singleplot.getData(inputFile)

        assert dataframe["TCENTER"].isnull().values.any() == True

        assert dataframe["ECENTER"].isnull().values.all() == False


        inputFile = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_t_e.csv")

        dataframe = singleplot.getData(inputFile)

        assert dataframe["TCENTER"].isnull().values.any() == False

        assert dataframe["ECENTER"].isnull().values.all() == False
        
        print("\n",dataframe)


    def test_getWindowSizes(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv")
        singleplot = PhotometrySinglePlot()

        inputFile = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_t.csv")

        dataframe = singleplot.getData(inputFile)

        t_window_size, e_window_size = singleplot.getWindowSizes(dataframe)

        assert t_window_size == 100
        assert e_window_size == None



        inputFile = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_e.csv")

        dataframe = singleplot.getData(inputFile)

        t_window_size, e_window_size = singleplot.getWindowSizes(dataframe)

        assert t_window_size == None
        assert e_window_size != None



        inputFile = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_t_e.csv")

        dataframe = singleplot.getData(inputFile)

        t_window_size, e_window_size = singleplot.getWindowSizes(dataframe)

        assert t_window_size == 100
        assert e_window_size != None


    def test_singlePlot_time_integration(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_t.csv")
        inputFileGrb = dataDir.joinpath("grb000002_simtype_bkg_onset_0_integration_t.csv")

        singleplot = PhotometrySinglePlot(title="Time integration")
        
        #  (self, photometryCsvFile, integration, labelOn, vertical_line=False, vertical_line_x=None, as_baseline=False, baseline_color="black"):
        
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)")

        _ = singleplot.plotScatter(0, "T")
        _ = singleplot.plotHist(1, "T")
        
        singleplot.save(outputDir, "test_singlePlot_time_integration_scatter_and_histo")  



    def test_singlePlot_energy_integration(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_e.csv")
        inputFileGrb = dataDir.joinpath("grb000002_simtype_bkg_onset_0_integration_e.csv")

        singleplot = PhotometrySinglePlot(title="Integration on ENERGY")
        
        #  (self, photometryCsvFile, integration, labelOn, vertical_line=False, vertical_line_x=None, as_baseline=False, baseline_color="black"):
        
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)")

        _ = singleplot.plotScatter(0, "E")
        _ = singleplot.plotHist(1, "E")

        singleplot.save(outputDir, "test_singlePlot_energy_integration_scatter_and_histo")  
        

    def test_singlePlot_time_energy_integration(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_simtype_bkg_onset_0_integration_t_e.csv")
        inputFileGrb = dataDir.joinpath("grb000002_simtype_bkg_onset_0_integration_t_e.csv")

        singleplot = PhotometrySinglePlot(title="Single trial")
        
        #  (self, photometryCsvFile, integration, labelOn, vertical_line=False, vertical_line_x=None, as_baseline=False, baseline_color="black"):
        
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)")

        _ = singleplot.plotScatter(0, "T_E")
        _ = singleplot.plotHist(1, "T_E")

        singleplot.save(outputDir, "test_singlePlot_time_energy_integration_scatter_and_histo")  
        
                