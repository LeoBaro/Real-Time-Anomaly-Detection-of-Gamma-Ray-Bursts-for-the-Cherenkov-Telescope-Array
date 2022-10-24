from pathlib import Path

from rtapipe.lib.plotting.PhotometrySinglePlot import PhotometrySinglePlot

class TestPhotometryPlot:

    def test_getData(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv_v2")
        singleplot = PhotometrySinglePlot()

        inputFile = dataDir.joinpath("bkg000002_t_simtype_bkg_onset_0.csv")

        dataframe = singleplot.getData(inputFile)

        assert list(dataframe.columns.values) == ["TMIN","TMAX","COUNTS","ERROR","TCENTER"]
        assert dataframe.isnull().values.any() == False


        inputFile = dataDir.joinpath("bkg000002_e_simtype_bkg_onset_0.csv")

        dataframe = singleplot.getData(inputFile)

        assert list(dataframe.columns.values) == ["EMIN","EMAX","COUNTS","ERROR","ECENTER"]
        assert dataframe.isnull().values.any() == False


        inputFile = dataDir.joinpath("bkg000002_te_simtype_bkg_onset_0.csv")

        dataframe = singleplot.getData(inputFile)

        assert list(dataframe.columns.values) == ["TMIN","TMAX","COUNTS_0.03-0.0449","ERROR_0.03-0.0449","COUNTS_0.0449-0.0671","ERROR_0.0449-0.0671","COUNTS_0.0671-0.1003","ERROR_0.0671-0.1003","COUNTS_0.1003-0.15","ERROR_0.1003-0.15","TCENTER"]
        assert dataframe.isnull().values.any() == False
        


    def test_getWindowSizes(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv_v2")
        singleplot = PhotometrySinglePlot()

        inputFile = dataDir.joinpath("bkg000002_t_simtype_bkg_onset_0.csv")

        dataframe = singleplot.getData(inputFile)

        t_window_size, e_window_size = singleplot.getWindowSizes(dataframe)

        assert t_window_size == 100
        assert e_window_size == None



        inputFile = dataDir.joinpath("bkg000002_e_simtype_bkg_onset_0.csv")

        dataframe = singleplot.getData(inputFile)

        t_window_size, e_window_size = singleplot.getWindowSizes(dataframe)

        assert t_window_size == None
        assert e_window_size != None



        inputFile = dataDir.joinpath("bkg000002_te_simtype_bkg_onset_0.csv")

        dataframe = singleplot.getData(inputFile)

        t_window_size, e_window_size = singleplot.getWindowSizes(dataframe)

        assert t_window_size == 100
        assert e_window_size == None


    def test_singlePlot_time_integration_not_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv_v2")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_t_simtype_bkg_onset_0.csv")
        inputFileGrb = dataDir.joinpath("grb000002_t_simtype_grb_onset_900.csv")

        singleplot = PhotometrySinglePlot(title="Time integration")
                
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background", marker="x", color="blue")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)", marker="D", color="red")

        _ = singleplot.plotScatter(0, "T", verticalLine=True, verticalLineX=900)
        
        singleplot.save(outputDir, "test_singlePlot_time_integration_scatter_and_histo")  


    def test_singlePlot_time_integration_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv_normalized")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_t_simtype_bkg_onset_0_normalized_True.csv")
        inputFileGrb = dataDir.joinpath("grb000002_t_simtype_grb_onset_900_normalized_True.csv")

        singleplot = PhotometrySinglePlot(title="Time integration")
                
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background", marker="x", color="blue")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)", marker="D", color="red")

        _ = singleplot.plotScatter(0, "T", plotError=False, verticalLine=True, verticalLineX=900)
        
        singleplot.save(outputDir, "test_singlePlot_time_integration_scatter_and_histo_normalized")  


    def test_singlePlot_energy_integration_not_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv_v2")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_e_simtype_bkg_onset_0.csv")
        inputFileGrb = dataDir.joinpath("grb000002_e_simtype_grb_onset_900.csv")

        singleplot = PhotometrySinglePlot(title="Integration on ENERGY")
                
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background", marker="x", color="blue")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)", marker="D", color="red")

        _ = singleplot.plotScatter(0, "E")

        singleplot.save(outputDir, "test_singlePlot_energy_integration_scatter_and_histo")  
        

    def test_singlePlot_energy_integration_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv_normalized")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_e_simtype_bkg_onset_0_normalized_True.csv")
        inputFileGrb = dataDir.joinpath("grb000002_e_simtype_grb_onset_900_normalized_True.csv")

        singleplot = PhotometrySinglePlot(title="Integration on ENERGY")
                
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background", marker="x", color="blue")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)", marker="D", color="red")

        _ = singleplot.plotScatter(0, "E")

        singleplot.save(outputDir, "test_singlePlot_energy_integration_scatter_and_histo_normalized")  


    def test_singlePlot_time_energy_integration_not_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv_v2")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_te_simtype_bkg_onset_0.csv")
        inputFileGrb = dataDir.joinpath("grb000002_te_simtype_grb_onset_900.csv")

        singleplot = PhotometrySinglePlot(title="Single trial")
        
        #  (self, photometryCsvFile, integration, labelOn, vertical_line=False, vertical_line_x=None, as_baseline=False, baseline_color="black"):
        
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background", marker="x", color="blue")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)", marker="D", color="red")

        _ = singleplot.plotScatter(0, "TE", verticalLine=True, verticalLineX=900)

        singleplot.save(outputDir, "test_singlePlot_time_energy_integration_scatter_and_histo")  
        


    def test_singlePlot_time_energy_integration_normalized(self):

        dataDir = Path(__file__).parent.joinpath("test_data", "csv_normalized")
        outputDir = Path(__file__).parent.joinpath("photometry_single_plot_test_output", "test_plot")

        inputFileBkg = dataDir.joinpath("bkg000002_te_simtype_bkg_onset_0_normalized_True.csv")
        inputFileGrb = dataDir.joinpath("grb000002_te_simtype_grb_onset_900_normalized_True.csv")

        singleplot = PhotometrySinglePlot(title="Single trial")
        
        #  (self, photometryCsvFile, integration, labelOn, vertical_line=False, vertical_line_x=None, as_baseline=False, baseline_color="black"):
        
        _ = singleplot.addData(inputFileBkg, labelPrefix="Background", marker="x", color="blue")
        _ = singleplot.addData(inputFileGrb, labelPrefix="GRB (onset 900s)", marker="D", color="red")

        _ = singleplot.plotScatter(0, "TE", plotError=False, verticalLine=True, verticalLineX=900)

        singleplot.save(outputDir, "test_singlePlot_time_energy_integration_scatter_and_histo_normalized")  
