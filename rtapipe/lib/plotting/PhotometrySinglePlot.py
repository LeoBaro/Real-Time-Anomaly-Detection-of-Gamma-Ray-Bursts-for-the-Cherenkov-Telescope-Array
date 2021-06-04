from pathlib import Path
import matplotlib.pyplot as plt

from rtapipe.lib.plotting.PhotometryPlot import PhotometryPlot

class PhotometrySinglePlot(PhotometryPlot):
    
    def __init__(self, title=None):
        super().__init__(title)
        self.FS, self.FSX  = plt.subplots()
        self.FS.set_size_inches(PhotometryPlot.inch_x, PhotometryPlot.inch_y)
        self.outputfile = None

    def getWindowSizes(self, dataframe):
        t_window_size = None
        e_window_size = None
        
        if not dataframe["TCENTER"].isnull().values.any():
            t_window_size = dataframe["TMAX"][0] - dataframe["TMIN"][0]
        
        if not dataframe["ECENTER"].isnull().values.any():
            e_window_size = dataframe["EMAX"][0] - dataframe["EMIN"][0]

        return (t_window_size, e_window_size)

    def addData(self, photometryCsvFile, integration, labelPrefix="", verticalLine=False, verticalLineX=None, as_baseline=False, baseline_color="black"):
        
        dataframe = super().getData(photometryCsvFile)
                
        assert dataframe["TCENTER"].isnull().values.any() == False or dataframe["ECENTER"].isnull().values.any() == False
            
        t_window_size, e_window_size = self.getWindowSizes(dataframe)


        if integration   == "T":
            label = f"{labelPrefix}"
            _ = self.FSX.scatter(dataframe["TCENTER"], dataframe["COUNTS"], s=0.1, color=PhotometryPlot.colors[self.ccount], label=label)
            _ = self.FSX.errorbar(dataframe["TCENTER"], dataframe["COUNTS"], xerr=t_window_size/2, yerr=dataframe["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
        
        elif integration == "E":
            label = f"{labelPrefix}"
            _ = self.FSX.scatter(dataframe["ECENTER"], dataframe["COUNTS"], s=0.1, color=PhotometryPlot.colors[self.ccount], label=label)
            _ = self.FSX.errorbar(dataframe["ECENTER"], dataframe["COUNTS"], xerr=e_window_size/2, yerr=dataframe["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
        
        elif integration == "T_E":

            for eminDf in dataframe.groupby(["EMIN", "EMAX"]):
                label = f"{labelPrefix} {eminDf[0]} TeV"
                _ = self.FSX.scatter(eminDf[1]["TCENTER"], eminDf[1]["COUNTS"], s=0.1, color=PhotometryPlot.colors[self.ccount], label=label)
                _ = self.FSX.errorbar(eminDf[1]["TCENTER"], eminDf[1]["COUNTS"], xerr=t_window_size/2, yerr=eminDf[1]["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 

        if verticalLine:
            _ = self.FSX.axvline(x=verticalLineX, color="red", linestyle="--")

        self.ccount += 1

        # self.FSX.set_title(self.getTitle(label_on, args))
        self.FSX.set_ylabel('Counts')
        self.FSX.set_xlabel(f'Window center (integration: "{integration}")')
        self.FSX.legend(loc="best")

    def show(self):
        plt.show()

    def save(self, outputDir, outputFilename):
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)
        outputFilePath = outputDir.joinpath(outputFilename).with_suffix(".png")
        self.FS.savefig(str(outputFilePath))
        print(f"Produced: {outputFilePath}")
        return str(outputFilePath)