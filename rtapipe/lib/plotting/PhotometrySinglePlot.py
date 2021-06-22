from pathlib import Path
import matplotlib.pyplot as plt

from rtapipe.lib.plotting.PhotometryPlot import PhotometryPlot

class PhotometrySinglePlot(PhotometryPlot):
    
    def __init__(self, title = None):
        super().__init__(title)
        self.fig, self.axes  = plt.subplots(2,1)
        self.fig.set_size_inches(PhotometryPlot.inch_x, PhotometryPlot.inch_y)
        self.outputfile = None
        self.data = []
        self.labels = []

    def getWindowSizes(self, dataframe):
        t_window_size = None
        e_window_size = None
        
        if not dataframe["TCENTER"].isnull().values.any():
            t_window_size = dataframe["TMAX"][0] - dataframe["TMIN"][0]
        
        if not dataframe["ECENTER"].isnull().values.any():
            e_window_size = dataframe["EMAX"][0] - dataframe["EMIN"][0]

        return (t_window_size, e_window_size)

    def getEnergyBinsFromDataframe(self, dataframe):
        energyBins = set()
        for eminDf in dataframe.groupby(["EMIN", "EMAX"]):
            energyBins.add(f"{eminDf[0][0]}-{eminDf[0][1]}")
        return energyBins



    def addData(self, photometryCsvFile, labelPrefix=""):
        
        dataframe = super().getData(photometryCsvFile)
                
        assert dataframe["TCENTER"].isnull().values.any() == False or dataframe["ECENTER"].isnull().values.any() == False
        
        self.data.append(dataframe)
        self.labels.append(labelPrefix)


    def plotScatter(self, axesID, integration, verticalLine=False, verticalLineX=None):
        
        assert axesID >= 0 and axesID <= 1 
        self.axes[axesID].clear()

        for ii in range(len(self.data)): 
        
            dataframe = self.data[ii]
            label = self.labels[ii]

            t_window_size, e_window_size = self.getWindowSizes(dataframe)

            if integration   == "T":
                label = f"{label}"
                _ = self.axes[axesID].scatter(dataframe["TCENTER"], dataframe["COUNTS"], s=0.1, color=PhotometryPlot.colors[self.ccount], label=label)
                _ = self.axes[axesID].errorbar(dataframe["TCENTER"], dataframe["COUNTS"], xerr=t_window_size/2, yerr=dataframe["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
            
            elif integration == "E":
                label = f"{label}"
                binSize = (dataframe["EMAX"] - dataframe["EMIN"]) / 2
                _ = self.axes[axesID].scatter(dataframe["ECENTER"], dataframe["COUNTS"], s=0.1, color=PhotometryPlot.colors[self.ccount], label=label)
                _ = self.axes[axesID].errorbar(dataframe["ECENTER"], dataframe["COUNTS"], xerr=binSize, yerr=dataframe["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
            
            elif integration == "T_E":

                for eminDf in dataframe.groupby(["EMIN", "EMAX"]):
                    energyBin = eminDf[0]
                    data = eminDf[1]
                    label = f"{energyBin} TeV"
                    _ = self.axes[axesID].scatter(data["TCENTER"], data["COUNTS"], s=0.1, label=label) #,color=PhotometryPlot.colors[self.ccount])
                    _ = self.axes[axesID].errorbar(data["TCENTER"], data["COUNTS"], xerr=t_window_size/2, yerr=data["ERROR"], fmt="o") #, color=PhotometryPlot.colors[self.ccount]) 
                    self.ccount += 1

                """
                textstr = "Energy bins:\n"
                textstr += "\n".join(list(self.getEnergyBinsFromDataframe(dataframe)))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                self.axes[0].text(0.05, 0.95, textstr, transform=self.axes[0].transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
                """
            if verticalLine:
                _ = self.axes[axesID].axvline(x=verticalLineX, color="red", linestyle="--")

            self.ccount += 1

            # self.axes[0].set_title(self.getTitle(label_on, args))
            self.axes[axesID].set_ylabel('Counts')
            self.axes[axesID].set_xlabel(f'Window center (integration: "{integration}")')
            self.axes[axesID].legend(loc="best")
        
        return None

    def plotHist(self, axesID, integration, bins=None, verticalLine=False, verticalLineX=None):
        
        assert axesID >= 0 and axesID <= 1 
        self.axes[axesID].clear()

        for ii in range(len(self.data)): 
        
            dataframe = self.data[ii]
            label = self.labels[ii]

            if integration   == "T":
                _ = self.axes[axesID].hist(dataframe["COUNTS"], bins=bins, alpha=0.5, label=label)
                #_ = self.axes[axesID].errorbar(dataframe["TCENTER"], dataframe["COUNTS"], xerr=t_window_size/2, yerr=dataframe["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
            
            elif integration == "E":
                binSize = (dataframe["EMAX"] - dataframe["EMIN"]) / 2
                _ = self.axes[axesID].hist(dataframe["COUNTS"], bins=bins, alpha=0.5, label=label)
                #_ = self.axes[axesID].errorbar(dataframe["ECENTER"], dataframe["COUNTS"], xerr=binSize, yerr=dataframe["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
            
            elif integration == "T_E":

                for eminDf in dataframe.groupby(["EMIN", "EMAX"]):
                    energyBin = eminDf[0]
                    data = eminDf[1]
                    label_eb = f"{label} - {energyBin} TeV"

                    _ = self.axes[axesID].hist(data["COUNTS"], bins=bins, alpha=0.5, label=label_eb) #,color=PhotometryPlot.colors[self.ccount])
                    #_ = self.axes[axesID].errorbar(data["TCENTER"], data["COUNTS"], xerr=t_window_size/2, yerr=data["ERROR"], fmt="o") #, color=PhotometryPlot.colors[self.ccount]) 
                    self.ccount += 1

                """
                textstr = "Energy bins:\n"
                textstr += "\n".join(list(self.getEnergyBinsFromDataframe(dataframe)))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                self.axes[0].text(0.05, 0.95, textstr, transform=self.axes[0].transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
                """
            if verticalLine:
                _ = self.axes[axesID].axvline(x=verticalLineX, color="red", linestyle="--")

            self.ccount += 1

            # self.axes[0].set_title(self.getTitle(label_on, args))
            self.axes[axesID].set_ylabel('Counts')
            self.axes[axesID].set_xlabel(f'Window center (integration: "{integration}")')
            self.axes[axesID].legend(loc="best")

    def show(self):
        plt.show()

    def save(self, outputDir, outputFilename):
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)
        outputFilePath = outputDir.joinpath(outputFilename).with_suffix(".png")
        self.fig.savefig(str(outputFilePath))
        print(f"Produced: {outputFilePath}")
        return str(outputFilePath)