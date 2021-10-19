from pathlib import Path
import matplotlib.pyplot as plt

from rtapipe.lib.plotting.PhotometryPlot import PhotometryPlot

class PhotometrySinglePlot(PhotometryPlot):
    
    def __init__(self, title = None):
        super().__init__(title)
        self.fig = None
        self.title = title
        self.outputfile = None
        self.data = []
        self.labels = []
        self.markers = []
        self.colors = []

    def getWindowSizes(self, dataframe):
        t_window_size = None
        e_window_size = None

        if "TCENTER" in dataframe.columns:
            t_window_size = dataframe["TMAX"][0] - dataframe["TMIN"][0]
        
        if "ECENTER" in dataframe.columns:
            e_window_size = dataframe["EMAX"][0] - dataframe["EMIN"][0]

        return (t_window_size, e_window_size)

    def getEnergyBinsFromDataframe(self, dataframe):
        energyBins = set()
        for eminDf in dataframe.groupby(["EMIN", "EMAX"]):
            energyBins.add(f"{eminDf[0][0]}-{eminDf[0][1]}")
        return energyBins



    def addData(self, photometryCsvFile, labelPrefix="", marker=".", color="blue"):
        
        dataframe = super().getData(photometryCsvFile)
                
        # assert dataframe["TCENTER"].isnull().values.any() == False or dataframe["ECENTER"].isnull().values.any() == False
        
        self.data.append(dataframe)
        self.labels.append(labelPrefix)
        self.markers.append(marker)
        self.colors.append(color)


    def plotScatter(self, axesID, integration, verticalLine=False, verticalLineX=None, plotError=True):
        
        #assert axesID >= 0 and axesID <= 1 
        #self.axes[axesID].clear()

        self.fig, axes  = plt.subplots(4,1, constrained_layout=True, figsize=(10,10))
        self.fig.suptitle(self.title, fontsize=15)
        self.fig.set_size_inches(PhotometryPlot.inch_x, PhotometryPlot.inch_y)
        axesID = 0

        if verticalLine:
            for ax in axes:
                _ = ax.axvline(x=verticalLineX, color="red", linestyle="--")

        
        for ii in range(len(self.data)): 
        
            dataframe = self.data[ii]
            label = self.labels[ii]
            marker = self.markers[ii]
            color = self.colors[ii]
            markerSize = 5

            t_window_size, e_window_size = self.getWindowSizes(dataframe)

            if integration == "T":

                label = f"{label}"
                axes[axesID].title.set_text('TIME integration')
                _ = axes[axesID].scatter(dataframe["TCENTER"], dataframe["COUNTS"], s=markerSize, color=color, label=label, marker=marker)
                if plotError:
                    _ = axes[axesID].errorbar(dataframe["TCENTER"], dataframe["COUNTS"], xerr=t_window_size/2, yerr=dataframe["ERROR"], fmt="o", color=color) 
                else:
                    _ = axes[axesID].errorbar(dataframe["TCENTER"], dataframe["COUNTS"], xerr=t_window_size/2, fmt="o", color=color)


                axes[0].set_ylabel('Gamma Photons Counts')
                axes[0].set_xlabel(f'Window center')
                axes[0].legend(loc="best")

            elif integration == "E":

                label = f"{label}"
                axes[axesID].title.set_text('ENERGY integration')
                binSize = (dataframe["EMAX"] - dataframe["EMIN"]) / 2
                _ = axes[axesID].scatter(dataframe["ECENTER"], dataframe["COUNTS"], s=markerSize, color=color, label=label, marker=marker)
                if plotError:
                    _ = axes[axesID].errorbar(dataframe["ECENTER"], dataframe["COUNTS"], xerr=binSize, yerr=dataframe["ERROR"], fmt="o", color=color) 

                axes[0].set_ylabel('Gamma Photons Counts')
                axes[0].set_xlabel(f'Window center')
                axes[0].legend(loc="best")

            elif integration == "TE":

                dataColNames = [col for col in dataframe.columns if "COUNTS" in col]
                errorDataColNames = [col for col in dataframe.columns if "ERROR" in col]

                for xx, dataColName in enumerate(dataColNames):
                    
                    axes[xx].title.set_text(f'ENERGY {dataColName}')
                    axes[xx].set_ylabel('Gamma Photons Counts')
                    axes[xx].set_xlabel(f'Window center')
                                        
                    errorColName = errorDataColNames[xx]
                    label = f"{dataColName} TeV"

                    _ = axes[xx].scatter(dataframe["TCENTER"], dataframe[dataColName], s=markerSize, label=label, marker=marker, color=color)

                    if plotError:
                        _ = axes[xx].errorbar(dataframe["TCENTER"], dataframe[dataColName], xerr=t_window_size/2, yerr=dataframe[errorColName], fmt="o", color=color) 
                    else:
                        _ = axes[xx].errorbar(dataframe["TCENTER"], dataframe[dataColName], xerr=t_window_size/2, fmt="o", color=color)

                    axes[xx].legend(loc="best")

            
            else:
                raise ValueError(f"integration value {integration} is not supported!")



        
        return None


    def show(self):
        plt.show()

    def save(self, outputDir, outputFilename):
        outputDir = Path(outputDir)
        outputDir.mkdir(parents=True, exist_ok=True)
        outputFilePath = outputDir.joinpath(outputFilename).with_suffix(".png")
        self.fig.savefig(str(outputFilePath), dpi=600)
        print(f"Produced: {outputFilePath}")
        return str(outputFilePath)










    # DEPRECATED
    def plotHist(self, axesID, integration, bins=None, verticalLine=False, verticalLineX=None):
        
        assert axesID >= 0 and axesID <= 1 
        self.axes[axesID].clear()

        for ii in range(len(self.data)): 
        
            dataframe = self.data[ii]
            label = self.labels[ii]

            if integration   == "T":
                self.axes[axesID].title.set_text('TIME integration') 
                _ = self.axes[axesID].hist(dataframe["COUNTS"], bins=bins, alpha=0.5, label=label)
                #_ = self.axes[axesID].errorbar(dataframe["TCENTER"], dataframe["COUNTS"], xerr=t_window_size/2, yerr=dataframe["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
            
            elif integration == "E":
                self.axes[axesID].title.set_text('ENERGY integration')
                binSize = (dataframe["EMAX"] - dataframe["EMIN"]) / 2
                _ = self.axes[axesID].hist(dataframe["COUNTS"], bins=bins, alpha=0.5, label=label)
                #_ = self.axes[axesID].errorbar(dataframe["ECENTER"], dataframe["COUNTS"], xerr=binSize, yerr=dataframe["ERROR"], fmt="o", color=PhotometryPlot.colors[self.ccount]) 
            
            elif integration == "TE":

                self.axes[axesID].title.set_text('TIME-ENERGY integration')

                print("\nDataframe: \n", dataframe)

                
                """
                for eminDf in dataframe.groupby(["EMIN", "EMAX"]):
                    energyBin = eminDf[0]
                    data = eminDf[1]
                    print("\nenergyBin: ",energyBin)
                    print(data)
                    label_eb = f"{label} - {energyBin} TeV"

                    _ = self.axes[axesID].hist(data["COUNTS"], bins=bins, alpha=0.5, label=label_eb) #,color=PhotometryPlot.colors[self.ccount])
                    #_ = self.axes[axesID].errorbar(data["TCENTER"], data["COUNTS"], xerr=t_window_size/2, yerr=data["ERROR"], fmt="o") #, color=PhotometryPlot.colors[self.ccount]) 
                    self.ccount += 1
                """

                """
                textstr = "Energy bins:\n"
                textstr += "\n".join(list(self.getEnergyBinsFromDataframe(dataframe)))
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
                self.axes[0].text(0.05, 0.95, textstr, transform=self.axes[0].transAxes, fontsize=12,
                        verticalalignment='top', bbox=props)
                """
            else:
                raise ValueError(f"integration value {integration} is not supported!")


            self.ccount += 1

            # self.axes[0].set_title(self.getTitle(label_on, args))
            self.axes[axesID].set_ylabel('Gamma Photons Counts')
            self.axes[axesID].set_xlabel(f'Counts')
            self.axes[axesID].legend(loc="best")






