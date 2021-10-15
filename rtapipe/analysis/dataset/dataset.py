import numpy as np
import pandas as pd
from pathlib import Path
from random import randrange
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rtapipe.lib.rtapipeutils.WindowsExtractor import WindowsExtractor

class APDataset:
    """
    This class can handle the .csv files of the AP dataset (generated witht the ?? API).
    """
    def __init__(self, tobs, onset, integrationTime, integrationType, featureColsNamesPattern, uselessColsNamesPattern, outDir="./"):
        
        # The data container
        self.data = {}
        
        # Original shape informations
        self.filesLoaded = {}
        self.singleFileDataShapes = {}

        # Dataset informations
        self.tobs = tobs
        self.onset = onset
        self.integrationTime = int(integrationTime)
        self.integrationType = integrationType
        self.featureCols = None
        self.featureColsNamesPattern = featureColsNamesPattern
        self.uselessColsNamesPattern = uselessColsNamesPattern

        # Scalers
        self.mmscaler = MinMaxScaler(feature_range=(0,1))
        self.stdscaler = StandardScaler()
        
        self.outDir = Path(outDir)

    def setOutDir(self, outDir):
        self.outDir = Path(outDir)

    def getFeaturesColsNames(self):
        return self.featureCols

    def _loadDataInDirectory(self, label, dataDir):
        """
        This method will read the csv files within a direcotry, loading the contents into the 'data' dictionary.
        """
        dfs = [pd.read_csv(file, sep=",") for file in Path(dataDir).iterdir()]
        self.filesLoaded[label] = len(dfs)
        self.singleFileDataShapes[label] = dfs[0].shape
        concat = pd.concat(dfs)

        if label not in self.data:
            self.data[label] = concat
        else:
            self.data[label] = pd.concat([self.data[label], concat], axis=0)

        print(f"\tLoaded {len(dfs)} csv files into data[{label}]") 
        print(f"\tSingle csv file shape: {self.singleFileDataShapes[label]}")
        print(f"\tDataframe shape: {self.data[label].shape}. Columns: {self.data[label].columns.values}")

    def _findColumns(self, patterns):
        """
        This method will search for columns names from the columns patterns passed as input.
        """
        cols = set()
        for df in self.data.values():
            for col in df.columns.values:
                for pattern in patterns:
                    if pattern in col:
                        cols.add(col)
        return list(cols)

    def loadData(self, label, dataDir):
        """
        This method will load the data from multiple directories into a dictionary of label : dataframe        
        """
        print(f"Loading dataset with label '{label}'")
        self._loadDataInDirectory(label, dataDir)

        uselessCols = self._findColumns(self.uselessColsNamesPattern)
        
        # dropping useless column
        self.data[label].drop(uselessCols, axis='columns', inplace=True)
        print(f"\tDropped columns={uselessCols} from {label} dataset")
        
        self.featureCols = self._findColumns(self.featureColsNamesPattern)



    def getSampleFraction(self, total, percentage):
        return int(total*(percentage/100))
    

    def _splitArrayWithPercentage(self, arr, percentage):
        splitPoint1 = self.getSampleFraction(len(arr), percentage)
        return np.split(arr, [splitPoint1])

    def getTrainingAndValidationSet(self, windowSize, stride, split=50, scaler=None):
        """
        Exctract time series from the "bkg" data 
        """
        print("Exctracting training set..")

        print("\tFitting scalers..")
        self.mmscaler.fit(self.data["bkg"])
        self.stdscaler.fit(self.data["bkg"])

        data = self.data["bkg"].values

        if scaler and scaler == "mm":
            data = self.mmscaler.transform(data)

        if scaler and scaler == "std":
            data = self.stdscaler.transform(data)

        windows = WindowsExtractor.test_extract_sub_windows(data, 0, data.shape[0], windowSize, stride)

        labels = np.array([False for i in range(len(windows))])
        
        windows = self._splitArrayWithPercentage(windows, split)
        labels = self._splitArrayWithPercentage(labels, split)

        print(f"\tTraining set: {windows[0].shape} Labels: {labels[0].shape}")
        print(f"\tValidation set: {windows[1].shape} Labels: {labels[1].shape}")

        return windows[0], labels[0], windows[1], labels[1]

    def getTestSet(self, windowSize, stride, onset, scaler=None):
        """
        At the moment it extracts subwindows from 1 GRB sample only
        
        """
        print("Exctracting test set..")

        numberOfFiles = self.filesLoaded["grb"]

        data = self.data["grb"].values

        if scaler and scaler == "mm":
            data = self.mmscaler.transform(data)

        if scaler and scaler == "std":
            data = self.stdscaler.transform(data)

        print("dataset shape:",data.shape)

        data = data.reshape((numberOfFiles, self.tobs, data.shape[1])) # e.g. (<number of files>, <tobs>)
        
        print("dataset shape after reshape:",data.shape)

        # TODO 
        # More samples in the test set..
        firstSample = data[0,:]

        # TODO
        # Fix the pivot
        beforeOnsetWindows, afterOnsetWindows = WindowsExtractor.test_extract_sub_windows_pivot(firstSample, windowSize, stride, onset)
        beforeOnsetLabels = np.array([False for i in range(beforeOnsetWindows.shape[0])])
        afterOnsetLabels = np.array([True for i in range(afterOnsetWindows.shape[0])])

        return beforeOnsetWindows, beforeOnsetLabels, afterOnsetWindows, afterOnsetLabels   


    def plotSamples(self, samples, labels, showFig=False, saveFig=True):

        
        numFeatures = samples[0].shape[1]
        numSamples = samples.shape[0]

        fig, axes = plt.subplots(nrows=numFeatures, ncols=numSamples, figsize=(15,10))
        print(numFeatures, numSamples, axes)

        if numFeatures == 1:
            axes = np.expand_dims(axes, axis=0)

        ylim = samples.max(axis=1).flatten().max()
        color=["blue","yellow","orange","red"]
        
        for idx, sample in enumerate(samples):

            for f in range(numFeatures):
                x = range(sample.shape[0])
                color="blue"
                if "+" in labels[idx]:
                    color="red"
                axes[f][idx].scatter(x, sample[:,f], label=self.featureCols[f], color=color, s=5)
                axes[f][idx].set_ylim(0,ylim)
                axes[f][idx].set_title(labels[idx])
                axes[f][0].legend()

        if showFig:
            plt.show()
            
        if saveFig:
            fig.savefig(self.outDir.joinpath(f"samples.png"), dpi=150)
            print(self.outDir.joinpath(f"samples.png"))

        plt.close()        
 


    def plotRandomSample(self, showFig=False, saveFig=True):

        print(self.data["bkg"].shape)
        print(self.data["grb"].shape)

        numFeatures = self.data["bkg"].shape[1]

        print(len(self.data["bkg"]),(self.tobs * self.filesLoaded["bkg"]) / self.integrationTime)
        #assert len(self.data["bkg"]) == (self.tobs * self.filesLoaded["bkg"]) / self.integrationTime
        #assert len(self.data["grb"]) == (self.tobs * self.filesLoaded["grb"]) / self.integrationTime

        bkgReshaped = self.data["bkg"].values.reshape((self.filesLoaded["bkg"], self.tobs, numFeatures))        
        grbReshaped = self.data["grb"].values.reshape((self.filesLoaded["grb"], self.tobs, numFeatures))

        random_bkg_sample_idx = randrange(0, bkgReshaped.shape[0])
        random_grb_sample_idx = randrange(0, grbReshaped.shape[0])

        randomBkgSample = bkgReshaped[random_bkg_sample_idx]
        randomGrbSample = grbReshaped[random_grb_sample_idx]

        fig, ax = plt.subplots(numFeatures, 1, figsize=(15,10))

        if numFeatures > 1:
            ax = ax.flatten()
        if numFeatures == 1:
            ax = [ax]
        
        x = range(1, randomBkgSample.shape[0]+1) # or range(1, randomGrbSample.shape[0]+1)

        for f in range(numFeatures):
            ax[f].scatter(x, randomBkgSample[:,f], label=f"bkg {self.featureCols[f]}", color="blue", s=5)
            ax[f].scatter(x, randomGrbSample[:,f], label=f"grb {self.featureCols[f]}", color="red", s=5)
            ax[f].axvline(x=self.onset, color="red", label=f"Onset: {self.onset}")
            ax[f].set_title(self.featureCols[f])
            ax[f].legend()

        if showFig:
            plt.show()

        if saveFig:
            fig.savefig(self.outDir.joinpath(f"random_sample.png"), dpi=400)

        plt.close()        
 















if __name__=='__main__':

    """
    # Single feature 
    bkgdata = Path("ap_data_for_training_and_testing/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_bkg_window_size_25_region_radius_0.5")
    grbdata = Path("ap_data_for_training_and_testing/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_grb_window_size_25_region_radius_0.5")
    ds = APDataset()
    ds.loadData(bkg=bkgdata, grb=grbdata)
    train, test, val = ds.getData()
    # print(train, test, val)
    ds.plotRandomSample(label="bkg", scaler=None, seed=1)
    ds.plotRandomSample(label="grb", scaler=None, seed=1)

    ds.plotRandomSample(label="bkg", scaler="mm", seed=1)
    ds.plotRandomSample(label="grb", scaler="mm", seed=1)

    ds.plotRandomSample(label="bkg", scaler="std", seed=1)
    ds.plotRandomSample(label="grb", scaler="std", seed=1)
    """

    from rtapipe.analysis.dataset.dataset_params import get_dataset_params

    # Multiple features 
    dataset_integration_time = "10"
    integration_type = "te"
    dataset_params = get_dataset_params(dataset_integration_time, integration_type)
    ds = APDataset(dataset_params["tobs"], dataset_params["onset"], dataset_integration_time, integration_type, ["COUNT"], ['TMIN', 'TMAX', 'LABEL', 'ERROR'], "./tmp")

    bkgdata = Path("/data/datasets/ap_data/t_1/ap_data_for_training_and_testing_NORMALIZED/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_te_type_bkg_window_size_10_region_radius_0.2")
    grbdata = Path("/data/datasets/ap_data/t_1/ap_data_for_training_and_testing_NORMALIZED/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_te_type_grb_window_size_10_region_radius_0.2")

    ds.loadData("bkg", dataset_params["bkg"])
    ds.loadData("grb", dataset_params["grb"])
    ds.plotRandomSample(saveFig=False)


    # Single features 
    dataset_integration_time = "10"
    integration_type = "t"
    dataset_params = get_dataset_params(dataset_integration_time, integration_type)
    ds = APDataset(dataset_params["tobs"], dataset_params["onset"], dataset_integration_time, integration_type, ["COUNT"], ['TMIN', 'TMAX', 'LABEL', 'ERROR'], "./tmp")

    bkgdata = Path("/data/datasets/ap_data/t_1/ap_data_for_training_and_testing_NORMALIZED/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_bkg_window_size_10_region_radius_0.2")
    grbdata = Path("/data/datasets/ap_data/t_1/ap_data_for_training_and_testing_NORMALIZED/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_grb_window_size_10_region_radius_0.2")

    ds.loadData("bkg", dataset_params["bkg"])
    ds.loadData("grb", dataset_params["grb"])
    ds.plotRandomSample(saveFig=False)


