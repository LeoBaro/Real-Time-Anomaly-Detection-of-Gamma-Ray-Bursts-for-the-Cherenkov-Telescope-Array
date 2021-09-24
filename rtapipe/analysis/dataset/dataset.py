from operator import index
import numpy as np
from random import randrange
from numpy.core.fromnumeric import shape
import pandas as pd
from time import sleep
from pathlib import Path
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rtapipe.lib.rtapipeutils.WindowsExtractor import WindowsExtractor

class APDataset(ABC):
    """
    This class can handle the .csv files of the AP dataset (generated witht the ?? API).
    """
    def __init__(self, tobs, onset, integrationTime, featureColsNamesPattern, uselessColsNamesPattern, outDir):
        
        # The data container
        self.data = {}
        
        # Original shape informations
        self.filesLoaded = {}
        self.singleFileDataShapes = {}

        # Dataset informations
        self.tobs = tobs
        self.onset = onset
        self.integrationTime = integrationTime
        self.featureCols = None
        self.featureColsNamesPattern = featureColsNamesPattern
        self.uselessColsNamesPattern = uselessColsNamesPattern

        # Scalers
        self.mmscaler = MinMaxScaler(feature_range=(0,1))
        self.stdscaler = StandardScaler()
        
        self.outDir = outDir


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
        return cols

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

    def getTrainingAndValidationSet(self, windowSize, stride, scaler=None):
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
        
        windows = self._splitArrayWithPercentage(windows, 70)
        labels = self._splitArrayWithPercentage(labels, 70)

        print(f"\tTraining set: {windows[0].shape} Labels: {labels[0].shape}")
        print(f"\tValidation set: {windows[1].shape} Labels: {labels[1].shape}")

        return windows[0], labels[0], windows[1], labels[1]

    def getTestSet(self, windowSize, stride, onset, scaler=None):

        print("Exctracting test set..")

        numberOfFiles = self.filesLoaded["grb"]

        data = self.data["grb"].values

        if scaler and scaler == "mm":
            data = self.mmscaler.transform(data)

        if scaler and scaler == "std":
            data = self.stdscaler.transform(data)

        data = data.reshape((numberOfFiles, self.tobs, 1)) # e.g. (<number of files>, <tobs>)
        
        print("dataset shape:",data.shape)

        firstSample = data[0,:]

        beforeOnsetWindows, afterOnsetWindows = WindowsExtractor.test_extract_sub_windows_pivot(firstSample,windowSize, stride, onset)
        beforeOnsetLabels = np.array([False for i in range(beforeOnsetWindows.shape[0])])
        afterOnsetLabels = np.array([True for i in range(afterOnsetWindows.shape[0])])

        return beforeOnsetWindows, beforeOnsetLabels, afterOnsetWindows, afterOnsetLabels   


    def plotSamples(self, samples, labels, filename=None, change_color_from_index=None):

        fig, axes = plt.subplots(nrows=2, ncols=samples.shape[0], figsize=(15,10))
        ylim = samples.max(axis=1).flatten().max()
        color="blue"
        for idx, sample in enumerate(samples):
            if change_color_from_index and idx >= change_color_from_index:
                color = "red"
            axes[0][idx].scatter(range(sample.shape[0]),sample,label=labels[idx],color=color, s=5)
            axes[1][idx].hist(sample, bins=15, alpha=0.5,color=color)
            axes[0][idx].set_ylim(0,ylim)

        fig.legend()
        if filename:
            fig.savefig(self.outDir.joinpath(f"{filename}.png"), dpi=300)
        plt.show()
        
    def plotSample(self, sample, label, filename=None):
        fig, axes = plt.subplots(nrows=2, ncols=1)
        axes[0].scatter( range(sample.shape[0]), sample , label=label, s=5)
        axes[1].hist(sample, bins=30, alpha=0.5)
        fig.legend()
        if filename:
            fig.savefig(self.outDir.joinpath(f"{filename}.png"), dpi=300)
        plt.show()
 


    def plotRandomSamples(self, filename=None):

        print(self.data["bkg"].shape)
        print(self.data["grb"].shape)

        print(len(self.data["bkg"]),(self.tobs * self.filesLoaded["bkg"]) / self.integrationTime)
        assert len(self.data["bkg"]) == (self.tobs * self.filesLoaded["bkg"]) / self.integrationTime
        assert len(self.data["grb"]) == (self.tobs * self.filesLoaded["grb"]) / self.integrationTime

        bkgReshaped = self.data["bkg"].values.reshape((self.filesLoaded["bkg"], self.tobs))        
        grbReshaped = self.data["grb"].values.reshape((self.filesLoaded["grb"], self.tobs))

        random_bkg_sample_idx = randrange(0, bkgReshaped.shape[0])
        random_grb_sample_idx = randrange(0, grbReshaped.shape[0])

        randomBkgSample = bkgReshaped[random_bkg_sample_idx]
        randomGrbSample = grbReshaped[random_grb_sample_idx]

        fig, ax = plt.subplots(2,1, figsize=(15,10))

        ax = ax.flatten()

        ax[0].scatter( range(1, randomBkgSample.shape[0]+1), randomBkgSample, label="bkg", color="blue", s=5)
        ax[0].scatter( range(1, randomGrbSample.shape[0]+1), randomGrbSample, label="grb", color="red", s=5)
        ax[0].axvline(x=self.onset, color="red", label=f"Onset: {self.onset}")

        ax[1].hist(randomBkgSample, bins=30, alpha=0.5, label="bkg", color="blue")
        ax[1].hist(randomGrbSample, bins=30, alpha=0.5, label="grb", color="red")
        

        #ax[0].set_ylim(100)
        #ax[0].set_ylim(100)
        
        if filename:
            fig.savefig(self.outDir.joinpath(f"{filename}.png"), dpi=300)

        ax[0].legend()
        plt.show()


















if __name__=='__main__':


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


    # Multiple features 
    bkgdata = Path("ap_data_for_training_and_testing/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_te_type_bkg_window_size_25_region_radius_0.5")
    grbdata = Path("ap_data_for_training_and_testing/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_te_type_grb_window_size_25_region_radius_0.5")
    ds = APDataset()
    ds.loadData(bkg=bkgdata, grb=grbdata)
    # train, test, val = ds.getData()
    # print(train, test, val)
    ds.plotRandomSample(label="bkg", scaler=None, seed=1)
    ds.plotRandomSample(label="grb", scaler=None, seed=1)

    ds.plotRandomSample(label="bkg", scaler="mm", seed=1)
    ds.plotRandomSample(label="grb", scaler="mm", seed=1)

    ds.plotRandomSample(label="bkg", scaler="std", seed=1)
    ds.plotRandomSample(label="grb", scaler="std", seed=1)

