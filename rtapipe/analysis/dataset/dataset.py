import yaml
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from random import randrange
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rtapipe.lib.rtapipeutils.WindowsExtractor import WindowsExtractor
from rtapipe.lib.rtapipeutils.FileSystemUtils import FileSystemUtils

class APDataset:

    def get_dataset(datasetsConfig, datasetID, outDir=None, scaler=None):

        with open(datasetsConfig, "r") as stream:
            datasets = yaml.safe_load(stream)

        if datasetID not in datasets:
            raise ValueError(f"Dataset with id={datasetID} not found. Available datasets: {datasets.keys()}")

        dataset_params = datasets[datasetID]

        dataset_params["outDir"] = outDir
        dataset_params["scaler"] = scaler

        return APDataset(dataset_params, outDir, scaler)


    def __init__(self, dataset_params, outDir=None, scaler=None):

        # The data container
        self.data = None

        # Original shape informations
        self.filesLoaded = 0
        self.singleFileDataShapes = ()

        # Dataset informations
        self.dataset_params = dataset_params
        self.featureCols = None
        self.featureColsNamesPattern = ["COUNT"]
        self.uselessColsNamesPattern = ['TMIN', 'TMAX', 'LABEL', 'ERROR']

        # Scalers
        if scaler is None:
            self.scaler = scaler
        elif scaler == "mm":
            self.scaler = MinMaxScaler(feature_range=(0,1))
        elif scaler == "ss":
            self.scaler = StandardScaler()
        else:
            raise ValueError("Supported scalers: mm, ss")

        if outDir is not None:
            self.outDir = Path(outDir)
        else:
            self.outDir = None

        self.batchIterator = None

    def checkCompatibilityWith(self, otherDatasetParams):

        if self.dataset_params["integration_type"] != otherDatasetParams["integration_type"]:
            print("integration_type is not compatible!")
            return False
        if self.dataset_params["integration_time"] != otherDatasetParams["integration_time"]:
            print("integration_time is not compatible!")
            return False
        if self.dataset_params["region_radius"] != otherDatasetParams["region_radius"]:
            print("region_radius is not compatible!")
            return False
        if self.dataset_params["irf"] != otherDatasetParams["irf"]:
            print("irf is not compatible!")
            return False
        if self.dataset_params["emin"] != otherDatasetParams["emin"]:
            print("emin is not compatible!")
            return False
        if self.dataset_params["emax"] != otherDatasetParams["emax"]:
            print("emax is not compatible!")
            return False
        if self.dataset_params["roi"] != otherDatasetParams["roi"]:
            print("roi is not compatible!")
            return False
        return True

    def setOutDir(self, outDir):
        self.outDir = Path(outDir)

    def setScalerFromPickle(self, pickleFile):
        with open(pickleFile, 'rb') as handle:
            self.scaler = pickle.load(handle)

    def getFeaturesColsNames(self):
        return self.featureCols

    def dumpDatasetParams(self, type):
        if self.outDir is None:
            raise ValueError("self.outDir is None. Call setOutDir()")
        if type == "pickle":
            with open(self.outDir.joinpath('dataset_params.pickle'), 'wb') as handle:
                pickle.dump(self.dataset_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif type == "ini":
            with open(self.outDir.joinpath('dataset_params.ini'), 'w') as handle:
                for key, val in self.dataset_params.items():
                    handle.write(f"{key}={val}\n")


    def preprocessData(self, verbose=True):
        if verbose:
            print(f"Single csv file shape: {self.singleFileDataShapes}")
            print(f"Dataframe shape: {self.data.shape}. Columns: {self.data.columns.values}")
        uselessCols = self._findColumns(self.uselessColsNamesPattern)
        # dropping useless column
        self.data.drop(uselessCols, axis='columns', inplace=True)
        if verbose:
            print(f"Dropped columns={uselessCols} from dataset")
            print(f"Dataframe shape: {self.data.shape}. Columns: {self.data.columns.values}")
        self.featureCols = self._findColumns(self.featureColsNamesPattern)


    def loadDataBatch(self, batchsize, verbose=False):
        # print(f"Loading batch of files ({batchsize}) from {Path(self.dataset_params['path'])}")
        if self.batchIterator is None:
            self.batchIterator = FileSystemUtils.iterDirBatch(Path(self.dataset_params["path"]), batchsize)
        try:
            batchFiles = next(self.batchIterator)
            self.singleFileDataShapes = pd.read_csv(batchFiles[0], sep=",").shape
            self.data = pd.concat([pd.read_csv(f, sep=",") for f in batchFiles])
            self.filesLoaded = len(batchFiles)
            self.preprocessData(verbose)
            return True
        except StopIteration:
            print("StopIteration exception!")
            return False
        except Exception as genericEx:
            print(f"Exception: {genericEx}")
            return False


    def loadData(self, size = None):

        print(f"Loading dataset..")

        count = 0
        for file in tqdm( Path(self.dataset_params["path"]).iterdir() ):
            try:
                if self.data is None:
                    self.data = pd.read_csv(file, sep=",")
                    self.singleFileDataShapes = self.data.shape
                else:
                    self.data = pd.concat([self.data, pd.read_csv(file, sep=",")])
            except pd.errors.EmptyDataError as exc:
                print(f"EmptyDataError exception! File {file} is empty!", exc)
            count += 1
            if size is not None and count >= size:
                break

        self.filesLoaded = count

        self.preprocessData()


    def getTrainingAndValidationSet(self, split=50, fitScaler=True, scale=True, verbose=True):

        numFeatures = self.data.shape[1]

        data = self.data.values

        if fitScaler:
            if verbose:
                print("\tFitting scalers..")
            self.scaler.fit(data)
            with open(self.outDir.joinpath('fitted_scaler.pickle'), 'wb') as handle:
                pickle.dump(self.scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if scale:
            if verbose:
                print("\tScaling data..")
            data = self.scaler.transform(data)

        # print(f"Data before reshape: {data.shape}")
        data = data.reshape((self.filesLoaded, self.dataset_params["timeseries_lenght"], numFeatures))

        # windows = WindowsExtractor.test_extract_sub_windows(data, 0, data.shape[0], windowSize, stride)

        labels = np.array([False for i in range(len(data))])

        windows = self._splitArrayWithPercentage(data, split)
        labels = self._splitArrayWithPercentage(labels, split)

        if verbose:
            print(f"Training set: {windows[0].shape} Labels: {labels[0].shape}")
            print(f"\tValidation set: {windows[1].shape} Labels: {labels[1].shape}")

        return windows[0], labels[0], windows[1], labels[1]



    # TODO different types of test sets
    def getTestSet(self, windowSize, stride):
        """
        At the moment it extracts subwindows from 1 GRB sample only
        """
        print("Exctracting test set..")

        data = self.scaler.transform(self.data.values)

        print(f"Dataset shape: {data.shape}")

        numFeatures = self.data.shape[1]

        data = data.reshape((self.filesLoaded, self.dataset_params["timeseries_lenght"], numFeatures))

        print(f"Dataset shape after reshape: {data.shape}")

        # TODO
        # More samples in the test set..
        sample = data[0,:]

        # TODO
        # Check/Fix the pivot
        onset_after_time_integration = int(self.dataset_params["onset"] / self.dataset_params["integration_time"])
        beforeOnsetWindows, afterOnsetWindows = WindowsExtractor.test_extract_sub_windows_pivot(sample, windowSize, stride, onset_after_time_integration)
        beforeOnsetLabels = np.array([False for i in range(beforeOnsetWindows.shape[0])])
        afterOnsetLabels = np.array([True for i in range(afterOnsetWindows.shape[0])])

        return beforeOnsetWindows, beforeOnsetLabels, afterOnsetWindows, afterOnsetLabels





    def plotRandomSample(self, howMany=1, showFig=False, saveFig=True):
        numFeatures = self.data.shape[1]

        #print(len(self.data),(self.dataset_params["timeseries_lenght"] * self.filesLoaded) / self.dataset_params["integration_time"])
        #assert len(self.data) == (self.dataset_params["timeseries_lenght"] * self.filesLoaded) / self.dataset_params["integrationTime"]

        fig, ax = plt.subplots(numFeatures, 1, figsize=(15,10))

        if numFeatures > 1:
            ax = ax.flatten()
        if numFeatures == 1:
            ax = [ax]

        x = range(1, self._getRandomSample().shape[0]+1) # or range(1, randomGrbSample.shape[0]+1)

        for f in range(numFeatures):

            for i in range(howMany):
                ax[f].scatter(x, self._getRandomSample()[:,f], label=f"{self.dataset_params['simtype']} {self.featureCols[f]}", s=5)

            if self.dataset_params["onset"] > 0:
                ax[f].axvline(x=self.onset, color="red", label=f"Onset: {self.dataset_params['onset']}")

            ax[f].set_title(self.featureCols[f])

        plt.tight_layout()

        if showFig:
            plt.show()

        if saveFig:
            fig.savefig(self.outDir.joinpath(f"random_sample.png"), dpi=400)

        plt.close()




    def _findColumns(self, patterns):
        """
        This method will search for columns names from the columns patterns passed as input.
        """
        cols = set()
        for col in self.data.columns.values:
            for pattern in patterns:
                if pattern in col:
                    cols.add(col)
        return list(cols)


    def _getSampleFraction(self, total, percentage):
        return int(total*(percentage/100))


    def _splitArrayWithPercentage(self, arr, percentage):
        splitPoint1 = self._getSampleFraction(len(arr), percentage)
        return np.split(arr, [splitPoint1])


    def _getRandomSample(self):
        numFeatures = self.data.shape[1]
        dataReshaped = self.data.values.reshape((self.filesLoaded, self.dataset_params["timeseries_lenght"], numFeatures))
        randomSampleID = randrange(0, dataReshaped.shape[0])
        return dataReshaped[randomSampleID]


    def plotSamples(self, samples, labels, showFig=False, saveFig=True):


        numFeatures = samples[0].shape[1]
        numSamples = samples.shape[0]

        fig, axes = plt.subplots(nrows=numFeatures, ncols=numSamples, figsize=(15,10))
        # print(numFeatures, numSamples, axes)

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
