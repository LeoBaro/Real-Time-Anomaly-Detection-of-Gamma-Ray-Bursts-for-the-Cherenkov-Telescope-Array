import os
import yaml
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from pathlib import Path
from random import randrange
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rtapipe.lib.rtapipeutils.SequenceUtils import extract_sub_windows, extract_sub_windows_pivot
from rtapipe.lib.rtapipeutils.FileSystemUtils import FileSystemUtils, parse_params

class APDataset(ABC):

    @abstractmethod
    def loadData(self, size = None):
        pass     

    @abstractmethod
    def train_val_split(self, split=70, scale=True, verbose=True):
        pass

    @abstractmethod
    def get_test_set(self, sequences_lenght, stride):
        pass

    def get_random_train_sample(self, scaled, tsl=10):
        if self.train_x is None:
            raise ValueError("Train set is None. Call train_val_split() first!")
        if self.train_x is not None:
            idx = np.random.randint(0, len(self.train_x))
        sample = self.train_x[idx]
        if scaled and self.scaler is not None:
            sample = self.scale(sample)
        elif scaled and self.scaler is None:
            raise ValueError("Scaler is None. Call train_val_split() first!")
        return sample

    @staticmethod
    def get_dataset(datasetsConfig, datasetID, out_dir=None, scaler_type=None):

        with open(datasetsConfig, "r") as stream:
            datasets = yaml.safe_load(stream)

        if datasetID not in datasets:
            raise ValueError(f"Dataset with id={datasetID} not found. Available datasets: {datasets.keys()}")
            
        dataset_params = datasets[datasetID]
        dataset_params["outDir"] = out_dir
        dataset_params["scaler"] = scaler_type

        if dataset_params["type"] == "single":
            return SinglePhList(dataset_params, out_dir, scaler_type)

        elif dataset_params["type"] == "multiple":
            return MultiplePhList(dataset_params, out_dir, scaler_type)


    def __init__(self, dataset_params, outDir=None, scaler_type=None):

        # The data container
        self.data = None
        self.test_data = None

        # Original shape informations
        self.filesLoaded = 0
        self.singleFileDataShapes = ()

        # Dataset informations
        self.dataset_params = dataset_params
        self.test_params = None

        self.featureCols = None
        self.featureColsNamesPattern = ["COUNT"]
        self.uselessColsNamesPattern = ['TMIN', 'TMAX', 'LABEL', 'ERROR']

        # Get filename of first file in the data dir
        dataDir = self.dataset_params["path"]
        # Get first filename:
        self.filenamePattern = None 
        for root, dirs, files in os.walk(dataDir, topdown=False):
            for name in files:
                self.filenamePattern = name
                break
        
        self.scaler_type = scaler_type
        self.scaler = None

        if self.scaler_type not in ["ss", "mm"]:
            raise ValueError("Scaler type not supported. Supported 'ss' and 'mm'.")

        self.outDir = None
        if outDir is not None:
            self.outDir = Path(outDir)

        self.batchIterator = None

        self.train_x, self.train_y = None, None
        self.val_x, self.val_y = None, None

    def checkCompatibilityWith(self, otherDatasetParams):

        params = ["integration_type", "integration_time", "region_radius", "irf", "emin", "emax", "roi"]

        for p in params:
            if self.dataset_params[p] != otherDatasetParams[p]:
                print(f"{p} is not compatible! {self.dataset_params[p]} != {otherDatasetParams[p]}")
                return False
        return True

    def setOutDir(self, outDir):
        self.outDir = Path(outDir)

    def setScalerFromPickle(self, pickleFile):
        with open(pickleFile, 'rb') as handle:
            self.scaler = pickle.load(handle)

    def getFeaturesColsNames(self):
        return self.featureCols

    def dumpDatasetParams(self, type, out_dir=None):
        if self.outDir is None:
            raise ValueError("self.outDir is None. Call setOutDir()")
        if out_dir is not None:
            out_dir = Path(out_dir)
        else:
            out_dir = self.outDir
        if type == "pickle":
            with open(out_dir.joinpath('dataset_params.pickle'), 'wb') as handle:
                pickle.dump(self.dataset_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif type == "ini":
            with open(out_dir.joinpath('dataset_params.ini'), 'w') as handle:
                for key, val in self.dataset_params.items():
                    handle.write(f"{key}={val}\n")
        elif type == "json":
            with open(out_dir.joinpath('dataset_params.json'), 'w') as handle:
                json.dump(self.dataset_params, handle)


    def preprocessData(self, verbose=True):
        if verbose:
            print(f"Single csv file shape: {self.singleFileDataShapes}")
            print(f"Dataframe shape: {self.data.shape}. Columns: {self.data.columns.values}")
        uselessCols = self._findColumns(self.uselessColsNamesPattern)
        # dropping useless column
        self.data.drop(uselessCols, axis='columns', inplace=True)
        if self.test_data is not None:
            self.test_data.drop(uselessCols, axis='columns', inplace=True)
        if verbose:
            print(f"Dropped columns={uselessCols} from dataset")
            print(f"Dataframe shape: {self.data.shape}. Columns: {self.data.columns.values}")
        self.featureCols = self._findColumns(self.featureColsNamesPattern)
        if len(self.featureCols) > 1:
            self.featureCols.sort(key = lambda col : float(col.split("COUNTS_")[1].split("-")[0]))



    """
    def loadBatchFromIDs(self, patternName, fromID, toID):
        ids: list of files' ids

        Load len(ids) files as a pandas.DataFrame and scale them. 
        bkg000001_t_simtype_bkg_onset_0_normalized_True.csv
        
        ids = list(range(fromID, toID))
        ids = [ f'{id:06d}' for id in ids if id < 10e6 ]

        dataDir = self.dataset_params["path"]
     
        s = time()
        countMissing = 0
        data = []
        for i,id in enumerate(ids):
            parts = self.filenamePattern.split("_t")
            fileName = f"bkg{id}_t{parts[-1]}"
            csvFile = Path(dataDir).joinpath(fileName)
            if csvFile.exists():
                df = pd.read_csv(csvFile, sep=",")
                data.append(df)
            else:
                print(f"[WARNING] File {csvFile} is missing!", flush=True)
                countMissing += 1

        self.singleFileDataShapes = data[0].shape
        self.data = pd.concat(data)
        self.filesLoaded = len(data)
        data = None
        print(f"Loaded: {self.filesLoaded}, took: {time()-s} seconds, missing files: {countMissing}.")
        
        s = time()
        self.preprocessData(verbose=False)
        print(f"Preprocessing took: {time()-s} seconds.")

        #print(self.data.values[0:10,2])
        s = time()
        data = self.scaler.transform(self.data.values)
        print(f"Scaling took: {time()-s} seconds.")

        # print(f"Dataset shape: {data.shape}")
        numFeatures = self.data.shape[1]
        data = data.reshape((self.filesLoaded, self.dataset_params["timeseries_lenght"], numFeatures))
        return data
    """

    """
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
    """


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


    def store_scaler(self, dest_path):
        
        if self.scaler is None:
            raise ValueError("Scaler is None. Call fit_scaler() first.")
        
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        with open(Path(dest_path).joinpath('fitted_scaler.pickle'), 'wb') as handle:
            pickle.dump(self.scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def fit_scaler(self, train_x):

        if self.scaler_type == "mm":
            self.scaler = MinMaxScaler(feature_range=(0,1))

        elif self.scaler_type == "ss":
            self.scaler = StandardScaler()

        self.scaler.fit(train_x.reshape(-1, train_x.shape[-1]))

        self.outDir.mkdir(parents=True, exist_ok=True)
        with open(self.outDir.joinpath('fitted_scaler.pickle'), 'wb') as handle:
            pickle.dump(self.scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def scale(self, data):
        if self.scaler is not None:
            return self.scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        else:
            print("Scaler not fitted yet!")

    def split_and_fit(self, data, split=70, scale=True, verbose=True):
        labels = np.array([False for i in range(len(data))])

        self.train_x, self.val_x = self._splitArrayWithPercentage(data, split)
        self.train_y, self.val_y = self._splitArrayWithPercentage(labels, split)
        
        if verbose:
            print(f"Training set: {self.train_x.shape} Labels: {self.train_y.shape}")
            print(f"Validation set: {self.val_x.shape} Labels: {self.val_y.shape}")

        if self.scaler is None:
            self.fit_scaler(self.train_x)

        if scale:
            return self.scale(self.train_x), self.train_y, self.scale(self.val_x), self.val_y
        else:
            return self.train_x, self.train_y, self.val_x, self.val_y

            




class SinglePhList(APDataset):

    def __init__(self, dataset_params, outDir=None, scaler_type=None):
        super().__init__(dataset_params, outDir, scaler_type)

    def loadData(self):
        print(f"Loading dataset from {self.dataset_params['path']}")
        params = parse_params(self.dataset_params["path"])
        for key, val in params.items():
            self.dataset_params[key] = val
        self.data = pd.read_csv(self.dataset_params['path'], sep=",")
        if 'test_path' in self.dataset_params:
            self.test_data = pd.read_csv(self.dataset_params['test_path'], sep=",")
            self.test_params = parse_params(self.dataset_params["test_path"])
        self.singleFileDataShapes = self.data.shape
        self.filesLoaded = 1
        self.preprocessData()

    def train_val_split(self, tsl, stride, split=70, scale=True, verbose=True):
        sequences = extract_sub_windows(self.data.values, start=0, stop=len(self.data), sub_window_size=tsl, stride_size=stride)
        return self.split_and_fit(sequences, split, scale, verbose)

    def get_test_set(self, tsl, stride):
        if self.test_params is None:
            raise Exception("Test set not loaded!")
        pivot_idx = self.test_params["onset"] // self.test_params["itime"]
        print("Pivot index: ", pivot_idx, self.test_data.shape)
        windows_before_pivot, windows_after_pivot = extract_sub_windows_pivot(self.test_data.values, sub_window_size=tsl, stride_size=stride, pivot_idx=pivot_idx)
        windows_before_pivot = self.scale(windows_before_pivot)
        windows_after_pivot = self.scale(windows_after_pivot)
        print("windows_before_pivot: ", windows_before_pivot.shape)
        print("windows_after_pivot: ", windows_after_pivot.shape)
        test_x = np.concatenate((windows_before_pivot, windows_after_pivot), axis=0)
        print("test_x: ", test_x.shape)
        labels = np.array([False for i in range(len(windows_before_pivot))]+[True for i in range(len(windows_after_pivot))])
        print("labels: ", labels.shape)
        return test_x, labels

class MultiplePhList(APDataset):
    
    def __init__(self, dataset_params, outDir=None, scaler_type=None):
        super().__init__(dataset_params, outDir, scaler_type)

    def loadData(self, size=None):
        print(f"Loading dataset from {self.dataset_params['path']}")
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


    def train_val_split(self, split=70, scale=True, verbose=True):

        numFeatures = self.data.shape[1]

        data = self.data.values

        data = data.reshape((self.filesLoaded, self.dataset_params["timeseries_lenght"], numFeatures))

        return self.split_and_fit(data, split, scale, verbose)



    def get_test_set(self, windowSize, stride):
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
        beforeOnsetWindows, afterOnsetWindows = extract_sub_windows_pivot(sample, windowSize, stride, onset_after_time_integration)
        beforeOnsetLabels = np.array([False for i in range(beforeOnsetWindows.shape[0])])
        afterOnsetLabels = np.array([True for i in range(afterOnsetWindows.shape[0])])

        return beforeOnsetWindows, beforeOnsetLabels, afterOnsetWindows, afterOnsetLabels

