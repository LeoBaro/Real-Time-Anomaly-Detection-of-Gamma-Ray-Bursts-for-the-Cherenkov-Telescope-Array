import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from random import randrange
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from rtapipe.lib.rtapipeutils.WindowsExtractor import WindowsExtractor

class APDataset:

    def get_dataset(datasetID, outDir=None, scaler=None):
        
        dataset_params = {
            # Dataset for training 
            1 : {
                "id": 1,
                "ap_command": "source generate_for_training.sh 1 5",
                "path": "/data/datasets/ap_data/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_1.0_roi_2.5/integration_t_integration_time_1_region_radius_0.2_timeseries_lenght_5",
                "simtype": "bkg",
                "onset": 0,
                "integration_type": "t",
                "integration_time": 1,
                "region_radius": 0.2,
                "timeseries_lenght": 5,
                "simulation_params_id": 1
            },
            # Dataset for training 
            2 : {
                "id": 2,
                "ap_command": "source generate_for_training.sh 1 5",
                "path": "/data/datasets/ap_data/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_1.0_roi_2.5/integration_te_integration_time_1_region_radius_0.2_timeseries_lenght_5",
                "simtype": "bkg",
                "onset": 0,
                "integration_type": "te",
                "integration_time": 1,
                "region_radius": 0.2,
                "timeseries_lenght": 5,
                "simulation_params_id": 1,                
            },
            # Dataset for testing
            3 : {
                "id": 3,
                "ap_command": "source generate_for_testing.sh 1 1800",
                "path": "/data/datasets/ap_data/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_1.0_roi_2.5/integration_t_integration_time_1_region_radius_0.2_timeseries_lenght_1800",
                "simtype": "grb",
                "onset": 900,
                "integration_type": "t",
                "integration_time": 1,
                "region_radius": 0.2,
                "timeseries_lenght": 1800,
                "simulation_params_id": 1,
            },
            4 : {
                "id": 3,
                "ap_command": "source generate_for_testing.sh 1 1800",
                "path": "/data/datasets/ap_data/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_1.0_roi_2.5/integration_te_integration_time_1_region_radius_0.2_timeseries_lenght_1800",
                "simtype": "grb",
                "onset": 900,
                "integration_type": "te",
                "integration_time": 1,
                "region_radius": 0.2,
                "timeseries_lenght": 1800,
                "simulation_params_id": 1,
            }
        }

        dataset_params = APDataset._addSimulationParams(dataset_params[datasetID])

        return APDataset(dataset_params, outDir, scaler)

    def _addSimulationParams(dataset_params):
        simulation_params = {
            1 : {
                "irf": "irf_South_z40_average_LST_30m",
                "emin": 0.03,
                "emax": 1,
                "roi": 2.5,
            }
        }
        for key,val in simulation_params[dataset_params["simulation_params_id"]].items():
            dataset_params[key] = val
        
        return dataset_params


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
            
        self.loadData()

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
        if self.dataset_params["simulation_params_id"] != otherDatasetParams["simulation_params_id"]:
            print("simulation_params_id is not compatible!")
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


    def loadData(self):
        
        print(f"Loading dataset..")

        dfs = []
        for file in tqdm( Path(self.dataset_params["path"]).iterdir() ):
            dfs.append(pd.read_csv(file, sep=","))
        self.filesLoaded = len(dfs)
        self.singleFileDataShapes = dfs[0].shape
        self.data = pd.concat(dfs)

        print(f"Loaded {len(dfs)} csv files.") 
        print(f"Single csv file shape: {self.singleFileDataShapes}")
        print(f"Dataframe shape: {self.data.shape}. Columns: {self.data.columns.values}")

        uselessCols = self._findColumns(self.uselessColsNamesPattern)
        
        # dropping useless column
        self.data.drop(uselessCols, axis='columns', inplace=True)
        print(f"Dropped columns={uselessCols} from dataset")
        print(f"Dataframe shape: {self.data.shape}. Columns: {self.data.columns.values}")

        self.featureCols = self._findColumns(self.featureColsNamesPattern)

    def getTrainingAndValidationSet(self, split=50, scale=True):

        numFeatures = self.data.shape[1]

        data = self.data.values 

        if scale:
            print("\tFitting scalers..")
            self.scaler.fit(data)
            data = self.scaler.transform(data)
            with open(self.outDir.joinpath('fitted_scaler.pickle'), 'wb') as handle:
                pickle.dump(self.scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

        data = data.reshape((self.filesLoaded, self.dataset_params["timeseries_lenght"], numFeatures)) 

        # windows = WindowsExtractor.test_extract_sub_windows(data, 0, data.shape[0], windowSize, stride)

        labels = np.array([False for i in range(len(data))])
        
        windows = self._splitArrayWithPercentage(data, split)
        labels = self._splitArrayWithPercentage(labels, split)

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

    



