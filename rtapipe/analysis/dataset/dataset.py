import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class APDataset(ABC):

    def __init__(self):
        self.data = None
        self.mmscaler = MinMaxScaler(feature_range=(0,1))
        self.stdscaler = StandardScaler()
        self.singleSampleShape = None
        self.uselessCols = []
        self.featureCols = []
    
    def loadData(self, **kwargs):
        count = 0
        for label in kwargs:
            dataDir = kwargs[label]
            dfs = [pd.read_csv(file, sep=",") for file in Path(dataDir).iterdir()]
            count += len(dfs)
            self.singleSampleShape = dfs[0].shape
            concat = pd.concat(dfs)
            concat["LABEL"] = label
            if self.data is None:
                self.data = concat
            else:
                self.data = pd.concat([self.data, concat], axis=0)

        self.uselessCols = ['TMIN', 'TMAX', 'LABEL'] + [col for col in self.data if "ERROR" in col]
        self.featureCols = [col for col in self.data if "COUNTS" in col]

        print(f"Loaded {count} dataframes. Single sample shape={self.singleSampleShape}. Total shape={self.data.shape}")
        print(f"Feature columns={self.featureCols}")
        print(f"Useless columns={self.uselessCols}")

    def getSampleFraction(self, total, percentage):
        return int(total*(percentage/100))

    def transformationPipeline(self, df, scaler):

        if scaler=="mm":
            df = self.mmscaler.transform(df)
            print("MinMax scaling applied.")
        elif scaler=="std":
            df = self.stdscaler.transform(df)
            print("Standard scaling applied.")
        else:
            df = df.values
            print("No scaling applied.")

        samples = int(df.shape[0]/self.singleSampleShape[0])
        timesteps = self.singleSampleShape[0]
        features = len(self.featureCols)
        df = df.reshape((samples, timesteps, features))
        print(f"Timeseries created. Shape={df.shape}")
        return df 

    def getData(self, trainP=(60,0), testP=(20,20), valP=(20,0), scaler="mm"):
        
        # total percentage of bkg samples
        assert trainP[0] + testP[0] + valP[0] <= 100

        # total percentage of grb samples
        assert trainP[1] + testP[1] + valP[1] <= 100

        # separating samples
        bkgData = self.data[self.data['LABEL']=="bkg"]
        grbData = self.data[self.data['LABEL']=="grb"]
        print(f"Total number of bkg samples={len(bkgData)}")
        print(f"Total number of grb samples={len(grbData)}")

        # dropping useless column
        bkgData = bkgData.drop(self.uselessCols, axis='columns')
        grbData = grbData.drop(self.uselessCols, axis='columns')
        print(f"Dropped columns={self.uselessCols}")

        # Splitting samples. Assumptions: train and val contains only bkg samples
        p1 = self.getSampleFraction(len(bkgData), trainP[0])
        p2 = self.getSampleFraction(len(bkgData), trainP[0] + testP[0])
        p3 = self.getSampleFraction(len(bkgData), trainP[0] + testP[0] + valP[0])
        p4 = self.getSampleFraction(len(grbData), testP[1])
        train = bkgData[:p1]
        test  = pd.concat([bkgData[p1:p2], grbData[:p4]])
        val   = bkgData[p2:p3]
        print(f"Training set: got {trainP[0]}% of bkg samples ({p1}). New shape={train.shape}.")
        print(f"Test set: got {testP[0]}% of bkg samples and {testP[1]}% of grb samples ({p2-p1 + p4}). New shape={test.shape}.")
        print(f"Validation set: got {valP[0]}% of bkg samples ({p3-p2}). New shape={val.shape}.")

        # scaling: before applying any scaling transformations it is very important to split your 
        # data into a train set and a test set. If you start scaling before, your training (and 
        # test) data might end up scaled around a mean value that is not actually the mean of the 
        # train or test data, and go past the whole reason why you’re scaling in the first place.
        # https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9
        self.mmscaler.fit(train)
        self.stdscaler.fit(train)
        
        train = self.transformationPipeline(train, scaler)
        test = self.transformationPipeline(test, scaler)
        val = self.transformationPipeline(val, scaler)

        trainLabels = np.array([False for l in range(train.shape[0])])
        testLabels = np.array([False for l in range(int(test.shape[0]/2))] + [True for l in range(int(test.shape[0]/2))])
        valLabels = np.array([False for l in range(train.shape[0])])

        return train, trainLabels, test, testLabels, val, valLabels

    def plotRandomSample(self, label, scaler=None, seed=None):
        fig, axes = plt.subplots(nrows=2, ncols=1)
        train, test, val = ds.getData(trainP=(60,0), testP=(0,20), valP=(20,0), scaler=scaler)
        if label == "bkg":
            print(train.shape)
            if seed is not None:
                randomSample = train[seed,:,:]
            else:
                randomSample = train[np.random.randint(train.shape[0]),:,:]
            axes[0].plot(randomSample)
            axes[1].hist(randomSample, bins=30, alpha=0.5)
        elif label == "grb":
            print(test.shape)
            if seed is not None:
                randomSample = test[seed,:,:]
            else:
                randomSample = test[np.random.randint(test.shape[0]),:,:]
            axes[0].plot(randomSample)
            axes[1].hist(randomSample, bins=30, alpha=0.5)
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
    train, test, val = ds.getData()
    # print(train, test, val)
    ds.plotRandomSample(label="bkg", scaler=None, seed=1)
    ds.plotRandomSample(label="grb", scaler=None, seed=1)

    ds.plotRandomSample(label="bkg", scaler="mm", seed=1)
    ds.plotRandomSample(label="grb", scaler="mm", seed=1)

    ds.plotRandomSample(label="bkg", scaler="std", seed=1)
    ds.plotRandomSample(label="grb", scaler="std", seed=1)

