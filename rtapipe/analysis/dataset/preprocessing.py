from os import times
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

class Dataset:

    def __init__(self):
        self.data = None
        self.dataNorm = None
        self.label = None
        self.shape = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def loadData(self, dataDir, label=""):
        dfs = [pd.read_csv(file, sep=",") for file in Path(dataDir).iterdir()]
        self.data = pd.concat(dfs, axis=0)
        self.label = label
        self.sampleShape = dfs[0].shape
        self.numberOfSamples = len(dfs)

        print(f"Loaded {self.numberOfSamples} files. Label={label} Shape single df = {self.sampleShape} Shape after concat = {self.data.shape}")

    def getTimeSeries(self, integrationType, minMaxScale = False):
        """
        The input data to an LSTM model is a 3-dimensional array. The shape of the array is 
        samples x timesteps x features.

        samples:   this is simply the number of observations, or in other words, the number of data points
        timesteps: LSTM models are meant to look at the past. Meaning, at time t the LSTM will process 
                   data up to (t-timesteps) to make a prediction.
        features:  it is the number of features present in the input data.
        """

        if integrationType == "t":
            self.data.drop(['TMIN', 'TMAX', 'ERROR'], axis='columns', inplace=True)
            if minMaxScale:
                # Fit the scaler using available training data. For normalization, this means the training data will be used to estimate the minimum and maximum observable values. This is done by calling the fit() function.
                scaler = self.scaler.fit(self.data)
                self.dataNorm = self.scaler.transform(self.data)
                timeseries = self.dataNorm.reshape((
                                    int(self.dataNorm.shape[0]/self.sampleShape[0]), 
                                    self.sampleShape[0], 
                                    1))
            else:
                timeseries = self.data.values.reshape((
                                int(self.data.shape[0]/self.sampleShape[0]), 
                                self.sampleShape[0], 
                                1))

            return timeseries

        elif integrationType == "te":
            errorCols = [col for col in self.data if "ERROR" in col]
            self.data.drop(["TMIN", "TMAX"]+errorCols, axis='columns', inplace=True)
            if minMaxScale:
                scaler = self.scaler.fit(self.data)
                self.dataNorm = self.scaler.transform(self.data)
                timeseries = self.dataNorm.reshape((
                                    int(self.dataNorm.shape[0]/self.sampleShape[0]), 
                                    self.sampleShape[0], 
                                    4))

            else:

                timeseries = self.data.values.reshape((
                                    int(self.data.shape[0]/self.sampleShape[0]), 
                                    self.sampleShape[0], 
                                    4))
            return timeseries

        else:
            raise ValueError(f"Integration type={integrationType} is not supported")

    




if __name__=='__main__':

    basepath = Path("/home/leobaro/Workspace/inaf/phd/rtapipe/analysis/dataset/ap_data_for_training_and_testing")

    ds = Dataset()    
    ds.loadData(basepath.joinpath("simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_bkg_window_size_25_region_radius_0.5"), label="bkg")
    ts = ds.getTimeSeries(integrationType="t", minMaxScale=True)
    print(ts)
    print(ts.shape)


    ds = Dataset()        
    ds.loadData(basepath.joinpath("simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_te_type_bkg_window_size_25_region_radius_0.5"), label="bkg")
    ts = ds.getTimeSeries(integrationType="te", minMaxScale=True)
    print(ts)
    print(ts.shape)
    

    #dp.loadData(basepath.joinpath("simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_grb_window_size_25_region_radius_0.5"), label="grb")
    #dp.loadData(basepath.joinpath("simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_te_type_grb_window_size_25_region_radius_0.5"), label="grb")

