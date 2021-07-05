import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from rtapipe.analysis.models.anomaly_detector import AnomalyDetector
from rtapipe.analysis.dataset.dataset import APDataset

if __name__=='__main__':
    
    # Loading data
    bkgdata = Path("dataset/ap_data_for_training_and_testing/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_bkg_window_size_25_region_radius_0.5")
    grbdata = Path("dataset/ap_data_for_training_and_testing/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_grb_window_size_25_region_radius_0.5")
    ds = APDataset()
    ds.loadData(bkg=bkgdata, grb=grbdata)
    train, trainLabels, test, testLabels, val, valLabels = ds.getData()
    print(train.shape)
    print(trainLabels.shape)
    print(test.shape)
    print(testLabels.shape)
    # Params
    units = 32
    dropoutrate = 0.3
    epochs = 20
    batchSize = 30

    # Building the model
    # loadModelFrom="./single_feature_model"
    loadModelFrom = None
    lstm = AnomalyDetector(train[0].shape, units=units, dropoutRate=dropoutrate, loadModelFrom=loadModelFrom)

    # Compiling the model
    lstm.compile()

    lstm.summary()

    if lstm.isFresh():

        # Fitting the model
        lstm.fit(train, train, epochs=epochs, batchSize=batchSize, verbose=1, validation_data=(val, val),
                     plotLosses=True)    

        # Saving the model
        lstm.save("single_feature_model")


    lstm.computeThreshold(train, plotError=True)

    recostructions, _, mask = lstm.classify(test, plotError=True)

    lstm.plotPredictions2(test, testLabels, recostructions, mask, howMany=40)

  