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
    train, test, val = ds.getData() 

    # Building the model
    lstm = AnomalyDetector(train[0].shape, units=64, dropoutRate=0.2, loadModelFrom="./single_feature_model")

    # Compiling the model
    lstm.compile()

    if lstm.isFresh():

        # Fitting the model
        lstm.fit(train, train, epochs=50, batch_size=32, verbose=1, validation_data=(val, val))    

        lstm.plotLosses()

        # Saving the model
        lstm.save("single_feature_model")

    """
    predictions = lstm.predict(test[0:int(test.shape[0]/2)])
    print(predictions)
    # lstm.plotPrediction(test[0, :, :], predictions[0, :, :])
    lstm.plotPredictions(test, predictions)

    predictions = lstm.predict(test[int(test.shape[0]/2):])
    print(predictions)
    # lstm.plotPrediction(test[0, :, :], predictions[0, :, :])
    lstm.plotPredictions(test, predictions)    
    """

    trainPred = lstm.predict(train)
    train_mae_loss = np.mean(np.abs(trainPred - train), axis=1)
    #plt.hist(train_mae_loss, bins=50)
    #plt.xlabel('Train MAE loss')
    #plt.ylabel('Number of Samples')
    #plt.show()

    threshold = np.max(train_mae_loss)
    print(f'Reconstruction error threshold: {threshold}')


    print("Test set shape: ",test.shape)
    predictions = lstm.predict(test)
    test_mae_loss = np.mean(np.abs(predictions-test), axis=1)
    plt.hist(test_mae_loss, bins=50)
    plt.xlabel('Test MAE loss')
    plt.ylabel('Number of samples')
    plt.show()

    anomalies = []
    preds = []
    for ii,loss in enumerate(test_mae_loss):
        if loss >= threshold:
            anomalies.append(test[ii])
            preds.append(predictions[ii])
    print("anomalies: ",len(anomalies))
    lstm.plotPredictions(anomalies, preds)