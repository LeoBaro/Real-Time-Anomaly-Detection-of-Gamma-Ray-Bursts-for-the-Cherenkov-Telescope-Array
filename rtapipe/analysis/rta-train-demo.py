import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from rtapipe.analysis.models.anomaly_detector import AnomalyDetector
from rtapipe.analysis.dataset.dataset import APDataset

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bkg", type=str, required=False, help="The folder containing AP data (background only)")
    parser.add_argument("--grb", type=str, required=False, help="The folder containing AP data (grb)")
    args = parser.parse_args()

    if args.bkg is None:
        args.bkg = "/data/datasets/ap_data/ap_data_for_training_and_testing_2021-07-20.21:47:42_NORMALIZED/simtype_bkg_os_0_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_bkg_window_size_10_region_radius_0.2"
        args.grb = "/data/datasets/ap_data/ap_data_for_training_and_testing_2021-07-20.21:47:42_NORMALIZED/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_0.15_roi_2.5/integration_t_type_grb_window_size_10_region_radius_0.2"

    ds = APDataset()
    ds.loadData(bkg=args.bkg, grb=args.grb)


    batchTime = 5
    batchSize = 5
    ws = 5
    units = 8
    dropoutrate = 0.3

    train, trainLabels, test, testLabels, val, valLabels = ds.getData()

    print((batchSize,ws,1))
    lstm = AnomalyDetector((ws,1), units=units, dropoutRate=dropoutrate)
    lstm.compile()
    lstm.summary()


    for train, trainLabels in ds.getStreamOfData(5, 5, 5):
        #print(train)
        #print(trainLabels)
        #print(f"train: {train.shape}")
        
        # Fitting the model
        lstm.fit(train, train, epochs=10, batchSize=batchSize, verbose=1, showLoss=False, validation_data=(val, val))

        recostructions, maeLosses = lstm.reconstruct(train)   

        plt.plot(recostructions[0], label="Reco")
        plt.plot(train[0], label="Orig")
        plt.title(f"Error: {maeLosses}")
        plt.legend()      
        plt.show()