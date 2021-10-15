import pickle
import argparse
import numpy as np
from pathlib import Path
from time import strftime

from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.lib.rtapipeutils.Chronometer import Chronometer
from rtapipe.analysis.models.anomaly_detector import AnomalyDetector
from rtapipe.analysis.dataset.dataset_params import get_dataset_params


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-it", "--dataset_integration_time", type=str, required=True, help="", choices=["1", "5", "10"])
    parser.add_argument("-ws", "--window_size", type=int, required=True, help="", choices=[5, 10])
    parser.add_argument("-i", "--integration", type=str, required=False, default="t", help="", choices=["t", "te"])
    parser.add_argument("-sa", "--save_after", type=int, required=False, default=1, help="Saving statistics and model after '-sa' epochs")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, choices = [0,1], help="If 1 plots will be shown")

    #parser.add_argument("-N", "--normalized", type=int, required=False, default=1, help="", choices=[0, 1])
    #parser.add_argument("--bkg", type=str, required=False, default=None, help="The folder containing AP data (background only)")
    #parser.add_argument("--grb", type=str, required=False, default=None, help="The folder containing AP data (grb)")
    args = parser.parse_args()

    showPlots = False
    if args.verbose == 1:
        showPlots = True

    # Output dir
    outDirRoot = Path(__file__).parent.resolve().joinpath(f"lstm_models_{strftime('%Y%m%d-%H%M%S')}")
    outDirRoot.mkdir(parents=True, exist_ok=True)
    
    # get dataset params
    dataset_params = get_dataset_params(args.dataset_integration_time, args.integration)
    
    dataset_params["ws"] = args.window_size
    dataset_params["stride"] = 1
    dataset_params["scaler"] = "mm"
    print(dataset_params)

    with open(outDirRoot.joinpath('dataset_params.pickle'), 'wb') as handle:
        pickle.dump(dataset_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # LSTM params
    units = 32
    dropoutrate = 0.2
    epochs = 20 # 2
    batchSize = 16 # 4

    ds = APDataset(dataset_params["tobs"], dataset_params["onset"], args.dataset_integration_time, args.integration, ["COUNT"], ['TMIN', 'TMAX', 'LABEL', 'ERROR'], outDirRoot)
    ds.loadData("bkg", dataset_params["bkg"])
    ds.loadData("grb", dataset_params["grb"])

    ds.plotRandomSample(showFig=showPlots)

    #train, trainLabels, test, testLabels, val, valLabels = ds.getData()
    train, trainLabels, val, valLabels = ds.getTrainingAndValidationSet(dataset_params["ws"], dataset_params["stride"], split=50, scaler=dataset_params["scaler"])

    print(f"Train shape: {train.shape}. Example: {train[0].flatten()}")
    print(f"Val shape: {val.shape}. Example: {val[0].flatten()}")


    # Building the model
    adLSTM = AnomalyDetector(train[0].shape, units, dropoutrate, outDirRoot)
    adLSTM.setFeaturesColsNames(ds.getFeaturesColsNames())

    # Compiling the model
    adLSTM.compile()
    adLSTM.summary()


    ######################################################################################################################################################################
    #
    #  Log file for parameters
    #
    with open(outDirRoot.joinpath("parameters.csv"), "w") as statFile:
        statFile.write("integrationtype,integrationtime,tobs,onset,ws,stride,scaler,units,dropoutrate,epochs,batchSize\n")
        statFile.write(f"{args.integration},{args.dataset_integration_time},{dataset_params['tobs']},{dataset_params['onset']},{dataset_params['ws']},{dataset_params['stride']},{dataset_params['scaler']},{units},{dropoutrate},{epochs},{batchSize}")
    #
    # Log file for statistics during training
    #
    with open(outDirRoot.joinpath("statistics.csv"), "w") as statFile:
        statFile.write("epoch,training_time_mean,training_time_dev,total_time\n")
    ######################################################################################################################################################################
    

    fit_cron = Chronometer()

    for ep in range(epochs):

        print(f"""
\nEpoch={ep}        
        """)
        # Fitting the model
        fit_cron.start()
        adLSTM.fit(train, train, epochs=1, batchSize=batchSize, verbose=1, validation_data=(val, val), plotTrainingLoss=False)    
        fit_cron.stop()
        print(f"Fitting time: {fit_cron.get_statistics()[0]} +- {fit_cron.get_statistics()[1]}")

        # Saving the model

        if (ep+1) % 1 == 0:

            outDir = outDirRoot.joinpath("epochs",f"epoch_{ep+1}")
            outDir.mkdir(exist_ok=True, parents=True)
            adLSTM.setOutputDir(outDir)

            # Saving the model
            adLSTM.save(outDir.joinpath("lstm_trained_model"))

            # Computing the threshold using a validation set
            maeThreshold, meaLossesVal = adLSTM.computeSimpleThreshold(val, showFig=showPlots)        
            with open(outDir.joinpath("reconstruction_errors.csv"), "w") as recoFile:
                for val in meaLossesVal.squeeze():
                    recoFile.write(f"{val}\n")

            # Plotting reconstruction error distribution on the validation set 
            adLSTM.recoErrorDistributionPlot(meaLossesVal, threshold=None, filenamePostfix=f"val_set", title=f"Reconstruction error distribution on validation set (epoch={ep+1})", showFig=showPlots)

            with open(outDir.joinpath("threshold.csv"), "w") as thresholdFile:
                thresholdFile.write(f"threshold\n{maeThreshold}")

            # Saving training loss plot
            adLSTM.plotTrainingLoss(showFig=showPlots)

            # Saving time statistics
            with open(outDirRoot.joinpath("statistics.csv"), "a") as statFile:
                statFile.write(f"{ep+1},{fit_cron.get_statistics()[0]},{fit_cron.get_statistics()[1]},{fit_cron.get_total_elapsed_time()}\n")



    # adLSTM.plotPrecisionRecall(testLabels, testMAE, showFig=showPlots)  
    print(f"Total time for training: {fit_cron.total_time} seconds")
