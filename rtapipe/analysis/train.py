import argparse
import numpy as np
from pathlib import Path
from time import strftime

from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.lib.rtapipeutils.Chronometer import Chronometer
from rtapipe.analysis.models.anomaly_detector import AnomalyDetector


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-di", "--dataset_id", type=int, required=True, help="")
    parser.add_argument('-sa', '--save-after', type=int, required=False, default=1, help="Store trained model after sa training epochs")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, choices = [0,1], help="If 1 plots will be shown")
    parser.add_argument("-dc", "--dataset_config", type=str, required=False, default="./dataset/config/agilehost3.yml")
    args = parser.parse_args()

    if args.model_name not in ["AnomalyDetector_2layers", "AnomalyDetector_2layers_small"]:
        raise ValueError(f"Model '{args.model_name}' does not exist.")

    showPlots = False
    if args.verbose == 1:
        showPlots = True

    # Output dir
    outDirRoot = Path(__file__) \
                    .parent \
                    .resolve() \
                    .joinpath("training_output",f"lstm_models_{strftime('%Y%m%d-%H%M%S')}")

    outDirRoot.mkdir(parents=True, exist_ok=True)

    # Dataset
    ds = APDataset.get_dataset(args.dataset_config, args.dataset_id, scaler="mm", outDir=outDirRoot)
    ds.loadData(size=24)

    ds.plotRandomSample(howMany=4, showFig=showPlots)

    ds.dumpDatasetParams(type="pickle")
    ds.dumpDatasetParams(type="ini")



    with open(outDirRoot.joinpath("model_params.ini"), "w") as handle:
        for key, val in model_params.items():
            handle.write(f"{key}={val}\n")

    #train, trainLabels, test, testLabels, val, valLabels = ds.getData()
    train, trainLabels, val, valLabels = ds.getTrainingAndValidationSet(split=20, scale=True)
    print(f"Train shape: {train.shape}. Example: {train[0].flatten()}")
    print(f"Val shape: {val.shape}. Example: {val[0].flatten()}")

    timesteps = train[0].shape[0]
    nfeatures = train[0].shape[1]

    # Building the model
    adLSTM = AnomalyDetector(timesteps, nfeatures, outDirRoot)
    adLSTM.setFeaturesColsNames(ds.getFeaturesColsNames())

    # Compiling the model
    adLSTM.compile()
    adLSTM.summary()


    # Log file for statistics during training
    #
    with open(outDirRoot.joinpath("statistics.csv"), "w") as statFile:
        statFile.write("epoch,training_time_mean,training_time_dev,total_time\n")
    ######################################################################################################################################################################


    fit_cron = Chronometer()

    for ep in range(model_params["epochs"]):

        print(f"\nEpoch={ep}")

        # Fitting the model
        fit_cron.start()
        adLSTM.fit(train, train, epochs=1, batchSize=model_params["batchSize"], verbose=1, validation_data=(val, val), plotTrainingLoss=False)
        fit_cron.stop()
        print(f"Fitting time: {fit_cron.get_statistics()[0]} +- {fit_cron.get_statistics()[1]}")

        # Saving the model

        if (ep+1) % args.save_after == 0:

            print("\nSaving the model and plots..")
            outDir = outDirRoot.joinpath("epochs",f"epoch_{ep+1}")
            outDir.mkdir(exist_ok=True, parents=True)
            adLSTM.setOutputDir(outDir)

            # Saving the model
            if args.save_after:
                adLSTM.save(outDir.joinpath("lstm_trained_model"))

            # Computing the threshold using a validation set
            maeThreshold, maeLossesVal = adLSTM.computeSimpleThreshold(val, showFig=showPlots)
            with open(outDir.joinpath("reconstruction_errors.csv"), "w") as recoFile:
                for vv in maeLossesVal.squeeze():
                    recoFile.write(f"{vv}\n")

            # Plotting reconstruction error distribution on the validation set
            adLSTM.recoErrorDistributionPlot(maeLossesVal, threshold=None, filenamePostfix=f"val_set", title=f"Reconstruction error distribution on validation set (epoch={ep+1})", showFig=showPlots)

            with open(outDir.joinpath("threshold.txt"), "w") as thresholdFile:
                thresholdFile.write(f"{maeThreshold}")

            # Saving training loss plot
            adLSTM.plotTrainingLoss(showFig=showPlots)

            # Saving time statistics
            with open(outDirRoot.joinpath("statistics.csv"), "a") as statFile:
                statFile.write(f"{ep+1},{fit_cron.get_statistics()[0]},{fit_cron.get_statistics()[1]},{fit_cron.get_total_elapsed_time()}\n")


    # with open(outDirRoot.joinpath("statistics.csv"), "r") as statFile:





    print(f"Total time for training: {fit_cron.total_time} seconds")
