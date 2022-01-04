import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import numpy as np
from pathlib import Path
from time import strftime

from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.lib.rtapipeutils.Chronometer import Chronometer
from rtapipe.analysis.models.anomaly_detector_builder import AnomalyDetectorBuilder


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-di", "--dataset_id", type=int, required=True)
    parser.add_argument("-tt", "--training_type", type=str, choices = ["light", "medium", "heavy"], required=True)
    parser.add_argument("-of", "--output_folder", type=str, required=True)
    parser.add_argument('-sa', '--save-after', type=int, required=False, default=1, help="Store trained model after each 'sa' training epochs")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, choices = [0,1], help="If 1 plots will be shown")
    parser.add_argument("-dc", "--dataset_config", type=str, required=False, default="./dataset/config/agilehost3.yml")
    args = parser.parse_args()

    showPlots = False
    if args.verbose == 1:
        showPlots = True

    # Output dir
    outDirBase = Path(__file__) \
                    .parent \
                    .resolve() \
                    .joinpath(args.output_folder)

    outDirRoot = outDirBase.joinpath(f"lstm_models_{strftime('%Y%m%d-%H%M%S')}")
    outDirRoot.mkdir(parents=True, exist_ok=True)

    # Dataset
    training_params = AnomalyDetectorBuilder.getTrainingParams(args.training_type)
    ds = APDataset.get_dataset(args.dataset_config, args.dataset_id, scaler="mm", outDir=outDirRoot)
    ds.loadData(size=training_params["sample_size"]*2)
    ds.plotRandomSample(howMany=4, showFig=showPlots)
    train, trainLabels, validationSet, valLabels = ds.getTrainingAndValidationSet(split=50, scale=True)
    #print(f"Train shape: {train.shape}. Example: {train[0].flatten()}")
    #print(f"Val shape: {val.shape}. Example: {val[0].flatten()}")

    timesteps = train[0].shape[0]
    nfeatures = train[0].shape[1]

    # Building the model
    adLSTM = AnomalyDetectorBuilder.getAnomalyDetector(args.model_name, timesteps, nfeatures, outDirRoot)
    adLSTM.setFeaturesColsNames(ds.getFeaturesColsNames())

    # Logging params to files
    model_params = adLSTM.getModelParams()
    with open(outDirRoot.joinpath("model_params.ini"), "w") as handle:
        for key, val in model_params.items():
            handle.write(f"{key}={val}\n")
        for key, val in training_params.items():
            handle.write(f"{key}={val}\n")
    ds.dumpDatasetParams(type="pickle")
    ds.dumpDatasetParams(type="ini")


    # Compiling the model
    adLSTM.compile()
    adLSTM.summary()

    # Log file for statistics during training
    with open(outDirRoot.joinpath("statistics.csv"), "w") as statFile:
        statFile.write("epoch,training_time_mean,training_time_dev,total_time\n")
    ######################################################################################################################################################################

    maxEpochs = 10
    convergence = False
    fit_cron = Chronometer()

    print(training_params)

    for ep in range(maxEpochs):

        print(f"\nEpoch={ep+1}")

        # Fitting the model
        fit_cron.start()
        adLSTM.fit(train, train, epochs=1, batchSize=training_params["batch_size"], verbose=1, validation_data=(validationSet, validationSet), plotTrainingLoss=False)
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
            maeThreshold, recostructions, maeLossPerEnergyBin, maeLossVal = adLSTM.computeThreshold(validationSet, recoErrMean="simple", showFig=showPlots)
            with open(outDir.joinpath("reconstruction_errors.csv"), "w") as recoFile:
                for vv in maeLossVal.squeeze():
                    recoFile.write(f"{vv}\n")

            # Plotting reconstruction error distribution on the validation set
            # For each energy bin
            adLSTM.recoErrorDistributionPlot(maeLossPerEnergyBin, threshold=None, filenamePostfix=f"reco_error_distribution_on_val_set_for_4_energy_bins", title=f"Reconstruction error distribution on validation set (epoch={ep+1})", showFig=showPlots)
            # Mean recostruction error
            adLSTM.recoErrorDistributionPlot(maeLossVal, threshold=None, filenamePostfix=f"reco_error_distribution_on_val_set_for_1_energy_bin", title=f"Reconstruction error distribution on validation set (epoch={ep+1})", showFig=showPlots)

            # Test for weighted mean
            _, _, _, maeLossVal = adLSTM.computeThreshold(validationSet, recoErrMean="weighted", showFig=showPlots)
            adLSTM.recoErrorDistributionPlot(maeLossVal, threshold=None, filenamePostfix=f"reco_error_distribution_on_val_set_for_1_energy_bin_weighted_mean", title=f"Reconstruction error distribution on validation set (epoch={ep+1})", showFig=showPlots)


            # Plotting some reconstructions
            # recostructions, maeLosses, maeLossesPerEnergyBin, mask = adLSTM.classify_with_mae(validationSet)
            # adLSTM.plotPredictions(validationSet, valLabels, recostructions, maeLossesPerEnergyBin, mask, showFig=False, saveFig=True)

            with open(outDir.joinpath("threshold.txt"), "w") as thresholdFile:
                thresholdFile.write(f"{maeThreshold}")

            # Saving training loss plot
            adLSTM.plotTrainingLoss(showFig=showPlots)

            # Saving time statistics
            with open(outDirRoot.joinpath("statistics.csv"), "a") as statFile:
                statFile.write(f"{ep+1},{fit_cron.get_statistics()[0]},{fit_cron.get_statistics()[1]},{fit_cron.get_total_elapsed_time()}\n")

        if convergence:
            print("Training loss has converged")
            break


    print(f"Total time for training: {fit_cron.total_time} seconds")

    print("Renaming output directory")
    name = f"datasetid_{args.dataset_id}-modelname_{args.model_name}-trainingtype_{args.training_type}-timestamp_{strftime('%Y%m%d-%H%M%S')}"
    target = outDirBase.joinpath(name)
    outDirRoot.rename(target)
    print(f"Results in: {target}")
