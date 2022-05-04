import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import numpy as np
from pathlib import Path
from time import strftime

from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.lib.rtapipeutils.Chronometer import Chronometer
from rtapipe.analysis.models.anomaly_detector_builder import AnomalyDetectorBuilder
from rtapipe.analysis.callbacks import CustomLogCallback

import wandb
from wandb.keras import WandbCallback


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-di", "--dataset_id", type=int, required=True)
    parser.add_argument("-tt", "--training_type", type=str, choices = ["light", "medium", "heavy"], required=True)
    parser.add_argument("-of", "--output_folder", type=str, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True, help="The number of training epochs")
    parser.add_argument('-sa', '--save-after', nargs='+', type=int, required=False, default=1, help="Store trained model after these epochs are reached")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, choices = [0,1], help="If 1 plots will be shown")
    parser.add_argument("-dc", "--dataset_config", type=str, required=False, default="./dataset/config/agilehost3.yml")
    parser.add_argument('-wb', "--weight_and_biases", type=int, choices=(0,1))
    args = parser.parse_args()

    showPlots = False
    if args.verbose == 1:
        showPlots = True

    # Weight and biases
    name = f"datasetid_{args.dataset_id}-modelname_{args.model_name}-trainingtype_{args.training_type}"

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
    train, trainLabels, validationSet, valLabels = ds.getTrainingAndValidationSet(split=10, scale=True)
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

    maxEpochs = args.epochs

    fit_cron = Chronometer()

    print(training_params)

    ## Per-batch fitting.
    ## An epoch is one batch.
    batch_size = training_params["batch_size"]
    start_index = 0

    callbacks = []#CustomLogCallback()]
    
    if args.weight_and_biases == 1:

        config = dict (
            entity="leobaro_",
            architecture = "LSTM",
            dataset_id = args.dataset_id,
            machine = "agilehost3",
            job_type='train',
            batch_size = batch_size,
            model = args.model_name
        )

        run = wandb.init(
            project="phd-lstm",
            config=config
        )


        callbacks.append(WandbCallback())

    for ep in range(1, maxEpochs+1):

        print(f"\nEpoch={ep}")

        titleStr = f"epoch: {ep} datasetid: {args.dataset_id} modelname: {args.model_name} trainingtype: {args.training_type}"

        train_batch = train[start_index : start_index + batch_size]

        print(f"\n batch size: ",train_batch.shape)
        # Fitting the model
        fit_cron.start()
        history = adLSTM.fit( train_batch, 
                              train_batch, 
                              epochs=1, 
                              batchSize=batch_size, 
                              verbose=0,
                              validation_data=(validationSet, validationSet),
                              callbacks=[callbacks]
        )
        fit_cron.stop()
        print(f"Fitting time: {fit_cron.get_statistics()[0]} +- {fit_cron.get_statistics()[1]}")



        # Saving the model

        if ep in args.save_after:

            print("\nSaving the model and plots..")
            outDir = outDirRoot.joinpath("epochs",f"epoch_{ep}")
            outDir.mkdir(exist_ok=True, parents=True)
            adLSTM.setOutputDir(outDir)

            # Saving the model
            if args.save_after:
                adLSTM.save(outDir.joinpath("lstm_trained_model"))

            # Write losses on file
            with open(outDir.joinpath("loss.csv"), "w") as lossFile:
                lossFile.write("trainLoss,valLoss\n")
                for lossHistory in history:
                    lossFile.write(f'{lossHistory.history["loss"][0]},{lossHistory.history["val_loss"][0]}\n')

            # Computing the threshold using a validation set
            maeThreshold, recostructions, maeLossPerEnergyBin, maeLossVal = adLSTM.computeThreshold(validationSet, recoErrMean="simple", showFig=showPlots)
            with open(outDir.joinpath("reconstruction_errors.csv"), "w") as recoFile:
                for vv in maeLossVal.squeeze():
                    recoFile.write(f"{vv}\n")

            # Plotting reconstruction error distribution on the validation set

            # Mean recostruction error
            adLSTM.recoErrorDistributionPlot(maeLossVal, threshold=None, filenamePostfix=f"reco_error_distribution_on_val_set_for_1_energy_bin", title=f"Reco error val set ({titleStr})", showFig=showPlots)
            # For each energy bin
            if ds.dataset_params["integration_type"] == "te":
                adLSTM.recoErrorDistributionPlot(maeLossPerEnergyBin, threshold=None, filenamePostfix=f"reco_error_distribution_on_val_set_for_4_energy_bins", title=f"Reco error val set ({titleStr})", showFig=showPlots)

            # Test for weighted mean
            if ds.dataset_params["integration_type"] == "te":
                _, _, _, maeLossVal = adLSTM.computeThreshold(validationSet, recoErrMean="weighted", showFig=showPlots)
                adLSTM.recoErrorDistributionPlot(maeLossVal, threshold=None, filenamePostfix=f"reco_error_distribution_on_val_set_for_1_energy_bin_weighted_mean", title=f"Reco error val set ({titleStr})", showFig=showPlots)


            # Plotting some reconstructions
            recostructions, maeLosses, maeLossesPerEnergyBin, mask = adLSTM.classifyWithMae(validationSet)
            adLSTM.plotPredictions(validationSet, valLabels, recostructions, maeLossesPerEnergyBin, mask, maxSamples=5, rows=2, cols=5, predictionsPerFigure=5, showFig=False, saveFig=True)

            with open(outDir.joinpath("threshold.txt"), "w") as thresholdFile:
                thresholdFile.write(f"{maeThreshold}")

            # Saving training loss plot
            adLSTM.plotTrainingLoss(showFig=showPlots, title=f"Training loss ({titleStr})")
            adLSTM.plotTrainingLoss(ylim=(0, 0.4), showFig=showPlots, figName="loss_plot_y_norm.png", title=f"Training loss ({titleStr})")

            # Saving time statistics
            with open(outDirRoot.joinpath("statistics.csv"), "a") as statFile:
                statFile.write(f"{ep},{round(fit_cron.get_statistics()[0], 2)},{round(fit_cron.get_statistics()[1], 2)},{round(fit_cron.get_total_elapsed_time(), 2)}\n")

            artifact = wandb.Artifact(f'run-{run.id}', type='result', metadata = {
                "run" : run.id,
                "epoch" : ep
            })
            artifact.add_file(outDir.joinpath("mae_distr_reco_error_distribution_on_val_set_for_1_energy_bin_weighted_mean.png"))
            artifact.add_file(outDir.joinpath("mae_distr_reco_error_distribution_on_val_set_for_1_energy_bin.png"))
            artifact.add_file(outDir.joinpath("mae_distr_reco_error_distribution_on_val_set_for_4_energy_bins.png"))
            artifact.add_file(outDir.joinpath("predictions_0_feature_0.png"))
            artifact.add_file(outDir.joinpath("predictions_0_feature_1.png"))
            artifact.add_file(outDir.joinpath("predictions_0_feature_2.png"))
            artifact.add_file(outDir.joinpath("predictions_0_feature_3.png"))
            artifact.add_file(outDirRoot.joinpath("statistics.csv"))
            run.log_artifact(artifact)

    print(f"Total time for training: {fit_cron.total_time} seconds")

    print("Renaming output directory")
    target = outDirBase.joinpath(name)
    if target.exists():
        target.unlink()
    outDirRoot.rename(target)
    print(f"Results in: {target}")
