import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import argparse
from pathlib import Path
from time import strftime
from shutil import rmtree

from rtapipe.lib.dataset.dataset import APDataset
from rtapipe.lib.rtapipeutils.Chronometer import Chronometer
from rtapipe.lib.models.anomaly_detector_builder import AnomalyDetectorBuilder
from rtapipe.training.callbacks import CustomLogCallback

import wandb
from wandb.keras import WandbCallback

def train(args):

    name = f"datasetid_{args.dataset_id}-modelname_{args.model_name}-trainingtype_{args.training_type}"

    # Output dir
    outDirBase = Path(__file__).parent.resolve().joinpath(args.output_folder)
    outDirRoot = outDirBase.joinpath(f"lstm_models_{strftime('%Y%m%d-%H%M%S')}")
    outDirRoot.mkdir(parents=True, exist_ok=True)

    # Dataset
    training_params = AnomalyDetectorBuilder.getTrainingParams(args.training_type)
    ds = APDataset.get_dataset(args.dataset_config, args.dataset_id, scaler="mm", outDir=outDirRoot)
    ds.loadData(size=training_params["sample_size"])
    ds.plotRandomSample(howMany=4, showFig=False)
    train, trainLabels, validationSet, valLabels = ds.getTrainingAndValidationSet(split=10, scale=True)
    #print(f"Train shape: {train.shape}. Example: {train[0].flatten()}")
    #print(f"Val shape: {val.shape}. Example: {val[0].flatten()}")

    timesteps = train[0].shape[0]
    nfeatures = train[0].shape[1]

    # Building the model
    anomalyDetector = AnomalyDetectorBuilder.getAnomalyDetector(args.model_name, timesteps, nfeatures)

    # Logging params to files
    with open(outDirRoot.joinpath("model_params.ini"), "w") as handle:
        for key, val in anomalyDetector.model_params.items():
            handle.write(f"{key}={val}\n")
        for key, val in training_params.items():
            handle.write(f"{key}={val}\n")
    ds.dumpDatasetParams(type="pickle")
    ds.dumpDatasetParams(type="ini")


    # Compiling the model
    anomalyDetector.model.compile(optimizer='adam', loss='mae')

    anomalyDetector.model.summary()

    # Log file for statistics during training
    with open(outDirRoot.joinpath("statistics.csv"), "w") as statFile:
        statFile.write("epoch,training_time_mean,training_time_dev,total_time\n")

    maxEpochs = args.epochs

    fit_cron = Chronometer()

    print(training_params)

    ## Per-batch fitting.
    ## An epoch is one batch.
    batch_size = training_params["batch_size"]

    callbacks = [
    ]
    
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

    callbacks.append(
        CustomLogCallback(args.save_after, validation_data=(validationSet, valLabels), out_dir_root=outDirRoot, wandb_run=run, metadata={"dataset_id":args.dataset_id,"model":args.model_name,"training":args.training_type})
    )

    start_index = 0

    for ep in range(1, maxEpochs+1):

        train_batch = train[start_index : start_index + batch_size]

        print(f"\nBatch: {ep} datasetid: {args.dataset_id} modelname: {args.model_name} trainingtype: {args.training_type} batchsize: {train_batch.shape}")

        # Fitting the model
        fit_cron.start()
        history = anomalyDetector.model.fit(train_batch, 
                                            train_batch, 
                                            epochs=1, 
                                            batch_size=batch_size, 
                                            verbose=1,
                                            validation_data=(validationSet, validationSet),
                                            callbacks=[callbacks]
        )
        fit_cron.stop()
        print(f"Fitting time: {fit_cron.get_statistics()[0]} +- {fit_cron.get_statistics()[1]}")

        # Saving time statistics
        with open(outDirRoot.joinpath("statistics.csv"), "a") as statFile:
            statFile.write(f"{ep},{round(fit_cron.get_statistics()[0], 2)},{round(fit_cron.get_statistics()[1], 2)},{round(fit_cron.get_total_elapsed_time(), 2)}\n")


    print(f"Total time for training: {fit_cron.total_time} seconds")

    print("Renaming output directory")
    target = outDirBase.joinpath(name)
    if target.exists():
        rmtree(target)
    outDirRoot.rename(target)
    print(f"Results in: {target}")


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-di", "--dataset_id", type=int, required=True)
    parser.add_argument("-tt", "--training_type", type=str, choices = ["light", "medium", "heavy"], required=True)
    parser.add_argument("-of", "--output_folder", type=str, required=True)
    parser.add_argument("-e", "--epochs", type=int, required=True, help="The number of training epochs")
    parser.add_argument('-sa', '--save-after', nargs='+', type=int, required=False, default=1, help="Store trained model after these epochs are reached")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, choices = [0,1], help="If 1 plots will be shown")
    parser.add_argument("-dc", "--dataset_config", type=str, required=False, default="./../lib/dataset/config/agilehost3.yml")
    parser.add_argument('-wb', "--weight_and_biases", type=int, choices=(0,1))
    args = parser.parse_args()

    train(args)

