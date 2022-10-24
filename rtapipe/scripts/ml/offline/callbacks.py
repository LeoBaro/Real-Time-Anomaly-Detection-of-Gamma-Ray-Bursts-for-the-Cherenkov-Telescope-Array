import wandb
import numpy as np
from tensorflow import keras
from rtapipe.lib.plotting import plotting

class EarlyStoppingCallback(keras.callbacks.Callback):

    def __init__(self, patience, delta, out_dir_root, metadata, wandb_run=None):
        self.count = 0
        self.out_dir_root = out_dir_root
        self.wandb_run = wandb_run
        self.metadata = metadata
        self.count_batch = 0
        self.patience = patience
        self.val_loss_prev = None
        self.val_loss_diff = None
        self.delta = delta 
        self.ok_count = 0
        self.ok_prev = False

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        self.count_batch += 1
        print("End epoch {} of training; got log keys: {}. Batch: {}".format(epoch, keys, self.count_batch))

        if self.val_loss_prev is not None:
            self.val_loss_diff = round(abs(self.val_loss_prev - logs["val_loss"]), 5)
            print(f"Val loss diff: {self.val_loss_diff:.8f}") 
            if self.val_loss_diff <= self.delta:
                if self.ok_prev:
                    self.ok_count += 1
                else:
                    self.ok_prev = True
            else:
                self.ok_prev = False
                self.ok_count = 0
            
            print(f"Ok count: {self.ok_count}")
            if self.ok_count == self.patience:
                self.model.stop_training = True
                self.stopped_epoch = epoch
        
        self.val_loss_prev = logs["val_loss"]

class CustomLogCallback(keras.callbacks.Callback):
    
    def __init__(self, trigger_after_epochs, validation_data, out_dir_root, metadata, wandb_run=None):
        self.count = 0
        self.trigger_after_epochs = trigger_after_epochs
        self.validation_data = validation_data[0]
        self.validation_data_labels = validation_data[1]
        self.out_dir_root = out_dir_root
        self.wandb_run = wandb_run
        self.metadata = metadata
    
    def on_train_end(self, batch, logs=None, force=False):
        print("--------------------- CustomLogCallback on_train_end:")

        self.count += 1
        epoch = self.count
        
        out_dir = self.out_dir_root.joinpath("epochs",f"epoch_{epoch}")
        out_dir.mkdir(exist_ok=True, parents=True)

        if epoch in self.trigger_after_epochs or force:
            
            print("Checkpoint! Saving data -----------------------------")

            # Saving the model
            self.model.save(out_dir.joinpath("lstm_trained_model"))

            # Saving the reconstruction errors 

            # The reconstructions:
            # [
            #   [
            #       [1],[2],[3],[4],[5]
            #   ],
            #   [
            #       [1],[2],[3],[4],[5]
            #   ],
            #   ... up to X samples
            # ]
            #
            # In the case of multiple features..
            # [
            #   [
            #       [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]
            #   ],
            #   [
            #       [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]
            #   ],
            #   ... up to X samples
            # ]
            recostructions = self.model.predict(self.validation_data, verbose=1)

            # For each sample, it computes X arrays of distances between the points, where X is the number of energy bins.
            distances = np.abs(recostructions - self.validation_data)
            # TODO: 
            # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

            # The mae loss is defined as the mean of those distances, for each sample, for each energy bin.
            # In this example we have 4 energy bins
            # [
            #   [0.11891678 0.11762658 0.11594792 0.08139625]
            #   [0.11626169 0.11343022 0.12967022 0.08330123]
            #   ... up to X samples
            # ]
            #
            maeLossPerEnergyBin = np.mean(distances, axis=1)

            plotting.recoErrorDistributionPlot(maeLossPerEnergyBin, threshold=None, title=f"Reco errors on val set ({self.metadata})", outputDir=out_dir, figName="reco_errors_distr_1_en_bin.png", showFig=False)

            # How do I "merge" the mae of each energy bin? Maybe with a weighted mean. For now I'll use a simple mean.
            # [
            #   0.10847188,
            #   0.11066584
            #   ... up to X samples
            # ]
            #
            maeLoss = np.mean(maeLossPerEnergyBin, axis=1)

            plotting.recoErrorDistributionPlot(maeLoss, threshold=None, title=f"Reco errors on val set ({self.metadata})", outputDir=out_dir, figName="reco_errors_distr_4_en_bins.png", showFig=False)

            with open(out_dir.joinpath("reconstruction_errors.csv"), "w") as recoFile:
                for vv in maeLoss.squeeze():
                    recoFile.write(f"{vv}\n")


            # The mean is weighted to give more importance to high energy photons
            maeLossAveraged = np.average(maeLossPerEnergyBin, axis=1, weights=[1./10, 1./5, 3./10, 2./5])

            plotting.recoErrorDistributionPlot(maeLossAveraged, threshold=None, title=f"Reco errors (averaged) on val set ({self.metadata})", outputDir=out_dir, figName="reco_errors_distr_4_en_bins_averaged.png", showFig=False)

            with open(out_dir.joinpath("reconstruction_errors_averaged.csv"), "w") as recoFile:
                for vv in maeLossAveraged.squeeze():
                    recoFile.write(f"{vv}\n")


            # The threshold is calculated as the 98% quantile of the mean absolute errors distribution for the normal examples of the training set,
            # then classify future examples as anomalous if the reconstruction error is higher than one standard
            # deviation from the training set.
            c_threshold = np.percentile(maeLoss, 98)
            c_threshold_avg = np.percentile(maeLossAveraged, 98)

            with open(out_dir.joinpath("threshold.txt"), "w") as thresholdFile:
                thresholdFile.write(f"{c_threshold}")
            with open(out_dir.joinpath("threshold_avg.txt"), "w") as thresholdFile:
                thresholdFile.write(f"{c_threshold_avg}")

            # Plotting some reconstructions
            mask = (maeLoss > c_threshold)
            mask_avg = (maeLossAveraged > c_threshold_avg)

            plotting.plotPredictions(self.validation_data, self.validation_data_labels, c_threshold, recostructions, maeLossPerEnergyBin, mask, maxSamples=5, rows=2, cols=5, predictionsPerFigure=5, showFig=False, saveFig=True, outputDir=out_dir, figName="predictions.png")
            
            if self.wandb_run is not None:
                artifact = wandb.Artifact(f'run-{self.wandb_run.id}', type='result', metadata = {
                    "run" : self.wandb_run.id,
                    "epoch" : epoch
                })
                artifact.add_file(out_dir.joinpath("reco_errors_distr_1_en_bin.png"))
                artifact.add_file(out_dir.joinpath("reco_errors_distr_4_en_bins.png"))
                artifact.add_file(out_dir.joinpath("reco_errors_distr_4_en_bins_averaged.png"))
                for i in range(4):
                    artifact.add_file(out_dir.joinpath(f"0_feature_{i}_predictions.png"))
                artifact.add_file(out_dir.parent.parent.joinpath("statistics.csv"))
                self.wandb_run.log_artifact(artifact)
            
            print("---------------Checkpoint ended! -----------------------------")
