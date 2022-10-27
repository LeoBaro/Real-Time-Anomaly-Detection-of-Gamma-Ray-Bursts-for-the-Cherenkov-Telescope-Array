import wandb
import numpy as np
import tensorflow as tf
from tensorflow import keras
from rtapipe.lib.plotting import plotting
from rtapipe.lib.evaluation.custom_mse import CustomMSE 

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
        self.count = 1
        self.trigger_after_epochs = trigger_after_epochs
        self.validation_data = validation_data[0]
        self.validation_data_labels = validation_data[1]
        self.out_dir_root = out_dir_root
        self.wandb_run = wandb_run
        self.metadata = metadata



    def on_epoch_end(self, batch, logs=None, force=False):

        epoch = self.count
        
        if epoch in self.trigger_after_epochs or force:
            
            print(f"\n\n----------------- Checkpoint! Saving data at epoch {epoch} (Triggered by Early Stopping={force}) -----------------")

            out_dir = self.out_dir_root.joinpath("epochs",f"epoch_{epoch}")
            out_dir.mkdir(exist_ok=True, parents=True)

            # Saving the model
            self.model.save(out_dir.joinpath("trained_model"))

            # Compute the loss plot before loss data is overwritten
            # plotting.loss_plot(self.model.history.history["loss"], self.model.history.history["val_loss"], outputDir=out_dir, figName="train_val_loss.png", showFig=False)

            recostructions = self.model.predict(self.validation_data, verbose=1)

            custom_mse = CustomMSE(n_features = self.validation_data.shape[-1], output_dir=out_dir)
            loss = custom_mse.call(tf.constant(self.validation_data, dtype=tf.float32), tf.constant(recostructions, dtype=tf.float32))
            custom_mse.write_reconstruction_errors()

            # The threshold is calculated as the 98% quantile of the mean absolute errors distribution for the normal examples of the training set,
            # then classify future examples as anomalous if the reconstruction error is higher than one standard
            # deviation from the training set.
            c_threshold = np.percentile(custom_mse.mse_per_sample.numpy(), 98)
            with open(out_dir.joinpath("threshold.txt"), "w") as tfile:
                tfile.write(f"{c_threshold}")


            # Plotting
            plotting.reco_error_distribution_plot(custom_mse.mse_per_sample_features, threshold=None, title=f"Reco errors on val set ({self.metadata})", outputDir=out_dir, figName="reco_errors_distr_per_features.png", showFig=False)
            plotting.reco_error_distribution_plot(custom_mse.mse_per_sample,          threshold=None, title=f"Reco errors on val set ({self.metadata})", outputDir=out_dir, figName="reco_errors_distr_per_sample.png", showFig=False)
            plotting.plot_predictions(self.validation_data, self.validation_data_labels, c_threshold, recostructions, custom_mse.mse_per_sample.numpy(), custom_mse.mse_per_sample_features.numpy(), max_plots=1, showFig=False, saveFig=True, outputDir=out_dir, figName="predictions.png")
                                      
            # Logging to wandb
            if self.wandb_run is not None:
                artifact = wandb.Artifact(f'run-{self.wandb_run.id}', type='result', metadata = {
                    "run" : self.wandb_run.id,
                    "epoch" : epoch
                })
                artifact.add_file(out_dir.joinpath("reco_errors_distr_per_features.png"))
                artifact.add_file(out_dir.joinpath("reco_errors_distr_per_sample.png"))
                for i in range(self.validation_data.shape[-1]):
                    artifact.add_file(out_dir.joinpath(f"0_feature_{i}_predictions.png"))
                artifact.add_file(out_dir.parent.parent.joinpath("statistics.csv"))
                self.wandb_run.log_artifact(artifact)
            
        self.count += 1
