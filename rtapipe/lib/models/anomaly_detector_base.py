import json
import tensorflow as tf
from pathlib import Path

from rtapipe.lib.evaluation.custom_mse import CustomMSE
from rtapipe.lib.plotting.plotting import plot_predictions

class AnomalyDetectorBase:
    
    def __init__(self, timesteps, nfeatures, threshold):
        self.model = None
        self.timesteps = timesteps
        self.nfeatures = nfeatures
        self.threshold = threshold
        self.loss_f = CustomMSE(nfeatures)

    def reconstruct(self, X):
        return self.model.predict(X)

    def predict(self, X):
        _ = self.loss_f.call(tf.constant(X, dtype=tf.float32), tf.constant(self.reconstruct(X), dtype=tf.float32))
        return [True if sample_loss > self.threshold else False for sample_loss in self.loss_f.mse_per_sample]

    def plot_predictions(self, X, y, features_names=[], max_plots=1, epoch="", showFig=True, saveFig=True, outputDir="./", figName="nn_predictions.png"):
        if self.loss_f.mse_per_sample is None:
            self.predict(X)
        plot_predictions(X, y, self.threshold, self.reconstruct(X), self.loss_f.mse_per_sample.numpy(), self.loss_f.mse_per_sample_features.numpy(), features_names=features_names, max_plots=max_plots, epoch=epoch, showFig=showFig, saveFig=saveFig, outputDir=outputDir, figName=figName)

    def store_parameters(self, dest_path):
        Path(dest_path).mkdir(exist_ok=True, parents=True)
        params = {
            "timesteps" : self.timesteps,
            "nfeatures" : self.nfeatures,
            "name" : self.__class__.__name__
        }
        with open(Path(dest_path).joinpath("model_parameters.json"), "w") as f:
            json.dump(params, f)

    def get_reconstruction_errors(self):
        if self.loss_f.mse_per_sample is None:
            raise Exception("You need to call predict() first")
        return self.loss_f.mse_per_sample.numpy()