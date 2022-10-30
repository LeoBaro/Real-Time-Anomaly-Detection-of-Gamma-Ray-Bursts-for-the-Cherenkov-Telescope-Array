import json
import tensorflow as tf
from pathlib import Path

from rtapipe.lib.evaluation.custom_mse import CustomMSE
from rtapipe.lib.plotting.plotting import plot_predictions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

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

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return self.evaluate_predictions(y, y_pred)

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


    def evaluate_predictions(self, y, y_pred):
        """
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)        
        """
        cm = confusion_matrix(y, y_pred)
        return {
            "accuracy" : accuracy_score(y, y_pred),
            "precision" : precision_score(y, y_pred),
            "recall" : recall_score(y, y_pred),
            "f1" : f1_score(y, y_pred),
            "roc_auc" : roc_auc_score(y, y_pred),
            "confusion_matrix" : cm.tolist(),
            "false_positive_rate" : cm[0][1] / (cm[0][1] + cm[1][1]),
        }