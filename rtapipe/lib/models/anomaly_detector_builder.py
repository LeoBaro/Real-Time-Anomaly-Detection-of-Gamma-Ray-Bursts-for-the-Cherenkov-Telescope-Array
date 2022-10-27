from pathlib import Path
from tensorflow import keras

from rtapipe.lib.models.anomaly_detector_lstm import *
from rtapipe.lib.models.anomaly_detector_rnn import *
from rtapipe.lib.models.anomaly_detector_cnn import *
from rtapipe.lib.evaluation.custom_mse import CustomMSE

class AnomalyDetectorBuilder:

    @staticmethod
    def getModelsName():
        return [class_name for class_name in globals() if "AnomalyDetector_" in class_name ]

    @staticmethod
    def getAnomalyDetector(name, timesteps, nfeatures, load_model=False, training_epoch_dir=None, training=True):
        
        klass = globals()[name]

        if not Path(training_epoch_dir).exists():
            raise Exception(f"Training epoch dir {training_epoch_dir} does not exist")

        model_dir = Path(training_epoch_dir).joinpath("trained_model")
        if not model_dir.exists():
            raise Exception(f"Model dir {model_dir} does not exist")

        threshold_file = Path(training_epoch_dir).joinpath("threshold.txt")
        if not threshold_file.exists():
            raise Exception(f"Threshold file {threshold_file} does not exist")

        with open(threshold_file, "r") as thf:
            threshold = float(thf.read().rstrip().strip())

        if load_model:
            ad = klass(timesteps, nfeatures, loadModel=True, threshold=threshold)
            ad.model = keras.models.load_model(model_dir, compile=training)
            return ad

        return klass(timesteps, nfeatures, False, None)


    @staticmethod
    def getTrainingParams(trainingType):
        if trainingType == "light":
            return {
                "sample_size" : 1024,
                "batch_size" : 4
            }
        elif trainingType == "medium":
            return {
                "sample_size" : 1024,
                "batch_size" : 8
            }
        elif trainingType == "heavy":
            return {
                "sample_size" : 1024,
                "batch_size" : 16
            }
        else:
            raise ValueError()

