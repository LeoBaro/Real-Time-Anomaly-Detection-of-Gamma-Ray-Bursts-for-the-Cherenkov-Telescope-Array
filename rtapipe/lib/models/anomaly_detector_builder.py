from tensorflow import keras

from rtapipe.lib.models.anomaly_detector_lstm import *
from rtapipe.lib.models.anomaly_detector_rnn import *
from rtapipe.lib.models.anomaly_detector_cnn import *

class AnomalyDetectorBuilder:

    @staticmethod
    def getAnomalyDetector(name, timesteps, nfeatures, load_model=False, model_dir=None):

        if name == "lstm-m1":
            if load_model:
                ad = AnomalyDetectorLSTM_m1(0, 0, True)
                ad.model = keras.models.load_model(model_dir)
                return ad
            return AnomalyDetectorLSTM_m1(timesteps, nfeatures, load_model)

        elif name == "lstm-m2":
            if load_model:
                ad = AnomalyDetectorLSTM_m2(0, 0, True)
                ad.model = keras.models.load_model(model_dir)
                return ad
            return AnomalyDetectorLSTM_m2(timesteps, nfeatures, load_model)

        elif name == "lstm-m3":
            if load_model:
                ad = AnomalyDetectorLSTM_m3(0, 0, True)
                ad.model = keras.models.load_model(model_dir)
                return ad
            return AnomalyDetectorLSTM_m3(timesteps, nfeatures, load_model)

        elif name == "lstm-m4":
            if load_model:
                ad = AnomalyDetectorLSTM_m4(0, 0, True)
                ad.model = keras.models.load_model(model_dir)
                return ad
            return AnomalyDetectorLSTM_m4(timesteps, nfeatures, load_model)

        else:
            raise ValueError("Model name not supported!")


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
