from tensorflow import keras

from rtapipe.lib.models.anomaly_detector_m1 import AnomalyDetector_m1
from rtapipe.lib.models.anomaly_detector_m2 import AnomalyDetector_m2
from rtapipe.lib.models.anomaly_detector_m3 import AnomalyDetector_m3
from rtapipe.lib.models.anomaly_detector_m4 import AnomalyDetector_m4

class AnomalyDetectorBuilder:

    @staticmethod
    def getAnomalyDetector(name, timesteps, nfeatures, load_model=False, model_dir=None):

        if name == "m1":
            if load_model:
                ad = AnomalyDetector_m1(0, 0, True)
                ad.model = keras.models.load_model(model_dir)
                return ad
            return AnomalyDetector_m1(timesteps, nfeatures, load_model)

        elif name == "m2":
            if load_model:
                ad = AnomalyDetector_m2(0, 0, True)
                ad.model = keras.models.load_model(model_dir)
                return ad
            return AnomalyDetector_m2(timesteps, nfeatures, load_model)

        elif name == "m3":
            if load_model:
                ad = AnomalyDetector_m3(0, 0, True)
                ad.model = keras.models.load_model(model_dir)
                return ad
            return AnomalyDetector_m3(timesteps, nfeatures, load_model)

        elif name == "m4":
            if load_model:
                ad = AnomalyDetector_m4(0, 0, True)
                ad.model = keras.models.load_model(model_dir)
                return ad
            return AnomalyDetector_m4(timesteps, nfeatures, load_model)

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
