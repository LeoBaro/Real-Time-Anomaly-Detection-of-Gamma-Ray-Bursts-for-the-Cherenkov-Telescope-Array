from pathlib import Path
from tensorflow import keras
from rtapipe.analysis.models.anomaly_detector_base import AnomalyDetectorBase

class AnomalyDetector_m1(AnomalyDetectorBase):

    @staticmethod
    def loadModel(modelDir):
        print(f"Loading model from {modelDir}")
        ad = AnomalyDetector_m1(0, 0, Path(modelDir).parent, True)
        ad.model = keras.models.load_model(modelDir)
        return ad


    def __init__(self, timesteps, nfeatures, outDir, loadModel = False):
        super().__init__(timesteps, nfeatures, outDir, loadModel)

        print(f"Building AnomalyDetector_m1 - input shape: ({timesteps},{nfeatures})")

        self.mp = {
            "units" : [8, 8],
            "dropoutrate" : [0.2, 0.2],
            "layers" : 2
        }

        if not loadModel:
            self.model = keras.Sequential()
            self.model.add(keras.layers.LSTM(units=self.mp["units"][0], input_shape=(timesteps, nfeatures), activation="tanh"))
            self.model.add(keras.layers.Dropout(rate=self.mp["dropoutrate"][0]))
            self.model.add(keras.layers.RepeatVector(n=timesteps)) # repeats the input n times
            self.model.add(keras.layers.LSTM(units=self.mp["units"][1], return_sequences=True, activation="tanh")) # makes it return the sequence
            self.model.add(keras.layers.Dropout(rate=self.mp["dropoutrate"][1]))
            self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=nfeatures)))# TimeDistributed: creates a vector with a length of the number of outputs from the previous layer
