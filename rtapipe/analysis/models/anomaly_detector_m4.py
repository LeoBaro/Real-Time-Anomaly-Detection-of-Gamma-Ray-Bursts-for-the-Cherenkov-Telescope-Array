from tensorflow import keras
from rtapipe.analysis.models.anomaly_detector_base import AnomalyDetectorBase

class AnomalyDetector_m4(AnomalyDetectorBase):

    @staticmethod
    def loadModel(modelDir):
        try:
            print(f"Loading model from {modelDir}")
            ad = AnomalyDetector_m4(0, 0, Path(modelDir).parent, True)
            ad.model = load_model(modelDir)
            return ad
        except Exception:
            print(f"Unable to load model from {modelDir}.")

    def __init__(self, timesteps, nfeatures, outDir, loadModel = False):
        super().__init__(timesteps, nfeatures, outDir, loadModel)

        print(f"Building AnomalyDetector_m3 - input shape: ({timesteps},{nfeatures})")

        self.mp = {
            "units" : [32, 32],
            "dropoutrate" : [0.2, 0.2],
            "layers" : 4
        }

        if not loadModel:
            self.model = keras.Sequential()
            self.model.add(keras.layers.LSTM(units=self.mp["units"][0], input_shape=(timesteps, nfeatures), activation="tanh", return_sequences=True))
            self.model.add(keras.layers.Dropout(rate=self.mp["dropoutrate"][0]))
            self.model.add(keras.layers.LSTM(units=self.mp["units"][0], activation="tanh"))
            self.model.add(keras.layers.Dropout(rate=self.mp["dropoutrate"][0]))
            self.model.add(keras.layers.RepeatVector(n=timesteps)) # repeats the input n times
            self.model.add(keras.layers.LSTM(units=self.mp["units"][0], activation="tanh", return_sequences=True))
            self.model.add(keras.layers.Dropout(rate=self.mp["dropoutrate"][0]))
            self.model.add(keras.layers.LSTM(units=self.mp["units"][1], return_sequences=True, activation="tanh")) # makes it return the sequence
            self.model.add(keras.layers.Dropout(rate=self.mp["dropoutrate"][1]))
            self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=nfeatures)))# TimeDistributed: creates a vector with a length of the number of outputs from the previous layer
