from tensorflow import keras

class AnomalyDetector_m2:

    def __init__(self, timesteps, nfeatures, loadModel = False):

        print(f"Building AnomalyDetector_m2 - input shape: ({timesteps},{nfeatures})")

        self.model_params = {
            "units" : [32, 32],
            "dropoutrate" : [0.2, 0.2],
            "layers" : 2
        }

        if not loadModel:
            self.model = keras.Sequential()
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][0], input_shape=(timesteps, nfeatures), activation="tanh"))
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]))
            self.model.add(keras.layers.RepeatVector(n=timesteps)) # repeats the input n times
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][1], return_sequences=True, activation="tanh")) # makes it return the sequence
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]))
            self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=nfeatures)))# TimeDistributed: creates a vector with a length of the number of outputs from the previous layer
