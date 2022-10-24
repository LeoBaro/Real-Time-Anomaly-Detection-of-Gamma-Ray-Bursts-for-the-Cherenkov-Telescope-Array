from tensorflow import keras

class AnomalyDetectorLSTM_m1:

    def __init__(self, timesteps, nfeatures, loadModel = False):

        print(f"AnomalyDetector_m1 - input shape: ({timesteps},{nfeatures})")

        self.model_params = {
            "units" : [8, 8],
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

class AnomalyDetectorLSTM_m2:

    def __init__(self, timesteps, nfeatures, loadModel = False):

        print(f"AnomalyDetectorLSTM_m2 - input shape: ({timesteps},{nfeatures})")

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

class AnomalyDetectorLSTM_m3:

    def __init__(self, timesteps, nfeatures, loadModel = False):

        print(f"AnomalyDetectorLSTM_m3 - input shape: ({timesteps},{nfeatures})")

        self.model_params = {
            "units" : [8, 8],
            "dropoutrate" : [0.2, 0.2],
            "layers" : 4
        }

        if not loadModel:
            self.model = keras.Sequential()
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][0], input_shape=(timesteps, nfeatures), activation="tanh", return_sequences=True))
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]))
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][0], activation="tanh"))
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]))
            self.model.add(keras.layers.RepeatVector(n=timesteps)) # repeats the input n times
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][0], activation="tanh", return_sequences=True))
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]))
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][1], return_sequences=True, activation="tanh")) # makes it return the sequence
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]))
            self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=nfeatures)))# TimeDistributed: creates a vector with a length of the number of outputs from the previous layer

class AnomalyDetectorLSTM_m4:

    def __init__(self, timesteps, nfeatures, loadModel = False):

        print(f"AnomalyDetectorLSTM_m4 - input shape: ({timesteps},{nfeatures})")

        self.model_params = {
            "units" : [32, 32],
            "dropoutrate" : [0.2, 0.2],
            "layers" : 4
        }

        if not loadModel:
            self.model = keras.Sequential()
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][0], input_shape=(timesteps, nfeatures), activation="tanh", return_sequences=True))
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]))
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][0], activation="tanh"))
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]))
            self.model.add(keras.layers.RepeatVector(n=timesteps)) # repeats the input n times
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][0], activation="tanh", return_sequences=True))
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]))
            self.model.add(keras.layers.LSTM(units=self.model_params["units"][1], return_sequences=True, activation="tanh")) # makes it return the sequence
            self.model.add(keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]))
            self.model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=nfeatures)))# TimeDistributed: creates a vector with a length of the number of outputs from the previous layer
