from tensorflow import keras

from rtapipe.lib.models.anomaly_detector_base import AnomalyDetectorBase

class AnomalyDetector_lstm_l2_u8(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

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

class AnomalyDetector_lstm_l2_u32(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

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

class AnomalyDetector_lstm_l2_u128(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

        self.model_params = {
            "units" : [128, 128],
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


class AnomalyDetector_lstm_l4_u8(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

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


class AnomalyDetector_lstm_l4_u32(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

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


class AnomalyDetector_lstm_l4_u128(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

        self.model_params = {
            "units" : [128, 128],
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
