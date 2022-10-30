from tensorflow import keras

from rtapipe.lib.models.anomaly_detector_base import AnomalyDetectorBase

class AnomalyDetector_cnn_l2_u8(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

        if timesteps == 5:

            self.model_params = {
                "units" : [8, 3],
                "dropoutrate" : [0.2, 0.2],
                "layers" : 2,
                "kernel_size" : [3, 3],
            }

            if not loadModel:
                self.model = keras.Sequential([
                    keras.layers.Conv1D(input_shape=(timesteps, nfeatures), filters=self.model_params["units"][0], kernel_size=self.model_params["kernel_size"][0], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.MaxPooling1D(pool_size=2, padding="same"),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][1], kernel_size=self.model_params["kernel_size"][1], padding="valid", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]),
                ])

        if timesteps == 10:

            self.model_params = {
                "units" : [16, 8, 8, 3],
                "dropoutrate" : [0.2, 0.2],
                "layers" : 2,
                "kernel_size" : [5, 3],
            }

            
            if not loadModel:
                self.model = keras.Sequential([
                    keras.layers.Conv1D(input_shape=(timesteps, nfeatures), filters=self.model_params["units"][0], kernel_size=self.model_params["kernel_size"][0], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.Conv1D(filters=self.model_params["units"][1], kernel_size=self.model_params["kernel_size"][1], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][2], kernel_size=self.model_params["kernel_size"][-1], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][3], kernel_size=self.model_params["kernel_size"][-2], padding="valid", strides=1, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1])
                ])

class AnomalyDetector_cnn_l2_u32(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

        if timesteps == 5:

            self.model_params = {
                "units" : [32, 3],
                "dropoutrate" : [0.2, 0.2],
                "layers" : 2,
                "kernel_size" : [3, 3],
            }

            if not loadModel:
                self.model = keras.Sequential([
                    keras.layers.Conv1D(input_shape=(timesteps, nfeatures), filters=self.model_params["units"][0], kernel_size=self.model_params["kernel_size"][0], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.MaxPooling1D(pool_size=2, padding="same"),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][1], kernel_size=self.model_params["kernel_size"][1], padding="valid", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]),
                ])

        if timesteps == 10:

            self.model_params = {
                "units" : [32, 16, 8, 3],
                "dropoutrate" : [0.2, 0.2],
                "layers" : 2,
                "kernel_size" : [5, 3],
            }

            
            if not loadModel:
                self.model = keras.Sequential([
                    keras.layers.Conv1D(input_shape=(timesteps, nfeatures), filters=self.model_params["units"][0], kernel_size=self.model_params["kernel_size"][0], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.Conv1D(filters=self.model_params["units"][1], kernel_size=self.model_params["kernel_size"][1], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][2], kernel_size=self.model_params["kernel_size"][-1], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][3], kernel_size=self.model_params["kernel_size"][-2], padding="valid", strides=1, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1])
                ])

# TO BE DEPRECATED..no need to many neurons
class AnomalyDetector_cnn_l2_u128(AnomalyDetectorBase):

    def __init__(self, timesteps, nfeatures, loadModel = False, threshold=None):
        super().__init__(timesteps, nfeatures, threshold)

        print(f"{self.__class__.__name__} - input shape: ({timesteps},{nfeatures})")

        if timesteps == 5:

            self.model_params = {
                "units" : [128, 3],
                "dropoutrate" : [0.2, 0.2],
                "layers" : 2,
                "kernel_size" : [3, 3],
            }

            if not loadModel:
                self.model = keras.Sequential([
                    keras.layers.Conv1D(input_shape=(timesteps, nfeatures), filters=self.model_params["units"][0], kernel_size=self.model_params["kernel_size"][0], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.MaxPooling1D(pool_size=2, padding="same"),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][1], kernel_size=self.model_params["kernel_size"][1], padding="valid", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]),
                ])


        if timesteps == 10:

            self.model_params = {
                "units" : [128, 64, 32, 3],
                "dropoutrate" : [0.2, 0.2],
                "layers" : 2,
                "kernel_size" : [5, 3],
            }

            
            if not loadModel:
                self.model = keras.Sequential([
                    keras.layers.Conv1D(input_shape=(timesteps, nfeatures), filters=self.model_params["units"][0], kernel_size=self.model_params["kernel_size"][0], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.Conv1D(filters=self.model_params["units"][1], kernel_size=self.model_params["kernel_size"][1], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][0]),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][2], kernel_size=self.model_params["kernel_size"][-1], padding="same", strides=2, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1]),
                    keras.layers.Conv1DTranspose(filters=self.model_params["units"][3], kernel_size=self.model_params["kernel_size"][-2], padding="valid", strides=1, activation="relu"),
                    keras.layers.Dropout(rate=self.model_params["dropoutrate"][1])
                ])  