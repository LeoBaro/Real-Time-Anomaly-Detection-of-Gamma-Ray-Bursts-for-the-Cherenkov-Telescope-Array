from tensorflow import keras
from tensorflow.keras import layers


class ModelBuilder:

    @staticmethod
    def buildLSTM_1(timesteps, nfeatures):

    @staticmethod
    def buildLSTM_2(inputShape, units, dropoutRate):

        print(f"Building LSTM 2 - input shape: {inputShape}")

        model = keras.Sequential()

        model.add(keras.layers.LSTM(units=units, activation=activation, input_shape=(inputShape[0],inputShape[1])))
        model.add(keras.layers.RepeatVector(n=inputShape[0]))
        model.add(keras.layers.LSTM(units=units, activation=activation, return_sequences=True))
        model.add(keras.layers.TimeDistributed(units=inputShape[1]))

        return model
