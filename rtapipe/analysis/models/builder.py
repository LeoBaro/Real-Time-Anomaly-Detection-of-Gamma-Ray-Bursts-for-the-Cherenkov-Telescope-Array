from tensorflow import keras
from tensorflow.keras import layers


class ModelBuilder:

  @staticmethod
  def buildLSTM_2layers(inputShape, units=64, dropoutRate=0.2):

    print(f"Building 2 layers - LSTM model. Input shape: {inputShape}")
    
    model = keras.Sequential()
    
    model.add(keras.layers.LSTM(
        units=units,
        input_shape=(inputShape[0], inputShape[1])
    ))
    model.add(keras.layers.Dropout(rate=dropoutRate))
    model.add(keras.layers.RepeatVector(n=inputShape[0])) # repeats the input n times
    
    model.add(keras.layers.LSTM(units=units, return_sequences=True)) # makes it return the sequence
    model.add(keras.layers.Dropout(rate=dropoutRate))
    
    model.add(
      keras.layers.TimeDistributed( # creates a vector with a length of the number of outputs from the previous layer
        keras.layers.Dense(units=inputShape[1])
      )
    )

    return model

