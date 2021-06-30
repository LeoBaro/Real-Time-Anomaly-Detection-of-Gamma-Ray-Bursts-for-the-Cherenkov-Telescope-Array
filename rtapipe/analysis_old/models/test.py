import tensorflow as tf
from os import getcwd
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

print(f"tf.__version__: {tf.__version__}")

master_url_root = "https://raw.githubusercontent.com/numenta/NAB/master/data/"

df_small_noise_url_suffix = "artificialNoAnomaly/art_daily_small_noise.csv"
df_small_noise_url = master_url_root + df_small_noise_url_suffix
df_small_noise = pd.read_csv(
    df_small_noise_url, parse_dates=True, index_col="timestamp"
)

df_daily_jumpsup_url_suffix = "artificialWithAnomaly/art_daily_jumpsup.csv"
df_daily_jumpsup_url = master_url_root + df_daily_jumpsup_url_suffix
df_daily_jumpsup = pd.read_csv(
    df_daily_jumpsup_url, parse_dates=True, index_col="timestamp"
)
print(df_small_noise.head())
print(df_daily_jumpsup.head())


training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
print("Number of training samples:", len(df_training_value))

df_training_value.head()

TIME_STEPS = 288

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS):
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


x_train = create_sequences(df_training_value.values)
print("Training input shape: ", x_train.shape)

"""
# ## Convolutional Autoencoder model
modelConv = keras.Sequential(
    [
        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
        layers.Conv1D(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1D(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(
            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Dropout(rate=0.2),
        layers.Conv1DTranspose(
            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
        ),
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
    ]
)
modelConv.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
modelConv.summary()
"""

# ## LSTM Autoencoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Input, Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.models import Model

from sklearn.preprocessing import MinMaxScaler, StandardScaler


modelLSTM = Sequential()
modelLSTM.add(LSTM(128, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True))
modelLSTM.add(LSTM(64, activation='relu', return_sequences=False))
modelLSTM.add(RepeatVector(x_train.shape[1]))
modelLSTM.add(LSTM(64, activation='relu', return_sequences=True))
modelLSTM.add(LSTM(128, activation='relu', return_sequences=True))
modelLSTM.add(TimeDistributed(Dense(x_train.shape[2])))

modelLSTM.compile(optimizer='adam', loss='mse')
modelLSTM.summary()


modelLSTM2 = Sequential()
modelLSTM2.add(LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
modelLSTM2.add(Dropout(rate=0.2))
modelLSTM2.add(RepeatVector(x_train.shape[1]))
modelLSTM2.add(LSTM(128, return_sequences=True))
modelLSTM2.add(Dropout(rate=0.2))
modelLSTM2.add(TimeDistributed(Dense(x_train.shape[2])))

modelLSTM2.compile(optimizer='adam', loss='mae')
modelLSTM2.summary()

epochs = 10

modelLSTMHistory = modelLSTM.fit(x_train, x_train, epochs=epochs, batch_size=128, validation_split=0.1, verbose=1)

modelLSTM2History = modelLSTM2.fit(x_train, x_train, epochs=epochs, batch_size=128, validation_split=0.1, verbose=1)


"""
modelConvHistory = modelConv.fit(
    x_train,
    x_train,
    epochs=50,
    batch_size=128,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
    ],
)
"""
plt.plot(modelLSTMHistory.history["loss"], label="lstm1 Training Loss", color="blue", linestyle='-.')
plt.plot(modelLSTMHistory.history["val_loss"], label="lstm1 Validation Loss", color="blue",)
plt.plot(modelLSTM2History.history["loss"], label="lstm2 Training Loss", color="purple", linestyle='-.')
plt.plot(modelLSTM2History.history["val_loss"], label="lstm2 Validation Loss", color="purple")

plt.legend()
plt.savefig("./los.png")


# ## Loss
def plotLoss(pred_data, real, labels=[""]):
    for i, pred in enumerate(pred_data):
        train_mae_loss = np.mean(np.abs(pred - real), axis=1)
        plt.hist(train_mae_loss, bins=50, label=labels[i])
        plt.xlabel("Train MAE loss")
        plt.ylabel("No of samples")
        # Get reconstruction loss threshold.
        threshold = np.max(train_mae_loss)
        print("Reconstruction error threshold: ", threshold)
    plt.legend()
    plt.savefig("./los2.png")

#autoencoder_predictions_on_training = modelConv.predict(x_train)
lstm_predictions_on_training = modelLSTM.predict(x_train)

#print(autoencoder_predictions_on_training.shape)
print(lstm_predictions_on_training.shape)

plotLoss([lstm_predictions_on_training], x_train, labels=["Lstm"])

plt.plot(x_train[0])
plt.plot(lstm_predictions_on_training[0], label="Lstm")
plt.legend()
plt.show()
plt.savefig("./los3.png")
