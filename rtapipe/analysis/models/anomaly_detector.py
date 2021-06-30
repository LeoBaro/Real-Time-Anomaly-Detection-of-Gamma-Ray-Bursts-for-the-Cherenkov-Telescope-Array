from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow.train import latest_checkpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from rtapipe.analysis.models.builder import ModelBuilder

class AnomalyDetector:

    def __init__(self, shape, units, dropoutRate, loadModelFrom=None):
        try:
            print(f"Loading model from {loadModelFrom}")
            self.model = load_model(loadModelFrom)
            self.fresh = False
        except Exception:
            print(f"Unable to load model from {loadModelFrom}. A new model will be created.")
            self.model = ModelBuilder.buildLSTM_2layers(shape, units, dropoutRate)
            self.fresh = True

        self.history = None
    
    def isFresh(self):
        return self.fresh

    def compile(self):
        # mean absolute error: computes the mean absolute error between labels and predictions.
        self.model.compile(optimizer='adam', loss='mae')

    def summary(self):
        self.model.summary()

    def fit(self, X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=None, validation_split=0.1):

        # batch_size: number of samples per gradient update. 
        # epoch: an epoch is an iteration over the entire x and y data provided
        # validation_split: fraction of the training data to be used as validation data. 
        #                   The model will set apart this fraction of the training data, will 
        #                   not train on it, and will evaluate the loss and any model metrics 
        #                   on this data at the end of each epoch.
        # validation_data: Data on which to evaluate the loss and any model metrics at the end 
        #                  of each epoch. The model will not be trained on this data. Thus, note 
        #                  the fact that the validation loss of data provided using validation_split 
        #                  or validation_data is not affected by regularization layers like noise and dropout.
        
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, validation_split=validation_split, verbose=verbose, callbacks=[])

    def predict(self, samples):
        return self.model.predict(samples, verbose=1)

    def save(self, dir):
        self.model.save(dir)

    def plotLosses(self):
        plt.figure(figsize = (10, 5))
        plt.plot(self.history.history["loss"], label="Training Loss")
        plt.plot(self.history.history["val_loss"], label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.show()

    def plotPredictions(self, realSamples, predictedSamples):
        fig,axes = plt.subplots(5,4)
        c = 0
        for row in range(5):
            for col in range(4):
                axes[row, col].plot(realSamples[c*row+col],color='red', label='Prediction')
                axes[row, col].plot(predictedSamples[c*row+col],color='blue', label='Real Data')
            c += 1

        plt.legend(loc='upper left')
        plt.grid()
        plt.legend()
        plt.show()

 