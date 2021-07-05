import random
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tensorflow import reduce_sum
from tensorflow.keras.losses import mae
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
        self.classificationThreshold = None
    
    def isFresh(self):
        return self.fresh

    def compile(self):
        # mean absolute error: computes the mean absolute error between labels and predictions.
        self.model.compile(optimizer='adam', loss='mae')

    def summary(self):
        self.model.summary()

    def fit(self, X_train, y_train, epochs=50, batchSize=32, verbose=1, validation_data=None, validation_split=0.1, plotLosses=True):

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
        
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, validation_data=validation_data, validation_split=validation_split, verbose=verbose, callbacks=[])

        if plotLosses:
            self.lossPlot(self.history)

    def predict(self, samples):
        return self.model.predict(samples, verbose=1)

    def save(self, dir):
        self.model.save(dir)

    def computeThreshold(self, trainSamples, plotError=True):
        # The threshold is calculated as the mean absolute error for normal examples from the training set, 
        # then classify future examples as anomalous if the reconstruction error is higher than one standard 
        # deviation from the training set.    
        trainPred = self.predict(trainSamples)

        # Computes the mean absolute error between labels and predictions.
        trainMAElosses = np.mean(np.abs(trainPred - trainSamples), axis=1)
        self.classificationThreshold = np.max(trainMAElosses)

        # Compute the threshold as the max of those errors
        print("Threshold: ",self.classificationThreshold)

        # Plotting the errors distributions
        if plotError:
            self.recoErrorDistributionPlot(trainMAElosses, threshold=self.classificationThreshold, saveFig=True,)

    def classify(self, samples, plotError=True):

        if self.classificationThreshold is None:
            print("The classification threshold is None. Call computeThreshold() to calculate it.")
            return None

        # encoding and decoding    
        recostructions = self.predict(samples)
        print("Recostructions:",recostructions)
        # computing the recostruction errors
        maeLosses = np.mean(np.abs(recostructions - samples), axis=1)

        if plotError:
            self.recoErrorDistributionPlot(maeLosses, threshold=self.classificationThreshold, saveFig=False)

        mask = (self.classificationThreshold > maeLosses).flatten()

        print("Anomalies: ",mask)
        
        return recostructions, maeLosses, mask

        
    def computeScore(self):
        pass


    def lossPlot(self, history):
        plt.figure(figsize = (10, 5))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend()
        plt.show()

    def recoErrorDistributionPlot(self, losses, threshold=None, saveFig=False, show=True):
        fig, ax = plt.subplots(1,1)
        ax.hist(losses, bins=25)
        ax.axvline(x=self.classificationThreshold, color="red")
        plt.xlabel('Train MAE loss')
        plt.ylabel('Number of Samples')
        if show:
            plt.show()
        if saveFig:    
            fig.savefig("./plots/mae_distribution.png")        



    def plotPredictions2(self, samples, samplesLabels, recostructions, mask, howMany):
        
        """
        classifiedAsAnomalies = samples[mask,:,:]
        classifiedAsNotAnomalies = samples[np.invert(mask),:,:]
    
        recoClassifiedAsAnomalies = recostructions[mask,:,:]
        recoClassifiedAsNotAnomalies = recostructions[np.invert(mask),:,:]
        """

        classifiedAsAnomalieslabels = samplesLabels[mask]
        classifiedAsNotAnomaliesLabels = samplesLabels[np.invert(mask)]
        print("shape:",classifiedAsAnomalieslabels.shape)


        #selectionArrayForAnomalies = random.sample(range(0, classifiedAsAnomalies.shape[0]), howMany)
        #selectionArrayForNotAnomalies = random.sample(range(0, classifiedAsNotAnomalies.shape[0]), howMany)
        #print("selectionArrayForAnomalies: ",selectionArrayForAnomalies)
        
        # Compute the number of subplots based on the number of samples to be drawn        
        rows = 1
        cols = 1
        while rows*cols < howMany:
            if (rows + cols) % 2 == 0:
                rows += 1  
            else: 
                cols += 1
        print(f"rows: {rows}, cols: {cols}")

        fig, ax = plt.subplots(rows,cols)
        fig.suptitle("Classifi")
        count = 0
        for row in range(rows):
            for col in range(cols):                
                if count >= len(samples):
                    break

                recoSample = recostructions[count].squeeze()
                sample = samples[count].squeeze()
                ax[row, col].plot(recoSample, color='red') #, label='Reconstructed')
                ax[row, col].plot(sample, color='blue') #, label=f'Real sample. grb={classifiedAsAnomalieslabels[selectionArrayForAnomalies[count]]}')
                ax[row, col].set_ylim(0, 1.5)
                if mask[count]:
                    label = "bkg"  
                else: 
                    label = "grb" 
                ax[row, col].set_title(f"LABEL={label}")

                # ax[row, col].legend(loc='upper left')
                ax[row, col].grid()
                count += 1

        plt.show()

