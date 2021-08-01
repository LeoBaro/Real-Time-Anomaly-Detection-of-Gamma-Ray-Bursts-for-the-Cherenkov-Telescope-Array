import random
import numpy as np
from pathlib import Path
from time import strftime
import matplotlib.pyplot as plt
from tensorflow import reduce_sum
from tensorflow.keras.losses import mae
from tensorflow.train import latest_checkpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from rtapipe.analysis.models.builder import ModelBuilder
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, precision_recall_curve, f1_score
from sklearn import metrics


plt.rcParams.update({'font.size': 14, 'lines.markersize': 0.5,'legend.markerscale': 3, 'lines.linewidth':1, 'lines.linestyle':'-'})

class AnomalyDetector:

    def __init__(self, shape, units, dropoutRate, loadModelFrom=None, outDir=None):
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

        if not outDir:
            self.outDir = Path(__file__).parent.resolve().joinpath(f"anomaly_detector_plots_{strftime('%Y%m%d-%H%M%S')}")
            self.outDir.mkdir(parents=True, exist_ok=True)

    def isFresh(self):
        return self.fresh

    def compile(self):
        # mean absolute error: computes the mean absolute error between labels and predictions.
        self.model.compile(optimizer='adam', loss='mae')

    def summary(self):
        self.model.summary()

    def fit(self, X_train, y_train, epochs=50, batchSize=32, verbose=1, validation_data=None, validation_split=0.1, showLoss=True):

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


        self.lossPlot(self.history, showFig=showLoss)

    def predict(self, samples):
        return self.model.predict(samples, verbose=1)

    def save(self, dir):
        self.model.save(dir)

    def computeThreshold(self, trainSamples, plotError=True, savePlot=True):
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
            self.recoErrorDistributionPlot(trainMAElosses, threshold=None, saveFig=savePlot, type="train")

    def classify(self, samples, showError=True, savePlot=True):

        if self.classificationThreshold is None:
            print("The classification threshold is None. Call computeThreshold() to calculate it.")
            return None

        # encoding and decoding    
        recostructions = self.predict(samples)

        # computing the recostruction errors
        maeLosses = np.mean(np.abs(recostructions - samples), axis=1).flatten()

        self.recoErrorDistributionPlot(maeLosses, threshold=self.classificationThreshold, showFig=showError, saveFig=savePlot, type="test")

        mask = (maeLosses > self.classificationThreshold).flatten()
        
        return recostructions, maeLosses, mask

        
    def computeScore(self):
        pass


    def lossPlot(self, history, showFig=False, saveFig=True):
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        fig.suptitle("Losses during training")
        ax.plot(history.history["loss"], label="Training Loss")
        ax.plot(history.history["val_loss"], label="Validation Loss")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid()
        plt.legend()
        if showFig:
            plt.show()
        if saveFig:
            fig.savefig(self.outDir.joinpath("loss_plot.png"), dpi=300)
        plt.close()

    def recoErrorDistributionPlot(self, losses, threshold, type=None, showFig=False, saveFig=False):
        fig, ax = plt.subplots(1,1)
        fig.suptitle("Reconstruction error distribution")
        ax.hist(losses, bins=30)
        if threshold:
            ax.axvline(x=threshold, color="red")
        plt.xlabel('Train MAE loss')
        plt.ylabel('Number of Samples')
        if showFig:
            plt.show()
        if saveFig:    
            fig.savefig(self.outDir.joinpath(f"mea_distribution_{type}.png"), dpi=300)
        plt.close()



    def plotPredictions2(self, samples, samplesLabels, recostructions, mask, howMany, showFig=False, saveFig=True):
        
        """
        classifiedAsAnomalies = samples[mask,:,:]
        classifiedAsNotAnomalies = samples[np.invert(mask),:,:]
    
        recoClassifiedAsAnomalies = recostructions[mask,:,:]
        recoClassifiedAsNotAnomalies = recostructions[np.invert(mask),:,:]
        """

        classifiedAsAnomalieslabels = samplesLabels[mask]
        classifiedAsNotAnomaliesLabels = samplesLabels[np.invert(mask)]

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

        fig, ax = plt.subplots(rows,cols, constrained_layout=True, figsize=(20,20))
        fig.suptitle("Anomaly detection")
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
                    label = "Pred: anomaly (grb)"  
                else: 
                    label = "Pred: not anomaly (bkg)" 
                ax[row, col].set_title(f"{label}")

                # ax[row, col].legend(loc='upper left')
                ax[row, col].grid()
                count += 1

        if showFig:
            plt.show()
    
        if saveFig:
            fig.savefig(self.outDir.joinpath(f"predictions.png"), dpi=300)
    
        plt.close()

    def F1Score(self, labels, predictions):      
        return f1_score(labels, predictions)        
    
    def plotROC(self, labels, reconstructions, showFig=False, saveFig=True):
        """
        It is a plot of the false positive rate (x-axis) versus the true positive rate (y-axis) for a 
        number of different candidate threshold values between 0.0 and 1.0. 
        Put another way, it plots the false alarm rate versus the hit rate.
        """
        
        # TPR: how good the model is at predicting the positive class when the actual outcome is positive.
        
        fpr, tpr, thresholds = roc_curve(labels, reconstructions, drop_intermediate=False)
        print("thresholds: ",thresholds)
        print("tpr: ",tpr)
        print("fpr: ",fpr)
        roc_auc = roc_auc_score(labels, reconstructions)

        print('AUC: %.3f' % roc_auc)

        fig, ax = plt.subplots(1,1, figsize=(10,10))
        fig.suptitle("ROC curve")
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='LSTM anomaly detector')
        display.plot(ax)  
        if showFig:
            plt.show()   
        if saveFig:
            fig.savefig(self.outDir.joinpath(f"roc.png"), dpi=300)
        plt.close()

    def plotPrecisionRecall(self, labels, reconstructions, showFig=False, saveFig=True):
        # calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(labels, reconstructions)
        auc_score = metrics.auc(recall, precision)

        print("precision: ",precision)
        print("recall: ",recall)
        print("thresholds: ",thresholds)
        print("auc: ",auc_score)

        fig, ax = plt.subplots(1,1, figsize=(10,10))
        ax.plot(recall, precision, marker='.', label=f'LSTM anomaly detector. AUC = {auc_score}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        if showFig:
            plt.show()   
        if saveFig:
            fig.savefig(self.outDir.joinpath(f"precision_recall.png"), dpi=300)
        plt.close()