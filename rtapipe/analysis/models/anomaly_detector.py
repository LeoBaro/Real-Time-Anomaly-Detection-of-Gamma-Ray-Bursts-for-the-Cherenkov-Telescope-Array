import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from math import floor
from numpy.lib.function_base import append
from scipy.stats import norm, chisquare
from tensorflow import reduce_sum
from tensorflow.keras.losses import mae
from tensorflow.python.types.core import Value
from tensorflow.train import latest_checkpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import mean_absolute_error
from rtapipe.analysis.models.builder import ModelBuilder
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay, precision_score, recall_score, precision_recall_curve, f1_score, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
import matplotlib.colors as mcolors

from sklearn import metrics
from tqdm import tqdm

COLORS = list(mcolors.BASE_COLORS)
plt.rcParams.update({'font.size': 18, 'lines.markersize': 0.5,'legend.markerscale': 3, 'lines.linewidth':1, 'lines.linestyle':'-'}) 
FIG_SIZE = (15,7)
DPI=300

class AnomalyDetector:

    def loadModel(modelDir):
        try:
            print(f"Loading model from {modelDir}")
            ad = AnomalyDetector(0, 0, 0, Path(modelDir).parent, True)
            ad.model = load_model(modelDir)
            return ad
        except Exception:
            print(f"Unable to load model from {modelDir}.")


    def __init__(self, shape, units, dropoutRate, outDir, loadModel = False):
        
        self.model = None

        if not loadModel:
            self.model = ModelBuilder.buildLSTM_2layers(shape, units, dropoutRate)

        self.history = []
        self.classificationThreshold = {
            "MAE":None,
            "AS": None
        }

        self.outDir = outDir
        self.outDir.mkdir(parents=True, exist_ok=True)

        self.featuresColsNames = []

    def setFeaturesColsNames(self, featuresColsNames):
        self.featuresColsNames = featuresColsNames

    def setOutputDir(self, outDir):
        self.outDir = outDir
        self.outDir.mkdir(parents=True, exist_ok=True)

    def setClassificationThreshold(self, maeThreshold, asThreshold):
        self.classificationThreshold["MAE"] = maeThreshold
        self.classificationThreshold["AS"] = asThreshold

    def compile(self):
        # mean absolute error: computes the mean absolute error between labels and predictions.
        self.model.compile(optimizer='adam', loss='mae')

    def summary(self):
        self.model.summary()

    def fit(self, X_train, y_train, epochs=50, batchSize=32, verbose=1, validation_data=None, validation_split=0.1, plotTrainingLoss=True):

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
        
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, validation_data=validation_data, validation_split=validation_split, verbose=verbose, callbacks=[])

        self.history.append(history)

        if plotTrainingLoss:
            self.lossPlot(history.history["loss"], history.history["val_loss"], showFig=True)

    def plotTrainingLoss(self, showFig=True, saveFig=True):
        loss = []
        val_loss = []
        for history in self.history:
            loss += history.history["loss"]
            val_loss += history.history["val_loss"]
        self.lossPlot(loss, val_loss, showFig=showFig, saveFig=True)

    def predict(self, samples):
        return self.model.predict(samples, verbose=1)

    def save(self, dir):
        self.model.save(dir)

    def computeSimpleThreshold(self, valSamples, showFig=False):
        """
        The threshold is calculated as the 98% quantile of the mean absolute errors distribution for the normal examples of the training set, 
        then classify future examples as anomalous if the reconstruction error is higher than one standard 
        deviation from the training set.    
        """
        reco = self.predict(valSamples)

        # Computes the mean absolute error between labels and predictions.
        maeLosses = np.mean(np.abs(reco - valSamples), axis=1)

        # Fits the reconstruction error distribution on the validation set
        mu, std = self.fitMAEonValidationSet(maeLosses)

        # Plotting the pdf and cdf of the recostruction errors on the validation set
        self.pdfPlot(maeLosses, mu, std, filenamePostfix=f"val_set", showFig=showFig)
        self.cdfPlot(maeLosses, mu, std, filenamePostfix=f"val_set", showFig=showFig)

        # Calcolarla attraverso il fitting della distrubuzione dell'errore in modo analitico
        self.classificationThreshold = np.percentile(maeLosses, 98)

        # Compute the threshold as the max of those errors
        return self.classificationThreshold, maeLosses


    def fitMAEonValidationSet(self, maeLosses):

        mu, std = norm.fit(maeLosses)

        return mu, std

        maeLosses = maeLosses.squeeze()
        binsNum = 50
        mu, std = norm.fit(maeLosses)

        hist, bins, patches = plt.hist(maeLosses, bins=binsNum, density=True, facecolor='none', edgecolor=COLORS[0])
        print("hist",hist)

        binCenters = []
        for idx in range(len(bins)-1):
            binCenters.append((bins[idx+1]+bins[idx])/2)       
        pdf2 = norm.pdf(binCenters, mu, std)

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, binsNum)
        pdf1 = norm.pdf(x, mu, std)

        #print(hist)
        #print(pdf1)
        #print(pdf2)
    
        plt.plot(x, pdf1, linestyle="--", linewidth=2, label=f"mu = {round(mu,2)},  std = {round(std,2)}")
        plt.plot(binCenters, pdf2, linestyle="-.", linewidth=2, label=f"Expected")
        plt.legend()
        plt.show()
        #chi2 = chisquare(hist, pdf1)
        #print(chi2)
        #chi2 = chisquare(hist, pdf2)
        #print(chi2)

        return mu, std

    # TODO
    def computeAnomalyScoreThreshold(self, valSamples):

        reco = self.predict(valSamples)

        # Computes the mean absolute error between labels and predictions.
        maeLosses = np.mean(mae(valSamples, reco), axis=1) # mae = np.abs(reco - valSamples)
        
        mu, std = norm.fit(maeLosses)


        return -666, maeLosses


    def reconstruct(self, samples):
        # encoding and decoding    
        recostructions = self.predict(samples)

        # computing the recostruction errors
        maeLosses = np.mean(np.abs(recostructions - samples), axis=1).flatten()


        return recostructions, maeLosses


    def classify_with_distance_from_distribution(self, samples):
        pass


    def classify_with_mae(self, samples):

        if self.classificationThreshold is None:
            raise ValueError("The classification threshold is None. Call computeThreshold() to calculate it or sei it with setClassificationThreshold()")

    
        # The reconstructions:
        # [
        #   [
        #       [1],[2],[3],[4],[5]
        #   ], 
        #   [
        #       [1],[2],[3],[4],[5]
        #   ],
        #   ... up to X samples
        # ]
        #
        # In the case of multiple features..
        # [
        #   [
        #       [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]
        #   ], 
        #   [
        #       [1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]
        #   ],
        #   ... up to X samples
        # ]
        recostructions = self.predict(samples)


        # The recostruction errors:
        # For each sample, it computes X arrays of distances between the points, where X is the number of energy bins.
        distances = np.abs(recostructions - samples)

        # The mae loss is defined as the mean of those distances, for each sample, for each energy bin.
        # [
        #   [0.11891678 0.11762658 0.11594792 0.08139625]
        #   [0.11626169 0.11343022 0.12967022 0.08330123]  
        #   ... up to X samples
        # ]
        #
        maeLossesPerEnergyBin = np.mean(distances, axis=1)

        # How do I "merge" the mae of each energy bin? Maybe with a weighted mean. For now I'll use a simple mean.
        # [
        #   0.10847188,
        #   0.11066584
        #   ... up to X samples
        # ]
        #
        maeLosses = np.mean(maeLossesPerEnergyBin, axis=1)



        mask = (maeLosses > self.classificationThreshold)


        
        return recostructions, maeLosses, maeLossesPerEnergyBin, mask

        























    """
    Plotting
    """

    def pdfPlot(self, values, mu, std, filenamePostfix="", showFig=False, saveFig=True):
        
        trials = len(values)
        fig, ax = plt.subplots(1,1, figsize=FIG_SIZE)

        n, bins, patches = ax.hist(values, bins=50, density=True, facecolor='none', edgecolor=COLORS[0])       
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        pdf = norm.pdf(x, mu, std)
        ax.plot(x, pdf, linestyle="--", linewidth=2, label=f"mu = {round(mu,2)},  std = {round(std,2)}")

        ax.set_title(f"Reconstruction errors PDF. Trials={trials}")
        ax.set_xlabel('Reconstruction errors')
        ax.set_ylabel('Counts (normalized)')
        ax.legend()
        if saveFig:
            fig.savefig(self.outDir.joinpath(f"mae_pdf_{filenamePostfix}.png"), dpi=DPI)
        if showFig:
            plt.show()
        plt.close()

    def cdfPlot(self, values, mu, std, filenamePostfix="", showFig=False, saveFig=True):
        # Test with 10e7 samples
        #values = np.random.normal(loc=mu, scale=std, size=int(10e7))
        n_bins = 100
        trials = len(values)
        fig, ax = plt.subplots(1,1, figsize=FIG_SIZE)

        # Overlay a reversed cumulative histogram.
        n, bins, patches = ax.hist(values, bins=n_bins, density=True, histtype='step', cumulative=-1, label='Reversed empirical')

        ax.set_title(f'Cumulative step histograms. Trials={trials}')
        ax.set_xlabel('Reconstruction errors')
        ax.set_ylabel('Likelihood of occurrence')
        ax.set_yscale('log')
        ax.legend()
        if saveFig:
            fig.savefig(self.outDir.joinpath(f"mae_cdf_{filenamePostfix}.png"), dpi=DPI)
        if showFig:
            plt.show()
        plt.close()

 
    def confusionMatrixPlot(self, realLabels, predLabels, showFig=False, saveFig=True):
        fig, ax = plt.subplots(1,1, figsize=FIG_SIZE)
        fig.suptitle("Confusion matrix")
        disp = ConfusionMatrixDisplay.from_predictions(realLabels, predLabels, display_labels=["bkg","grb"], ax=ax)
        if showFig:
            plt.show()
        if saveFig:
            fig.savefig(self.outDir.joinpath("confusion_matrix.png"), dpi=DPI)
        plt.close()

    def computeMetrics(self, realLabels, predLabels):   
        prec = precision_score(realLabels, predLabels)
        recall = recall_score(realLabels, predLabels)
        f1 = f1_score(realLabels, predLabels)      
        tn, fp, fn, tp = confusion_matrix(realLabels, predLabels).ravel()
        print(tp, fp, tn, fn)
        fpr = fp / (fp+tn)
        fnr = fn / (fn+tp)

        score_str = f"Precision={round(prec,3)}\nRecall={round(recall,3)}\nF1={round(f1, 3)}\nFalse Positive Rate={round(fpr, 3)}\nFalse Negative Rate={round(fnr,3 )}"
        
        print(score_str)
        
        with open(self.outDir.joinpath("performance_metrics.txt"), "w") as pf:
            pf.write(score_str)

    def lossPlot(self, loss, val_loss, showFig=False, saveFig=True):
        fig, ax = plt.subplots(1,1, figsize=FIG_SIZE)
        fig.suptitle("Losses during training")
        ax.plot(loss, label="Training Loss")
        ax.plot(val_loss, label="Validation Loss")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.grid()
        plt.legend()
        if showFig:
            plt.show()
        if saveFig:
            fig.savefig(self.outDir.joinpath("loss_plot.png"), dpi=DPI)
        plt.close()

    def recoErrorDistributionPlot(self, losses, threshold=None, filenamePostfix="", title="", showFig=False, saveFig=True):
        fig, ax = plt.subplots(1,1, figsize=FIG_SIZE)
        if title:
            fig.suptitle(title)

        # print(losses.shape)

        if len(losses.shape) == 1:
            numFeatures = 1
            losses = np.expand_dims(losses, axis=0)
        else:
            numFeatures = losses.shape[1]

        # print(losses.shape, numFeatures,losses[:, 0])

        for f in range(numFeatures):
            ax.hist(losses[:, f], bins=50, label=self.featuresColsNames[f], facecolor='none', edgecolor=COLORS[f])
        # plt.yscale('log', nonposy='clip')
        
        if threshold:
            ax.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold: {round(threshold,2)}")

        plt.xlabel(f'Mean Absolute Error loss')  
        plt.ylabel('Number of Samples')

        ax.legend()
        if showFig:
            plt.show()
        if saveFig:    
            fig.savefig(self.outDir.joinpath(f"mea_distribution_{filenamePostfix}.png"), dpi=DPI)
        plt.close()

    def plotPredictions(self, samples, samplesLabels, recostructions, maeLossePerEnergyBin, mask, showFig=False, saveFig=True):
        ylim = samples.max(axis=1).flatten().max()
        
        print(f"Number of predictions: {len(samples)}. Sample shape: {samples.shape}")

        featureNum = samples.shape[2]

        realLabels = []
        for lab in samplesLabels:
            if lab:
                realLabels.append("grb")
            else:
                realLabels.append("bkg")

        predLabels = []
        for lab in mask:
            if lab:
                predLabels.append("grb")
            else:
                predLabels.append("bkg")
        

        predictionsPerFigure = 40

        cols = 5
        rows = 10

        figsize_x = 20
        figsize_y = rows*4

        numberOfFigures = floor(len(samples) / predictionsPerFigure)
        if len(samples) % predictionsPerFigure > 0:
            numberOfFigures += 1

        # For each feature..
        for f in tqdm(range(featureNum)):

            count = 0   
            for i in tqdm(range(numberOfFigures), leave=False):

                fig, ax = plt.subplots(rows, cols, constrained_layout=True, figsize=(figsize_x, figsize_y))
                fig.suptitle(f"Reconstructions. Feature={f} Threshold={round(self.classificationThreshold, 2)}")
                
                for row in range(rows):
                    for col in range(cols):                
                        if count >= len(samples):
                            break
                        
                        # Chaning color for grb class
                        color="blue"
                        if realLabels[count] == "grb":
                            color="red"


                        # Get a sample and its recostruction
                        sample = samples[count][:,f]
                        recoSample = recostructions[count][:,f]

                        # And plot them
                        ax[row, col].plot(recoSample, color='red', linestyle='dashed', label=f'mae={round(maeLossePerEnergyBin[count,f],2)}')
                        ax[row, col].scatter(range(sample.shape[0]), sample, color=color, s=5)
                        ax[row, col].set_ylim(0, ylim)
                        ax[row, col].set_title(f"Real: {realLabels[count]} Pred: {predLabels[count]}")
                        ax[row, col].legend(loc='upper left')
                        ax[row, col].grid()
                        if realLabels[count] != predLabels[count]:
                            ax[row, col].set_facecolor('#e0e0eb')
                        count += 1

                if showFig:
                    plt.show()
            
                if saveFig:
                    fig.savefig(self.outDir.joinpath(f"predictions_{i}_feature_{f}.png"), dpi=DPI/8)
            
            plt.close()






  
    """
    def plotROC(self, testLabels, reconstructions, showFig=False, saveFig=True):
        # TPR: how good the model is at predicting the positive class when the actual outcome is positive.
        testLabels = [int(boolLabel) for boolLabel in testLabels]
        fpr, tpr, thresholds = roc_curve(testLabels, reconstructions, drop_intermediate=False)
        print("thresholds: ",thresholds)
        print("tpr: ",tpr)
        print("fpr: ",fpr)
        roc_auc = roc_auc_score(testLabels, reconstructions)

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


    def precisionRecallCurvePlot(self, realLabels, predLabels, showFig=False, saveFig=True):
        # https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#sphx-glr-auto-examples-model-selection-plot-precision-recall-py
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        fig.suptitle("Precision-Recall Curve\nPrecision: {} Recall: {}")
        disp = PrecisionRecallDisplay.from_predictions(realLabels, predLabels, name="Autoencoder/LSTM classifier", ax=ax)
        if showFig:
            plt.show()
        if saveFig:
            fig.savefig(self.outDir.joinpath("precision_recall_curve.png"), dpi=300)
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
    """