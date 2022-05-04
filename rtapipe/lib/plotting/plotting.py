import numpy as np
from tqdm import tqdm
from math import floor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import ConfusionMatrixDisplay

COLORS = list(mcolors.BASE_COLORS)
plt.rcParams.update({'font.size': 18, 'lines.markersize': 0.5,'legend.markerscale': 3, 'lines.linewidth':1, 'lines.linestyle':'-'})
FIG_SIZE = (15,7)
DPI=300

def confusionMatrixPlot(realLabels, predLabels, showFig=False, saveFig=True, outputDir="./", figName="confusion_matrix.png"):
    fig, ax = plt.subplots(1,1, figsize=FIG_SIZE)
    fig.suptitle("Confusion matrix")
    disp = ConfusionMatrixDisplay.from_predictions(realLabels, predLabels, display_labels=["bkg","grb"], ax=ax)
    if showFig:
        plt.show()
    if saveFig:
        outputPath = outputDir.joinpath(figName)
        fig.savefig(outputPath, dpi=DPI)
        print(f"Plot {outputPath} created.")
    plt.close()

def lossPlot(loss, val_loss, title="Training loss", ylim=None, showFig=False, saveFig=True, outputDir="./", figName="loss_plot.png"):
    fig, ax = plt.subplots(1,1, figsize=FIG_SIZE)
    fig.suptitle(title)
    ax.plot(loss, label="Training Loss")
    ax.plot(val_loss, label="Validation Loss")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    ax.grid()
    plt.legend()
    if showFig:
        plt.show()
    if saveFig:
        outputPath = outputDir.joinpath(figName)
        fig.savefig(outputPath, dpi=DPI)
        print(f"Plot {outputPath} created.")
    plt.close()

def recoErrorDistributionPlot(losses, threshold=None, title="", showFig=False, saveFig=True, outputDir="./", figName="reco_errors_distr"):
    fig, ax = plt.subplots(1,1, figsize=FIG_SIZE)
    if title:
        fig.suptitle(title)


    if len(losses.shape) == 1:
        numFeatures = 1
        losses = np.expand_dims(losses, axis=1)
        featuresColsNames = ["0.03-1.0"]
    else:
        numFeatures = losses.shape[1]
        featuresColsNames = ["0.03-0.0721", "0.0721-0.1732", "0.1732-0.4162", "0.4162-1.0"]

    # print(losses.shape, numFeatures,losses[:, 0])

    for f in range(numFeatures):
        ax.hist(losses[:, f], bins=50, label=featuresColsNames[f], facecolor='none', edgecolor=COLORS[f])
    # plt.yscale('log', nonposy='clip')

    if threshold:
        ax.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold: {round(threshold,2)}")

    plt.xlabel(f'Mean Absolute Error loss')
    plt.ylabel('Number of Samples')

    ax.legend()
    if showFig:
        plt.show()
    if saveFig:
        outputPath = outputDir.joinpath(figName)
        fig.savefig(outputPath, dpi=DPI)
        print(f"Plot {outputPath} created.")
    plt.close()

def plotPredictions(samples, samplesLabels, c_threshold, recostructions, maeLossePerEnergyBin, mask, maxSamples=None, rows=5, cols=10, predictionsPerFigure=40, showFig=False, saveFig=True, outputDir="./", figName="predictions.png"):
    if maxSamples is not None:
        samples = samples[:maxSamples]
        samplesLabels = samplesLabels[:maxSamples]
        recostructions = recostructions[:maxSamples]
        mask = mask[:maxSamples]

    # ylim = samples.max(axis=1).flatten().max()
    ymax = 3
    ymin = -3
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
            fig.suptitle(f"Reconstructions. Feature={f} Threshold={round(c_threshold)}")

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
                    ax[row, col].set_ylim(ymin, ymax)
                    ax[row, col].set_title(f"Real: {realLabels[count]} Pred: {predLabels[count]}")
                    ax[row, col].legend(loc='upper left')
                    ax[row, col].grid()
                    if realLabels[count] != predLabels[count]:
                        ax[row, col].set_facecolor('#e0e0eb')
                    count += 1

            if showFig:
                plt.show()

            if saveFig:
                outputPath = outputDir.joinpath(f"{i}_feature_{f}_{figName}")
                fig.savefig(outputPath, dpi=DPI/8)

            plt.close()

        #print(f"Plot {outputPath} created.")






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
