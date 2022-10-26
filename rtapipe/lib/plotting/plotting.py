import numpy as np
from tqdm import tqdm
from math import floor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import ConfusionMatrixDisplay
from rtapipe.lib.plotting.PlotConfig import PlotConfig

COLORS = list(mcolors.BASE_COLORS)

def confusion_matrix_plot(realLabels, predLabels, showFig=False, saveFig=True, outputDir="./", figName="confusion_matrix.png"):
    pc = PlotConfig()
    fig, ax = plt.subplots(1,1, figsize=pc.fig_size)
    fig.suptitle("Confusion matrix")
    disp = ConfusionMatrixDisplay.from_predictions(realLabels, predLabels, display_labels=["bkg","grb"], ax=ax)
    if showFig:
        plt.show()
    if saveFig:
        outputPath = outputDir.joinpath(figName)
        fig.savefig(outputPath, dpi=pc.dpi)
        print(f"Plot {outputPath} created.")
    plt.close()

def loss_plot(loss, val_loss, title="Training loss", ylim=None, showFig=False, saveFig=True, outputDir="./", figName="loss_plot.png"):
    pc = PlotConfig()
    fig, ax = plt.subplots(1,1, figsize=pc.fig_size)
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
        fig.savefig(outputPath, dpi=pc.dpi)
        print(f"Plot {outputPath} created.")
    plt.close()

def reco_error_distribution_plot(losses, threshold=None, title="", showFig=False, saveFig=True, outputDir="./", figName="reco_errors_distr"):
    
    pc = PlotConfig()

    fig, ax = plt.subplots(1,1, figsize=pc.fig_size)
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
        fig.savefig(outputPath, dpi=pc.dpi)
        print(f"Plot {outputPath} created.")
    plt.close()

def plot_predictions(samples, samplesLabels, c_threshold, recostructions, mse_per_sample, mse_per_sample_features, max_plots=5, showFig=False, saveFig=True, outputDir="./", figName="predictions.png"):

    pc = PlotConfig()

    total_samples = samples.shape[0]
    
    max_samples = 5
    features_num = samples.shape[2]
    num_plots = total_samples // max_samples

    mask = (mse_per_sample > c_threshold)

    start = 0
    for p in tqdm(range(num_plots)):

        if p == max_plots:
            print("Max plots reached")
            break

        current_samples = samples[start:start+max_samples, :, :]
        current_samplesLabels = samplesLabels[start:start+max_samples]
        current_recostructions = recostructions[start:start+max_samples, :, :]
        current_mask = mask[start:start+max_samples]
        
        start += max_samples

        ymax = 1.5 
        ymin = 0
        
        #print(f"Plot {p}. \nNumber of predictions: {len(current_samples)}. \nSample shape: {current_samples.shape} \n Number of features: {features_num}")

        real_labels = ["grb" if lab==1 else "bkg" for lab in current_samplesLabels ]
        pred_labels = ["grb" if lab==1 else "bkg" for lab in current_mask          ]

        fig, ax = plt.subplots(features_num, max_samples, figsize=pc.fig_size)
        fig.suptitle(f"Predictions (using threshold={round(c_threshold, 3)})")

        # For each feature..
        for f in range(features_num):

            for i in range(max_samples):

                # Get a sample and its recostruction
                sample = current_samples[i][:,f]
                recoSample = current_recostructions[i][:,f]

                # And plot them                
                ax[f, i].plot(recoSample, color='red',  marker='o', markersize=8, linestyle='dashed', label="reconstruction")
                ax[f, i].plot(sample,     color="blue", marker='o', markersize=8, linestyle='dashed', label="ground truth")
                ax[f, i].set_ylim(ymin, ymax)

                if real_labels[i] != pred_labels[i]:
                    ax[f, i].set_facecolor('#e6e6e6')
                #else:
                #    ax[f, i].set_facecolor('#a6f5aa')

                if i == 0:
                    ax[f, i].set_ylabel(f"Feature {f}")

                ax[f, i].set_xticks([])
                ax[f, i].set_xlabel("mse={:.4f}".format(mse_per_sample_features[i,f]))

                if real_labels[i] == "grb" and real_labels[i] == pred_labels[i]:
                    ax[f, i].set_title("TP")
                elif real_labels[i] == "grb" and real_labels[i] != pred_labels[i]:
                    ax[f, i].set_title("FN")
                elif real_labels[i] == "bkg" and real_labels[i] == pred_labels[i]:  
                    ax[f, i].set_title("TN")
                elif real_labels[i] == "bkg" and real_labels[i] != pred_labels[i]:
                    ax[f, i].set_title("FP")

        handles, labels = ax[f, i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')

        plt.tight_layout()

        if showFig:
            plt.show()

        if saveFig:
            outputDir.mkdir(parents=True, exist_ok=True)
            outputPath = outputDir.joinpath(f"{p}_{figName}")
            fig.savefig(outputPath, dpi=200)

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
