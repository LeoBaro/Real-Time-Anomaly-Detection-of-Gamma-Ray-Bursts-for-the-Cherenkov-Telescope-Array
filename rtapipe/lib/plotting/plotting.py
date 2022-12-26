import numpy as np
from tqdm import tqdm
from pathlib import Path
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

def loss_plot(loss, val_loss, model_name="", title="Training loss", ylim=None, showFig=False, saveFig=True, outputDir="./", figName="loss_plot.png"):
    pc = PlotConfig()
    fig, ax = plt.subplots(1,1, figsize=pc.fig_size)
    fig.suptitle(title+" "+model_name)
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
        outputPath = outputDir.joinpath(model_name+"_"+figName)
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

def plot_sequences(sequences, scaled, features_names=[], labels=[], showFig=False, saveFig=True, outputDir="./", figName="sample.png"):
    """
    Shape of sequence: (n_samples, n_timesteps, n_features)
    """
    pc = PlotConfig()
    ymax, ymin = 1.5, 0
    n_samples = sequences.shape[0] 
    n_features = sequences.shape[-1]
    if len(labels) != n_samples:
        labels = [None] * n_samples
    if len(features_names) != n_features:
        features_names = [f"Feature {i}" for i in range(n_features)]
    fig, ax = plt.subplots(n_features, 1, figsize=pc.fig_size)
    fig.suptitle(f"Sequence with {n_features} features")
    for j in range(n_samples):
        color = pc.colors[j]
        for i in range(n_features):
            ax[i].plot(sequences[j, :, i], color=color, marker='o', markersize=6, linestyle='dashed', label=labels[j])
            ax[i].set_ylabel(features_names[i])
            ax[i].set_xticks([])
            if scaled: ax[i].set_ylim(ymin, ymax)
    handles, labels = ax[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left')            
    if showFig:
        plt.show()
    if saveFig:
        Path(outputDir).mkdir(parents=True, exist_ok=True)
        outputPath = Path(outputDir).joinpath(figName)
        print(f"Saving {outputPath}")
        fig.savefig(outputPath, dpi=200)

    plt.close()

def plot_recostruction_errors_distribution(recostructions, output_dir):
    pc = PlotConfig()
    fig, ax = plt.subplots(1,1,figsize=pc.fig_size)
    ax.hist(recostructions, bins=50, **pc.get_histogram_colors())  
    fig.suptitle("Distribution of the reconstruction erros", fontsize=pc.fig_suptitle_size)
    ax.set_title("Test set", fontsize=pc.fig_title_size)
    ax.set_xlabel("Recostruction errors")
    ax.set_ylabel("Counts (log)")
    ax.set_yscale('log')
    fig.savefig(Path(output_dir).joinpath("reco_distributions.png"))


def plot_predictions(samples, samplesLabels, c_threshold, recostructions, mse_per_sample, mse_per_sample_features, features_names=[], integration_time=5, epoch="", max_plots=5, showFig=False, saveFig=True, outputDir="./", figName="predictions.png"):

    pc = PlotConfig()

    total_samples = samples.shape[0]
    
    max_samples = 5
    n_features = samples.shape[2]
    if len(features_names) != n_features:
        features_names = [f"Feature {i}" for i in range(n_features)]

    num_plots = total_samples // max_samples

    mask = (mse_per_sample > c_threshold)

    start = 0
    for p in tqdm(range(num_plots)):

        if p == max_plots:
            print("Max plots reached")
            break



                
        annotations = [f"{i*integration_time}-{i*integration_time+samples.shape[1]}" for i in range(0,96)]
        xticks = [i*5 for i in range(0,96)]

        current_samples = samples[start:start+max_samples, :, :]
        current_samplesLabels = samplesLabels[start:start+max_samples]
        current_samples_annotations = annotations[start:start+max_samples]
        current_samples_xticks = xticks[start:start+max_samples]
        current_recostructions = recostructions[start:start+max_samples, :, :]
        current_mask = mask[start:start+max_samples]
        current_mse_per_sample_features = mse_per_sample_features[start:start+max_samples]
        current_mse_per_sample = mse_per_sample[start:start+max_samples]

        #print("current_samples:",current_samples)
        #print("current_mse_per_sample_features: ", current_mse_per_sample_features)
        start += max_samples

        ymax, ymin = 1, 0
        
        #print(f"Plot {p}. \nNumber of predictions: {len(current_samples)}. \nSample shape: {current_samples.shape} \n Number of features: {n_features}")

        real_labels = ["grb" if lab==1 else "bkg" for lab in current_samplesLabels ]
        pred_labels = ["grb" if lab==1 else "bkg" for lab in current_mask          ]

        fig, ax = plt.subplots(n_features, max_samples, figsize=(pc.fig_size[0]*2,pc.fig_size[1]*2))
        fig.suptitle(f"5σ threshold={round(c_threshold, 3)}")
        #fig.suptitle(f"Predictions (using threshold={round(c_threshold, 3)})")
        fig.supylabel('Energy bins (TeV)')

      

        # For each feature..
        for f in range(n_features):

            for i in range(max_samples):

                # Get a sample and its recostruction
                sample = current_samples[i][:,f]
                recoSample = current_recostructions[i][:,f]

                # And plot them                
                ax[f, i].plot(recoSample, color='red',  marker='o', markersize=6, linestyle='dashed', label="reconstruction")
                ax[f, i].plot(sample,     color="blue", marker='o', markersize=6, linestyle='dashed', label="ground truth")
                ax[f, i].set_ylim(ymin, ymax)

                
                ax[f, i].set_xticks(current_samples_xticks, current_samples_annotations, fontsize=10)

                if real_labels[i] != pred_labels[i]:
                    ax[f, i].set_facecolor('#e6e6e6')

                # Only the first column will show the Y labels
                if i == 0:
                    ax[f, i].set_ylabel(features_names[f].split("EB_")[1].split(" TeV")[0])


                ax[f, i].set_xlabel("mse={:.6f}".format(current_mse_per_sample_features[i, f]))

                # Only the first row will show the TN/FP/FN/TP labels and the averaged mse
                if f == 0:
                    title = "Sample {}\nW. avg mse={:.3f}\nClassification=".format(start+i-max_samples, current_mse_per_sample[i])
                    if real_labels[i] == "grb" and real_labels[i] == pred_labels[i]:
                        title += "TP"
                    elif real_labels[i] == "grb" and real_labels[i] != pred_labels[i]:
                        title += "FN"
                    elif real_labels[i] == "bkg" and real_labels[i] == pred_labels[i]:  
                        title += "TN"
                    elif real_labels[i] == "bkg" and real_labels[i] != pred_labels[i]:
                        title += "FP"
                    ax[f, i].set_title(title)

        handles, labels = ax[f, i].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper left')

        plt.tight_layout()

        if showFig:
            plt.show()

        if saveFig:
            Path(outputDir).mkdir(parents=True, exist_ok=True)
            outputPath = Path(outputDir).joinpath(f"{figName}_epoch_{epoch}_plot_{p}.png")
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
