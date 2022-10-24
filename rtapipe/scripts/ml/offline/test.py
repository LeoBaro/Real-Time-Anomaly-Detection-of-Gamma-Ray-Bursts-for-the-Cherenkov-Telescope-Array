import pickle
import argparse
import numpy as np
from pathlib import Path
from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.analysis.models.anomaly_detector import AnomalyDetector
from multiprocessing import Pool


def testModel(modelDir):

    # Loading the trained LSTM model
    adLSTM = AnomalyDetector.loadModel(modelDir)
    adLSTM.setFeaturesColsNames(ds.getFeaturesColsNames())

    print(f"\n*************************************")
    print(f"Testing model {modelDir}")

    # Loading the threshold
    with open(modelDir.parent.joinpath("threshold.txt"), "r") as tf:
        threshold = float(tf.read().rstrip().strip())

    adLSTM.setClassificationThreshold(threshold)

    print(f"Threshold: {threshold}")
    print(test.shape)
    recostructions, testMAE, testMAEPerEnergyBin, mask = adLSTM.classify_with_mae(test)

    #print("recostructions: ",recostructions)
    print("testMAE: ",testMAE)
    print("testMAEPerEnergyBin: ",testMAEPerEnergyBin)
    print("testMAE: ",testMAE.shape)
    print("testMAEPerEnergyBin: ",testMAEPerEnergyBin.shape)

    #print("mask: ",mask)

    # Plotting
    adLSTM.setOutputDir(modelDir.parent)
    adLSTM.recoErrorDistributionPlot(testMAE, threshold=threshold, filenamePostfix="test_set-1", title="Reconstruction error distribution on test set", showFig=showPlots, saveFig=True)
    adLSTM.recoErrorDistributionPlot(testMAEPerEnergyBin, threshold=threshold, filenamePostfix="test_set-2", title="Reconstruction error distribution test set", showFig=showPlots, saveFig=True)
    adLSTM.confusionMatrixPlot(testLabels, mask)
    adLSTM.computeMetrics(testLabels, mask)
    adLSTM.plotPredictions(test, testLabels, recostructions, testMAEPerEnergyBin, mask, showFig=showPlots, saveFig=True)

    #adLSTM.precisionRecallCurvePlot(testLabels, mask)
    #adLSTM.plotROC(testLabels, testMAE, showFig=showPlots, saveFig=True)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-tmd", "--trained-model-dir", type=str, required=True, help="")
    parser.add_argument("-di", "--dataset_id", type=int, required=True, help="The dataset to be used as test set", choices=[1,2,3,4])
    parser.add_argument("-e", "--epoch", type=int, required=False, default=-1, help="-1 for testing all epoch")
    parser.add_argument("-mp", "--multiprocessing", type=int, required=False, default=1, choices = [0,1], help="If 1 multiprocessing will be used")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, choices = [0,1], help="If 1 plots will be shown")

    args = parser.parse_args()

    showPlots = False
    if args.verbose == 1:
        showPlots = True

    # Loading training dataset params
    with open(Path(args.trained_model_dir).joinpath('dataset_params.pickle'), 'rb') as handle:
        training_dataset_params = pickle.load(handle)
        print(training_dataset_params)

    # Loading the dataset for testing
    ds = APDataset.get_dataset(args.dataset_id)
    if not ds.checkCompatibilityWith(training_dataset_params):
        print("The test set is not compatible with the dataset used for training..")
        exit(1)

    ds.setScalerFromPickle(Path(args.trained_model_dir).joinpath('fitted_scaler.pickle'))
    ds.setOutDir(args.trained_model_dir)

    #wmin = 0
    #wmax = ds.dataset_params["timeseries_lenght"]
    new_timeseries_lenght = training_dataset_params["timeseries_lenght"]
    stride = 1
    beforeOnsetWindows, beforeOnsetLabels, afterOnsetWindows, afterOnsetLabels = ds.getTestSet(new_timeseries_lenght, stride=new_timeseries_lenght)

    test = np.concatenate((beforeOnsetWindows,afterOnsetWindows), axis=0)
    testLabels = np.concatenate((beforeOnsetLabels,afterOnsetLabels), axis=0)
    print(f"Test shape: {test.shape}. Example: {test[0].flatten()}")
    print(f"Test labels: {testLabels.shape}. Examples: {testLabels.flatten()}")

    ds.plotSamples(np.concatenate((beforeOnsetWindows[-3:], afterOnsetWindows[0:5]), axis=0), ["onset-3","onset-2","onset-1","onset+0","onset+1","onset+2","onset+3","onset+4"], showFig=showPlots)



    # Locating the trained LSTM models
    modelsDirs = []
    if args.epoch == -1:
        for path in Path(args.trained_model_dir).joinpath("epochs").iterdir():
            if path.is_dir():
                modelsDirs.append(path.joinpath("lstm_trained_model"))
    else:
        modelDir = Path(args.trained_model_dir).joinpath("epochs",f"epoch_{args.epoch}","lstm_trained_model")
        if modelDir.is_dir():
            modelsDirs.append(modelDir)
        else:
            raise FileNotFoundError(f"The directory {modelDir} does not exist!")

    if args.multiprocessing == 1:
        with Pool() as p:
            p.map(testModel, modelsDirs)
    else:
        for modelDir in modelsDirs:
            testModel(modelDir)
