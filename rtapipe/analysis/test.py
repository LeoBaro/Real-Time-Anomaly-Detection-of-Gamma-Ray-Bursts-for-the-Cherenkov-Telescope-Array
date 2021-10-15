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
    # TODO READCSV
    with open(modelDir.parent.joinpath("threshold.csv"), "r") as tf:
        threshold = float(tf.read().rstrip().strip())
    adLSTM.setClassificationThreshold(threshold)

    print(f"threshold: {threshold}")

    print(test.shape)
    recostructions, testMAE, testMAEPerEnergyBin, mask = adLSTM.classify_with_mae(test)

    #print("recostructions: ",recostructions)
    #print("testMAE: ",testMAE)
    #print("testMAEPerEnergyBin: ",testMAEPerEnergyBin)
    #print("mask: ",mask)

    # Plotting
    adLSTM.setOutputDir(modelDir.parent)
    adLSTM.recoErrorDistributionPlot(testMAE, threshold=threshold, showFig=showPlots, saveFig=True, type="test")
    adLSTM.recoErrorDistributionPlot(testMAEPerEnergyBin, threshold=threshold, showFig=showPlots, saveFig=True, type="test-2")
    adLSTM.confusionMatrixPlot(testLabels, mask)
    adLSTM.computeMetrics(testLabels, mask)
    adLSTM.plotPredictions(test, testLabels, recostructions, testMAEPerEnergyBin, mask, showFig=showPlots, saveFig=True)
    
    #adLSTM.precisionRecallCurvePlot(testLabels, mask)
    #adLSTM.plotROC(testLabels, testMAE, showFig=showPlots, saveFig=True)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", type=str, required=True, help="")
    parser.add_argument("-e", "--epoch", type=int, required=False, default=-1, help="-1 for testing all epoch")
    parser.add_argument("-mp", "--multiprocessing", type=int, required=False, default=1, choices = [0,1], help="If 1 multiprocessing will be used")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=0, choices = [0,1], help="If 1 plots will be shown")

    args = parser.parse_args()

    showPlots = False
    if args.verbose == 1:
        showPlots = True

    # Loading dataset params
    with open(Path(args.dir).joinpath('dataset_params.pickle'), 'rb') as handle:
        dataset_params = pickle.load(handle)
        print(dataset_params)

    # Loading the dataset
    ds = APDataset(dataset_params["tobs"], dataset_params["onset"], dataset_params["integration_time"], dataset_params["integration_type"], ["COUNT"], ['TMIN', 'TMAX', 'LABEL', 'ERROR'])
    ds.loadData("bkg", dataset_params["bkg"])
    ds.loadData("grb", dataset_params["grb"])
    ds.setOutDir(args.dir)

    train, trainLabels, val, valLabels = ds.getTrainingAndValidationSet(dataset_params["ws"], dataset_params["stride"], scaler="mm")
    beforeOnsetWindows, beforeOnsetLabels, afterOnsetWindows, afterOnsetLabels = ds.getTestSet(dataset_params["ws"], dataset_params["stride"], dataset_params["onset"], scaler="mm")
    
    test = np.concatenate((beforeOnsetWindows,afterOnsetWindows), axis=0)
    testLabels = np.concatenate((beforeOnsetLabels,afterOnsetLabels), axis=0)
  
    ds.plotSamples(np.concatenate((beforeOnsetWindows[-3:], afterOnsetWindows[0:5]), axis=0), ["onset-3","onset-2","onset-1","onset+0","onset+1","onset+2","onset+3","onset+4"], showFig=showPlots)

    print(f"Train shape: {train.shape}. Example: {train[0].flatten()}")
    print(f"Val shape: {val.shape}. Example: {val[0].flatten()}")
    print(f"Test shape: {test.shape}. Example: {test[0].flatten()}")    
    print(f"Test labels: {testLabels.shape}. Examples: {testLabels.flatten()}")    

    # Locating the trained LSTM models
    modelsDirs = []
    if args.epoch == -1:
        for path in Path(args.dir).joinpath("epochs").iterdir():
            if path.is_dir():
                modelsDirs.append(path.joinpath("lstm_trained_model"))
    else:
        modelDir = Path(args.dir).joinpath("epochs",f"epoch_{args.epoch}","lstm_trained_model")
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

