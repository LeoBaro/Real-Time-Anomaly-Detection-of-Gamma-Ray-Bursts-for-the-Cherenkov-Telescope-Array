import argparse
from re import A
from rtapipe.lib.rtapipeutils.Chronometer import Chronometer
from time import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from time import strftime

from rtapipe.analysis.models.anomaly_detector import AnomalyDetector
from rtapipe.analysis.dataset.dataset import APDataset


if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bkg", type=str, required=True, help="The folder containing AP data (background only)")
    parser.add_argument("--grb", type=str, required=True, help="The folder containing AP data (grb)")
    args = parser.parse_args()

    # Output dir
    outDirRoot = Path(__file__).parent.resolve().joinpath(f"rtapipe_training_plots_{strftime('%Y%m%d-%H%M%S')}")
    outDirRoot.mkdir(parents=True, exist_ok=True)
    ###########
    # Params  # 
    ###########

    
    # Dataset params
    integration_time = "1"

    dataset_params = {
        "10" : {"tobs" : 180, "onset" : 90},
        "1" : {"tobs" : 1800, "onset" : 900}
    }
    dataset_params = dataset_params[integration_time]

    # Training set - Test set params
    ws = 25
    stride = 1
    scaler = None # mm, std

    # LSTM params
    units = 32 #1
    dropoutrate = 0.2 # 0
    epochs = 500 # 2
    batchSize = 64 #30

    with open(outDirRoot.joinpath("parameters.csv"), "w") as statFile:
        statFile.write("integrationtime,tobs,onset,ws,stride,units,dropoutrate,epochs,batchSize\n")
        statFile.write(f"{integration_time},{dataset_params['tobs']},{dataset_params['onset']},{ws},{stride},{units},{dropoutrate},{epochs},{batchSize}")

    ds = APDataset(dataset_params["tobs"], dataset_params["onset"], 1, ["COUNT"], ['TMIN', 'TMAX', 'LABEL', 'ERROR'], outDirRoot)
    ds.loadData("bkg", args.bkg)
    ds.loadData("grb", args.grb)

    ds.plotRandomSamples()

    #train, trainLabels, test, testLabels, val, valLabels = ds.getData()
    train, trainLabels, val, valLabels = ds.getTrainingAndValidationSet(ws, stride, scaler=scaler)
    beforeOnsetWindows, beforeOnsetLabels, afterOnsetWindows, afterOnsetLabels = ds.getTestSet(ws, stride, dataset_params["onset"], scaler=scaler)
    
    beforeOnsetWindows = beforeOnsetWindows[-11:]  
    beforeOnsetLabels = beforeOnsetLabels[-11:]
    afterOnsetWindows = afterOnsetWindows[:11] 
    afterOnsetLabels = afterOnsetLabels[:11] 


    test = np.concatenate((beforeOnsetWindows,afterOnsetWindows), axis=0)
    testLabels = np.concatenate((beforeOnsetLabels,afterOnsetLabels), axis=0)

    print(f"Train shape: {train.shape}. Example: {train[0].flatten()}")
    print(f"Val shape: {val.shape}. Example: {val[0].flatten()}")
    print(f"Test shape: {test.shape}. Example: {test[0].flatten()}")    
    print(f"Test labels: {testLabels.shape}. Examples: {testLabels.flatten()}")    

    ds.plotSamples(np.concatenate((beforeOnsetWindows[-3:], afterOnsetWindows[0:5]), axis=0), ["onset-3","onset-2","onset-1","onset+0","onset+1","onset+2","onset+3","onset+4"], "samples", change_color_from_index=3)



    # Building the model
    # loadModelFrom="./single_feature_model"
    loadModelFrom = None
    adLSTM = AnomalyDetector(train[0].shape, units, dropoutrate, outDirRoot, loadModelFrom)

    # Compiling the model
    adLSTM.compile()

    adLSTM.summary()

    #if lstm.isFresh():

    fit_cron = Chronometer()

    with open(outDirRoot.joinpath("statistics.csv"), "w") as statFile:
        statFile.write("epoch,training_time_mean,training_time_dev,total_time,f1_score\n")

    showPlots = False
    for ep in range(epochs):

        print(f"""
\nEpoch={ep}        
        """)
        # Fitting the model
        fit_cron.start()
        adLSTM.fit(train, train, epochs=1, batchSize=batchSize, verbose=1, validation_data=(val, val), plotTrainingLoss=False)    
        fit_cron.stop()
        print(f"Fitting time: {fit_cron.get_statistics()[0]} +- {fit_cron.get_statistics()[1]}")

        # Saving the model
        # lstm.save("single_feature_model")


        threshold, trainMAE = adLSTM.computeThreshold(train)        
        print(f"Threshold={round(threshold,2)}")


        print(f"Testing...")
        recostructions, testMAE, mask = adLSTM.classify(test)

        #print("recostructions: ",recostructions) 
        #print("maeLosses:", maeLosses)
        #print("testLabels:", testLabels)
        if (ep+1) % 10 == 0:
            outDir = outDirRoot.joinpath("epochs",f"epoch_{ep+1}")
            outDir.mkdir(exist_ok=True, parents=True)
            # Plotting
            adLSTM.setOutputDir(outDir)
            adLSTM.plotTrainingLoss(showFig=showPlots)
            adLSTM.plotPredictions2(test, testLabels, recostructions, testMAE, mask, howMany=30, showFig=showPlots, saveFig=True)
            adLSTM.recoErrorDistributionPlot(trainMAE, threshold=threshold, showFig=showPlots, saveFig=True, type="train")
            adLSTM.recoErrorDistributionPlot(testMAE, threshold=threshold, showFig=showPlots, saveFig=True, type="test")
            adLSTM.plotROC(testLabels, testMAE, showFig=showPlots, saveFig=True)
            f1 = adLSTM.F1Score(testLabels, mask)   
            print("F1 score: ", f1)

            with open(outDirRoot.joinpath("statistics.csv"), "a") as statFile:
                statFile.write(f"{ep+1},{fit_cron.get_statistics()[0]},{fit_cron.get_statistics()[1]},{fit_cron.get_total_elapsed_time()},{f1}\n")

    # adLSTM.plotPrecisionRecall(testLabels, testMAE, showFig=True)  
    print(f"Total time for training: {fit_cron.total_time} seconds")
