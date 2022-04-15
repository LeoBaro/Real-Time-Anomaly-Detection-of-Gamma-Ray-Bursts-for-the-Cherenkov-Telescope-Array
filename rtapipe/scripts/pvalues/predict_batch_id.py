import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pickle
import shutil
import argparse
import numpy as np
from time import time
from pathlib import Path
from datetime import datetime
from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.analysis.models.anomaly_detector_builder import AnomalyDetectorBuilder

# 1 feature (t)
# python predict_batch_id.py -tmd /data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs/datasetid_600-modelname_m4-trainingtype_heavy-timestamp_20220109-155649 -e 10 -pdi 620 -pn bkg*_t_simtype_bkg_onset_0_normalized_True.csv -fid 1 -tid 101 -bs 10 -od job_1


# 4 features (te)
# python predict_batch_id.py -tmd /data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs/datasetid_601-modelname_m4-trainingtype_heavy-timestamp_20220109-161654 -e 10 -pdi 621 -pn bkg*_te_simtype_bkg_onset_0_normalized_True.csv -fid 1 -tid 101 -bs 10 -od job_1

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-tmd", "--trained_model_dir", type=str, required=True, help="")
    parser.add_argument("-e", "--epoch", type=int, required=True, help="The epoch of the training")
    parser.add_argument("-pdi", "--pvalue_dataset_id", type=int, required=True, help="The dataset to be used for the p-value computation")
    parser.add_argument("-pn", "--pattern_name", type=str, required=True, help="")
    parser.add_argument("-fid", "--from_id", type=int, required=True, help="")
    parser.add_argument("-tid", "--to_id", type=int, required=True, help="")
    parser.add_argument("-bs", "--batch_size", type=int, required=True, help="tid-fid % batchsize == 0")
    parser.add_argument("-od", "--output_dir", type=str, required=True, help="If this is not None or empty, the files containing the reconstruction errors will be placed into this folder.")
    parser.add_argument("-dc", "--dataset_config", type=str, required=False, default="/data01/homes/baroncelli/phd/rtapipe/analysis/dataset/config/agilehost3.yml", help="The configuration file that contains the descriptions of all the datasets")
    args = parser.parse_args()


    now = datetime.now()
    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Start:", dt_string)	

    # Loading training dataset params
    with open(Path(args.trained_model_dir).joinpath('dataset_params.pickle'), 'rb') as handle:
        training_dataset_params = pickle.load(handle)
        # print(training_dataset_params)

    # Loading the dataset for pvalue computation
    ds = APDataset.get_dataset(args.dataset_config, args.pvalue_dataset_id)
    if not ds.checkCompatibilityWith(training_dataset_params):
        print(f"The pvalue dataset {args.pvalue_dataset_id} is not compatible with the dataset used for training..")
        exit(1)

    ds.setScalerFromPickle(Path(args.trained_model_dir).joinpath('fitted_scaler.pickle'))
    ds.setOutDir(args.trained_model_dir)

    modelDir = Path(args.trained_model_dir).joinpath("epochs",f"epoch_{args.epoch}","lstm_trained_model")

    if not modelDir.is_dir():
        raise FileNotFoundError(f"The directory {modelDir} does not exist!")

    # Loading the trained LSTM model
    model_name = args.trained_model_dir.split("modelname_")[1][:2].strip()

    timesteps = training_dataset_params["timeseries_lenght"]
    nfeatures = 1 
    if training_dataset_params["integration_type"] == "t":
        nfeatures = 4 # TODO: write this into the params

    outputDir = Path(args.trained_model_dir).joinpath("epochs",f"epoch_{args.epoch}","pvalues", "jobs", args.output_dir)
    
    s = time()
    adLSTM = AnomalyDetectorBuilder.getAnomalyDetector(model_name, timesteps, nfeatures, outputDir, loadModel=True, modelDir=modelDir)
    print(f"Loading the model took: {time()-s} seconds.")
    
    adLSTM.setFeaturesColsNames(ds.getFeaturesColsNames())

    assert args.to_id > args.from_id

    delta = args.to_id - args.from_id

    assert delta % args.batch_size == 0

    nintervals = delta // args.batch_size

    # print(f"delta: {delta} nintervals: {nintervals}")

    if outputDir.exists():
        shutil.rmtree(outputDir)

    outputDir.mkdir(parents=True, exist_ok=True)

    startID = args.from_id

    for i in range(nintervals):

        print(f"Batch {i} start!", flush=True)

        start = time()

        endID = startID + args.batch_size

        print(f"Loading batch: {startID}-{endID}..")

        testData = ds.loadBatchFromIDs(args.pattern_name, startID, endID)

        s = time()
        _, _, maeLoss = adLSTM.computeReconstructionError(testData, recoErrMean="simple",  verbose=1)
        print(len(maeLoss)) 
        print(f"Predict took {time()-s} seconds.")

        startID = startID + args.batch_size

        outFile = outputDir.joinpath(f"predictions_{i:05d}.txt")

        np.savetxt(outFile, maeLoss, fmt='%.10f')
        
        end_time_batch = time() - start

        print(f"Total time for batch {end_time_batch} seconds.")
        print(f"File {outFile} saved.")
 