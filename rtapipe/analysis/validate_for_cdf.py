import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.analysis.models.anomaly_detector import AnomalyDetector
from multiprocessing import Pool

def validateModel(modelDir, training_dataset_params):

    adLSTM = AnomalyDetector.loadModel(modelDir)
    adLSTM.setFeaturesColsNames(ds.getFeaturesColsNames())

    outfile = Path(args.trained_model_dir).joinpath("epochs", f"epoch_{args.epoch}", "reconstruction_error_on_validation_set.txt")

    print(f"\n*************************************************")
    print(f"Validating model {modelDir}. \nOutput file: {outfile}\nNumber of input files: {training_dataset_params['size']}")

    with open(outfile, 'w') as handle:

        ccontinue = True
        batchsize = 100
        for i in tqdm(range(int(training_dataset_params["size"] / batchsize))):
            ccontinue = ds.loadDataBatch(batchsize, verbose=False)
            train, trainLabels, val, valLabels = ds.getTrainingAndValidationSet(split=1, fitScaler=False, scale=True, verbose=False)
            # print(f"Loaded {batchsize} files, validation set shape: {val.shape}")
            recostructions, maeLossesPerEnergyBin, maeLosses = adLSTM.compute_reconstruction_error(val, verbose=0)
            handle.write('\n'.join(['{:.5f}'.format(num) for num in maeLosses]))
            handle.write('\n')
            if not ccontinue:
                break

    file_data = np.genfromtxt(outfile, usecols=(0), skip_header=1, dtype=float)
    print(file_data)
    # Plotting the pdf and cdf of the recostruction errors on the validation set
    adLSTM.pdfPlot(file_data, filenamePostfix=f"far_validation", showFig=True)
    adLSTM.cdfPlot(file_data, filenamePostfix=f"far_validation", showFig=True)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-tmd", "--trained-model-dir", type=str, required=True, help="")
    parser.add_argument("-e", "--epoch", type=int, required=False, default=-1, help="-1 for testing all epoch")
    parser.add_argument("-di", "--dataset_id", type=int, required=True, help="The dataset to be used as validation set", choices=[1,2,3,4])
    parser.add_argument("-dc", "--dataset_config", type=str, required=False, default="./dataset/config/agilehost3.yml")
    args = parser.parse_args()



    # Loading training dataset params
    with open(Path(args.trained_model_dir).joinpath('dataset_params.pickle'), 'rb') as handle:
        training_dataset_params = pickle.load(handle)
        print(training_dataset_params)

    # Loading the dataset for testing
    ds = APDataset.get_dataset(args.dataset_config, args.dataset_id)
    if not ds.checkCompatibilityWith(training_dataset_params):
        print("The test set is not compatible with the dataset used for training..")
        exit(1)

    ds.setScalerFromPickle(Path(args.trained_model_dir).joinpath('fitted_scaler.pickle'))
    ds.setOutDir(args.trained_model_dir)

    modelDir = Path(args.trained_model_dir).joinpath("epochs",f"epoch_{args.epoch}","lstm_trained_model")
    validateModel(modelDir, ds.dataset_params)
