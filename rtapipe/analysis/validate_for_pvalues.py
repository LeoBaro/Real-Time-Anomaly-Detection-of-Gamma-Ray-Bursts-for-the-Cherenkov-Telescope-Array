import pickle
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from rtapipe.analysis.dataset.dataset import APDataset
from rtapipe.analysis.models.anomaly_detector import AnomalyDetector
from multiprocessing import Pool
from RTAscience.lib.RTAStats import ts_wilks, p_values, ts_wilks_cumulative

def rm_tree(pth):
    pth = Path(pth)
    for child in pth.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

def validateModel(modelDir, ds):

    adLSTM = AnomalyDetector.loadModel(modelDir)
    adLSTM.setFeaturesColsNames(ds.getFeaturesColsNames())
    training_dataset_params = ds.dataset_params

    outDir = Path(args.trained_model_dir).joinpath("epochs", f"epoch_{args.epoch}", "pvalue_validation")
    if outDir.exists():
        rm_tree(outDir)
    outDir.mkdir(parents=True)

    reconstructionErrorsFile = outDir.joinpath("reconstruction_error_on_validation_set.txt")

    print(f"\n*************************************************")
    print(f"Validating model {modelDir}. \nOutput file: {reconstructionErrorsFile}\nNumber of input files: {training_dataset_params['size']}")

    ccontinue = True
    batchsize = 10000
    for i in tqdm(range(int(training_dataset_params["size"] / batchsize))):

        # Loading a batch of files
        ccontinue = ds.loadDataBatch(batchsize, verbose=False)

        # Computing the network reconstruction errors and write them on a file
        with open(reconstructionErrorsFile, 'a') as handle:
            train, trainLabels, val, valLabels = ds.getTrainingAndValidationSet(split=1, fitScaler=False, scale=True, verbose=False)
            # print(f"Loaded {batchsize} files, validation set shape: {val.shape}")
            recostructions, maeLossesPerEnergyBin, maeLosses = adLSTM.compute_reconstruction_error(val, verbose=0)
            handle.write('\n'.join(['{:.5f}'.format(num) for num in maeLosses]))
            handle.write('\n')

        if not ccontinue:
            break

    data = np.genfromtxt(reconstructionErrorsFile, usecols=(0), skip_header=0, dtype=float)
    print("len(data)=",len(data))
    ts_wilks(data, trials=data.shape[0], nbin=100, width=None, filename = outDir.joinpath(f"ts_wilks_{i}"))
    ts_wilks_cumulative(data, trials=data.shape[0], nbin=100, width=None, filename = outDir.joinpath(f"ts_wilks_cumulative_{i}"))
    p_values(data, trials=data.shape[0], nbin=100, width=None, filename = outDir.joinpath(f"p_values_{i}"))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-tmd", "--trained-model-dir", type=str, required=True, help="")
    parser.add_argument("-e", "--epoch", type=int, required=True)
    parser.add_argument("-di", "--dataset_id", type=int, required=True, help="The dataset to be used as validation set")
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
    validateModel(modelDir, ds)
