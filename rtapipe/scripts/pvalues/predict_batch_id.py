import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import json
import tqdm
import pickle
import argparse
import numpy as np
from time import time
from pathlib import Path
from datetime import datetime

from rtapipe.lib.datasource.Photometry3 import OnlinePhotometry
from rtapipe.lib.models.anomaly_detector_builder import AnomalyDetectorBuilder


"""
predict_batch_id -plp /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_cnn_l2_u32_dataset_1201_tsl_5/epochs/epoch_117/pvalues/slurm/jobs_input/input_pred_job_0.pickle -dc /scratch/baroncelli/DATA/obs/backgrounds_prod5b_10mln/backgrounds/config.yaml -tmd /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_cnn_l2_u32_dataset_1201_tsl_5 -e 117 -od /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_cnn_l2_u32_dataset_1201_tsl_5/epochs/epoch_117/pvalues -l 10 -bs 5 

predict_batch_id 
    -plp /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_cnn_l2_u32_dataset_1201_tsl_5/epochs/epoch_117/pvalues/slurm/jobs_input/input_pred_job_0.pickle 
    -dc /scratch/baroncelli/DATA/obs/backgrounds_prod5b_10mln/backgrounds/config.yaml 
    -tmd /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_cnn_l2_u32_dataset_1201_tsl_5 
    -e 117 
    -od /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221027-134533_T_5_TSL_5/model_AnomalyDetector_cnn_l2_u32_dataset_1201_tsl_5/epochs/epoch_117/pvalues 
    -l 10 
    -bs 5 
"""

def load_scaler(scaler_path):
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler

def load_parameters(params_path):
    with open(params_path, "r") as f:
        params = json.load(f)
    return params

REGION_RADIUS=0.2

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-jn", "--job_name", type=str, required=True, help="")
    parser.add_argument("-plp", "--photon_lists_pickled", type=str, required=True, help="")
    parser.add_argument("-dc", "--dataset_config", type=str, required=True, help="The yaml configuration file")
    parser.add_argument("-tmd", "--trained_model_dir", type=str, required=True, help="")
    parser.add_argument("-e", "--epoch", type=int, required=True, help="The epoch of the training")
    parser.add_argument("-od", "--output_dir", type=str, required=True, help="If this is not None or empty, the files containing the reconstruction errors will be placed into this folder.")
    parser.add_argument("-bs", "--batch_size", type=int, required=False, default=0, help="The input photons lists will be divided in batches. Photometry and models predictions will be applied per-batch")
    parser.add_argument("-l", "--limit", type=int, required=False, default=0, help="")
    parser.add_argument("-v", "--verbose", type=int, required=False, default=1, help="")
    args = parser.parse_args()

    now = datetime.now()
    
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Start:", dt_string)	

    # Check input dir
    model_dir = Path(args.trained_model_dir).joinpath("epochs",f"epoch_{args.epoch}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"The directory {model_dir} does not exist!")

    # Load the model parameters
    model_params = load_parameters(Path(args.trained_model_dir).joinpath("model_parameters.json"))

    # Load the trained model
    ad = AnomalyDetectorBuilder.getAnomalyDetector(name=model_params["name"], timesteps=model_params["timesteps"], nfeatures=model_params["nfeatures"], load_model="True", training_epoch_dir=model_dir, training=False)

    # Load the fitted scaler -> must be saved on disk
    scaler = load_scaler(Path(args.trained_model_dir).joinpath("fitted_scaler.pickle"))

    # Load the dataset parameters
    dataset_params = load_parameters(Path(args.trained_model_dir).joinpath("dataset_params.json"))

    # Set output dir
    outputDir = Path(args.output_dir).joinpath("ts_values")
    outputDir.mkdir(parents=True, exist_ok=True)

    # The class that will be used to do photometry
    o_phm = OnlinePhotometry(Path(args.dataset_config))

    # Create the regions configuration for the photometry
    MAX_OFFSET = 2
    REFLECTION = True
    regions_dict = o_phm.create_photometry_configuration(REGION_RADIUS, model_params["nfeatures"], max_offset=MAX_OFFSET, reflection=True)
    
    # Load pickled input
    with open(Path(args.photon_lists_pickled), "rb") as f:
        photon_lists = pickle.load(f)
        if args.limit != 0:
            photon_lists = photon_lists[:args.limit]

    print("Number of photon lists:", len(photon_lists))

    # Create the batches
    if args.batch_size == 0:
        batches = [photon_lists]
    else:
        # create args.batch_size batches
        num_batches = len(photon_lists) // args.batch_size
        batches = []
        for i in range(num_batches):
            batches.append(photon_lists[i*args.batch_size:(i+1)*args.batch_size])
            
    
    print("Number of batches:", len(batches))
    print("Batch size:", args.batch_size)
    
    # TODO: in principle it could be parallelized
    start = time()
    for i, b in enumerate(batches):
        process_batch(i, b, o_phm, regions_dict, scaler, ad, dataset_params, model_params, outputDir, args.verbose, args.job_name)

    print(f"Total time: {time()-start} seconds.")


def process_batch(batch_index, batch, o_phm, regions_dict, scaler, ad, dataset_params, model_params, outputDir, verbose, job_name):

    start = time()

    data = []
    for pht_list in tqdm.tqdm(batch, disable=bool(verbose==0)):           
            
        # Apply photometry with normalization (need to now T and TSL) --> get the object not the csv file!!
        #s = time()
        flux, flux_err = o_phm.integrate(
            pht_list, regions_dict, REGION_RADIUS, dataset_params["itime"], model_params["nfeatures"], model_params["timesteps"], normalize=True, threads=10
        ) 
        #print(f"Photometry took {time()-s} seconds.")
        data.append(flux)

    data = np.concatenate(data, axis=0)
    if verbose:
        print(f"Photometry took {time()-start} seconds.")

    # Apply the scaler
    s = time()
    data = scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    if verbose:
        print(f"Scaler took {time()-s} seconds.")

    # Predict the reconstruction error
    s = time()
    ad.predict(data)
    losses = ad.get_reconstruction_errors()
    if verbose:
        print(len(losses)) 
        print(f"Predict took {time()-s} seconds.")

    # Save the reconstruction error in a file 
    outFile = outputDir.joinpath(f"ts_for_pvalues_{job_name}_batch_{batch_index:05d}.txt")
    np.savetxt(outFile, losses, fmt='%.10f')
    if verbose:
        print(f"File {outFile} saved.")

    end_time_batch = time() - start

    print(f"Total time for batch {end_time_batch} seconds.")

if __name__=='__main__':
    main()