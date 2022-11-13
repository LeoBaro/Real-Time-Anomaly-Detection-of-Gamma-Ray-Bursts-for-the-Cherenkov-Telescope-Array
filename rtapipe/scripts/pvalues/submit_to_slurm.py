import os
import pickle
import argparse
import subprocess
from time import sleep
from pathlib import Path
from shutil import rmtree

PARTITION="large"
CPU_PER_TASK = 1

def setup_filesystem(args):

    pvalueFolder = Path(args.trained_model_dir).joinpath("epochs", f"epoch_{args.epoch}", "pvalues")

    jobsFileDir = pvalueFolder.joinpath("slurm", "jobs_file")
    jobsOutDir = pvalueFolder.joinpath("slurm", "jobs_out")
    jobsInputDir = pvalueFolder.joinpath("slurm", "jobs_input")

    jobsFileDir.mkdir(parents=True, exist_ok=True)
    jobsOutDir.mkdir(parents=True, exist_ok=True)
    jobsInputDir.mkdir(parents=True, exist_ok=True)

    return pvalueFolder, jobsFileDir, jobsOutDir, jobsInputDir


def create_input_file_for_predictions(jobName, files, jobsInputDir):
    input_pickle = jobsInputDir.joinpath(f"input_{jobName}.pickle")
    with open(input_pickle, "wb") as ff:
         pickle.dump(files, ff)
    return input_pickle

def create_job_file_for_predictions(jobName, jobsFileDir, jobsOutDir, input_pickle, trained_model_dir, epoch, pvalueFolder, batch_size):

    jobFile = jobsFileDir.joinpath(f"{jobName}.ll")
    jobOut = jobsOutDir.joinpath(f"{jobName}.out")
    
    with open(jobFile, "w") as jf:
        jf.write( "#!/bin/bash\n")
        jf.write(f"#SBATCH --job-name={jobName}\n")
        jf.write(f"#SBATCH --output={jobOut}\n")
        jf.write(f"#SBATCH --cpus-per-task={CPU_PER_TASK}\n")
        jf.write(f"#SBATCH --partition={PARTITION}\n")
        jf.write(f"predict_batch_id -jn {jobName} -plp {input_pickle} -tmd {trained_model_dir} -e {epoch} -od {pvalueFolder} -bs {batch_size} -v 0 \n")

    return jobFile


def main():
    """
    export CTOOLS=/data01/homes/baroncelli/.conda/envs/bphd
    export PYTHONPATH=/data01/homes/baroncelli/phd/cta-sag-sci

    cnn
    submit_to_slurm -tmd /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221112-170625_T_5_TSL_5_multiple_regions/model_AnomalyDetector_cnn_l2_u32_dataset_train_itime_5_a_tsl_5_nbins_3_tsl_3600 -e 39 -pvdp /scratch/baroncelli/DATA/obs/backgrounds_prod5b_10mln/backgrounds  -nf 5000000 -nj 500
 
    rnn
    submit_to_slurm -tmd /data01/homes/baroncelli/phd/rtapipe/notebooks/run_20221112-170625_T_5_TSL_5_multiple_regions/model_AnomalyDetector_rnn_l2_u32_dataset_train_itime_5_a_tsl_5_nbins_3_tsl_3600 -e 15 -pvdp /scratch/baroncelli/DATA/obs/backgrounds_prod5b_10mln/backgrounds  -nf 5000000 -nj 500


    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-tmd", "--trained-model-dir", type=str, required=True, help="")
    parser.add_argument("-e", "--epoch", type=int, required=True, help="The epoch of the training")
    parser.add_argument("-pvdp", "--pvalue_dataset_path", type=str, required=True, help="The dataset to be used for the p-value computation")
    parser.add_argument("-nf", "--num_files", type=int, required=True, help="To limit the total number of input files")
    parser.add_argument("-nj", "--num_jobs", type=int, required=True, help="The max number of jobs. The number of files per job is num_files/num_jobs")
    parser.add_argument("-bs", "--batch_size", type=int, required=False, default=500, help="For each job, the number of files that will be processed with photometry then model predictions")
    parser.add_argument("-s", "--submit", type=int, required=False, default=0, help="If 1, the jobs will be submitted to the cluster")
    args = parser.parse_args()

    arguments = vars(args)

    ## Delete current results
    pvalueFolder = Path(args.trained_model_dir).joinpath("epochs", f"epoch_{args.epoch}", "pvalues")
    if pvalueFolder.exists():
        rmtree(pvalueFolder)

    ## Create fresh directories
    pvalueFolder, jobsFileDir, jobsOutDir, jobsInputDir = setup_filesystem(args)

    print("Loading files...")
    files = [os.path.join(args.pvalue_dataset_path, f) for f in os.listdir(args.pvalue_dataset_path) if f.endswith(".fits")][:args.num_files]

    ## Split 'num_files' files in 'X' batches for 'num_jobs' jobs 
    njobs = args.num_jobs
    total_files = len(files)
    total_samples_per_job = total_files // njobs

    print("Files to be processed: ", len(files))
    print("njobs: ", njobs)
    print("total_samples_per_job: ", total_samples_per_job)

    jobNames = [f"pred_job_{i}" for i in range(njobs)]

    startId = 1
    jobFiles = []
    for jobName in jobNames:
        arguments["output_dir"] = jobName
        input_pickle = create_input_file_for_predictions(jobName, files[startId:startId+total_samples_per_job], jobsInputDir)
        jobFiles.append(
            create_job_file_for_predictions(jobName, jobsFileDir, jobsOutDir, input_pickle, args.trained_model_dir, args.epoch, pvalueFolder, args.batch_size)
        )
        startId += total_samples_per_job

    if args.submit:
        for jobFile in jobFiles:
            stdoutdata = subprocess.getoutput(f"sbatch {jobFile}").split()
            if len(stdoutdata) != 4:
                print(f"Error submitting the job: {stdoutdata}")
            else:
                print(stdoutdata)




if __name__=='__main__':
    main()