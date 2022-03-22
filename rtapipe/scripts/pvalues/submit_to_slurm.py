import os
import argparse
from time import sleep
from pathlib import Path
from shutil import rmtree

def setup_filesystem(args):

    pvalueFolder = Path(args.trained_model_dir).joinpath("epochs", f"epoch_{args.epoch}", "pvalues")

    jobsFileDir = pvalueFolder.joinpath("slurm", "jobs_file")
    jobsOutDir = pvalueFolder.joinpath("slurm", "jobs_out")

    jobsFileDir.mkdir(parents=True, exist_ok=True)
    jobsOutDir.mkdir(parents=True, exist_ok=True)

    return pvalueFolder, jobsFileDir, jobsOutDir



def create_job_file_for_predictions(jobName, args, jobsFileDir, jobsOutDir):

    jobFile = jobsFileDir.joinpath(f"{jobName}.ll")
    jobOut = jobsOutDir.joinpath(f"{jobName}.out")

    argsStr = ""
    for key,val in args.items():
        argsStr += f" --{key} {val}"
    
    with open(jobFile, "w") as jf:
        jf.write("#!/bin/bash\n")
        jf.write(f"#SBATCH --job-name={jobName}\n")
        jf.write(f"#SBATCH --output={jobOut}\n")
        jf.write("#SBATCH --cpus-per-task=1\n")
        jf.write("#SBATCH --partition=small\n")
        jf.write(f"python {Path(__file__).parent}/predict_batch_id.py {argsStr}\n")

    import subprocess
    stdoutdata = subprocess.getoutput(f"sbatch {jobFile}").split()
    if len(stdoutdata) != 4:
        print(f"Error submitting the job: {stdoutdata}")
    else:
        print(stdoutdata)
    return stdoutdata[-1]

def create_job_file_for_pvalue(jobIDs, pvalueFolder, jobsFileDir, jobsOutDir):

    script_path = Path(__file__).parent

    tsDataPath = pvalueFolder.joinpath("jobs")
    tsMergedDataFilePath = pvalueFolder.joinpath("merged_ts_for_pvalues.pickle.npy")

    jobFile = jobsFileDir.joinpath(f"pvalue_job.ll")
    jobOut = jobsOutDir.joinpath(f"pvalue_job.out")

    dependencyStr = "afterok:"
    for jobId in jobIDs:
        dependencyStr += jobId+":"
    dependencyStr = dependencyStr[:-1]

    with open(jobFile, "w") as jf:
        jf.write("#!/bin/bash\n")
        jf.write(f"#SBATCH --job-name=pvalue_job\n")
        jf.write(f"#SBATCH --output={jobOut}\n")
        jf.write("#SBATCH --cpus-per-task=1\n")
        jf.write("#SBATCH --partition=small\n")
        jf.write(f"#SBATCH --dependency={dependencyStr}\n")
        jf.write(f"python {Path(__file__).parent}/merge_ts_files.py -p {tsDataPath}\n")
        jf.write(f"python {Path(__file__).parent}/compute_pvalues.py -p {tsMergedDataFilePath}")
    
    os.system(f"sbatch {jobFile}")

"""
Choose:
    * model
    * epoch
    * the corresponding dataset
    * the corresponding filenames 

python submit_to_slurm.py.py -tmd /data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs/datasetid_601-modelname_m4-trainingtype_heavy-timestamp_20220109-161654 -e 10 -pdi 621 -pn bkg*_te_simtype_bkg_onset_0_normalized_True.csv
"""

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-tmd", "--trained-model-dir", type=str, required=True, help="")
    parser.add_argument("-e", "--epoch", type=int, required=True, help="The epoch of the training")
    parser.add_argument("-pdi", "--pvalue_dataset_id", type=int, required=True, help="The dataset to be used for the p-value computation")
    parser.add_argument("-pn", "--pattern_name", type=str, required=True, help="")
    args = parser.parse_args()

    arguments = vars(args)

    ## Delete current results
    pvalueFolder = Path(args.trained_model_dir).joinpath("epochs", f"epoch_{args.epoch}", "pvalues")
    if pvalueFolder.exists():
        rmtree(pvalueFolder)

    ## Create fresh directories
    pvalueFolder, jobsFileDir, jobsOutDir = setup_filesystem(args)


    ## Split 10 millions files in 10 batches for 10 jobs 
    njobs = 10
    #totalSamplesPerJob = 10000
    #arguments["batch_size"] = 1000

    totalSamplesPerJob = 1000000
    arguments["batch_size"] = 10000

    jobNames = [f"pred_job_{i}" for i in range(njobs)]
    jobsIds = []

    startId = 1
    for jobName in jobNames:
        arguments["from_id"] = startId
        arguments["to_id"] = startId + totalSamplesPerJob
        arguments["output_dir"] = jobName
        startId += totalSamplesPerJob
        jobId = create_job_file_for_predictions(jobName, arguments, jobsFileDir, jobsOutDir)
        jobsIds.append(jobId)

    # Create job for pvalue
    create_job_file_for_pvalue(jobsIds, pvalueFolder, jobsFileDir, jobsOutDir)


