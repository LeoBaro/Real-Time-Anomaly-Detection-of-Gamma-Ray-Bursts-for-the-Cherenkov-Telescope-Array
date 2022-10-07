#!/bin/bash


./generate_for_training.sh backgrounds_prod5b_1K 1 5 /scratch/baroncelli/AP_DATA_PROD5 10 te yes sim2
sbatch /data01/homes/baroncelli/phd/rtapipe/lib/dataset/job_file.tmp

./generate_for_training.sh backgrounds_prod5b_1K 1 10 /scratch/baroncelli/AP_DATA_PROD5 10 te yes sim2
sbatch /data01/homes/baroncelli/phd/rtapipe/lib/dataset/job_file.tmp

./generate_for_training.sh backgrounds_prod5b_1K 5 5 /scratch/baroncelli/AP_DATA_PROD5 10 te yes sim2
sbatch /data01/homes/baroncelli/phd/rtapipe/lib/dataset/job_file.tmp

./generate_for_training.sh backgrounds_prod5b_1K 5 10 /scratch/baroncelli/AP_DATA_PROD5 10 te yes sim2
sbatch /data01/homes/baroncelli/phd/rtapipe/lib/dataset/job_file.tmp

./generate_for_training.sh backgrounds_prod5b_1K 10 5 /scratch/baroncelli/AP_DATA_PROD5 10 te yes sim2
sbatch /data01/homes/baroncelli/phd/rtapipe/lib/dataset/job_file.tmp

./generate_for_training.sh backgrounds_prod5b_1K 10 10 /scratch/baroncelli/AP_DATA_PROD5 10 te yes sim2
sbatch /data01/homes/baroncelli/phd/rtapipe/lib/dataset/job_file.tmp
