#!/bin/bash

if [[ -z $DATASET_CONFIG_FILE ]]; then
    printf "\nExport DATASET_CONFIG_FILE. It must point to the datasets config file.\n\n"
else

    base_path="/data01/homes/baroncelli/phd/rtapipe/analysis/training_output_10_epochs"

    #python submit_to_slurm.py -tmd TRAINED_MODEL_DIR -e EPOCH -pdi PVALUE_DATASET_ID -pn PATTERN_NAME
    #python submit_to_slurm.py -tmd datasetid_601-modelname_m4-trainingtype_heavy-timestamp_20220109-161654 -e 10 -pdi 621 -pn bkg*_te_simtype_bkg_onset_0_normalized_True.csv

    for d in $base_path/* ; do
        printf "\n\nConsidering $d"
        line=$(head -n 1 $d/dataset_params.ini)
        printf "\nDataset: $line"
        pvalid=$(python get_pvalue_dataset_id.py -l $line)
        printf "\nP-val dataset: $pvalid"
        if [ $pvalid -eq "0" ]; then
            printf "\nSkipping this pvalue.."
        else
            cmd="python submit_to_slurm.py -tmd $d -e 10 -pdi $pvalid -pn bkg*_te_simtype_bkg_onset_0_normalized_True.csv"
            printf "\n$cmd\n"
            $($cmd)
        fi
    done

fi