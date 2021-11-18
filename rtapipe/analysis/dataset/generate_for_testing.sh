#!/bin/bash

if [ $# -lt 2 ];
    then
      printf "\nNot enought arguments supplied.\nArguments:\n\t- simulation dataset folder \n\t- integration time\n\t- lenght of the training time series sample\n\tthe output directory\n\tthe number of processes\n\n"
    else

        today=`date +%Y-%m-%d.%H:%M:%S`
        simdata=$1
        it=$2
        tsl=$3
        outdir="$4/ap_data_grb_T_${it}_TSL_${tsl}"
        processes=$5

        printf "Generating test set\n"
        printf "\tsimtype=grb\n"
        printf "\tintegration time=$it\n"
        printf "\ttimeseries lenght=$tsl\n"
        printf "\toutputdir=$outdir\n"
        printf "\tprocesses=$processes\n"

        python generate_ap_data.py \
                -dd $DATA/obs/$simdata \
                -t grb \
                -mp no \
                -it $it \
                -rr 0.2 \
                -norm yes \
                -tsl $tsl \
                -out $outdir \
                -proc $processes              
fi
