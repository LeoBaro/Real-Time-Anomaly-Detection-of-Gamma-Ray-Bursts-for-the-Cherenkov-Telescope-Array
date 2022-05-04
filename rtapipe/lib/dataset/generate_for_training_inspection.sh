#!/bin/bash

if [ $# -lt 2 ];
    then
        printf "\nNot enought arguments supplied.\nArguments:\n\t- dataset folder \n\t- integration time\n\t- lenght of the training time series sample.\n\n"
    else

        dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
        today=`date +%Y-%m-%d.%H:%M:%S`
        simdata=$1
        it=$2
        tsl=$3
        outdir="${dir}/ap_data_for_training_inspection_date_${today}"

        printf "Generating training set\n"
        printf "\tsimtype=bkg\n"
        printf "\tintegration time=$it\n"
        printf "\ttimeseries lenght=$tsl\n"
        printf "\toutputdir=$outdir\n"

        python generate_ap_data.py \
                -dd $DATA/obs/$simdata \
                -t bkg \
                -mp yes \
                -it $it \
                -rr 0.2 \
                -norm yes \
                -tsl $tsl \
                -lim 5 \
                -out $outdir
fi
