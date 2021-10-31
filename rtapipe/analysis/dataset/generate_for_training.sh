#!/bin/bash

if [ $# -lt 2 ];
    then
        printf "\nNot enought arguments supplied.\nArguments:\n\t- dataset folder \n\t- simulation dataset \n\t- integration time\n\t- lenght of the training time series sample\n\tthe output directory\n\n"
    else

        today=`date +%Y-%m-%d.%H:%M:%S`
        simdata=$1
        it=$2
        tsl=$3
        outdir="$4/ap_data_bkg_T_${it}_TSL_${tsl}"

        printf "Generating training set\n"
        printf "\tsimtype=bkg\n"
        printf "\tintegration time=$it\n"
        printf "\ttimeseries lenght=$tsl\n"
        printf "\toutputdir=$outdir\n"

        python generate_ap_data.py \
                -dd $DATA/obs/$simdata \
                -t bkg \
                -mp no \
                -it $it \
                -rr 0.2 \
                -norm yes \
                -tsl $tsl \
                -out $outdir
fi
