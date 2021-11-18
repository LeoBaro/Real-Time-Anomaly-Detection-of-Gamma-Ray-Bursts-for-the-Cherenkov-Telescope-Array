#!/bin/bash

if [ $# -lt 2 ];
    then
      printf """
Not enought arguments supplied.
        \nArguments:
          - simulation dataset folder
          - integration time
          - lenght of the training time series sample
          - the output directory
          - the number of processes
          - integration type
"""
    else

        today=`date +%Y-%m-%d.%H:%M:%S`
        simdata=$1
        it=$2
        tsl=$3
        outdir="$4/ap_data_bkg_T_${it}_TSL_${tsl}"
        processes=$5
        integration_type=$6

        printf "Generating training set\n"
        printf "\tsimtype=bkg\n"
        printf "\tintegration time=$it\n"
        printf "\ttimeseries lenght=$tsl\n"
        printf "\toutputdir=$outdir\n"
        printf "\tprocesses=$processes\n"
        printf "\tintegration_type=$integration_type\n"

        if [ "$integration_type" = "t" ] || [ "$integration_type" = "te" ]; then

          nohup python generate_ap_data.py \
                  -dd $DATA/obs/$simdata \
                  -t bkg \
                  -mp no \
                  -itype $integration_type \
                  -itime $it \
                  -rr 0.2 \
                  -norm yes \
                  -tsl $tsl \
                  -out $outdir \
                  -proc $processes \
          > nohup.log 2>&1 &

        else

          printf "\nError! Integration type must be t or te\n\n"

        fi
fi
