#!/bin/bash

if [ $# -lt 2 ]; 
    then
        printf "\nNot enought arguments supplied.\nArguments: \n\t- integration time\n\t- lenght of the training time series sample.\n\n"
    else

        dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
        today=`date +%Y-%m-%d.%H:%M:%S`
        it=$1
        tsl=$2
        outdir="${dir}/ap_data_for_testing_inspection_date_${today}"

        printf "Generating test set\n"
        printf "\tsimtype=grb\n"
        printf "\tintegration time=$it\n"
        printf "\ttimeseries lenght=$tsl\n"
        printf "\toutputdir=$outdir\n"

        python generate_ap_data.py \
                -dd $DATA/obs/simtype_grb_os_900_tobs_1800_irf_South_z40_average_LST_30m_emin_0.03_emax_1.0_roi_2.5 \
                -t grb \
                -mp yes \
                -it $it \
                -rr 0.2 \
                -norm yes \
                -tsl $tsl \
                -lim 5 \
                -out $outdir
fi