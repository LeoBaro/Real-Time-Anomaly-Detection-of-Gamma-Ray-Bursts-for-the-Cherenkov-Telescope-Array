#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters. > source clean_obs_dir.sh run0406_ID000126"

else

    out_dir=$1

    if [[ -z "${DATA}" ]]; then
        echo "DATA is undefined. Please, export DATA=.."
    else

        echo "removing *log in out_dir=$1"
        rm "$DATA/obs/$1/*.log"
        rm "$DATA/obs/$1/*_tbin*"

    fi

fi

