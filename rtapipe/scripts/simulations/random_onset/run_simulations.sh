#!/bin/bash

# Check if arg exist
: '
if [ -z "$1" ]
then
    echo "No argument supplied"
    exit 1
fi
' 

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# path to cta-sag-sci TODO: get from python
cta_sag_sci_path="/data01/homes/baroncelli/phd/cta-sag-sci"



#date_suffix=$(date +%Y-%m-%d_%H-%M-%S)
date_suffix="dev"

output_dir_sim="$this_dir/sim_output_${date_suffix}"
mkdir -p $output_dir_sim

: '
python $cta_sag_sci_path/RTAscience/simGRBcatalogWithRandomization.py  \
                                        -f config.yml  \
                                        --output-dir $output_dir_sim  \
                                        --print yes \
                                        --mp-threads 20

'

output_dir_ap="$this_dir/ap_output_${date_suffix}"
python /data01/homes/baroncelli/phd/rtapipe/lib/dataset/generate_ap_data.py \
        -dd $output_dir_sim/run0406_ID000126 \
        -t grb \
        -mp no \
        -itype te \
        -itime 1 \
        -rr 0.2 \
        -norm no \
        -tsl 100 \
        -lim 5 \
        -out $output_dir_ap


output_dir_plot="$this_dir/plot_output_${date_suffix}"
python /data01/homes/baroncelli/phd/rtapipe/scripts/plot_ap_timeseries.py \
    --dir $output_dir_ap/run0406_ID000126/integration_te_integration_time_1_region_radius_0.2_timeseries_lenght_100/ \
    --output-dir $output_dir_plot
