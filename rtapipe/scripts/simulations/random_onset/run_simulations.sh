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

rtapipe_path=/data01/homes/baroncelli/phd/rtapipe

date_suffix=$(date +%Y-%m-%d_%H-%M-%S)
date_suffix="2022-09-21_09-51-20"

output_dir_sim="$this_dir/sim_output_${date_suffix}"
: '
mkdir -p $output_dir_sim

python $cta_sag_sci_path/RTAscience/simGRBcatalogWithRandomization.py  \
                                        -f config.yml  \
                                        --output-dir $output_dir_sim  \
                                        --print yes \
                                        --mp-threads 20
'


output_dir_ap="$this_dir/ap_output_${date_suffix}"
python $rtapipe_path/lib/dataset/generate_ap_data.py \
        -dd $output_dir_sim/run0406_ID000126 \
        -itype t \
        -itime 1 \
        -rr 0.2 \
        -norm no \
        -lim 5 \
        -out $output_dir_ap \
        -proc 20


output_dir_plot="$this_dir/plot_output_${date_suffix}"
python $rtapipe_path/scripts/plot_ap_timeseries.py \
    --dir $output_dir_ap \
    --output-dir $output_dir_plot
