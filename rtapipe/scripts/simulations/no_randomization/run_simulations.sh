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

output_dir_sim="$this_dir/sim_output"

mkdir -p $output_dir_sim

python $cta_sag_sci_path/RTAscience/simGRBcatalogWithRandomization.py  \
                                        -f config_grb_onset50.yml  \
                                        --output-dir $output_dir_sim  \
                                        --print yes \
                                        --mp-threads 20

python $cta_sag_sci_path/RTAscience/simGRBcatalogWithRandomization.py  \
                                        -f config_grb_delay50.yml  \
                                        --output-dir $output_dir_sim  \
                                        --print yes \
                                        --mp-threads 20

python $cta_sag_sci_path/RTAscience/simGRBcatalogWithRandomization.py  \
                                        -f config_bkg.yml  \
                                        --output-dir $output_dir_sim  \
                                        --print yes \
                                        --mp-threads 20