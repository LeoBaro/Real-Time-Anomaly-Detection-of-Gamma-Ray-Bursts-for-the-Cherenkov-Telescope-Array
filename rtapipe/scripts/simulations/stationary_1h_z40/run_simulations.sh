#!/bin/bash

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# path to cta-sag-sci TODO: get from python
cta_sag_sci_path="/data01/homes/baroncelli/phd/cta-sag-sci"

rtapipe_path=/data01/homes/baroncelli/phd/rtapipe

date_suffix=$(date +%Y-%m-%d_%H-%M-%S)

output_dir_sim="$this_dir/sim_output_${date_suffix}"

mkdir -p $output_dir_sim

python $cta_sag_sci_path/RTAscience/simGRBcatalogWithRandomization.py  \
                                        -f config.yml  \
                                        --output-dir $output_dir_sim  \
                                        --print yes \
                                        --mp-threads 6

