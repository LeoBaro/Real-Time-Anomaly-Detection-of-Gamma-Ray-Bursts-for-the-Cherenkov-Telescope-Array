#!/bin/bash

set -e 

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# path to cta-sag-sci TODO: get from python
cta_sag_sci_path="/data01/homes/baroncelli/phd/cta-sag-sci"

python $cta_sag_sci_path/RTAscience/simGRBcatalog.py -f config.yml --output-dir $this_dir/output


python /data01/homes/baroncelli/phd/rtapipe/lib/dataset/generate_ap_data.py \
        -dd $this_dir/output/run0406_ID000126 \
        -t grb \
        -mp no \
        -itype te \
        -itime 5 \
        -rr 0.2 \
        -norm yes \
        -tsl 10 \
        -lim 5 \
        -out $this_dir/output

python /data01/homes/baroncelli/phd/rtapipe/scripts/plot_ap_timeseries.py \
    --dir $this_dir/output/run0406_ID000126/integration_te_integration_time_5_region_radius_0.2_timeseries_lenght_10/ \
    --output-dir $this_dir/output/run0406_ID000126