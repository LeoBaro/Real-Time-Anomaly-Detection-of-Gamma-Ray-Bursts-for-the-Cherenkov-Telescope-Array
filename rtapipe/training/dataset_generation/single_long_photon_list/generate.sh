#!/bin/bash

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# path to cta-sag-sci TODO: get from python
cta_sag_sci_path="/data01/homes/baroncelli/phd/cta-sag-sci"
generate_ap_data_script=$this_dir/../../../lib/dataset/generate_ap_data.py

output_dir_sim="$this_dir/sim_output"
echo "output_dir_sim=$output_dir_sim"

#echo "Generating the photon list.."
#python $cta_sag_sci_path/RTAscience/simGRBcatalogWithRandomization.py -f config.yml --output-dir $output_dir_sim --print no --mp-threads 10 2>&1 > $this_dir/simulation.log

echo "Generating AP data.."
data_dir="$this_dir/sim_output/backgrounds"
ap_output_dir="$this_dir/ap_output"

#T=1 with-normalization
#python "$generate_ap_data_script" -c "$this_dir/config.yml" -dd "$data_dir" -itype te -itime 1 -rr 0.2 -norm yes -mp 50 -out "$ap_output_dir" -proc 10
#T=1 no-normalization
#python "$generate_ap_data_script" -c "$this_dir/config.yml" -dd "$data_dir" -itype te -itime 1 -rr 0.2 -norm no -mp 50 -out "$ap_output_dir" -proc 10

#T=5 with-normalization
#python "$generate_ap_data_script" -c "$this_dir/config.yml" -dd "$data_dir" -itype te -itime 5 -rr 0.2 -norm yes -mp 50 -out "$ap_output_dir" -proc 10
#T=5 no-normalization
#python "$generate_ap_data_script" -c "$this_dir/config.yml" -dd "$data_dir" -itype te -itime 5 -rr 0.2 -norm no -mp 50 -out "$ap_output_dir" -proc 10

#T=10 with-normalization
#python "$generate_ap_data_script" -c "$this_dir/config.yml" -dd "$data_dir" -itype te -itime 10 -rr 0.2 -norm yes -mp 50 -out "$ap_output_dir" -proc 10
#T=10 no-normalization
#python "$generate_ap_data_script" -c "$this_dir/config.yml" -dd "$data_dir" -itype te -itime 10 -rr 0.2 -norm no -mp 50 -out "$ap_output_dir" -proc 10

plot_ap_timeseries -d "$this_dir/ap_output" -o "$this_dir/plot_output"

