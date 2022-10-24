#!/bin/bash

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# path to cta-sag-sci TODO: get from python
cta_sag_sci_path="/data01/homes/baroncelli/phd/cta-sag-sci"
generate_ap_data_script=$this_dir/../../../lib/dataset/generate_ap_data.py

base_output_dir="$this_dir/training"
mkdir -p $base_output_dir
echo "base_output_dir=$base_output_dir"

output_dir_sim="$base_output_dir/sim_output"

#echo "Generating the photon list.."
#python $cta_sag_sci_path/RTAscience/simGRBcatalogWithRandomization.py -f config_train_set.yml --output-dir $output_dir_sim --print no --mp-threads 10 2>&1 > $this_dir/simulation.log

echo "Generating AP data.."
data_dir="$base_output_dir/sim_output/backgrounds"
ap_output_dir="$base_output_dir/ap_output"
ap_logs_output_dir="$base_output_dir/ap_logs_output"
mkdir -p $ap_logs_output_dir

' :
#T=1 with-normalization
#sbatch --output="$ap_logs_output_dir/t1_norm.log" --wrap="python $generate_ap_data_script -c $this_dir/config_train_set.yml -dd $data_dir -itype te -itime 1 -rr 0.2 -norm yes -out $ap_output_dir -proc 10"
#T=1 no-normalization
sbatch --output="$ap_logs_output_dir/t1_no_norm.log" --wrap="python $generate_ap_data_script -c $this_dir/config_train_set.yml -dd $data_dir -itype te -itime 1 -rr 0.2 -norm no -out $ap_output_dir -proc 10"
#T=5 with-normalization
sbatch --output="$ap_logs_output_dir/t5_norm.log" --wrap="python $generate_ap_data_script -c $this_dir/config_train_set.yml -dd $data_dir -itype te -itime 5 -rr 0.2 -norm yes -out $ap_output_dir -proc 10"
#T=5 no-normalization
sbatch --output="$ap_logs_output_dir/t5_no_norm.log" --wrap="python $generate_ap_data_script -c $this_dir/config_train_set.yml -dd $data_dir -itype te -itime 5 -rr 0.2 -norm no -out $ap_output_dir -proc 10"
#T=10 with-normalization
sbatch --output="$ap_logs_output_dir/t10_norm.log" --wrap="python $generate_ap_data_script -c $this_dir/config_train_set.yml -dd $data_dir -itype te -itime 10 -rr 0.2 -norm yes -out $ap_output_dir -proc 10"
#T=10 no-normalization
sbatch --output="$ap_logs_output_dir/t10_no_norm.log" --wrap="python $generate_ap_data_script -c $this_dir/config_train_set.yml -dd $data_dir -itype te -itime 10 -rr 0.2 -norm no -out $ap_output_dir -proc 10"
'

#echo "Generating plots.."
plot_ap_timeseries -d "$base_output_dir/ap_output" -o "$base_output_dir/plot_output"

