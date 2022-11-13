#!/bin/bash

# if the number of arguments is less than 1, print the usage and exit
if [ $# -lt 1 ]; then
    printf "Usage:   $0 <do_simulation> <do_photometry> <do_plots>\nExample: $0 1 1 1\n"
    exit 1
fi

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
base_output_dir="$this_dir"
mkdir -p $base_output_dir
echo "base_output_dir=$base_output_dir"

# if first argument is 1
if [ "$1" = "1" ]; then
    echo "Generating the photon list.."
    output_dir_sim="$base_output_dir/sim_output"
    nohup simGRBcatalogWithRandomization -f $this_dir/config.yml --output-dir $output_dir_sim --print no --mp-threads 50 --remove yes 2>&1 > $this_dir/simulation.log &
    echo "When the simulation ends, please call move_fits.sh to move the FITS files to the correct directory."
fi

if [ "$2" = "1" ]; then
    echo "Generating AP data (T=5 with-normalization).."
    data_dir="$base_output_dir/sim_output/backgrounds"
    ap_output_dir="$base_output_dir/ap_output"
    ap_logs_output_dir="$base_output_dir/ap_logs_output"
    mkdir -p $ap_logs_output_dir
    sbatch --output="$ap_logs_output_dir/t5_norm.log" --wrap="generate_ap_data -c $this_dir/config_train_set.yml -dd $data_dir -off 2 -itype te -itime 5 -rr 0.2 -norm yes -out $ap_output_dir -proc 10"
fi

if [ "$3" = "1" ]; then
    echo "Generating plots.."
    plot_ap_timeseries -d "$base_output_dir/ap_output" -o "$base_output_dir/plot_output"
fi
