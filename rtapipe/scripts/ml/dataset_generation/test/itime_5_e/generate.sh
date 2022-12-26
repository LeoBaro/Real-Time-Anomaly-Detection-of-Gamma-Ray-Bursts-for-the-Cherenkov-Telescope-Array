#!/bin/bash

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Generating the photon list.."
output_dir_sim="$this_dir/sim_output"
mkdir -p $output_dir_sim
nohup simGRBcatalogWithRandomization -f $this_dir/config.yml --output-dir $output_dir_sim --print no --mp-threads 50 --remove yes 2>&1 > $this_dir/simulation.log &
echo "When the simulation ends, please call move_fits.sh to move the FITS files to the correct directory."
