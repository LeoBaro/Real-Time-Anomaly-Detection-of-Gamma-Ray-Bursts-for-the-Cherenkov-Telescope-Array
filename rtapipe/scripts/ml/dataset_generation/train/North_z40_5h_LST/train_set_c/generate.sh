#!/bin/bash

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# if first argument is 1
echo "Generating the photon list.."
output_dir_sim="$this_dir/fits_data"
mkdir -p $output_dir_sim
simGRBcatalogWithRandomization -f $this_dir/config.yml --output-dir $output_dir_sim --print no --remove yes --mp-threads 2 2>&1 > $this_dir/simulation.log

