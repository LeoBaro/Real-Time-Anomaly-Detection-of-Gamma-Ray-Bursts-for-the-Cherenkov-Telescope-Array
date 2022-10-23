#!/bin/bash

# get the path to the current script
this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

echo "Generating the photon list.."
#simGRBcatalogWithRandomization -f config_bkg.yml --output-dir $this_dir --print no --mp-threads 10 2>&1 > $this_dir/bkg_simulation.log
phlists_to_skymaps --data-dir "$this_dir/backgrounds" --subtraction IRF

#simGRBcatalogWithRandomization -f config_grb_onset_0.yml --output-dir $this_dir --print no --mp-threads 10 2>&1 > $this_dir/grb_simulation_new_code.log
#mv "$this_dir/run0406_ID000126" "$this_dir/run0406_ID000126_onset_0" 
phlists_to_skymaps --data-dir "$this_dir/run0406_ID000126_onset_0" --subtraction IRF

#simGRBcatalogWithRandomization -f config_grb_onset_100.yml --output-dir $this_dir --print no --mp-threads 10 2>&1 > $this_dir/grb_simulation_new_code.log
#mv "$this_dir/run0406_ID000126" "$this_dir/run0406_ID000126_onset_100" 
phlists_to_skymaps --data-dir "$this_dir/run0406_ID000126_onset_100" --subtraction IRF
