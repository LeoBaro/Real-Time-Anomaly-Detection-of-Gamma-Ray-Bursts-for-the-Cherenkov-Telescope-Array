#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

rm -rf $DIR/logs
mkdir -p $DIR/logs

function execute_window () {
    printf "\n\n Command:     nohup python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/$1 -md windowed -wsize $2 -wstep $3 -rad $4 -pl $5 -ov 1 2>&1 > $DIR/logs/$1_$2_$3_$4_$5.log &" 
    nohup python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/$1 -md windowed -wsize $2 -wstep $3 -rad $4 -pl $5 -ov 1 -cc 0 2>&1 > "$DIR/logs/$1_$2_$3_$4_$5.log" &
}

function execute_cumul () {
    printf "\n\n Command:     nohup python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/$1 -md cumulative -wsize $2 -wstep $3 -rad $4 -pl $5 -ov 1 2>&1 > $DIR/logs/$1_$2_$3_$4_$5.log &"
    nohup python pyscripts/phlists_to_photometry_plot.py -obs ~/phd/DATA/obs/$1 -md cumulative -wsize $2 -wstep $3 -rad $4 -pl $5 -ov 1 2>&1 > "$DIR/logs/$1_$2_$3_$4_$5.log" &
}
function execute_for_folder () {
    execute_window "$1" 5 6 0.5 1
    execute_window "$1" 25 26 0.5 1
    execute_window "$1" 25 5 0.5 1
    execute_window "$1" 50 5 0.5 1
    #execute_cumul "obs_st_bkg_tr_1_os_0_emin_0.03_emax_0.15_roi_2.5"  50 0 1 1
}

declare -a folders=( "obs_st_bkg_tr_1_os_0_emin_0.03_emax_0.15_roi_2.5"  
                     "obs_st_grb_tr_1_os_0_emin_0.03_emax_0.15_roi_2.5"  
                     "obs_st_grb_tr_1_os_1800_emin_0.03_emax_0.15_roi_2.5"
                    )



## now loop through the above array
for i in "${folders[@]}"
do
   echo "$i"
   execute_for_folder "$i"

done
