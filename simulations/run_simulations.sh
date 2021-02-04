#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

mkdir -p temp

echo " 
setup:
  simtype: XXX                      # grb -> src+bkg; bkg -> empty fields
  runid: run0406_ID000126           # can be all or any template (list or str) in catalog
  trials: 1                         # realisations per runid
  start_count: 0                    # starting count for seed

simulation:
  caldb: prod3b                     # calibration database
  irf: South_z40_average_LST_30m    # istrument response function
  tobs: 1200                        # total obs time (s)
  onset: XXX                        # time of bkg only a.k.a. delayed onset of burst (s)
  emin: 0.03                        # simulation minimum energy (TeV)
  emax: 0.15                        # simulation maximum energy (TeV)
  roi: 2.5                          # region of interest radius (deg)

options:
  set_ebl: True                     # uses the EBL absorbed template
  extract_data: True                # if True extracts lightcurves and spectra 

path: 
  data: $DATA                       # all data should be under this folder
  ebl: $DATA/ebl_tables/gilmore_tau_fiducial.csv
  model: $DATA/models
  catalog: $DATA/templates/grb_afterglow/GammaCatalogV1.0
" > $DIR/config.yaml


echo "Only background simulations"

mkdir -p $DIR/temp/temp1 

sed 's/simtype: XXX/simtype: bkg/g'  $DIR/config.yaml > $DIR/temp/temp1/config.yaml
sed 's/onset: XXX/onset: 0/g' -i $DIR/temp/temp1/config.yaml

python ~/phd/repos/cta-sag-sci/RTAscience/simGRB.py --cfgfile $DIR/temp/temp1/config.yaml



echo "GRB+BKG"

mkdir -p $DIR/temp/temp2 

sed 's/simtype: XXX/simtype: grb/g' $DIR/config.yaml > $DIR/temp/temp2/config.yaml
sed 's/onset: XXX/onset: 0/g' -i $DIR/temp/temp2/config.yaml

python ~/phd/repos/cta-sag-sci/RTAscience/simGRB.py --cfgfile $DIR/temp/temp2/config.yaml



echo "BKG only + GRB+BKG"

mkdir -p $DIR/temp/temp3

sed 's/simtype: XXX/simtype: grb/g' $DIR/config.yaml > $DIR/temp/temp3/config.yaml
sed 's/onset: XXX/onset: 600/g' -i $DIR/temp/temp3/config.yaml

python ~/phd/repos/cta-sag-sci/RTAscience/simGRB.py --cfgfile $DIR/temp/temp3/config.yaml