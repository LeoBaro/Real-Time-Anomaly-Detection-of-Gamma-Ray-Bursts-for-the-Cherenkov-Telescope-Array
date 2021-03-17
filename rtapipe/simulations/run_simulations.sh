#!/bin/bash

function getConfiguration {
  #for i; do 
  #  echo $i 
  #done  
  DIR="$1"
  LOG_ID="$2"
  SIMTYPE="$3"
  LOG_DIR="$DIR/temp/temp_$LOG_ID"
  ONSET="$4"
  TOBS="$5"
  TRIALS="$6"
  START_COUNT="$7"
  EMIN="$8"
  EMAX="$9"
  ROI="${10}"

  mkdir -p "$LOG_DIR"

  sed "s/simtype: XXX/simtype: $SIMTYPE/g"  $DIR/config.yaml > $LOG_DIR/config.yaml
  sed "s/onset: XXX/onset: $ONSET/g" -i $LOG_DIR/config.yaml
  sed "s/tobs: XXX/tobs: $TOBS/g" -i $LOG_DIR/config.yaml
  sed "s/trials: XXX/trials: $TRIALS/g" -i $LOG_DIR/config.yaml
  sed "s/start_count: XXX/start_count: $START_COUNT/g" -i $LOG_DIR/config.yaml
  sed "s/emin: XXX/emin: $EMIN/g" -i $LOG_DIR/config.yaml
  sed "s/emax: XXX/emax: $EMAX/g" -i $LOG_DIR/config.yaml
  sed "s/roi: XXX/roi: $ROI/g" -i $LOG_DIR/config.yaml

  configFile="$LOG_DIR/config.yaml"

  echo $configFile # this is the return of the function
}

if [[ -z "${DATA}" ]]; then
  echo "DATA is undefined. Please, export DATA=.."
else

  DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


  echo " 
  setup:
    simtype: XXX                      # grb -> src+bkg; bkg -> empty fields
    runid: run0406_ID000126           # can be all or any template (list or str) in catalog
    trials: XXX                       # realisations per runid
    start_count: XXX                  # starting count for seed

  simulation:
    caldb: prod3b                     # calibration database
    irf: South_z40_average_LST_5h    # istrument response function
    tobs: XXX                         # total obs time (s)
    onset: XXX                        # time of bkg only a.k.a. delayed onset of burst (s)
    emin: XXX                        # simulation minimum energy (TeV)
    emax: XXX                        # simulation maximum energy (TeV)
    roi: XXX                          # region of interest radius (deg)

  options:
    set_ebl: True                     # uses the EBL absorbed template
    extract_data: True                # if True extracts lightcurves and spectra 

  path: 
    data: $DATA                       # all data should be under this folder
    ebl: $DATA/ebl_tables/gilmore_tau_fiducial.csv
    model: $DATA/models
    catalog: $DATA/templates/grb_afterglow/GammaCatalogV1.0
  " > $DIR/config.yaml

  log_id=0
  mkdir -p $DIR/logs

  echo "BKG only + GRB+BKG at different offsets"
  for ONSET in 5000
  do
    printf "ONSET: -->$ONSET<--"
    cfgfile=$(getConfiguration $DIR $log_id "grb" $ONSET 18000 1 4 0.03 0.15 2.5)
    printf "cfgfile: -->$cfgfile<--"
    # nohup python ~/phd/repos/cta-sag-sci/RTAscience/simGRB.py  --ncpu 50 --cfgfile $cfgfile 2>&1 > "$DIR/logs/simGRB_$log_id.log" &
    python ~/phd/repos/cta-sag-sci/RTAscience/simGRB.py  --ncpu 5 --cfgfile $cfgfile 2>&1 > "$DIR/logs/simGRB_$log_id.log"

    log_id=$((log_id+1))
  done


  #cfgfile=$(getConfiguration $DIR $log_id "bkg" 0 18000 95 5 0.03 0.15 2.5)
  #printf "cfgfile: -->$cfgfile<--"
  # nohup python ~/phd/repos/cta-sag-sci/RTAscience/simGRB.py  --ncpu 50 --cfgfile $cfgfile 2>&1 > "$DIR/logs/simGRB_$log_id.log" &
  #nohup python ~/phd/repos/cta-sag-sci/RTAscience/simGRB.py  --ncpu 5 --cfgfile $cfgfile 2>&1 > "$DIR/logs/simGRB_$log_id.log" &
  #log_id=$((log_id+1))



  rm $DIR/config.yaml

fi


