#!/bin/bash

function createConfigTemplate {
    if [[ -z "${1}" ]]; then
        printf "Error: no arguments have been passed to the function. Expected: directory to store the configuration file\n"
        echo "" 
    else
        configFilePath="$1/config.yaml"

        echo " 
setup:
    simtype: XXX                      # grb -> src+bkg; bkg -> empty fields; skip -> skip sim; 
                                    # wobble -> LST-like runs (str)
    runid: run0406_ID000126           # can be all or any template (list or str) in catalog
    trials: XXX                       # realisations per runid
    start_count: XXX                  # starting count for seed (int)
    scalefluxfactor: XXX                # scale src nominal flux by factor (float)

simulation:
    caldb: prod3b                     # calibration database
    irf: South_z40_average_LST_30m    # istrument response function
    tobs: XXX                         # total obs time (s)
    onset: XXX                        # time of bkg only a.k.a. delayed onset of burst (s)
    delay: 0                         # delayed start of observation (s) (float)
    emin: XXX                        # simulation minimum energy (TeV)
    emax: XXX                        # simulation maximum energy (TeV)
    roi: XXX                          # region of interest radius (deg)
    offset: 0                      # 'gw' -> from alert; value -> otherwise (deg) (str/float)
    nruns:                        # numer of runs (of lenght=tobs) for wobble simtype (int)


analysis:
    skypix:                       # pixel size in skymap (deg) (float)
    skyroifrac:                   # ratio between skymap axis and roi (float)
    smooth:                       # Gaussian corr. kernel rad. (deg) (float)
    maxsrc:                       # number of hotspot to search for (float)
    sgmthresh: 3                  # blind-search acc. thresh. in Gaussian sigma (float)
    usepnt: yes                   # use pointing for RA/DEC (bool)
    exposure:                     # exposure times for the analysis (s) (float)
    binned: no                    # perform binned or unbinned analysis (bool)
    blind: yes                    # requires blind-search (bool)
    tool: ctools                  # which science tool (str) 
    type: 3d                      # 1d on/off or 3d full-fov (str)
    cumulative: no
    lightcurve: no
    index: -2.1

options:
    set_ebl: True                     # uses the EBL absorbed template
    extract_data: True                # if True extracts lightcurves and spectra 
    plotsky:                      # if True generates skymap plot (bool)

path: 
    data: $DATA                       # all data should be under this folder
    ebl: $DATA/ebl_tables/gilmore_tau_fiducial.csv
    model: $DATA/models
    merger: $DATA/mergers                     # folder of alerts (str)
    bkg: $DATA/models/CTAIrfBackground.xml    # file of background model (str)
    catalog: $DATA/templates/grb_afterglow/GammaCatalogV1.0
        " > $configFilePath
        echo $configFilePath
    fi
}

function getConfiguration {
    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    CONFIG_FILE_TEMPLATE="$1"
    LOG_ID="$2"
    SIMTYPE="$3"
    LOG_DIR="$DIR/conf/temp_$LOG_ID"
    ONSET="$4"
    TOBS="$5"
    TRIALS="$6"
    START_COUNT="$7"
    EMIN="$8"
    EMAX="$9"
    ROI="${10}"
    SCALE="${11}"

    mkdir -p "$LOG_DIR"

    sed "s/simtype: XXX/simtype: $SIMTYPE/g"  $CONFIG_FILE_TEMPLATE > $LOG_DIR/config.yaml
    sed "s/onset: XXX/onset: $ONSET/g" -i $LOG_DIR/config.yaml
    sed "s/tobs: XXX/tobs: $TOBS/g" -i $LOG_DIR/config.yaml
    sed "s/trials: XXX/trials: $TRIALS/g" -i $LOG_DIR/config.yaml
    sed "s/start_count: XXX/start_count: $START_COUNT/g" -i $LOG_DIR/config.yaml
    sed "s/emin: XXX/emin: $EMIN/g" -i $LOG_DIR/config.yaml
    sed "s/emax: XXX/emax: $EMAX/g" -i $LOG_DIR/config.yaml
    sed "s/roi: XXX/roi: $ROI/g" -i $LOG_DIR/config.yaml
    sed "s/scalefluxfactor: XXX/scalefluxfactor: $SCALE/g" -i $LOG_DIR/config.yaml

    configFile="$LOG_DIR/config.yaml"

    echo "$configFile"
}

function startSimulation {
    if [[ -z "${1}" ]]; then
        printf "Error: no arguments have been passed to the function. Expected: the path to the configuration file\n"
        echo "" 
    else
        DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
        printf "Starting simulations..please wait..\n"
        current_time=$(date "+%Y.%m.%d-%H.%M.%S")
        logFile="$DIR/logs/simGRBcatalog_$current_time.log"
        nohup python ~/Workspace/inaf/phd/cta-sag-sci/RTAscience/simBkg.py -f "$1" --mp-enabled "true" --mp-threads 50 2>&1 > "$logFile" &
    fi
}

function message {
    printf "${GREEN}\n${1}${NORMC}\n";
}


if [[ -z "${DATA}" ]]; then
    
    printf "DATA is undefined. Please, export DATA=..\n"

else

    DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
    mkdir -p $DIR/logs


    message "Creating configuration file template"    
    configFileTemplate=$(createConfigTemplate $DIR)  
    printf "Configuration file template=$configFileTemplate\n"

    message "BKG simulations"
    log_id=0
    configFile=$(getConfiguration "$configFileTemplate" "$log_id" "bkg" 0   1800 100 1   0.03 0.15 0.5 1)
    printf "Configuration file=$configFile\n"
    
    message "GRB simulations"
    log_id=$((log_id+1))
    configFile=$(getConfiguration "$configFileTemplate" "$log_id" "grb" 900 1800 100 100 0.03 0.15 0.5 2)
    printf "Configuration file=$configFile\n"
    # TODO startSimulation



    rm $DIR/config.yaml
fi