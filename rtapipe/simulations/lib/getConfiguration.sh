#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
CONFIG_FILE_TEMPLATE="$1"
LOG_ID="$2"
SIMTYPE="$3"
LOG_DIR="$DIR/../conf/temp_$LOG_ID"
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
